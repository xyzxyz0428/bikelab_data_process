#!/usr/bin/env python3
"""Summarise recorded chrony evidence for the bicycle common time base.

The input files are treated as read-only.  Some legacy CSV writers inserted
the resolved reference IP without adding it to the header and split the
``chronyc sources`` last-sample expression into three numeric columns.  This
script normalises those rows explicitly instead of trusting the legacy header.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACKING_FIELDS = [
    "unix_time",
    "ref_id",
    "reference_ip",
    "stratum",
    "ref_time_utc",
    "system_time_offset_s",
    "last_offset_s",
    "rms_offset_s",
    "frequency_ppm",
    "residual_freq_ppm",
    "skew_ppm",
    "root_delay_s",
    "root_dispersion_s",
    "update_interval_s",
    "leap_status",
]

SOURCE_FIELDS = [
    "unix_time",
    "mode",
    "state",
    "source_ip",
    "stratum",
    "poll",
    "reachability",
    "last_rx_s",
    "raw_sample_offset_s",
    "adjusted_sample_offset_s",
    "sample_error_s",
]


def read_rows(paths: list[Path], expected_columns: int, fields: list[str]) -> list[dict]:
    rows: dict[float, dict] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            next(reader, None)
            for values in reader:
                if len(values) != expected_columns:
                    continue
                row = dict(zip(fields, values))
                try:
                    key = float(row["unix_time"])
                except ValueError:
                    continue
                row["source_file"] = path.name
                rows[key] = row
    return [rows[key] for key in sorted(rows)]


def as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if finite.size else math.nan


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-ip", default="192.168.1.102")
    args = parser.parse_args()

    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    tracking = read_rows(
        sorted(args.input.glob("chrony_tracking*.csv")), 15, TRACKING_FIELDS
    )
    sources = read_rows(
        sorted(args.input.glob("chrony_sources*.csv")), 11, SOURCE_FIELDS
    )
    client_tracking = [
        row for row in tracking if row["reference_ip"] == args.reference_ip
    ]
    client_sources = [
        row
        for row in sources
        if row["source_ip"] == args.reference_ip and row["state"] == "*"
    ]
    if not client_tracking:
        raise RuntimeError(f"No chrony tracking rows for {args.reference_ip}")

    write_csv(
        tables / "computer1_chrony_tracking_normalised.csv",
        client_tracking,
        TRACKING_FIELDS + ["source_file"],
    )
    write_csv(
        tables / "computer1_chrony_sources_normalised.csv",
        client_sources,
        SOURCE_FIELDS + ["source_file"],
    )

    system_offset = np.asarray(
        [as_float(row, "system_time_offset_s") for row in client_tracking]
    )
    last_offset = np.asarray(
        [as_float(row, "last_offset_s") for row in client_tracking]
    )
    rms_offset = np.asarray(
        [as_float(row, "rms_offset_s") for row in client_tracking]
    )
    root_delay = np.asarray(
        [as_float(row, "root_delay_s") for row in client_tracking]
    )
    root_dispersion = np.asarray(
        [as_float(row, "root_dispersion_s") for row in client_tracking]
    )
    unix_time = np.asarray([as_float(row, "unix_time") for row in client_tracking])

    summary = {
        "scope": "Historical short chrony test logs; not a 2026 dataset acquisition session",
        "reference_ip": args.reference_ip,
        "tracking_sample_count": int(len(client_tracking)),
        "unique_clock_update_count": int(
            len({row["ref_time_utc"] for row in client_tracking})
        ),
        "source_sample_count": int(len(client_sources)),
        "capture_span_s": float(unix_time.max() - unix_time.min()),
        "log_start_utc": datetime.fromtimestamp(
            float(unix_time.min()), tz=timezone.utc
        ).isoformat(),
        "log_end_utc": datetime.fromtimestamp(
            float(unix_time.max()), tz=timezone.utc
        ).isoformat(),
        "system_time_offset_abs_median_us": percentile(abs(system_offset) * 1e6, 50),
        "system_time_offset_abs_p95_us": percentile(abs(system_offset) * 1e6, 95),
        "system_time_offset_abs_max_us": percentile(abs(system_offset) * 1e6, 100),
        "last_offset_abs_median_us": percentile(abs(last_offset) * 1e6, 50),
        "last_offset_abs_p95_us": percentile(abs(last_offset) * 1e6, 95),
        "rms_offset_median_us": percentile(rms_offset * 1e6, 50),
        "rms_offset_max_us": percentile(rms_offset * 1e6, 100),
        "root_delay_median_us": percentile(root_delay * 1e6, 50),
        "root_dispersion_median_us": percentile(root_dispersion * 1e6, 50),
        "root_dispersion_max_us": percentile(root_dispersion * 1e6, 100),
        "all_leap_status_normal": all(
            row["leap_status"] == "Normal" for row in client_tracking
        ),
        "all_selected_source_state": all(
            row["state"] == "*" for row in client_sources
        ),
        "all_reachability_377": all(
            row["reachability"] == "377" for row in client_sources
        ),
        "limitations": [
            "The logs contain two short windows rather than a full ride.",
            "The local-reference rows (127.127.1.1) are excluded because they do not measure the Computer 1--Computer 2 offset.",
            "The empty ptp4l log does not quantify LiDAR clock offset.",
            "No saved Tobii NTP offset or request round-trip-time series is available.",
        ],
    }
    (tables / "computer1_computer2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    split_points = np.flatnonzero(np.diff(unix_time) > 30.0) + 1
    segments = np.split(np.arange(len(unix_time)), split_points)
    fig, axes = plt.subplots(
        2, len(segments), figsize=(7.6, 5.2), squeeze=False,
        constrained_layout=True, sharey="row"
    )
    for column, indices in enumerate(segments):
        elapsed = unix_time[indices] - unix_time[indices][0]
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(elapsed, system_offset[indices] * 1e6, "o-", lw=0.9, ms=3.5,
                 label="Estimated system offset")
        top.plot(elapsed, last_offset[indices] * 1e6, "s--", lw=0.8, ms=3.2,
                 label="Last clock update")
        top.axhline(0.0, color="0.45", lw=0.7)
        top.set_title(
            f"Log window {chr(ord('A') + column)} "
            f"({len(indices)} snapshots)", fontsize=9
        )
        bottom.plot(elapsed, rms_offset[indices] * 1e6, "o-", lw=0.9, ms=3.5,
                    label="RMS offset")
        bottom.plot(elapsed, root_dispersion[indices] * 1e6, "s--", lw=0.8, ms=3.2,
                    label="Root dispersion")
        bottom.plot(elapsed, root_delay[indices] * 1e6, "^:", lw=0.8, ms=3.2,
                    label="Root delay")
        bottom.set_xlabel("Elapsed time (s)")
        for axis in (top, bottom):
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Offset (µs)")
    axes[1, 0].set_ylabel("Chrony estimate (µs)")
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="best")
    axes[1, 0].legend(frameon=False, fontsize=7.5, loc="best")
    fig.savefig(figures / "computer1_computer2_chrony_quality.png", dpi=300)
    fig.savefig(figures / "computer1_computer2_chrony_quality.svg")
    plt.close(fig)

    text = (
        "The available historical chrony test record (18 June 2024) contains "
        f"{summary['tracking_sample_count']} logger snapshots in two short windows, "
        f"representing only {summary['unique_clock_update_count']} distinct chrony updates. "
        "Computer 1 selected Computer 2 (192.168.1.102) as its time source, "
        "and the source reachability register was 377 in all saved source rows. "
        "The absolute estimated system-clock offset snapshots ranged from "
        f"{float(np.min(abs(system_offset) * 1e6)):.3f} to "
        f"{summary['system_time_offset_abs_max_us']:.3f} microseconds. "
        f"The median chrony RMS offset was {summary['rms_offset_median_us']:.3f} "
        "microseconds. These values demonstrate the available Computer 1--Computer 2 "
        "test condition. They do not validate the 2026 dataset sessions or establish "
        "stability over a complete ride because continuous per-session logs were not recorded."
    )
    (args.output / "computer1_computer2_result_text.txt").write_text(
        text + "\n", encoding="utf-8"
    )
    (args.output / "figure_caption.txt").write_text(
        "Computer 1--Computer 2 clock alignment in two available historical chrony test "
        "windows. The upper panels show the estimated system offset and the most recent "
        "clock update relative to Computer 2. The lower panels show the chrony RMS offset, "
        "root dispersion, and network root delay. The record is dated 18 June 2024 and is "
        "a representative setup check rather than a continuous 2026 session assessment.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
