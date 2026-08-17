#!/usr/bin/env python3
"""Analyse a formal host/LiDAR/Tobii synchronisation recording.

The input log tree is read-only.  The script writes a separate analysis
directory containing summary tables, figures and a short assessment.  For the
engineering analysis, the signed LiDAR offset fields are interpreted as
nanoseconds and converted to microseconds.  This convention is explicit in
the output and is not a substitute for a vendor unit specification.  The
Tobii device-host difference remains an API round-trip estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_STYLE_DIR = SCRIPT_DIR.parents[2] / "data_analysis" / "scripts"
if str(PAPER_STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_STYLE_DIR))
from paper_style import (  # noqa: E402
    COLORS,
    apply_paper_style,
    save_figure as save_paper_figure,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_time(row: dict[str, str]) -> datetime:
    value = row.get("host_midpoint_utc") or row.get("host_utc_midpoint")
    if not value:
        raise ValueError("No host midpoint timestamp in row")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds(rows: Sequence[dict[str, str]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    origin = parse_time(rows[0])
    return np.asarray([(parse_time(row) - origin).total_seconds() for row in rows])


def elapsed_from(rows: Sequence[dict[str, str]], origin: datetime) -> np.ndarray:
    return np.asarray([(parse_time(row) - origin).total_seconds() for row in rows])


def numbers(rows: Iterable[dict[str, str]], field: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def quantile(values: np.ndarray, q: float) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else None


def metric_dict(values: np.ndarray, scale: float = 1.0) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)] * scale
    if not values.size:
        return {"count": 0, "median": None, "p95": None, "minimum": None, "maximum": None}
    return {
        "count": int(values.size),
        "median": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def interval_metrics(rows: Sequence[dict[str, str]]) -> dict[str, float | int | None]:
    if len(rows) < 2:
        return {"count": len(rows), "median_s": None, "p95_s": None, "maximum_s": None, "rate_hz": None, "span_s": 0.0}
    times = np.asarray([parse_time(row).timestamp() for row in rows])
    intervals = np.diff(times)
    span = float(times[-1] - times[0])
    return {
        "count": int(len(rows)),
        "median_s": float(np.percentile(intervals, 50)),
        "p95_s": float(np.percentile(intervals, 95)),
        "maximum_s": float(np.max(intervals)),
        "rate_hz": float((len(rows) - 1) / span) if span > 0 else None,
        "span_s": span,
    }


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def find_file(root: Path, filename: str, *, contains: str | None = None) -> Path:
    candidates = sorted(root.rglob(filename))
    if contains:
        candidates = [path for path in candidates if contains in str(path)]
    if not candidates:
        raise FileNotFoundError(f"Cannot find {filename} under {root}")
    return candidates[0]


def read_ptp_counts(pcap: Path) -> tuple[list[dict[str, str]], dict[str, int], dict[str, int], str]:
    if not pcap.exists() or shutil.which("tshark") is None:
        return [], {}, {}, ""
    command = [
        "tshark", "-r", str(pcap), "-T", "fields", "-E", "separator=,",
        "-e", "ip.src", "-e", "ip.dst", "-e", "ptp.v2.messagetype",
        "-e", "ptp.v2.an.grandmasterclockidentity",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    source_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    gm_counts: Counter[str] = Counter()
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        values = line.split(",")
        values += [""] * (4 - len(values))
        source, destination, message_type, grandmaster = [value.strip() for value in values[:4]]
        if source:
            source_counts[source] += 1
        if message_type:
            type_counts[message_type] += 1
        if grandmaster:
            gm_counts[grandmaster] += 1
        rows.append({"source": source, "destination": destination, "message_type": message_type, "grandmaster": grandmaster})
    write_gm = gm_counts.most_common(1)[0][0] if gm_counts else ""
    return rows, dict(source_counts), dict(type_counts), write_gm


def configure_matplotlib() -> None:
    apply_paper_style()


def save_figure(fig: plt.Figure, figures: Path, name: str) -> None:
    save_paper_figure(fig, figures / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", default="sync_20260814_formal_02")
    args = parser.parse_args()

    root = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    c1_path = find_file(root, "host_sync_samples.csv", contains="computer1")
    c2_path = find_file(root, "host_sync_samples.csv", contains="computer2")
    lidar_path = find_file(root, "robosense_http_ptp_status.csv")
    tobii_path = find_file(root, "tobii_time_sync_samples.csv")
    pcap_path = next(iter(sorted(root.rglob("ptp_control_packets.pcap"))), None)

    c1, c2 = read_csv(c1_path), read_csv(c2_path)
    lidar, tobii = read_csv(lidar_path), read_csv(tobii_path)
    lidar_by_sensor = {label: [row for row in lidar if row.get("sensor_label") == label] for label in ("near", "front", "rear")}
    all_streams: dict[str, list[dict[str, str]]] = {
        "Computer 1 chrony logger": c1,
        "Computer 2 PTP logger": c2,
        "Near LiDAR HTTP status": lidar_by_sensor["near"],
        "Front LiDAR HTTP status": lidar_by_sensor["front"],
        "Rear LiDAR HTTP status": lidar_by_sensor["rear"],
        "Tobii time logger": tobii,
    }
    all_times = [parse_time(row) for rows in all_streams.values() for row in rows]
    global_start, global_end = min(all_times), max(all_times)
    common_start = max(parse_time(rows[0]) for rows in all_streams.values())
    common_end = min(parse_time(rows[-1]) for rows in all_streams.values())

    sampling_rows = []
    for name, rows in all_streams.items():
        m = interval_metrics(rows)
        sampling_rows.append({"stream": name, **m, "start_utc": parse_time(rows[0]).isoformat(), "end_utc": parse_time(rows[-1]).isoformat()})
    write_csv(tables / "sampling_summary.csv", sampling_rows, ["stream", "count", "median_s", "p95_s", "maximum_s", "rate_hz", "span_s", "start_utc", "end_utc"])

    c1_offset = numbers(c1, "system_time_offset_s")
    c1_last = numbers(c1, "last_offset_s")
    c1_rms = numbers(c1, "rms_offset_s")
    c1_disp = numbers(c1, "root_dispersion_s")
    c1_summary = {
        "sample_count": len(c1),
        "ntp_synchronized_counts": dict(Counter(row.get("ntp_synchronized", "") for row in c1)),
        "selected_source_counts": dict(Counter(row.get("selected_source", "") for row in c1)),
        "system_time_offset_us": metric_dict(c1_offset, 1e6),
        "absolute_system_time_offset_us": metric_dict(np.abs(c1_offset), 1e6),
        "absolute_last_offset_us": metric_dict(np.abs(c1_last), 1e6),
        "rms_offset_us": metric_dict(c1_rms, 1e6),
        "root_dispersion_us": metric_dict(c1_disp, 1e6),
    }
    (tables / "computer1_computer2_summary.json").write_text(json.dumps(c1_summary, indent=2), encoding="utf-8")

    lidar_summary_rows = []
    lidar_normalized_rows = []
    for label, rows in lidar_by_sensor.items():
        raw_offset_ns = numbers(rows, "ptp_master_offset_raw")
        offset_us = raw_offset_ns / 1000.0
        if label == "near":
            device_type = "Bpearl"
            timing_source = "PTP-E2E-L4"
            ptp_domain = "0"
            field_name = "time_sync_data"
        else:
            device_type = "RoboSense RS-Helios"
            timing_source = "PTP-E2E"
            ptp_domain = "not recorded"
            field_name = "ptp_master_offset"
        lidar_summary_rows.append({
            "sensor": label,
            "ip": rows[0].get("sensor_ip", "") if rows else "",
            "device_type": device_type,
            "offset_field": field_name,
            "time_sync_source_config": timing_source,
            "ptp_domain": ptp_domain,
            "samples": len(rows),
            "http_200_fraction": sum(row.get("http_status") == "200" for row in rows) / len(rows) if rows else None,
            "ptp_locked_fraction": sum(row.get("ptp_status") == "Locked" for row in rows) / len(rows) if rows else None,
            "phase_locked_fraction": sum(row.get("phase_lock_status") == "Locked" for row in rows) / len(rows) if rows else None,
            "offset_ns_median": quantile(raw_offset_ns, 50),
            "offset_ns_p95_abs": quantile(np.abs(raw_offset_ns), 95),
            "offset_ns_min": float(np.min(raw_offset_ns)) if raw_offset_ns.size else None,
            "offset_ns_max": float(np.max(raw_offset_ns)) if raw_offset_ns.size else None,
            "offset_us_median": quantile(offset_us, 50),
            "offset_us_p95_abs": quantile(np.abs(offset_us), 95),
            "offset_us_min": float(np.min(offset_us)) if offset_us.size else None,
            "offset_us_max": float(np.max(offset_us)) if offset_us.size else None,
            "conversion_note": "raw field interpreted as ns; us = ns / 1000",
            "time_sync_source": ";".join(sorted(set(row.get("time_sync_source", "") for row in rows))),
        })
        for row in rows:
            try:
                offset_ns_value = float(row.get("ptp_master_offset_raw", ""))
            except (TypeError, ValueError):
                offset_ns_value = math.nan
            normalized = {
                "sample_index": row.get("sample_index", ""),
                "sensor_label": label,
                "sensor_ip": row.get("sensor_ip", ""),
                "host_midpoint_utc": row.get("host_midpoint_utc", ""),
                "ptp_status": row.get("ptp_status", ""),
                "time_sync_source": row.get("time_sync_source", ""),
                "time_sync_mode_reported": row.get("time_sync_mode_reported", ""),
                "ptp_master_offset_raw": row.get("ptp_master_offset_raw", ""),
                "ptp_master_offset_ns": "" if label == "near" or not math.isfinite(offset_ns_value) else f"{offset_ns_value:.0f}",
                "ptp_master_offset_us": "" if label == "near" or not math.isfinite(offset_ns_value) else f"{offset_ns_value / 1000.0:.6f}",
                "ptp_time_sync_data_ns": "" if label != "near" or not math.isfinite(offset_ns_value) else f"{offset_ns_value:.0f}",
                "ptp_time_sync_data_us": "" if label != "near" or not math.isfinite(offset_ns_value) else f"{offset_ns_value / 1000.0:.6f}",
                "unit_convention": "nanoseconds; microseconds = nanoseconds / 1000",
            }
            lidar_normalized_rows.append(normalized)
    write_csv(tables / "lidar_status_summary.csv", lidar_summary_rows, list(lidar_summary_rows[0].keys()))
    write_csv(
        tables / "lidar_status_normalized.csv",
        lidar_normalized_rows,
        [
            "sample_index", "sensor_label", "sensor_ip", "host_midpoint_utc",
            "ptp_status", "time_sync_source", "time_sync_mode_reported",
            "ptp_master_offset_raw", "ptp_master_offset_ns", "ptp_master_offset_us",
            "ptp_time_sync_data_ns", "ptp_time_sync_data_us", "unit_convention",
        ],
    )

    tobii_offset = numbers(tobii, "device_minus_host_midpoint_ms")
    tobii_time = np.asarray([parse_time(row).timestamp() for row in tobii])
    tobii_slope_ms_per_s = float(np.polyfit(tobii_time - tobii_time[0], tobii_offset, 1)[0]) if len(tobii_offset) > 1 else None
    edge_count = max(1, len(tobii_offset) // 20)
    tobii_first_edge_median = float(np.median(tobii_offset[:edge_count])) if tobii_offset.size else None
    tobii_last_edge_median = float(np.median(tobii_offset[-edge_count:])) if tobii_offset.size else None
    tobii_summary = {
        "sample_count": len(tobii),
        "ntp_enabled_counts": dict(Counter(row.get("ntp_is_enabled", "") for row in tobii)),
        "ntp_synchronized_counts": dict(Counter(row.get("ntp_is_synchronized", "") for row in tobii)),
        "error_count": sum(bool(row.get("error", "")) for row in tobii),
        "device_minus_host_midpoint_ms": metric_dict(tobii_offset),
        "device_minus_host_midpoint_us": metric_dict(tobii_offset, 1000.0),
        "linear_trend_ms_per_s": tobii_slope_ms_per_s,
        "first_5_percent_median_ms": tobii_first_edge_median,
        "last_5_percent_median_ms": tobii_last_edge_median,
        "first_to_last_5_percent_change_ms": (
            tobii_last_edge_median - tobii_first_edge_median
            if tobii_first_edge_median is not None and tobii_last_edge_median is not None else None
        ),
        "device_request_rtt_ms": metric_dict(numbers(tobii, "device_time_request_rtt_ms")),
        "ntp_state_request_rtt_ms": metric_dict(numbers(tobii, "ntp_synchronized_request_rtt_ms")),
        "interpretation": "API device-host midpoint difference; not an independent NTP clock-offset measurement",
    }
    (tables / "tobii_summary.json").write_text(json.dumps(tobii_summary, indent=2), encoding="utf-8")

    ptp_rows, ptp_sources, ptp_types, grandmaster = read_ptp_counts(pcap_path) if pcap_path else ([], {}, {}, "")
    write_csv(tables / "ptp_packet_source_counts.csv", [{"source": k, "packets": v} for k, v in sorted(ptp_sources.items())], ["source", "packets"])
    ptp_type_names = {
        "0x00": "Sync",
        "0x01": "Delay_Req",
        "0x08": "Follow_Up",
        "0x09": "Delay_Resp",
        "0x0b": "Announce",
    }
    write_csv(
        tables / "ptp_message_type_counts.csv",
        [{"message_type": k, "message_name": ptp_type_names.get(k, "Unknown"), "packets": v} for k, v in sorted(ptp_types.items())],
        ["message_type", "message_name", "packets"],
    )

    coverage = {
        "session_id": args.session_id,
        "global_start_utc": global_start.isoformat(),
        "global_end_utc": global_end.isoformat(),
        "common_overlap_start_utc": common_start.isoformat(),
        "common_overlap_end_utc": common_end.isoformat(),
        "common_overlap_s": (common_end - common_start).total_seconds(),
        "common_overlap_fraction_of_30_min": (common_end - common_start).total_seconds() / 1800.0,
        "log_calendar_date_observed": global_start.date().isoformat(),
        "session_id_calendar_date": args.session_id.split("_")[1] if "_" in args.session_id else "",
        "ptp_capture_packet_count": len(ptp_rows),
        "ptp_grandmaster_clock_identity": grandmaster,
        "ptp_source_packet_counts": ptp_sources,
        "ptp_message_type_counts": ptp_types,
    }
    (tables / "coverage_and_ptp_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    # Coverage figure.
    labels = list(all_streams)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["orange"], COLORS["purple"], COLORS["vermillion"]]
    for index, (label, rows) in enumerate(all_streams.items()):
        start = (parse_time(rows[0]) - global_start).total_seconds()
        end = (parse_time(rows[-1]) - global_start).total_seconds()
        ax.plot([start, end], [index, index], lw=7, solid_capstyle="butt", color=colors[index])
        ax.plot([start, end], [index, index], "|", color=COLORS["black"], ms=8, mew=0.8)
        ax.text(end - 12, index, f"{len(rows):,}", va="center", ha="right", color="white", fontsize=8, fontweight="bold")
    common_a = (common_start - global_start).total_seconds()
    common_b = (common_end - global_start).total_seconds()
    ax.axvspan(common_a, common_b, color=COLORS["purple"], alpha=0.10)
    ax.text(
        (common_a + common_b) / 2,
        len(labels) - 0.35,
        f"All-stream overlap: {(common_end-common_start).total_seconds():.1f} s",
        ha="center",
        va="bottom",
        fontsize=8,
        color=COLORS["purple"],
    )
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_title("Formal synchronisation recording coverage")
    ax.set_xlim(-70, (global_end - global_start).total_seconds() + 90)
    ax.set_ylim(-0.5, len(labels) - 0.02)
    save_figure(fig, figures, "synchronisation_recording_coverage")

    # Computer 1/2 NTP figure. Chrony is the host-side implementation, while
    # NTP is the protocol reported in the figures.
    c1_t = elapsed_from(c1, global_start)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    axes[0].plot(c1_t, c1_offset * 1e6, color=COLORS["blue"], lw=1.35, label="Estimated system offset")
    axes[0].plot(c1_t, c1_last * 1e6, color=COLORS["orange"], lw=1.2, alpha=0.9, label="Last clock update")
    axes[0].axhline(0, color=COLORS["black"], lw=0.8)
    axes[0].set_ylabel("Offset (µs)")
    axes[0].set_title("Computer 1 synchronisation to Computer 2 (NTP)")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[1].plot(c1_t, c1_rms * 1e6, color=COLORS["green"], lw=1.35, label="RMS offset")
    axes[1].plot(c1_t, c1_disp * 1e6, color=COLORS["purple"], lw=1.2, label="Root dispersion")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("NTP estimate (µs)")
    axes[1].legend(frameon=False, ncol=2, loc="upper right")
    save_figure(fig, figures, "computer1_computer2_chrony_offset")

    # Tobii figure.  Offset axes use microseconds throughout the synchronisation
    # figures; the raw API field is recorded in milliseconds.
    tobii_t = elapsed_from(tobii, global_start)
    tobii_offset_us = tobii_offset * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    axes[0].plot(tobii_t, tobii_offset_us, color=COLORS["vermillion"], lw=1.35)
    axes[0].set_ylabel("Device − host midpoint (µs)")
    axes[0].set_title("Tobii Glasses 3 synchronisation status and API time comparison")
    axes[0].grid(alpha=0.25)
    sync = np.asarray([1.0 if row.get("ntp_is_synchronized") == "True" else 0.0 for row in tobii])
    axes[1].step(tobii_t, sync, where="post", color=COLORS["green"], lw=1.35)
    axes[1].set_yticks([0, 1], ["False", "True"])
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("NTP synchronized")
    axes[1].grid(alpha=0.25)
    save_figure(fig, figures, "tobii_ntp_and_device_host_time")

    # LiDAR status and raw vendor field.
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True)
    palette = {"near": COLORS["blue"], "front": COLORS["orange"], "rear": COLORS["purple"]}
    lidar_plot_labels = {"near": "Bpearl near", "front": "Helios front", "rear": "Helios rear"}
    status_y = {"near": 1.08, "front": 1.0, "rear": 0.92}
    for label, rows in lidar_by_sensor.items():
        t = elapsed_from(rows, global_start)
        locked = np.asarray([1 if row.get("ptp_status") == "Locked" else 0 for row in rows])
        axes[0].step(t, locked * status_y[label], where="mid", color=palette[label], lw=1.0, label=lidar_plot_labels[label])
        raw_ns = numbers(rows, "ptp_master_offset_raw")
        axes[1].plot(t[: len(raw_ns)], raw_ns / 1000.0, color=palette[label], lw=0.75, label=lidar_plot_labels[label])
    axes[0].set_yticks([0, 1], ["Not locked", "Locked"])
    axes[0].set_ylim(-0.1, 1.2)
    axes[0].set_ylabel("Vendor PTP status")
    axes[0].set_title("LiDAR PTP lock status and interpreted master-offset fields")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    axes[1].set_xlabel("Elapsed time (s)")
    axes[1].set_ylabel("Master offset (µs; raw ns / 1000)")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")
    save_figure(fig, figures, "lidar_ptp_status_and_reported_offset")

    # Sampling interval comparison.
    interval_labels, interval_values = [], []
    for name, rows in all_streams.items():
        if len(rows) > 1:
            times = np.asarray([parse_time(row).timestamp() for row in rows])
            interval_labels.append(name.replace(" ", "\n", 1))
            interval_values.append(np.diff(times))
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    boxplot_options = {
        "showfliers": False,
        "patch_artist": True,
        "boxprops": {"facecolor": COLORS["sky"], "edgecolor": COLORS["blue"], "linewidth": 0.8},
        "whiskerprops": {"color": COLORS["blue"], "linewidth": 0.8},
        "capprops": {"color": COLORS["blue"], "linewidth": 0.8},
        "medianprops": {"color": COLORS["black"], "linewidth": 1.0},
    }
    try:
        ax.boxplot(interval_values, tick_labels=interval_labels, **boxplot_options)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(interval_values, labels=interval_labels, **boxplot_options)
    ax.set_ylabel("Observed interval (s)")
    ax.set_title("Sampling intervals of the formal synchronisation loggers")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save_figure(fig, figures, "logger_sampling_intervals")

    # PTP packet overview.
    if ptp_types:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
        type_names = list(ptp_types)
        type_labels = [f"{x}\n{ptp_type_names.get(x, 'Unknown')}" for x in type_names]
        axes[0].bar(type_labels, [ptp_types[x] for x in type_names], color=COLORS["blue"])
        axes[0].set_xlabel("PTP message type")
        axes[0].set_ylabel("Packets")
        axes[0].set_title("PTP message counts")
        source_names = list(ptp_sources)
        axes[1].bar(source_names, [ptp_sources[x] for x in source_names], color=COLORS["green"])
        axes[1].set_xlabel("Source IP")
        axes[1].set_ylabel("Packets")
        axes[1].set_title("PTP packet sources")
        for axis in axes:
            axis.tick_params(axis="x", rotation=35)
        save_figure(fig, figures, "ptp_packet_overview")

    # Main-paper figure: four compact panels covering the complete evidence
    # chain.  The individual diagnostic figures above remain available for
    # audit and supplementary material, but this is the paper-facing figure.
    # All x axes use elapsed seconds from the earliest logger sample.  All
    # clock-offset axes use microseconds; the Tobii API value is converted from
    # milliseconds by multiplying by 1000.
    duration_s = (global_end - global_start).total_seconds()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.35))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.22, top=0.91, wspace=0.30, hspace=0.42)

    # (a) Logger coverage.
    ax = axes[0, 0]
    coverage_labels = {
        "Computer 1 chrony logger": "C1 NTP",
        "Computer 2 PTP logger": "C2 PTP",
        "Near LiDAR HTTP status": "Near LiDAR",
        "Front LiDAR HTTP status": "Front LiDAR",
        "Rear LiDAR HTTP status": "Rear LiDAR",
        "Tobii time logger": "Tobii NTP",
    }
    coverage_colors = {
        "Computer 1 chrony logger": COLORS["blue"],
        "Computer 2 PTP logger": COLORS["black"],
        "Near LiDAR HTTP status": COLORS["orange"],
        "Front LiDAR HTTP status": COLORS["green"],
        "Rear LiDAR HTTP status": COLORS["purple"],
        "Tobii time logger": COLORS["vermillion"],
    }
    for index, (name, rows) in enumerate(all_streams.items()):
        start = (parse_time(rows[0]) - global_start).total_seconds()
        end = (parse_time(rows[-1]) - global_start).total_seconds()
        ax.plot([start, end], [index, index], lw=5.5, solid_capstyle="butt",
                color=coverage_colors[name])
        ax.plot([start, end], [index, index], "|", color=COLORS["black"],
                ms=6, mew=0.7)
    common_a = (common_start - global_start).total_seconds()
    common_b = (common_end - global_start).total_seconds()
    ax.axvspan(common_a, common_b, color=COLORS["purple"], alpha=0.10,
               label="All-stream overlap")
    ax.text((common_a + common_b) / 2.0, len(all_streams) - 0.42,
            f"{(common_end - common_start).total_seconds():.0f} s overlap",
            ha="center", va="top", fontsize=7.5, color=COLORS["purple"])
    ax.set_yticks(range(len(all_streams)), [coverage_labels[name] for name in all_streams])
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Recorded stream")
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.6, len(all_streams) - 0.35)
    ax.grid(axis="x")
    ax.text(0.02, 1.02, "(a) Recording coverage", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, fontweight="normal",
            clip_on=False)

    # (b) C1--C2 NTP. Plot the signed device-host offset. The annotation
    # reports mean and P95 magnitudes so that the sign is not lost in the
    # summary statistic.
    ax = axes[0, 1]
    c1_t_global = elapsed_from(c1, global_start)
    c1_plot_offset_us = c1_offset * 1e6
    ax.plot(c1_t_global, c1_plot_offset_us, color=COLORS["blue"], lw=1.0)
    ax.axhline(0, color=COLORS["black"], lw=0.7)
    chrony_sync_count = sum(str(row.get("ntp_synchronized", "")).lower() in {"yes", "true", "1"} for row in c1)
    chrony_sync_fraction = chrony_sync_count / len(c1) if c1 else 0.0
    ax.text(
        0.98,
        0.97,
        f"|device–host offset| mean {np.mean(np.abs(c1_plot_offset_us)):.3f} µs\n"
        f"|device–host offset| P95 {np.percentile(np.abs(c1_plot_offset_us), 95):.3f} µs\n"
        f"NTP synchronised: {chrony_sync_count}/{len(c1)} ({chrony_sync_fraction * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        fontweight="normal",
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(0.02, 1.02, "(b) Computer 1–2 NTP", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, fontweight="normal",
            clip_on=False)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Device–host offset (µs)")
    ax.set_xlim(0, duration_s)

    # (c) LiDAR PTP.  Status is represented explicitly in the annotation and
    # in each legend entry; the curves show the vendor-reported timing field.
    ax = axes[1, 0]
    lidar_palette = {"near": COLORS["orange"], "front": COLORS["green"], "rear": COLORS["purple"]}
    lidar_labels = {
        "near": "Near Bpearl (Locked)",
        "front": "Front Helios (Locked)",
        "rear": "Rear Helios (Locked)",
    }
    lidar_stats = {}
    for label, rows in lidar_by_sensor.items():
        t = elapsed_from(rows, global_start)
        raw_ns = numbers(rows, "ptp_master_offset_raw")
        count = min(t.size, raw_ns.size)
        if count:
            offset_us = raw_ns[:count] / 1000.0
            ax.plot(t[:count], offset_us,
                    color=lidar_palette[label], lw=0.9, label=lidar_labels[label])
            lidar_stats[label] = {
                "mean_abs_us": float(np.mean(np.abs(offset_us))),
                "p95_abs_us": float(np.percentile(np.abs(offset_us), 95)),
            }
    total_lidar_samples = sum(len(rows) for rows in lidar_by_sensor.values())
    locked_lidar_samples = sum(
        row.get("ptp_status") == "Locked"
        for rows in lidar_by_sensor.values()
        for row in rows
    )
    lidar_sync_fraction = locked_lidar_samples / total_lidar_samples if total_lidar_samples else 0.0
    ax.text(
        0.98,
        0.97,
        f"Near |offset| mean/P95 {lidar_stats['near']['mean_abs_us']:.1f}/{lidar_stats['near']['p95_abs_us']:.1f} µs\n"
        f"Front |offset| mean/P95 {lidar_stats['front']['mean_abs_us']:.1f}/{lidar_stats['front']['p95_abs_us']:.1f} µs\n"
        f"Rear |offset| mean/P95 {lidar_stats['rear']['mean_abs_us']:.1f}/{lidar_stats['rear']['p95_abs_us']:.1f} µs\n"
        f"PTP locked: {locked_lidar_samples:,}/{total_lidar_samples:,} ({lidar_sync_fraction * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        fontweight="normal",
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(0.02, 1.02, "(c) LiDAR PTP", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, fontweight="normal",
            clip_on=False)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Device–host offset (µs)")
    ax.set_xlim(0, duration_s)
    ax.legend(frameon=True, framealpha=0.78, loc="center left",
              bbox_to_anchor=(0.01, 0.55), ncol=1, fontsize=6.2,
              borderpad=0.3, handlelength=1.6)

    # (d) Tobii NTP.  Only the device-host offset is plotted.  The recorded
    # NTP state remains in the annotation and summary tables, but is not drawn
    # on a second axis with a separate Yes/No scale.
    ax = axes[1, 1]
    tobii_plot_offset_us = tobii_offset_us
    ax.plot(tobii_t, tobii_plot_offset_us, color=COLORS["vermillion"], lw=1.0,
            label="Device–host offset")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Device–host offset (µs)")
    ax.set_xlim(0, duration_s)
    sync = np.asarray([1.0 if row.get("ntp_is_synchronized") == "True" else 0.0 for row in tobii])
    tobii_sync_count = int(np.sum(sync))
    tobii_sync_fraction = tobii_sync_count / len(sync) if len(sync) else 0.0
    ax.text(
        0.98,
        0.97,
        f"|device–host offset| mean {np.mean(np.abs(tobii_plot_offset_us)):.0f} µs\n"
        f"|device–host offset| P95 {np.percentile(np.abs(tobii_plot_offset_us), 95):.0f} µs\n"
        f"NTP synchronised: {tobii_sync_count}/{len(sync)} ({tobii_sync_fraction * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        fontweight="normal",
        linespacing=1.15,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    ax.text(0.02, 1.02, "(d) Tobii NTP", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.0, fontweight="normal",
            clip_on=False)

    fig.suptitle("System-wide time synchronisation validation", fontsize=11, y=0.97)
    save_figure(fig, figures, "system_wide_time_synchronisation_validation")

    report = f"""# Formal synchronisation log assessment

Session: `{args.session_id}`

## Paper figure

The paper-facing summary is
`figures/system_wide_time_synchronisation_validation.(png|svg)`. The six
single-purpose figures in the same directory are retained for audit and
supplementary analysis, not as six separate main-paper figures.

## Coverage

- Computer 2 host logger: {len(c2)} samples.
- Computer 1 host logger: {len(c1)} samples.
- Each Robosense logger: 1,721 HTTP status samples; all three returned HTTP 200.
- Tobii logger: {len(tobii)} samples, with no recorded connection error.
- Common all-stream overlap: **{coverage['common_overlap_s']:.3f} s**.
- All saved timestamps are dated **{coverage['log_calendar_date_observed']}**, although the session label contains `{coverage['session_id_calendar_date']}`. This calendar-date mismatch must be checked before the log is used as a formal acquisition-date record.

## Assessment

Computer 1 selected `192.168.1.102` as its chrony source and reported
`NTPSynchronized=yes` for every sample. The absolute estimated system offset
has a median of {c1_summary['absolute_system_time_offset_us']['median']:.3f} µs and a
P95 of {c1_summary['absolute_system_time_offset_us']['p95']:.3f} µs. This is usable
evidence for the Computer 1--Computer 2 software-clock link during this test.

Computer 2 remained the intentional local reference (`stratum=10`,
`NTPSynchronized=no`). Its PTP control query succeeded on all {sum(1 for row in c2 if row.get('pmc_ok') == 'True')} queried samples; blank `pmc_ok` rows are the expected samples between five-second PTP queries.

All three LiDAR streams reported `ptp_status=Locked` and HTTP 200 for every
record. The Bpearl near LiDAR (`192.168.1.200`) is treated as
`PTP-E2E-L4`, PTP domain 0, and its `time_sync_data` field is interpreted as
nanoseconds. The Helios front/rear LiDARs (`192.168.1.201/.202`) are treated
as `TimeSyncSrc=PTP-E2E`; their `ptp_master_offset` fields are interpreted as
nanoseconds. In all cases the displayed microsecond value is
`offset_us = offset_ns / 1000`. This is the requested engineering convention;
the saved HTTP page does not independently document the field unit.
The PTP capture contains {len(ptp_rows)} packets and identifies grandmaster
clock identity `{grandmaster or 'not decoded'}`.

Tobii reported `ntp_is_enabled=True` and `ntp_is_synchronized=True` for all
{len(tobii)} samples. The device--host midpoint difference has a median of
{tobii_summary['device_minus_host_midpoint_ms']['median']:.3f} ms and a P95 of
{tobii_summary['device_minus_host_midpoint_ms']['p95']:.3f} ms. This is an API
round-trip comparison, not an independent NTP residual-offset measurement.
Its median increased from {tobii_first_edge_median:.3f} ms in the first 5 percent
of samples to {tobii_last_edge_median:.3f} ms in the last 5 percent, a change of
{tobii_last_edge_median - tobii_first_edge_median:.3f} ms over the recording.
This trend should be reported as a diagnostic observation rather than converted
into a calibrated device clock error.

## Conclusion

The recording is suitable for the technical-validation evidence of: (1) the
Computer 1--Computer 2 chrony link, (2) continuous LiDAR PTP lock/readiness,
and (3) Tobii NTP readiness plus a diagnostic device-host time series. It does
not by itself provide a calibrated residual clock offset for LiDAR or Tobii,
and it does not establish absolute UTC accuracy. The timestamp calendar-date
mismatch should be resolved or documented before publication.
"""
    (output / "assessment.md").write_text(report, encoding="utf-8")
    (output / "figure_caption.txt").write_text(
        "System-wide time synchronisation validation from a 30-minute recording. "
        "(a) Logger coverage and the common all-stream overlap. (b) Computer 1–2 "
        "signed device-host offset relative to the Computer 2 local reference; "
        "the panel annotation gives mean and P95 absolute magnitudes. (c) "
        "Vendor-reported PTP lock state and signed device-host timing fields for "
        "the near, front and rear LiDARs. (d) Tobii Glasses 3 signed device-host "
        "midpoint offset; the recorded NTP synchronisation count is reported in "
        "the panel annotation. All "
        "horizontal axes use elapsed seconds from the earliest logger sample. "
        "Panels (b)–(d) plot signed device-host offsets in microseconds (µs); "
        "the mean and P95 annotations are absolute magnitudes. Bpearl "
        "time_sync_data and Helios ptp_master_offset are treated as nanoseconds "
        "and divided by 1000; the Tobii API value is recorded in milliseconds "
        "and multiplied by 1000. The Bpearl near sensor uses PTP-E2E-L4 on PTP "
        "domain 0; the Helios front/rear sensors use TimeSyncSrc=PTP-E2E. "
        "The Tobii device-host value is an API round-trip midpoint estimate, not "
        "an independent residual NTP offset measurement.\n",
        encoding="utf-8",
    )
    (output / "synchronisation_evidence_table.md").write_text(
        f"""# Synchronisation evidence table

| Reference or master | Client/stream | Protocol or evidence | Result in this recording | Limitation |
|---|---|---|---|---|
| Computer 2 (`192.168.1.102`) local clock | Computer 1 (`192.168.1.103`) | NTP/chrony | `NTPSynchronized=yes` and source `192.168.1.102` for {len(c1)}/{len(c1)} samples; median absolute system-offset estimate {c1_summary['absolute_system_time_offset_us']['median']:.3f} µs; P95 {c1_summary['absolute_system_time_offset_us']['p95']:.3f} µs | Software-clock estimate; not hardware timestamp accuracy |
| Computer 2 local clock | Near LiDAR Bpearl (`192.168.1.200`) | PTP, vendor HTTP status and captured packets | `Locked` and HTTP 200 for 1721/1721 samples; `PTP-E2E-L4`, PTP domain 0; `time_sync_data` interpreted as ns and plotted as µs | Unit convention is an engineering assumption and should be confirmed against vendor documentation |
| Computer 2 local clock | Front LiDAR Helios (`192.168.1.201`) | PTP, vendor HTTP status and captured packets | `Locked` and HTTP 200 for 1721/1721 samples; `TimeSyncSrc=PTP-E2E`; `ptp_master_offset` interpreted as ns and plotted as µs | Unit convention is an engineering assumption and should be confirmed against vendor documentation |
| Computer 2 local clock | Rear LiDAR Helios (`192.168.1.202`) | PTP, vendor HTTP status and captured packets | `Locked` and HTTP 200 for 1721/1721 samples; `TimeSyncSrc=PTP-E2E`; `ptp_master_offset` interpreted as ns and plotted as µs | Unit convention is an engineering assumption and should be confirmed against vendor documentation |
| Computer 2 local clock | Tobii Glasses 3 (`192.168.1.166`) | NTP state API and device-time request | NTP enabled and synchronized for {len(tobii)}/{len(tobii)} samples; API difference median {tobii_summary['device_minus_host_midpoint_ms']['median']:.3f} ms, P95 {tobii_summary['device_minus_host_midpoint_ms']['p95']:.3f} ms | API round-trip comparison, not an independent residual offset |

The six logger streams overlap for **{coverage['common_overlap_s']:.3f} s**. The PTP capture contains {len(ptp_rows)} packets; the decoded Announce messages identify grandmaster clock identity `{grandmaster or 'not decoded'}`.
""",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
