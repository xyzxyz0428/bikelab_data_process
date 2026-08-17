#!/usr/bin/env python3
"""Audit raw/AHRS pairing and tilt compensation for fixed IMU windows."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


PACKAGE_SOURCE = (
    Path(__file__).resolve().parent
    / "ros2_ws" / "src" / "bikelab_process"
)
sys.path.insert(0, str(PACKAGE_SOURCE))

from bikelab_process.imu_file_player import (  # noqa: E402
    tilt_compensated_yaw_rate_native_frd,
)


def file_sha256(path):
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(row, name):
    """Read one required finite float or return None."""
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_imu(path):
    """Load valid dtype 64 and 65 rows needed by this audit."""
    raw = []
    ahrs = []
    total = 0
    invalid_frame = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            total += 1
            checks = [row.get(name, "") for name in (
                "crc8_ok", "crc16_ok", "end_ok",
            )]
            if any(value and value.strip() != "1" for value in checks):
                invalid_frame += 1
                continue
            try:
                stamp = int(row["t_unix_ns"])
                dtype = int(row["dtype"])
            except (KeyError, TypeError, ValueError):
                continue
            if dtype == 64:
                values = [finite_float(row, name) for name in (
                    "gyro_x", "gyro_y", "gyro_z",
                )]
                if all(value is not None for value in values):
                    raw.append((stamp, *values))
            elif dtype == 65:
                values = [finite_float(row, name) for name in (
                    "roll", "pitch", "heading",
                )]
                if all(value is not None for value in values):
                    ahrs.append((stamp, *values))
    raw.sort()
    ahrs.sort()
    return total, invalid_frame, np.asarray(raw), np.asarray(ahrs)


def nearest_raw_indices(raw_time, ahrs_time):
    """Return nearest raw-row index and signed raw-minus-AHRS time."""
    right = np.searchsorted(raw_time, ahrs_time)
    right = np.clip(right, 0, len(raw_time) - 1)
    left = np.clip(right - 1, 0, len(raw_time) - 1)
    choose_left = (
        np.abs(raw_time[left] - ahrs_time)
        <= np.abs(raw_time[right] - ahrs_time)
    )
    index = np.where(choose_left, left, right)
    return index, raw_time[index] - ahrs_time


def observed_rate(rows):
    """Return count, time span, and effective rate."""
    span_s = (rows[-1, 0] - rows[0, 0]) / 1e9
    return {
        "count": len(rows),
        "span_s": float(span_s),
        "effective_rate_hz": float((len(rows) - 1) / span_s),
    }


def interval(rows, start_ns, end_ns):
    """Select one inclusive nanosecond interval."""
    return rows[(rows[:, 0] >= start_ns) & (rows[:, 0] <= end_ns)]


def summarize(values):
    """Return selected descriptive statistics."""
    values = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def paired_rates(raw, ahrs, bias, maximum_dt_ns):
    """Pair streams and calculate raw-z and tilt-compensated ENU rates."""
    indices, signed_dt = nearest_raw_indices(raw[:, 0], ahrs[:, 0])
    valid = np.abs(signed_dt) <= maximum_dt_ns
    matched_raw = raw[indices[valid]]
    matched_ahrs = ahrs[valid]
    gyro_y = matched_raw[:, 2] - bias[1]
    gyro_z = matched_raw[:, 3] - bias[2]
    raw_yaw_rate = -gyro_z
    tilt_rate = np.asarray([
        tilt_compensated_yaw_rate_native_frd(
            q,
            r,
            roll,
            pitch,
        )
        for q, r, roll, pitch in zip(
            gyro_y,
            gyro_z,
            matched_ahrs[:, 1],
            matched_ahrs[:, 2],
        )
    ], dtype=float)
    finite = np.isfinite(tilt_rate)
    return (
        matched_ahrs[finite],
        raw_yaw_rate[finite],
        tilt_rate[finite],
        signed_dt[valid][finite],
    )


def main():
    """Write a read-only source audit to a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--static-start-ns", type=int, required=True)
    parser.add_argument("--static-end-ns", type=int, required=True)
    parser.add_argument("--turn-start-ns", type=int, required=True)
    parser.add_argument("--turn-end-ns", type=int, required=True)
    parser.add_argument("--max-merge-dt-ms", type=float, default=10.0)
    parser.add_argument("--gyro-covariance-scale", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.imu.resolve()
    output = args.out.resolve()
    if not source.is_file():
        parser.error(f"IMU file not found: {source}")
    if output.exists():
        parser.error(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    total, invalid_frame, raw, ahrs = load_imu(source)
    if not len(raw) or not len(ahrs):
        raise RuntimeError("Required dtype 64 or dtype 65 rows are missing")
    all_indices, all_dt = nearest_raw_indices(raw[:, 0], ahrs[:, 0])
    del all_indices
    maximum_dt_ns = int(args.max_merge_dt_ms * 1e6)
    all_pair_valid = np.abs(all_dt) <= maximum_dt_ns

    static_raw = interval(raw, args.static_start_ns, args.static_end_ns)
    static_ahrs = interval(ahrs, args.static_start_ns, args.static_end_ns)
    bias = np.mean(static_raw[:, 1:4], axis=0)
    static_pair, _, static_tilt_rate, static_dt = paired_rates(
        static_raw,
        static_ahrs,
        bias,
        maximum_dt_ns,
    )
    del static_pair

    turn_raw = interval(raw, args.turn_start_ns, args.turn_end_ns)
    turn_ahrs = interval(ahrs, args.turn_start_ns, args.turn_end_ns)
    paired_ahrs, raw_rate, tilt_rate, turn_dt = paired_rates(
        turn_raw,
        turn_ahrs,
        bias,
        maximum_dt_ns,
    )
    relative_time = (paired_ahrs[:, 0] - paired_ahrs[0, 0]) / 1e9
    rate_difference = np.abs(tilt_rate - raw_rate)

    report = {
        "schema_version": 1,
        "source": {
            "path": str(source),
            "sha256": file_sha256(source),
            "csv_rows": total,
            "invalid_crc_or_end_rows": invalid_frame,
        },
        "timestamp_note": (
            "t_unix_ns is host receive/log time and is used for this "
            "workflow alignment; it is not the IMU device clock"
        ),
        "streams": {
            "dtype64_raw_imu": observed_rate(raw),
            "dtype65_ahrs": observed_rate(ahrs),
        },
        "all_stream_pairing": {
            "maximum_allowed_ms": args.max_merge_dt_ms,
            "matched_count": int(np.count_nonzero(all_pair_valid)),
            "unmatched_count": int(np.count_nonzero(~all_pair_valid)),
            "matched_absolute_difference_ms": summarize(
                np.abs(all_dt[all_pair_valid]) / 1e6
            ),
            "nearest_difference_max_ms_before_gate": float(
                np.max(np.abs(all_dt)) / 1e6
            ),
        },
        "static_interval": {
            "start_ns": args.static_start_ns,
            "end_ns": args.static_end_ns,
            "raw_count": len(static_raw),
            "ahrs_count": len(static_ahrs),
            "paired_count": len(static_dt),
            "native_gyro_bias_rad_s": {
                "x": float(bias[0]),
                "y": float(bias[1]),
                "z": float(bias[2]),
            },
            "raw_yaw_rate_variance_scaled_rad2_s2": float(
                np.var(-(static_raw[:, 3] - bias[2]), ddof=1)
                * args.gyro_covariance_scale
            ),
            "tilt_yaw_rate_variance_scaled_rad2_s2": float(
                np.var(static_tilt_rate, ddof=1)
                * args.gyro_covariance_scale
            ),
        },
        "turn_interval": {
            "start_ns": args.turn_start_ns,
            "end_ns": args.turn_end_ns,
            "raw_count": len(turn_raw),
            "ahrs_count": len(turn_ahrs),
            "paired_count": len(turn_dt),
            "absolute_pair_difference_ms": summarize(
                np.abs(turn_dt) / 1e6
            ),
            "native_roll_deg": summarize(
                np.degrees(paired_ahrs[:, 1])
            ),
            "native_pitch_deg": summarize(
                np.degrees(paired_ahrs[:, 2])
            ),
            "absolute_tilt_minus_raw_rate_rad_s": summarize(
                rate_difference
            ),
            "raw_yaw_rate_integral_deg": math.degrees(float(
                np.trapz(raw_rate, relative_time)
            )),
            "tilt_yaw_rate_integral_deg": math.degrees(float(
                np.trapz(tilt_rate, relative_time)
            )),
        },
        "method": {
            "name": "AHRS-assisted tilt-compensated raw gyro",
            "equation": (
                "yaw_rate_ENU = -(gyro_y*sin(roll) + "
                "gyro_z*cos(roll))/cos(pitch)"
            ),
            "coordinates": "native FRD gyro and native AHRS roll/pitch",
            "limitation": (
                "roll and pitch are AHRS outputs and may be affected by "
                "dynamic acceleration"
            ),
        },
    }
    (output / "source_input_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    with (output / "turn_rate_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "t_unix_ns",
            "elapsed_s",
            "raw_yaw_rate_rad_s",
            "tilt_compensated_yaw_rate_rad_s",
            "tilt_minus_raw_rad_s",
            "native_roll_rad",
            "native_pitch_rad",
            "raw_minus_ahrs_pair_dt_ms",
        ])
        for row, raw_value, tilt_value, dt in zip(
            paired_ahrs,
            raw_rate,
            tilt_rate,
            turn_dt,
        ):
            writer.writerow([
                int(row[0]),
                (row[0] - args.turn_start_ns) / 1e9,
                raw_value,
                tilt_value,
                tilt_value - raw_value,
                row[1],
                row[2],
                dt / 1e6,
            ])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
