#!/usr/bin/env python3
"""Clean P9 steering limits and crop timestamped tables to one GNSS interval.

The raw steering logger clamps values outside its calibrated ADC range to
+/-45 degrees. This script distinguishes continuous, near-boundary readings
from implausible clamped readings using the raw ADC value and the entry/exit
angular rates of each limit run. It does not interpolate measurements and
does not modify source files or existing result directories.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from riding_input_sensor_validation_raw import (  # noqa: E402
    NS_PER_SECOND,
    read_gnss_speed,
    select_gnss_positive_interval,
)


STEERING_FILE = "steering_angle_20260603_134654.csv"
TIMESTAMPED_CSV_FILES = {
    "brake_sensors_force_20260603_134654.csv": "t_unix_ns",
    "imu_20260603_134654.csv": "t_unix_ns",
    "rally_payload_decoded_20260603_134654.csv": "t_unix_ns",
    "speed_decoded_20260603_134656.csv": "t_unix_ns",
    "camera_20260603_135432/timestamps.csv": "unix_ns",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def crop(frame: pd.DataFrame, time_column: str, start_ns: int, end_ns: int):
    time = pd.to_numeric(frame[time_column], errors="coerce")
    return frame.loc[(time >= start_ns) & (time <= end_ns)].copy()


def infer_linear_calibration(steering: pd.DataFrame, limit_deg: float):
    adc = pd.to_numeric(steering["adc_raw"], errors="coerce").to_numpy(float)
    angle = pd.to_numeric(steering["angle_deg"], errors="coerce").to_numpy(float)
    regular = (
        np.isfinite(adc)
        & np.isfinite(angle)
        & (np.abs(angle) < limit_deg - 1.0e-9)
    )
    if np.count_nonzero(regular) < 100:
        raise RuntimeError("Too few non-limit steering samples for calibration")
    slope, intercept = np.polyfit(adc[regular], angle[regular], 1)
    predicted = slope * adc[regular] + intercept
    residual = angle[regular] - predicted
    total = angle[regular] - np.mean(angle[regular])
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
    if not math.isfinite(slope) or abs(slope) < 1.0e-9 or r_squared < 0.999:
        raise RuntimeError(
            f"Steering ADC-angle relation is not sufficiently linear (R2={r_squared:.6f})"
        )
    return {
        "slope_deg_per_adc_count": float(slope),
        "intercept_deg": float(intercept),
        "r_squared": r_squared,
        "positive_limit_adc": float((limit_deg - intercept) / slope),
        "negative_limit_adc": float((-limit_deg - intercept) / slope),
        "fit_sample_count": int(np.count_nonzero(regular)),
    }


def classify_steering(
    steering: pd.DataFrame,
    *,
    limit_deg: float,
    maximum_extrapolation_deg: float,
    maximum_transition_rate_deg_s: float,
    maximum_contiguous_gap_s: float,
):
    result = steering.copy().reset_index(drop=True)
    time = pd.to_numeric(result["t_unix_ns"], errors="coerce").to_numpy(float)
    ok = pd.to_numeric(result["ok"], errors="coerce").eq(1).to_numpy()
    adc = pd.to_numeric(result["adc_raw"], errors="coerce").to_numpy(float)
    angle = pd.to_numeric(result["angle_deg"], errors="coerce").to_numpy(float)
    calibration = infer_linear_calibration(result, limit_deg)
    extrapolated = (
        calibration["slope_deg_per_adc_count"] * adc
        + calibration["intercept_deg"]
    )
    finite = np.isfinite(time) & np.isfinite(adc) & np.isfinite(angle)
    limit = finite & np.isclose(np.abs(angle), limit_deg, atol=1.0e-9)
    valid = finite & ok & ~limit
    classification = np.full(len(result), "invalid_decode_or_nonfinite", dtype=object)
    classification[finite & ok & ~limit] = "regular_measurement"
    classification[limit] = "abnormal_clamped_limit"
    run_id = np.full(len(result), -1, dtype=int)
    entry_rate = np.full(len(result), np.nan)
    exit_rate = np.full(len(result), np.nan)
    run_duration = np.full(len(result), np.nan)
    abnormal_runs = []
    run_counter = 0
    index = 0
    maximum_gap_ns = maximum_contiguous_gap_s * NS_PER_SECOND
    while index < len(result):
        if not limit[index]:
            index += 1
            continue
        first = index
        sign = 1.0 if angle[first] > 0 else -1.0
        last = first
        while (
            last + 1 < len(result)
            and limit[last + 1]
            and np.sign(angle[last + 1]) == sign
            and time[last + 1] - time[last] <= maximum_gap_ns
        ):
            last += 1
        indices = np.arange(first, last + 1)
        run_id[indices] = run_counter
        duration_s = float((time[last] - time[first]) / NS_PER_SECOND)
        run_duration[indices] = duration_s

        has_neighbors = first > 0 and last + 1 < len(result)
        neighbor_regular = (
            has_neighbors
            and finite[first - 1]
            and finite[last + 1]
            and ok[first - 1]
            and ok[last + 1]
            and not limit[first - 1]
            and not limit[last + 1]
        )
        same_side = (
            neighbor_regular
            and angle[first - 1] * sign > 0
            and angle[last + 1] * sign > 0
        )
        if neighbor_regular:
            entry_dt_s = (time[first] - time[first - 1]) / NS_PER_SECOND
            exit_dt_s = (time[last + 1] - time[last]) / NS_PER_SECOND
            entry_value = (
                abs(sign * limit_deg - angle[first - 1]) / entry_dt_s
                if entry_dt_s > 0 else math.inf
            )
            exit_value = (
                abs(angle[last + 1] - sign * limit_deg) / exit_dt_s
                if exit_dt_s > 0 else math.inf
            )
            entry_rate[indices] = entry_value
            exit_rate[indices] = exit_value
        else:
            entry_value = math.inf
            exit_value = math.inf

        if sign > 0:
            adc_near_boundary = bool(
                np.all(extrapolated[indices] <= limit_deg + maximum_extrapolation_deg)
            )
        else:
            adc_near_boundary = bool(
                np.all(extrapolated[indices] >= -limit_deg - maximum_extrapolation_deg)
            )
        transition_continuous = (
            entry_value <= maximum_transition_rate_deg_s
            and exit_value <= maximum_transition_rate_deg_s
        )
        plausible = bool(
            np.all(ok[indices])
            and adc_near_boundary
            and same_side
            and transition_continuous
        )
        if plausible:
            valid[indices] = True
            classification[indices] = "plausible_boundary_measurement"
        else:
            abnormal_runs.append((first, last, sign))
        run_counter += 1
        index = last + 1

    # Remove near-limit bridge samples that connect an otherwise normal signal
    # to an abnormal clamped run. These samples are inside the calibrated range,
    # but the implied one-sample transition is still physically implausible.
    fringe_threshold_deg = limit_deg - maximum_extrapolation_deg
    for first, last, sign in abnormal_runs:
        left_candidates = []
        cursor = first - 1
        while (
            cursor >= 0
            and valid[cursor]
            and angle[cursor] * sign > 0
            and abs(angle[cursor]) >= fringe_threshold_deg
        ):
            left_candidates.append(cursor)
            cursor -= 1
        if left_candidates and cursor >= 0 and valid[cursor]:
            edge = left_candidates[-1]
            dt_s = (time[edge] - time[cursor]) / NS_PER_SECOND
            rate = abs(angle[edge] - angle[cursor]) / dt_s if dt_s > 0 else math.inf
            if rate > maximum_transition_rate_deg_s:
                valid[left_candidates] = False
                classification[left_candidates] = "abnormal_transition_to_clamped_run"

        right_candidates = []
        cursor = last + 1
        while (
            cursor < len(result)
            and valid[cursor]
            and angle[cursor] * sign > 0
            and abs(angle[cursor]) >= fringe_threshold_deg
        ):
            right_candidates.append(cursor)
            cursor += 1
        if right_candidates and cursor < len(result) and valid[cursor]:
            edge = right_candidates[-1]
            dt_s = (time[cursor] - time[edge]) / NS_PER_SECOND
            rate = abs(angle[cursor] - angle[edge]) / dt_s if dt_s > 0 else math.inf
            if rate > maximum_transition_rate_deg_s:
                valid[right_candidates] = False
                classification[right_candidates] = "abnormal_transition_from_clamped_run"

    # Reject isolated one-sample spikes when both transitions exceed the same
    # rate limit but the two surrounding samples are mutually continuous.
    spike_indices = []
    for index in range(1, len(result) - 1):
        if not valid[index] or classification[index] != "regular_measurement":
            continue
        if not valid[index - 1] or not valid[index + 1]:
            continue
        dt_before = (time[index] - time[index - 1]) / NS_PER_SECOND
        dt_after = (time[index + 1] - time[index]) / NS_PER_SECOND
        dt_across = (time[index + 1] - time[index - 1]) / NS_PER_SECOND
        if min(dt_before, dt_after, dt_across) <= 0:
            continue
        if max(dt_before, dt_after) > maximum_contiguous_gap_s:
            continue
        incoming = abs(angle[index] - angle[index - 1]) / dt_before
        outgoing = abs(angle[index + 1] - angle[index]) / dt_after
        across = abs(angle[index + 1] - angle[index - 1]) / dt_across
        if (
            incoming > maximum_transition_rate_deg_s
            and outgoing > maximum_transition_rate_deg_s
            and across <= maximum_transition_rate_deg_s
        ):
            spike_indices.append(index)
    if spike_indices:
        valid[spike_indices] = False
        classification[spike_indices] = "isolated_rate_spike"

    audit = result.copy()
    audit.insert(0, "source_row_index", np.arange(len(result), dtype=int))
    audit["is_limit_value"] = limit
    audit["unclamped_angle_from_fitted_calibration_deg"] = extrapolated
    audit["limit_run_id"] = run_id
    audit["limit_run_duration_s"] = run_duration
    audit["entry_rate_deg_s"] = entry_rate
    audit["exit_rate_deg_s"] = exit_rate
    audit["quality_classification"] = classification
    audit["is_valid_for_analysis"] = valid
    return valid, audit, calibration


def crop_tobii_json_lines(
    source: Path,
    destination: Path,
    created_ns: int,
    start_ns: int,
    end_ns: int,
):
    count = 0
    with gzip.open(source, "rt", encoding="utf-8") as input_stream:
        with gzip.open(destination, "wt", encoding="utf-8") as output_stream:
            for line in input_stream:
                item = json.loads(line)
                timestamp_s = float(item["timestamp"])
                unix_ns = created_ns + int(round(timestamp_s * NS_PER_SECOND))
                if start_ns <= unix_ns <= end_ns:
                    output_stream.write(line)
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed-threshold-mps", type=float, default=0.1)
    parser.add_argument("--steering-limit-deg", type=float, default=45.0)
    parser.add_argument("--maximum-extrapolation-deg", type=float, default=10.0)
    parser.add_argument("--maximum-transition-rate-deg-s", type=float, default=250.0)
    parser.add_argument("--maximum-contiguous-gap-s", type=float, default=0.25)
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory does not exist: {session_dir}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    bag_matches = sorted(session_dir.glob("rosbag2_*/metadata.yaml"))
    if len(bag_matches) != 1:
        raise SystemExit("Expected exactly one rosbag2 metadata.yaml")
    bag_dir = bag_matches[0].parent

    speed = read_gnss_speed(bag_dir)
    start_ns, end_ns, selected_speed = select_gnss_positive_interval(
        speed, args.speed_threshold_mps
    )
    steering_path = session_dir / STEERING_FILE
    steering_raw = pd.read_csv(steering_path, low_memory=False)
    original_columns = list(steering_raw.columns)
    valid, steering_audit, calibration = classify_steering(
        steering_raw,
        limit_deg=args.steering_limit_deg,
        maximum_extrapolation_deg=args.maximum_extrapolation_deg,
        maximum_transition_rate_deg_s=args.maximum_transition_rate_deg_s,
        maximum_contiguous_gap_s=args.maximum_contiguous_gap_s,
    )
    interval_mask = (
        pd.to_numeric(steering_raw["t_unix_ns"], errors="coerce").between(
            start_ns, end_ns, inclusive="both"
        ).to_numpy()
    )
    cleaned_steering = steering_raw.loc[valid & interval_mask, original_columns].copy()

    data_dir = output / "data" / "common_interval"
    tables = output / "tables"
    data_dir.mkdir(parents=True)
    tables.mkdir(parents=True)
    cleaned_csv = data_dir / STEERING_FILE
    cleaned_xlsx = data_dir / Path(STEERING_FILE).with_suffix(".xlsx").name
    cleaned_steering.to_csv(cleaned_csv, index=False)
    steering_for_excel = cleaned_steering.copy()
    steering_for_excel["t_unix_ns"] = (
        pd.to_numeric(steering_for_excel["t_unix_ns"], errors="raise")
        .astype("int64").astype(str)
    )
    steering_for_excel.to_excel(
        cleaned_xlsx, index=False, sheet_name="steering"
    )
    steering_audit.loc[interval_mask].to_csv(
        tables / "steering_quality_audit.csv", index=False
    )

    cropped_inputs = {}
    for relative_name, time_column in TIMESTAMPED_CSV_FILES.items():
        source = session_dir / relative_name
        frame = pd.read_csv(source, low_memory=False)
        selected = crop(frame, time_column, start_ns, end_ns)
        destination = data_dir / Path(relative_name).name
        selected.to_csv(destination, index=False)
        cropped_inputs[relative_name] = {
            "source": str(source),
            "source_rows": int(len(frame)),
            "cropped_rows": int(len(selected)),
            "output": str(destination),
            "time_column": time_column,
            "columns_preserved": list(frame.columns) == list(selected.columns),
        }

    speed_interval = speed[
        speed["t_unix_ns"].between(start_ns, end_ns, inclusive="both")
    ].copy()
    speed_interval.to_csv(data_dir / "ubx_nav_vel_ned.csv", index=False)

    recording_dir = session_dir / "20260603T115600Z"
    recording_path = recording_dir / "recording.g3"
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    created = datetime.fromisoformat(recording["created"].replace("Z", "+00:00"))
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))
    tobii_counts = {}
    for filename in ["gazedata.gz", "imudata.gz", "eventdata.gz"]:
        source = recording_dir / filename
        destination = data_dir / filename
        tobii_counts[filename] = crop_tobii_json_lines(
            source, destination, created_ns, start_ns, end_ns
        )

    interval_audit = steering_audit.loc[interval_mask]
    classification_counts = (
        interval_audit["quality_classification"].value_counts().to_dict()
    )
    summary = {
        "selection": "first to last /ubx_nav_vel_ned ground speed above threshold",
        "speed_threshold_mps": args.speed_threshold_mps,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": (end_ns - start_ns) / NS_PER_SECOND,
        "gnss_samples_above_threshold": int(len(selected_speed)),
        "steering_source_rows_in_interval": int(np.count_nonzero(interval_mask)),
        "steering_cleaned_rows_in_interval": int(len(cleaned_steering)),
        "steering_limit_rows_in_interval": int(interval_audit["is_limit_value"].sum()),
        "steering_plausible_boundary_rows_kept": int(
            interval_audit["quality_classification"]
            .eq("plausible_boundary_measurement").sum()
        ),
        "steering_abnormal_clamped_rows_removed": int(
            interval_audit["quality_classification"]
            .eq("abnormal_clamped_limit").sum()
        ),
        "steering_other_invalid_rows_removed": int(
            (~interval_audit["is_valid_for_analysis"]
             & ~interval_audit["quality_classification"].eq("abnormal_clamped_limit")).sum()
        ),
        "steering_classification_counts": classification_counts,
        "steering_excel_timestamp_storage": (
            "t_unix_ns stored as text to preserve all 19 decimal digits"
        ),
        "steering_calibration_fit": calibration,
        "cleaning_parameters": {
            "steering_limit_deg": args.steering_limit_deg,
            "maximum_extrapolation_deg": args.maximum_extrapolation_deg,
            "maximum_transition_rate_deg_s": args.maximum_transition_rate_deg_s,
            "maximum_contiguous_gap_s": args.maximum_contiguous_gap_s,
            "near_limit_fringe_threshold_deg": (
                args.steering_limit_deg - args.maximum_extrapolation_deg
            ),
            "isolated_spike_rule": (
                "both adjacent rates exceed the limit while the two neighbours remain continuous"
            ),
            "interpolation_applied": False,
        },
        "cropped_csv_inputs": cropped_inputs,
        "cropped_tobii_line_counts": tobii_counts,
        "excluded_large_binary_sources": [
            "scenevideo.mp4",
            "camera video_mjpg.avi",
            "lidar pcap",
            "rosbag db3",
        ],
    }
    write_json(tables / "common_interval_and_steering_cleaning.json", summary)

    readme = (
        "The common interval is defined by the first and last UBX-NAV-VEL-NED "
        f"sample with ground speed above {args.speed_threshold_mps:g} m/s. "
        "Timestamped CSV and Tobii JSON streams used by the validation figures are "
        "cropped to these exact boundaries. Large video, pcap, and rosbag files are "
        "not duplicated; downstream scripts apply the same boundaries while reading "
        "them. The steering CSV and XLSX contain only valid rows and preserve the six "
        "source columns. In XLSX, t_unix_ns is stored as text because Excel numeric "
        "cells cannot preserve all 19 digits. No value is interpolated. The audit table records every "
        "classification decision. Source files are unchanged.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    source_files = [steering_path, recording_path]
    source_files.extend(session_dir / name for name in TIMESTAMPED_CSV_FILES)
    source_files.extend(recording_dir / name for name in tobii_counts)
    bag_files = sorted(item for item in bag_dir.iterdir() if item.is_file())
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session_dir),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in [*source_files, *bag_files]
        ],
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "summary": summary,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    checksum_lines = []
    for path in sorted(
        item for item in output.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    ):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote cleaned and cropped P9 data to {output}")


if __name__ == "__main__":
    main()
