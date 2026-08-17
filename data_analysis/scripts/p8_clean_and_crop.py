#!/usr/bin/env python3
"""Create a new, cropped P8 validation input set.

The source session is never changed.  The common interval is the first to the
last ``/ubx_nav_vel_ned`` sample whose ground speed is above the threshold.
All timestamped CSV and Tobii JSON streams are copied as cropped text data;
the rosbag and camera video remain referenced by their original paths.
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

from p9_clean_and_crop import (  # noqa: E402
    NS_PER_SECOND,
    classify_steering,
    crop_tobii_json_lines,
    read_gnss_speed,
    select_gnss_positive_interval,
    sha256_file,
    write_json,
)
import video_time_mapping as VIDEO_TIME  # noqa: E402


def crop_frame(frame: pd.DataFrame, column: str, start_ns: int, end_ns: int):
    t = pd.to_numeric(frame[column], errors="coerce")
    return frame.loc[t.between(start_ns, end_ns, inclusive="both")].copy()


def find_one(session: Path, pattern: str) -> Path:
    matches = sorted(session.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speed-threshold-mps", type=float, default=0.1)
    parser.add_argument("--video-start-s", type=float, default=None)
    parser.add_argument("--video-end-s", type=float, default=None)
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session.is_dir():
        raise SystemExit(f"Session directory does not exist: {session}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    bag_files = sorted(session.glob("rosbag2_*/*.db3"))
    if len(bag_files) != 1:
        raise SystemExit("Expected exactly one rosbag2 db3 file")
    bag_file = bag_files[0]
    speed = read_gnss_speed(bag_file)
    start_ns, end_ns, selected_speed = select_gnss_positive_interval(
        speed, args.speed_threshold_mps
    )

    steering_path = find_one(session, "steering_angle_*.csv")
    brake_path = find_one(session, "brake_sensors_force_*.csv")
    imu_path = find_one(session, "imu_*.csv")
    power_path = find_one(session, "rally_payload_decoded_*.csv")
    wheel_path = find_one(session, "speed_decoded_*.csv")
    camera_path = find_one(session, "camera_*/timestamps.csv")
    video_path = find_one(session, "camera_*/video_mjpg.avi")
    video_time_mapping = None
    if (args.video_start_s is None) != (args.video_end_s is None):
        raise SystemExit("Both --video-start-s and --video-end-s are required together")
    if args.video_start_s is not None:
        start_ns, end_ns, video_time_mapping = VIDEO_TIME.map_playback_interval(
            video_path, camera_path, args.video_start_s, args.video_end_s
        )
        selected_speed = speed[speed["t_unix_ns"].between(start_ns, end_ns, inclusive="both")].copy()
        selection_mode = "explicit camera video interval"
    else:
        selection_mode = "first to last GNSS speed sample above threshold"
    recording_matches = sorted(session.glob("**/recording.g3"))
    if len(recording_matches) != 1:
        raise SystemExit(f"Expected one recording.g3, found {len(recording_matches)}")
    recording_path = recording_matches[0]
    recording_dir = recording_path.parent

    raw_steering = pd.read_csv(steering_path, low_memory=False)
    original_columns = list(raw_steering.columns)
    valid, audit, calibration = classify_steering(
        raw_steering,
        limit_deg=45.0,
        maximum_extrapolation_deg=10.0,
        maximum_transition_rate_deg_s=250.0,
        maximum_contiguous_gap_s=0.25,
    )
    in_interval = pd.to_numeric(raw_steering["t_unix_ns"], errors="coerce").between(
        start_ns, end_ns, inclusive="both"
    ).to_numpy()
    clean_steering = raw_steering.loc[valid & in_interval, original_columns].copy()

    data_dir = output / "data" / "common_interval"
    tables = output / "tables"
    data_dir.mkdir(parents=True)
    tables.mkdir(parents=True)
    clean_steering.to_csv(data_dir / steering_path.name, index=False)
    excel_steering = clean_steering.copy()
    excel_steering["t_unix_ns"] = pd.to_numeric(
        excel_steering["t_unix_ns"], errors="raise"
    ).astype("int64").astype(str)
    excel_steering.to_excel(
        data_dir / steering_path.with_suffix(".xlsx").name,
        index=False,
        sheet_name="steering",
    )
    audit_interval = audit.loc[in_interval].copy()
    audit_interval.to_csv(tables / "steering_quality_audit.csv", index=False)

    csv_inputs = {
        brake_path: "t_unix_ns",
        imu_path: "t_unix_ns",
        power_path: "t_unix_ns",
        wheel_path: "t_unix_ns",
        camera_path: "unix_ns",
    }
    cropped_inputs = {}
    for source, column in csv_inputs.items():
        frame = pd.read_csv(source, low_memory=False)
        selected = crop_frame(frame, column, start_ns, end_ns)
        destination = data_dir / source.name
        selected.to_csv(destination, index=False)
        cropped_inputs[str(source)] = {
            "output": str(destination),
            "source_rows": int(len(frame)),
            "cropped_rows": int(len(selected)),
            "time_column": column,
            "columns_preserved": list(frame.columns) == list(selected.columns),
        }
    speed_interval = speed.loc[
        speed["t_unix_ns"].between(start_ns, end_ns, inclusive="both")
    ].copy()
    speed_interval.to_csv(data_dir / "ubx_nav_vel_ned.csv", index=False)

    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    created = datetime.fromisoformat(recording["created"].replace("Z", "+00:00"))
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))
    tobii_counts = {}
    for filename in ["gazedata.gz", "imudata.gz", "eventdata.gz"]:
        source = recording_dir / filename
        if not source.is_file():
            continue
        destination = data_dir / filename
        tobii_counts[filename] = crop_tobii_json_lines(
            source, destination, created_ns, start_ns, end_ns
        )

    summary = {
        "selection": selection_mode,
        "speed_topic": "/ubx_nav_vel_ned",
        "speed_timestamp_source": "rosbag record time",
        "speed_threshold_mps": args.speed_threshold_mps,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "start_utc": datetime.fromtimestamp(start_ns / NS_PER_SECOND, tz=timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(end_ns / NS_PER_SECOND, tz=timezone.utc).isoformat(),
        "duration_s": (end_ns - start_ns) / NS_PER_SECOND,
        "gnss_samples_above_threshold": int(len(selected_speed)),
        "minimum_selected_speed_mps": float(selected_speed["ground_speed_mps"].min()),
        "median_selected_speed_mps": float(selected_speed["ground_speed_mps"].median()),
        "maximum_selected_speed_mps": float(selected_speed["ground_speed_mps"].max()),
        "video_start_s": args.video_start_s,
        "video_end_s": args.video_end_s,
        "video_time_mapping": video_time_mapping,
        "steering_source_rows_in_interval": int(in_interval.sum()),
        "steering_cleaned_rows_in_interval": int(len(clean_steering)),
        "steering_limit_rows_in_interval": int(audit_interval["is_limit_value"].sum()),
        "steering_classification_counts": audit_interval["quality_classification"].value_counts().to_dict(),
        "steering_calibration_fit": calibration,
        "cleaning_parameters": {
            "steering_limit_deg": 45.0,
            "maximum_extrapolation_deg": 10.0,
            "maximum_transition_rate_deg_s": 250.0,
            "maximum_contiguous_gap_s": 0.25,
            "interpolation_applied": False,
        },
        "cropped_csv_inputs": cropped_inputs,
        "cropped_tobii_line_counts": tobii_counts,
        "source_files_not_copied": [str(bag_file), str(video_path)],
    }
    write_json(tables / "common_interval_and_steering_cleaning.json", summary)
    interval_description = (
        f"camera video seconds {args.video_start_s:g}--{args.video_end_s:g}"
        if args.video_start_s is not None
        else f"the first to the last GNSS velocity sample with ground speed > {args.speed_threshold_mps:.1f} m/s"
    )
    readme = (
        "This directory is a derived P8 validation input set. Raw files, rosbag, and video "
        "are unchanged and are not copied. The interval is selected from "
        f"{interval_description} using rosbag record time for GNSS alignment; intermittent stops inside this interval "
        "AVI player seconds are converted to frame indices using the nominal video frame rate, then mapped through "
        "timestamps.csv to the recorded Unix time. "
        "are retained. Timestamped CSV and Tobii streams are cropped "
        "to the same nanosecond boundaries. Steering cleaning uses ADC calibration and rate "
        "continuity; no interpolation is applied.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    source_list = [steering_path, brake_path, imu_path, power_path, wheel_path, camera_path, recording_path]
    source_list += [recording_dir / name for name in tobii_counts]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in source_list + [bag_file]],
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "summary": summary,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(output / "run_manifest.json", manifest)
    checksums = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256"):
        checksums.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Wrote P8 cropped validation inputs to {output}")


if __name__ == "__main__":
    main()
