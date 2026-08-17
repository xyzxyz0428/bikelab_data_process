#!/usr/bin/env python3
"""Generate the P1 close-loop figure with bike-ego-frame gaze angles.

The script reads all source data without modification. It sequentially decodes
only the selected rear-camera frame range into a new output directory, runs the
project AprilTag head-pose estimator, pairs each quality-approved camera pose
with the nearest raw Tobii gaze sample, and rotates Methods A--C into
``base_link`` (x forward, y left, z up).
"""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
HEADPOSE_DIR = PROJECT_ROOT / "headpose_estimation" / "scripts"
for path in (SCRIPT_DIR, HEADPOSE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import p9_speed_timing_closed_loop as P9  # noqa: E402
from paper_style import apply_paper_style  # noqa: E402
from pose_utils import invert_T, rot_to_rpy_deg  # noqa: E402
from video_time_mapping import (  # noqa: E402
    load_video_clock,
    playback_seconds_to_unix_ns,
    unix_ns_to_playback_seconds,
)


NS_PER_SECOND = 1_000_000_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_output_checksums(output: Path):
    checksum_path = output / "CHECKSUMS.sha256"
    paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output)}"
        for path in paths
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {pattern} under {directory}, found {len(matches)}"
        )
    return matches[0]


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1.0e-12:
        return np.full(3, np.nan)
    return vector / norm


def angle_between_deg(first, second):
    first = normalize(first)
    second = normalize(second)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return np.nan
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def wrapped_angle_difference_deg(final_deg, initial_deg):
    """Return the signed shortest angular change from initial to final."""
    return (float(final_deg) - float(initial_deg) + 180.0) % 360.0 - 180.0


def smoothed_absolute_peak(frame, time_column, values, window_samples):
    """Locate the largest absolute value after a short centred mean filter."""
    table = pd.DataFrame({
        "time_ns": pd.to_numeric(frame[time_column], errors="coerce"),
        "value": pd.to_numeric(values, errors="coerce"),
    }).dropna().sort_values("time_ns")
    if table.empty:
        return None
    table["smoothed"] = table["value"].rolling(
        window=window_samples, center=True, min_periods=1
    ).mean()
    index = table["smoothed"].abs().idxmax()
    row = table.loc[index]
    return {
        "time_ns": int(row["time_ns"]),
        "raw_value": float(row["value"]),
        "smoothed_value": float(row["smoothed"]),
        "smoothing": f"centred mean, {window_samples} samples",
    }


def rotation_rpy_static(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = np.radians([roll_deg, pitch_deg, yaw_deg])
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])
    ry = np.array([
        [math.cos(pitch), 0.0, math.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch), 0.0, math.cos(pitch)],
    ])
    rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return rz @ ry @ rx


def rotation_headpose(row):
    values = pd.to_numeric(
        pd.Series([
            row.get("cam_head_roll_deg"),
            row.get("cam_head_pitch_deg"),
            row.get("cam_head_yaw_deg"),
        ]), errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None
    roll, pitch, yaw = np.radians(values)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(roll), -math.sin(roll)],
        [0.0, math.sin(roll), math.cos(roll)],
    ])
    ry = np.array([
        [math.cos(pitch), 0.0, math.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-math.sin(pitch), 0.0, math.cos(pitch)],
    ])
    rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0],
        [math.sin(yaw), math.cos(yaw), 0.0],
        [0.0, 0.0, 1.0],
    ])
    # Inverse of pose_utils.rot_to_rpy_deg(): scipy extrinsic z-y-x.
    return rx @ ry @ rz


def base_to_camera_optical_rotation(extrinsics_path: Path):
    data = read_json(extrinsics_path)
    graph = {}
    for transform in data["transforms"]:
        parent, child = transform["parent"], transform["child"]
        translation = transform["translation"]
        rpy = transform["rotation_rpy_deg"]
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_rpy_static(
            float(rpy["roll"]), float(rpy["pitch"]), float(rpy["yaw"])
        )
        matrix[:3, 3] = [
            float(translation["x"]),
            float(translation["y"]),
            float(translation["z"]),
        ]
        graph[(parent, child)] = matrix
        graph[(child, parent)] = invert_T(matrix)
    queue = [("base_link", np.eye(4))]
    visited = {"base_link"}
    while queue:
        current, transform = queue.pop(0)
        if current == "camera_optical_frame":
            return transform[:3, :3]
        for (parent, child), edge in graph.items():
            if parent != current or child in visited:
                continue
            visited.add(child)
            queue.append((child, transform @ edge))
    raise RuntimeError("No base_link -> camera_optical_frame transform")


def extract_frame_window(video_path, timestamps_path, start_ns, end_ns, output):
    timestamps = pd.read_csv(timestamps_path)
    timestamps["frame_idx"] = pd.to_numeric(
        timestamps["frame_idx"], errors="coerce"
    )
    timestamps["unix_ns"] = pd.to_numeric(
        timestamps["unix_ns"], errors="coerce"
    )
    selected = timestamps[
        timestamps["frame_idx"].notna()
        & timestamps["unix_ns"].between(start_ns, end_ns)
    ].copy()
    selected["frame_idx"] = selected["frame_idx"].astype(int)
    selected["unix_ns"] = selected["unix_ns"].astype("int64")
    if selected.empty:
        raise RuntimeError("No camera timestamps in selected window")
    first_index = int(selected["frame_idx"].min())
    last_index = int(selected["frame_idx"].max())
    if first_index < 1:
        raise RuntimeError("Expected one-based camera frame indices")

    frames = output / "camera_frames"
    frames.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    seek_target = first_index - 1
    seek_used = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, seek_target))
    seek_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if not seek_used or abs(seek_position - seek_target) > 0.5:
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot reopen {video_path} after failed seek")
        decoded = 0
        seek_used = False
    else:
        decoded = seek_target
    decoded_in_call = 0
    written = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame_index = decoded + 1
        decoded_in_call += 1
        if frame_index >= first_index:
            path = frames / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(
                str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise RuntimeError(f"Cannot write {path}")
            written += 1
        decoded += 1
        if frame_index >= last_index:
            break
    capture.release()
    expected = last_index - first_index + 1
    if written != expected:
        raise RuntimeError(
            f"Decoded {written}/{expected} selected camera frames; "
            f"last sequential frame was {decoded}"
        )
    selected.to_csv(output / "camera_timestamps_window.csv", index=False)
    return frames, selected, {
        "first_frame_idx": first_index,
        "last_frame_idx": last_index,
        "selected_frame_count": int(len(selected)),
        "written_frame_count": written,
        "sequentially_decoded_through_frame": decoded,
        "decoder_seek_used": seek_used,
        "frames_decoded_in_this_call": decoded_in_call,
    }


def run_headpose(frames, timestamps_path, output_csv, args):
    command = [
        sys.executable,
        str(args.headpose_script),
        "--camera", str(args.rear_camera_calibration),
        "--config", str(args.head_rig_config),
        "--rig-calib", str(args.rig_calibration),
        "--frame-dir", str(frames),
        "--timestamps-csv", str(timestamps_path),
        "--output-csv", str(output_csv),
        "--min-head-tags", str(args.min_head_tags),
        "--max-head-rmse-px", str(args.max_head_rmse_px),
        "--detector-threads", str(args.detector_threads),
    ]
    result = subprocess.run(
        command,
        check=False,
        cwd=str(HEADPOSE_DIR),
        text=True,
        capture_output=True,
    )
    log_dir = output_csv.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    stdout_path = log_dir / "headpose_stdout.log"
    stderr_path = log_dir / "headpose_stderr.log"
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Head-pose estimator returned {result.returncode}; see {stderr_path}"
        )
    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        raise RuntimeError(
            f"Head-pose estimator returned {result.returncode} without a usable output"
        )
    expected = pd.read_csv(timestamps_path)
    observed = pd.read_csv(output_csv)
    expected_frames = pd.to_numeric(
        expected["frame_idx"], errors="coerce"
    ).dropna().astype(int).to_numpy()
    observed_frames = pd.to_numeric(
        observed["frame_idx"], errors="coerce"
    ).dropna().astype(int).to_numpy()
    complete = (
        len(expected_frames) == len(observed_frames)
        and len(np.unique(observed_frames)) == len(observed_frames)
        and np.array_equal(expected_frames, observed_frames)
    )
    if not complete:
        raise RuntimeError(
            f"Incomplete head-pose output after return code {result.returncode}: "
            f"expected {len(expected_frames)} ordered frames, got "
            f"{len(observed_frames)}"
        )
    return {
        "command": command,
        "return_code": int(result.returncode),
        "complete_output_validated": True,
        "expected_frame_count": int(len(expected_frames)),
        "observed_frame_count": int(len(observed_frames)),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "note": "Return code 0 and exact ordered frame-index output were required.",
    }


def build_gaze_ego(gaze, eye_validity, headpose, camera_timestamps, recording,
                   transforms, extrinsics, sync_tolerance_ms,
                   eye_validity_tolerance_ms,
                   eye_direction_consistency_deg):
    timestamp_map = camera_timestamps[["frame_idx", "unix_ns"]].rename(
        columns={"unix_ns": "camera_timestamp_ns"}
    )
    pose = headpose.merge(timestamp_map, on="frame_idx", how="left")
    pose["camera_timestamp_ns"] = pd.to_numeric(
        pose["camera_timestamp_ns"], errors="coerce"
    )
    pose = pose.dropna(subset=["camera_timestamp_ns"]).copy()
    pose["camera_timestamp_ns"] = pose["camera_timestamp_ns"].astype("int64")
    pose = pose.sort_values("camera_timestamp_ns")

    gaze_join = gaze.rename(columns={"t_unix_ns": "gaze_timestamp_ns"}).copy()
    gaze_join["gaze_timestamp_ns"] = gaze_join["gaze_timestamp_ns"].astype("int64")
    validity_join = eye_validity.rename(
        columns={"t_unix_ns": "validity_timestamp_ns"}
    ).copy()
    validity_join["validity_timestamp_ns"] = validity_join[
        "validity_timestamp_ns"
    ].astype("int64")
    gaze_join = pd.merge_asof(
        gaze_join.sort_values("gaze_timestamp_ns"),
        validity_join.sort_values("validity_timestamp_ns"),
        left_on="gaze_timestamp_ns",
        right_on="validity_timestamp_ns",
        direction="nearest",
        tolerance=int(round(eye_validity_tolerance_ms * 1.0e6)),
    )
    gaze_join["validity_sync_dt_ms"] = (
        gaze_join["validity_timestamp_ns"] - gaze_join["gaze_timestamp_ns"]
    ).abs() / 1.0e6
    merged = pd.merge_asof(
        pose,
        gaze_join.sort_values("gaze_timestamp_ns"),
        left_on="camera_timestamp_ns",
        right_on="gaze_timestamp_ns",
        direction="nearest",
        tolerance=int(round(sync_tolerance_ms * 1.0e6)),
    )
    merged["sync_dt_ms"] = (
        merged["gaze_timestamp_ns"] - merged["camera_timestamp_ns"]
    ).abs() / 1.0e6
    merged["t_unix_ns"] = merged["camera_timestamp_ns"]

    transforms_data = read_json(transforms)
    t_h_c1 = np.asarray(transforms_data["T_H_C1"], dtype=float).reshape(4, 4)
    recording_data = read_json(recording)
    calibration = recording_data["scenecamera"]["camera-calibration"]
    r_c1_hucs = np.asarray(calibration["rotation"], dtype=float).reshape(3, 3)
    r_h_hucs = t_h_c1[:3, :3] @ r_c1_hucs
    r_h_c1 = t_h_c1[:3, :3]
    fx, fy = map(float, calibration["focal-length"])
    cx, cy = map(float, calibration["principal-point"])
    width, height = map(float, calibration["resolution"])
    radial = list(map(float, calibration.get("radial-distortion", [0, 0, 0])))
    tangential = list(map(float, calibration.get("tangential-distortion", [0, 0])))
    while len(radial) < 3:
        radial.append(0.0)
    while len(tangential) < 2:
        tangential.append(0.0)
    scene_camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])
    scene_distortion = np.array([
        radial[0], radial[1], tangential[0], tangential[1], radial[2]
    ])
    r_base_camera = base_to_camera_optical_rotation(extrinsics)

    output_rows = []
    for _, row in merged.iterrows():
        pose_valid = (
            bool(pd.to_numeric(row.get("head_quality_ok"), errors="coerce") == 1)
            and pd.notna(row.get("gaze_timestamp_ns"))
            and np.isfinite(float(row.get("sync_dt_ms", np.nan)))
            and float(row.get("sync_dt_ms")) <= sync_tolerance_ms
        )
        r_camera_head = rotation_headpose(row) if pose_valid else None
        r_base_head = (
            r_base_camera @ r_camera_head if r_camera_head is not None else None
        )

        left = np.asarray([
            row.get("left_direction_x"), row.get("left_direction_y"),
            row.get("left_direction_z")
        ], dtype=float)
        right = np.asarray([
            row.get("right_direction_x"), row.get("right_direction_y"),
            row.get("right_direction_z")
        ], dtype=float)
        gaze_point = np.asarray([
            row.get("gaze3d_x_mm"), row.get("gaze3d_y_mm"),
            row.get("gaze3d_z_mm")
        ], dtype=float)
        left_origin = np.asarray([
            row.get("left_origin_x_mm"), row.get("left_origin_y_mm"),
            row.get("left_origin_z_mm")
        ], dtype=float)
        right_origin = np.asarray([
            row.get("right_origin_x_mm"), row.get("right_origin_y_mm"),
            row.get("right_origin_z_mm")
        ], dtype=float)
        left_export_valid = bool(row.get("left_eye_valid", False))
        right_export_valid = bool(row.get("right_eye_valid", False))
        left_direction_residual = angle_between_deg(
            left, gaze_point - left_origin
        )
        right_direction_residual = angle_between_deg(
            right, gaze_point - right_origin
        )
        left_direction_consistent = bool(
            np.isfinite(left_direction_residual)
            and left_direction_residual <= eye_direction_consistency_deg
        )
        right_direction_consistent = bool(
            np.isfinite(right_direction_residual)
            and right_direction_residual <= eye_direction_consistency_deg
        )
        left_eye_valid = left_export_valid and left_direction_consistent
        right_eye_valid = right_export_valid and right_direction_consistent
        any_export_eye_valid = left_export_valid or right_export_valid
        any_direction_eye_valid = left_eye_valid or right_eye_valid
        valid_origins = []
        valid_directions = []
        if left_export_valid and np.isfinite(left_origin).all():
            valid_origins.append(left_origin)
        if right_export_valid and np.isfinite(right_origin).all():
            valid_origins.append(right_origin)
        if left_eye_valid and np.isfinite(left).all():
            valid_directions.append(left)
        if right_eye_valid and np.isfinite(right).all():
            valid_directions.append(right)
        binocular_origin = (
            np.mean(valid_origins, axis=0)
            if valid_origins else np.full(3, np.nan)
        )

        method_a_raw = normalize(gaze_point - binocular_origin)
        method_b_raw = normalize(
            np.mean(valid_directions, axis=0)
            if valid_directions else np.full(3, np.nan)
        )
        x_norm = float(row.get("gaze_x_norm", np.nan))
        y_norm = float(row.get("gaze_y_norm", np.nan))
        method_c_raw = np.full(3, np.nan)
        c_in_image = (
            any_export_eye_valid and np.isfinite(x_norm) and np.isfinite(y_norm)
            and 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0
        )
        if c_in_image:
            u, v = x_norm * width, y_norm * height
            undistorted = cv2.undistortPoints(
                np.array([[[u, v]]], dtype=np.float64),
                scene_camera_matrix,
                scene_distortion,
            ).reshape(2)
            method_c_raw = normalize(
                np.array([undistorted[0], undistorted[1], 1.0])
            )

        directions_head = {
            "a": normalize(r_h_hucs @ method_a_raw),
            "b": normalize(r_h_hucs @ method_b_raw),
            "c": normalize(r_h_c1 @ method_c_raw),
        }
        num_head_tags = pd.to_numeric(row.get("num_head_tags"), errors="coerce")
        head_rmse_px = pd.to_numeric(row.get("head_rmse_px"), errors="coerce")
        result = {
            "t_unix_ns": int(row["camera_timestamp_ns"]),
            "camera_frame_idx": int(row["frame_idx"]),
            "camera_timestamp_ns": int(row["camera_timestamp_ns"]),
            "gaze_timestamp_ns": (
                int(row["gaze_timestamp_ns"])
                if pd.notna(row.get("gaze_timestamp_ns")) else None
            ),
            "sync_dt_ms": float(row["sync_dt_ms"]) if pd.notna(row.get("sync_dt_ms")) else np.nan,
            "validity_timestamp_ns": (
                int(row["validity_timestamp_ns"])
                if pd.notna(row.get("validity_timestamp_ns")) else None
            ),
            "validity_sync_dt_ms": (
                float(row["validity_sync_dt_ms"])
                if pd.notna(row.get("validity_sync_dt_ms")) else np.nan
            ),
            "left_eye_export_valid": left_export_valid,
            "right_eye_export_valid": right_export_valid,
            "left_eye_direction_residual_deg": left_direction_residual,
            "right_eye_direction_residual_deg": right_direction_residual,
            "left_eye_direction_consistent": left_direction_consistent,
            "right_eye_direction_consistent": right_direction_consistent,
            "left_eye_valid": left_eye_valid,
            "right_eye_valid": right_eye_valid,
            "any_eye_export_valid": any_export_eye_valid,
            "any_eye_direction_valid": any_direction_eye_valid,
            "method_b_eye_count": int(len(valid_directions)),
            "camera_frame_readable": str(row.get("status", "")) not in {
                "image_not_found", "image_read_failed"
            },
            "num_head_tags": int(num_head_tags) if pd.notna(num_head_tags) else 0,
            "head_rmse_px": float(head_rmse_px) if pd.notna(head_rmse_px) else np.nan,
            "head_quality_ok": bool(pd.to_numeric(row.get("head_quality_ok"), errors="coerce") == 1),
            "back_tag_available": bool(str(row.get("status")) == "ok"),
            "pose_and_time_valid": bool(pose_valid),
        }
        for method, direction_head in directions_head.items():
            raw_valid = bool(np.isfinite(direction_head).all())
            direction_base = (
                normalize(r_base_head @ direction_head)
                if pose_valid and raw_valid else np.full(3, np.nan)
            )
            horizontal_projection_valid = bool(
                np.isfinite(direction_base).all()
                and math.hypot(direction_base[0], direction_base[1]) > 1.0e-6
            )
            angle = (
                math.degrees(math.atan2(direction_base[1], direction_base[0]))
                if horizontal_projection_valid else np.nan
            )
            result[f"method_{method}_fields_valid"] = raw_valid
            result[f"method_{method}_horizontal_projection_valid"] = (
                horizontal_projection_valid
            )
            result[f"method_{method}_valid"] = bool(np.isfinite(angle))
            result[f"method_{method}_ego_angle_deg"] = angle
            result[f"method_{method}_dir_base_x"] = direction_base[0]
            result[f"method_{method}_dir_base_y"] = direction_base[1]
            result[f"method_{method}_dir_base_z"] = direction_base[2]
        result["gaze_2d_image_valid"] = bool(c_in_image)
        output_rows.append(result)
    return pd.DataFrame(output_rows)


def merged_duration_s(intervals, start_ns, end_ns):
    clipped = []
    for _, row in intervals.iterrows():
        start = max(start_ns, int(row["start_ns"]))
        end = min(end_ns, int(row["end_ns"]))
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / NS_PER_SECOND


def boolean_column(frame, column):
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.fillna(False).astype(str).str.lower().eq("true")


def timestamp_rate_summary(stream, timestamps, window_duration_s):
    values = pd.to_numeric(timestamps, errors="coerce").dropna()
    unique = np.sort(values.astype("int64").unique())
    intervals_ms = (
        np.diff(unique).astype(float) / 1.0e6
        if len(unique) >= 2 else np.asarray([], dtype=float)
    )
    span_s = (
        float(unique[-1] - unique[0]) / NS_PER_SECOND
        if len(unique) >= 2 else 0.0
    )
    return {
        "stream": stream,
        "sample_count": int(len(values)),
        "unique_timestamp_count": int(len(unique)),
        "span_s": span_s,
        "effective_rate_hz": (
            float((len(unique) - 1) / span_s) if span_s > 0 else None
        ),
        "sample_density_hz": (
            float(len(values) / window_duration_s)
            if window_duration_s > 0 else None
        ),
        "median_interval_ms": (
            float(np.median(intervals_ms)) if len(intervals_ms) else None
        ),
        "p95_interval_ms": (
            float(np.quantile(intervals_ms, 0.95))
            if len(intervals_ms) else None
        ),
        "maximum_interval_ms": (
            float(np.max(intervals_ms)) if len(intervals_ms) else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, type=Path)
    parser.add_argument("--common-dir", required=True, type=Path)
    parser.add_argument("--workflow-dir", required=True, type=Path)
    parser.add_argument("--riding-input-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fixation-table", required=True, type=Path)
    parser.add_argument(
        "--gnss-route", type=Path, default=None,
        help="GNSS RTK position table with record_ns, east_m, and north_m",
    )
    parser.add_argument("--rear-camera-calibration", type=Path, default=HEADPOSE_DIR / "camera.json")
    parser.add_argument("--head-rig-config", type=Path, default=HEADPOSE_DIR / "head_rig_config.json")
    parser.add_argument("--rig-calibration", type=Path, default=HEADPOSE_DIR / "rig_calib.json")
    parser.add_argument("--tobii-head-transforms", type=Path, default=HEADPOSE_DIR / "transforms.json")
    parser.add_argument("--bike-extrinsics", type=Path, default=SCRIPT_DIR / "bike_extrinsics.json")
    parser.add_argument("--headpose-script", type=Path, default=HEADPOSE_DIR / "estimate_headpose_from_frames.py")
    parser.add_argument("--min-head-tags", type=int, default=2)
    parser.add_argument("--max-head-rmse-px", type=float, default=5.0)
    parser.add_argument("--detector-threads", type=int, default=1)
    parser.add_argument("--sync-tolerance-ms", type=float, default=50.0)
    parser.add_argument("--eye-validity-tolerance-ms", type=float, default=2.0)
    parser.add_argument(
        "--eye-direction-consistency-deg", type=float, default=10.0
    )
    parser.add_argument(
        "--plot-duration-s",
        type=float,
        default=None,
        help=(
            "Optional record-time duration. All plotted streams and reported "
            "window statistics use this crop."
        ),
    )
    parser.add_argument(
        "--plot-start-video-s",
        type=float,
        default=None,
        help=(
            "Optional start on the AVI player timeline. The nominal-rate frame "
            "index is mapped through timestamps.csv to rosbag record time."
        ),
    )
    parser.add_argument(
        "--trajectory-event-label",
        default="Peak gaze deflection",
        help="Neutral label for the gaze event marked by the vertical line.",
    )
    parser.add_argument(
        "--gaze-event-offset-s",
        type=float,
        default=None,
        help=(
            "Optional selected gaze-event time relative to the plot start. "
            "The nearest quality-approved Method A estimate is used."
        ),
    )
    parser.add_argument(
        "--video-confirmed-event",
        action="store_true",
        help=(
            "Record that the marked gaze event was checked in the corresponding "
            "video frames."
        ),
    )
    parser.add_argument(
        "--figure-title",
        default="Gaze, rider input, and bicycle response during a selected turn",
    )
    args = parser.parse_args()
    args.session_dir = args.session_dir.resolve()
    args.common_dir = args.common_dir.resolve()
    args.workflow_dir = args.workflow_dir.resolve()
    args.riding_input_dir = args.riding_input_dir.resolve()
    args.out = args.out.resolve()
    for name in (
        "fixation_table", "rear_camera_calibration", "head_rig_config",
        "rig_calibration", "tobii_head_transforms", "bike_extrinsics",
        "headpose_script",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.gnss_route is None:
        args.gnss_route = (
            args.workflow_dir.parent
            / "ego_motion" / "tables" / "gnss_route_epochs.csv"
        ).resolve()
    else:
        args.gnss_route = args.gnss_route.resolve()
    interval = read_json(args.workflow_dir / "tables" / "video_interval_global_time.json")
    source_start_ns = int(interval["start_ns"])
    source_end_ns = int(interval["end_ns"])
    camera_dir = find_one(args.session_dir, "camera_*")
    video = camera_dir / "video_mjpg.avi"
    timestamps = camera_dir / "timestamps.csv"
    video_clock = load_video_clock(video, timestamps)

    window_basis = "source workflow interval"
    start_ns, end_ns = source_start_ns, source_end_ns
    common_start_ns = None
    common_end_ns = None
    requested_video_start_s = None
    if args.plot_start_video_s is not None:
        if args.plot_duration_s is None:
            raise ValueError(
                "--plot-duration-s is required with --plot-start-video-s"
            )
        if (
            not math.isfinite(args.plot_start_video_s)
            or args.plot_start_video_s < 0
        ):
            raise ValueError("--plot-start-video-s must be finite and non-negative")
        common_metadata_path = (
            args.common_dir.parents[1]
            / "tables" / "common_interval_and_steering_cleaning.json"
        )
        if not common_metadata_path.is_file():
            raise RuntimeError(
                "Cannot resolve the full common interval: "
                f"{common_metadata_path}"
            )
        common_metadata = read_json(common_metadata_path)
        common_start_ns = int(common_metadata["start_ns"])
        common_end_ns = int(common_metadata["end_ns"])
        start_ns, _ = playback_seconds_to_unix_ns(
            video_clock, args.plot_start_video_s
        )
        requested_video_start_s = float(args.plot_start_video_s)
        window_basis = "AVI playback frame mapped to record time"
    if args.plot_duration_s is not None:
        if not math.isfinite(args.plot_duration_s) or args.plot_duration_s <= 0:
            raise ValueError("--plot-duration-s must be positive and finite")
        requested_end_ns = start_ns + int(
            round(args.plot_duration_s * NS_PER_SECOND)
        )
        upper_bound_ns = (
            common_end_ns if args.plot_start_video_s is not None
            else source_end_ns
        )
        if requested_end_ns > upper_bound_ns:
            raise ValueError(
                "Requested plot duration exceeds the available analysis interval"
            )
        end_ns = requested_end_ns
    if common_start_ns is not None and start_ns < common_start_ns:
        raise ValueError("Requested video start precedes the common analysis interval")
    actual_video_start_s = unix_ns_to_playback_seconds(video_clock, start_ns)
    actual_video_end_s = unix_ns_to_playback_seconds(video_clock, end_ns)
    annotation_kind = (
        "attention"
        if "attention" in args.fixation_table.stem.casefold()
        else "fixation"
    )
    annotation_label = (
        "Tobii attention intervals"
        if annotation_kind == "attention"
        else "Tobii I-VT fixations"
    )

    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.mkdir(parents=True)
    figures, tables = args.out / "figures", args.out / "tables"
    figures.mkdir()
    tables.mkdir()

    frames, camera_window, extraction = extract_frame_window(
        video, timestamps, start_ns, end_ns, args.out
    )
    headpose_path = tables / "headpose_window.csv"
    headpose_run = run_headpose(
        frames, args.out / "camera_timestamps_window.csv", headpose_path, args
    )

    recording = find_one(args.session_dir, "*/recording.g3")
    recording_data = read_json(recording)
    created = datetime.fromisoformat(
        recording_data["created"].replace("Z", "+00:00")
    )
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))
    gaze_path = find_one(args.common_dir, "gazedata.gz")
    gaze = P9.read_tobii_gaze(gaze_path, created_ns)
    (
        fixation_intervals,
        fixation_summary,
        eye_validity,
        eye_validity_summary,
    ) = P9.read_tobii_fixation_and_eye_validity(
        args.fixation_table, recording_data["created"]
    )
    gaze_ego = build_gaze_ego(
        gaze, eye_validity, pd.read_csv(headpose_path), camera_window, recording,
        args.tobii_head_transforms, args.bike_extrinsics,
        args.sync_tolerance_ms, args.eye_validity_tolerance_ms,
        args.eye_direction_consistency_deg,
    )
    gaze_ego.to_csv(tables / "gaze_bike_ego_series.csv", index=False)

    selected_eye_validity = eye_validity[
        eye_validity["t_unix_ns"].between(start_ns, end_ns)
    ].copy()
    selected_eye_validity.to_csv(
        tables / "tobii_eye_validity_window.csv", index=False
    )
    eye_validity_summary["selected_window_row_count"] = int(
        len(selected_eye_validity)
    )
    eye_validity_summary["selected_window_any_eye_valid_count"] = int(
        (
            selected_eye_validity["left_eye_valid"]
            | selected_eye_validity["right_eye_valid"]
        ).sum()
    )
    eye_validity_summary["selected_window_both_eyes_valid_count"] = int(
        (
            selected_eye_validity["left_eye_valid"]
            & selected_eye_validity["right_eye_valid"]
        ).sum()
    )
    P9.write_json(
        tables / "tobii_eye_validity_summary.json", eye_validity_summary
    )
    selected_fixations = fixation_intervals[
        (fixation_intervals["end_ns"] >= start_ns)
        & (fixation_intervals["start_ns"] <= end_ns)
    ].copy()
    selected_fixations.to_csv(tables / "tobii_fixation_intervals.csv", index=False)
    fixation_duration = merged_duration_s(selected_fixations, start_ns, end_ns)
    fixation_summary.update({
        "selected_window_event_count": int(len(selected_fixations)),
        "selected_window_covered_duration_s": fixation_duration,
        "selected_window_covered_fraction": fixation_duration / ((end_ns - start_ns) / NS_PER_SECOND),
    })
    P9.write_json(tables / "tobii_fixation_summary.json", fixation_summary)

    steering = P9.read_csv(find_one(args.common_dir, "steering_angle_*.csv"))
    brake = P9.read_csv(find_one(args.common_dir, "brake_sensors_force_*.csv"))
    imu = P9.read_csv(find_one(args.common_dir, "imu_*.csv"))
    power = P9.read_csv(find_one(args.common_dir, "rally_payload_decoded_*.csv"))
    wheel = P9.read_csv(find_one(args.common_dir, "speed_decoded_*.csv"))
    velocity = P9.read_csv(find_one(args.common_dir, "ubx_nav_vel_ned.csv"))
    if "record_ns" in velocity.columns:
        velocity["t_unix_ns"] = pd.to_numeric(
            velocity["record_ns"], errors="coerce"
        )
        velocity = velocity[
            velocity["t_unix_ns"].notna() & (velocity["t_unix_ns"] > 0)
        ].copy()
        velocity["t_unix_ns"] = velocity["t_unix_ns"].astype("int64")
        velocity = velocity.sort_values("t_unix_ns").drop_duplicates(
            "t_unix_ns"
        )

    route_source = pd.read_csv(args.gnss_route, low_memory=False)
    required_route_columns = {"record_ns", "east_m", "north_m"}
    missing_route_columns = required_route_columns - set(route_source.columns)
    if missing_route_columns:
        raise RuntimeError(
            f"GNSS route table is missing: {sorted(missing_route_columns)}"
        )
    trajectory = route_source.rename(columns={"record_ns": "t_unix_ns"}).copy()
    for column in ("t_unix_ns", "east_m", "north_m"):
        trajectory[column] = pd.to_numeric(trajectory[column], errors="coerce")
    trajectory = trajectory.dropna(
        subset=["t_unix_ns", "east_m", "north_m"]
    ).copy()
    trajectory["t_unix_ns"] = trajectory["t_unix_ns"].astype("int64")
    trajectory = trajectory[
        trajectory["t_unix_ns"].between(start_ns, end_ns)
    ].sort_values("t_unix_ns").drop_duplicates("t_unix_ns")
    if len(trajectory) < 2:
        raise RuntimeError(
            "Fewer than two valid GNSS RTK positions in the plot window"
        )
    trajectory["elapsed_s"] = (
        trajectory["t_unix_ns"] - start_ns
    ) / NS_PER_SECOND
    trajectory["relative_east_m"] = (
        trajectory["east_m"] - float(trajectory["east_m"].iloc[0])
    )
    trajectory["relative_north_m"] = (
        trajectory["north_m"] - float(trajectory["north_m"].iloc[0])
    )
    trajectory.to_csv(tables / "close_loop_gnss_trajectory.csv", index=False)

    wheel["wheel_speed_mps"] = pd.to_numeric(wheel["speed_mps"], errors="coerce") / 3.6
    raw_imu = imu[pd.to_numeric(imu["dtype"], errors="coerce").eq(64)].copy()
    power_p10 = power[power["page_name"].eq("standard_power")].copy()
    steering_neutral = read_json(
        args.riding_input_dir / "tables" / "steering_neutral_reference.json"
    )
    brake_band_table = pd.read_csv(
        args.riding_input_dir / "tables" / "brake_zero_input_band.csv"
    )
    brake_bands = {
        str(row["force_column"]): (
            float(row["zero_band_lower_n"]), float(row["zero_band_upper_n"])
        ) for _, row in brake_band_table.iterrows()
    }

    method_a = pd.to_numeric(
        gaze_ego["method_a_ego_angle_deg"], errors="coerce"
    )
    valid_method_a = method_a.notna()
    if not valid_method_a.any():
        raise RuntimeError("No quality-approved Method A gaze estimate in window")
    if args.gaze_event_offset_s is not None:
        if (
            not math.isfinite(args.gaze_event_offset_s)
            or not 0.0 <= args.gaze_event_offset_s <= (end_ns - start_ns) / NS_PER_SECOND
        ):
            raise ValueError("--gaze-event-offset-s is outside the plot window")
        requested_gaze_event_ns = start_ns + int(round(
            args.gaze_event_offset_s * NS_PER_SECOND
        ))
        valid_indices = gaze_ego.index[valid_method_a]
        gaze_peak_index = valid_indices[
            np.argmin(np.abs(
                gaze_ego.loc[valid_indices, "t_unix_ns"].to_numpy(dtype=np.int64)
                - requested_gaze_event_ns
            ))
        ]
        gaze_event_selection = "explicit candidate-screening offset"
    else:
        gaze_peak_index = method_a.abs().idxmax()
        gaze_event_selection = "largest absolute accepted Method A angle"
    gaze_peak_ns = int(gaze_ego.loc[gaze_peak_index, "t_unix_ns"])

    P9.plot_closed_loop(
        gaze, steering, brake, power_p10, raw_imu,
        P9.gnss_course_rate(velocity), velocity, wheel,
        start_ns, end_ns, figures / "rider_input_and_bicycle_response",
        args.bike_extrinsics,
        steering_neutral=steering_neutral,
        brake_bands=brake_bands,
        fixation_intervals=selected_fixations,
        fixation_label=annotation_label,
        precomputed_gaze_angles=gaze_ego,
        gaze_axis_label="Bike ego-frame\ngaze angle (deg)",
        trajectory=trajectory,
        trajectory_event_label=args.trajectory_event_label,
        trajectory_event_time_ns=gaze_peak_ns,
        trajectory_layout="components_last",
        figure_title=args.figure_title,
    )

    route_steps = np.hypot(
        np.diff(trajectory["east_m"].to_numpy(dtype=float)),
        np.diff(trajectory["north_m"].to_numpy(dtype=float)),
    )
    route_nearest_index = int(
        np.argmin(
            np.abs(
                trajectory["t_unix_ns"].to_numpy(dtype=np.int64)
                - gaze_peak_ns
            )
        )
    )
    route_nearest = trajectory.iloc[route_nearest_index]
    route_summary = {
        "source_topic": "/ubx_nav_pvt",
        "time_source": "rosbag record_ns",
        "pvt_epoch_count": int(len(trajectory)),
        "first_elapsed_s": float(trajectory["elapsed_s"].iloc[0]),
        "last_elapsed_s": float(trajectory["elapsed_s"].iloc[-1]),
        "covered_duration_s": float(
            trajectory["elapsed_s"].iloc[-1]
            - trajectory["elapsed_s"].iloc[0]
        ),
        "path_length_m": float(route_steps.sum()),
        "net_displacement_m": float(math.hypot(
            trajectory["relative_east_m"].iloc[-1],
            trajectory["relative_north_m"].iloc[-1],
        )),
        "selected_gaze_event": {
            "camera_elapsed_s": (gaze_peak_ns - start_ns) / NS_PER_SECOND,
            "method_a_angle_deg": float(method_a.loc[gaze_peak_index]),
            "direction": (
                "left" if float(method_a.loc[gaze_peak_index]) >= 0.0
                else "right"
            ),
            "label": args.trajectory_event_label,
            "selection": gaze_event_selection,
            "requested_elapsed_s": args.gaze_event_offset_s,
            "video_confirmed": bool(args.video_confirmed_event),
            "nearest_pvt_time_difference_ms": abs(
                int(route_nearest["t_unix_ns"]) - gaze_peak_ns
            ) / 1.0e6,
            "relative_east_m": float(route_nearest["relative_east_m"]),
            "relative_north_m": float(route_nearest["relative_north_m"]),
        },
        "interpretation": (
            "GNSS receiver positions with RTK fixed or float carrier solutions "
            "provide east and north components as "
            "time-aligned trajectory context; they are not ground truth or a fused "
            "trajectory."
        ),
        "plot_layout": "relative east versus time and relative north versus time",
    }
    P9.write_json(
        tables / "close_loop_gnss_trajectory_summary.json", route_summary
    )

    event_search_end_ns = min(
        end_ns, gaze_peak_ns + int(round(4.0 * NS_PER_SECOND))
    )
    steering_after_gaze = steering[
        steering["t_unix_ns"].between(gaze_peak_ns, event_search_end_ns)
    ].copy()
    relative_steering = (
        pd.to_numeric(steering_after_gaze["angle_deg"], errors="coerce")
        - float(steering_neutral["neutral_angle_deg"])
    )
    steering_peak = smoothed_absolute_peak(
        steering_after_gaze, "t_unix_ns", relative_steering, 5
    )
    yaw_after_gaze = raw_imu[
        raw_imu["t_unix_ns"].between(gaze_peak_ns, event_search_end_ns)
    ].copy()
    yaw_peak = smoothed_absolute_peak(
        yaw_after_gaze, "t_unix_ns", yaw_after_gaze["gyro_z"], 9
    )
    for peak in (steering_peak, yaw_peak):
        if peak is not None:
            peak["elapsed_s"] = (peak["time_ns"] - start_ns) / NS_PER_SECOND

    course_window = velocity[
        velocity["t_unix_ns"].between(start_ns, end_ns)
    ].copy()
    course_window["course_deg_numeric"] = pd.to_numeric(
        course_window["course_deg"], errors="coerce"
    )
    course_window["speed_numeric"] = pd.to_numeric(
        course_window["ground_speed_mps"], errors="coerce"
    )
    course_window["accuracy_numeric"] = pd.to_numeric(
        course_window["course_accuracy_deg"], errors="coerce"
    )
    course_window = course_window[
        course_window["course_deg_numeric"].notna()
        & course_window["speed_numeric"].ge(2.0)
        & course_window["accuracy_numeric"].le(30.0)
    ].copy()
    course_window["elapsed_s"] = (
        course_window["t_unix_ns"] - start_ns
    ) / NS_PER_SECOND
    course_start = course_window[
        course_window["elapsed_s"] <= 1.5
    ]["course_deg_numeric"]
    course_end = course_window[
        course_window["elapsed_s"]
        >= (end_ns - start_ns) / NS_PER_SECOND - 1.5
    ]["course_deg_numeric"]
    course_change_deg = None
    course_start_deg = None
    course_end_deg = None
    if len(course_start) and len(course_end):
        course_start_deg = float(course_start.median())
        course_end_deg = float(course_end.median())
        course_change_deg = wrapped_angle_difference_deg(
            course_end_deg, course_start_deg
        )

    sequence_summary = {
        "purpose": (
            "Descriptive screening of temporal order; it is not a causal "
            "driver-response estimate."
        ),
        "gaze_event": {
            "elapsed_s": (gaze_peak_ns - start_ns) / NS_PER_SECOND,
            "method_a_angle_deg": float(method_a.loc[gaze_peak_index]),
            "selection": gaze_event_selection,
        },
        "search_after_gaze_s": 4.0,
        "relative_steering_peak": steering_peak,
        "raw_gyro_z_peak": yaw_peak,
        "peak_time_differences_s": {
            "gaze_to_steering": (
                (steering_peak["time_ns"] - gaze_peak_ns) / NS_PER_SECOND
                if steering_peak is not None else None
            ),
            "steering_to_yaw_rate": (
                (yaw_peak["time_ns"] - steering_peak["time_ns"])
                / NS_PER_SECOND
                if steering_peak is not None and yaw_peak is not None else None
            ),
        },
        "gnss_derived_course_over_ground": {
            "quality_gate": "speed >= 2 m/s and course accuracy <= 30 deg",
            "start_median_deg": course_start_deg,
            "end_median_deg": course_end_deg,
            "signed_change_deg": course_change_deg,
            "interpretation": (
                "Receiver-derived turn context, not an independent heading "
                "ground truth."
            ),
        },
    }
    P9.write_json(tables / "cycling_sequence_summary.json", sequence_summary)

    pose_valid = gaze_ego["pose_and_time_valid"].fillna(False).astype(bool)
    summary_rows = []
    for method in ("a", "b", "c"):
        fields = gaze_ego[f"method_{method}_fields_valid"].fillna(False).astype(bool)
        final = gaze_ego[f"method_{method}_valid"].fillna(False).astype(bool)
        summary_rows.append({
            "method": f"Method {method.upper()}",
            "camera_frames_in_window": int(len(gaze_ego)),
            "pose_and_time_valid": int(pose_valid.sum()),
            "gaze_fields_valid_among_all_frames": int(fields.sum()),
            "final_valid": int(final.sum()),
            "final_valid_percent_of_camera_frames": 100.0 * float(final.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "gaze_bike_ego_validity_summary.csv", index=False)

    window_duration_s = (end_ns - start_ns) / NS_PER_SECOND
    gaze_window = gaze[
        gaze["t_unix_ns"].between(start_ns, end_ns)
    ].copy()
    rate_rows = [
        timestamp_rate_summary(
            "Raw Tobii gaze", gaze_window["t_unix_ns"], window_duration_s
        ),
        timestamp_rate_summary(
            "Rear camera", camera_window["unix_ns"], window_duration_s
        ),
    ]
    for method in ("a", "b", "c"):
        method_valid = boolean_column(gaze_ego, f"method_{method}_valid")
        rate_rows.append(timestamp_rate_summary(
            f"Method {method.upper()} valid output",
            gaze_ego.loc[method_valid, "camera_timestamp_ns"],
            window_duration_s,
        ))
    rate_table = pd.DataFrame(rate_rows)
    rate_table.to_csv(tables / "gaze_output_rate_summary.csv", index=False)

    left_export = boolean_column(gaze_ego, "left_eye_export_valid")
    right_export = boolean_column(gaze_ego, "right_eye_export_valid")
    both_export = left_export & right_export
    left_only_export = left_export & ~right_export
    right_only_export = ~left_export & right_export
    neither_export = ~left_export & ~right_export
    camera_eye_rows = [
        {
            "eye_validity_category": "both eyes valid",
            "camera_frame_count": int(both_export.sum()),
        },
        {
            "eye_validity_category": "left eye only valid",
            "camera_frame_count": int(left_only_export.sum()),
        },
        {
            "eye_validity_category": "right eye only valid",
            "camera_frame_count": int(right_only_export.sum()),
        },
        {
            "eye_validity_category": "both eyes invalid",
            "camera_frame_count": int(neither_export.sum()),
        },
    ]
    camera_eye_table = pd.DataFrame(camera_eye_rows)
    camera_eye_table["percent_of_camera_frames"] = (
        100.0 * camera_eye_table["camera_frame_count"] / len(gaze_ego)
    )
    camera_eye_table.to_csv(
        tables / "gaze_camera_eye_validity_breakdown.csv", index=False
    )

    method_eye_breakdown = {}
    for method in ("a", "b", "c"):
        final = boolean_column(gaze_ego, f"method_{method}_valid")
        method_eye_breakdown[f"method_{method}"] = {
            "final_valid_count": int(final.sum()),
            "both_export_valid_count": int((final & both_export).sum()),
            "single_export_valid_count": int(
                (final & (left_only_export | right_only_export)).sum()
            ),
            "both_export_invalid_count": int(
                (final & neither_export).sum()
            ),
        }
    method_b_valid = boolean_column(gaze_ego, "method_b_valid")
    method_b_eye_count = pd.to_numeric(
        gaze_ego["method_b_eye_count"], errors="coerce"
    )
    method_eye_breakdown["method_b"].update({
        "two_accepted_direction_rays": int(
            (method_b_valid & method_b_eye_count.eq(2)).sum()
        ),
        "one_accepted_direction_ray": int(
            (method_b_valid & method_b_eye_count.eq(1)).sum()
        ),
    })

    paired_gaze = pd.to_numeric(
        gaze_ego["gaze_timestamp_ns"], errors="coerce"
    ).dropna().astype("int64")
    sync_dt = pd.to_numeric(
        gaze_ego["sync_dt_ms"], errors="coerce"
    ).dropna()
    temporal_summary = {
        "window_duration_s": window_duration_s,
        "pairing": {
            "direction": "camera timestamp to nearest raw Tobii gaze timestamp",
            "maximum_allowed_difference_ms": args.sync_tolerance_ms,
            "interpolation_or_upsampling": False,
            "camera_frame_count": int(len(gaze_ego)),
            "matched_camera_frame_count": int(len(paired_gaze)),
            "unique_matched_gaze_timestamp_count": int(
                paired_gaze.nunique()
            ),
            "reused_gaze_sample_count_in_this_window": int(
                len(paired_gaze) - paired_gaze.nunique()
            ),
            "raw_gaze_samples_not_selected_for_camera_frames": int(
                len(gaze_window) - paired_gaze.nunique()
            ),
            "absolute_difference_ms": {
                "median": float(sync_dt.median()),
                "p95": float(sync_dt.quantile(0.95)),
                "maximum": float(sync_dt.max()),
            },
        },
        "rates": {
            row["stream"]: row for row in rate_rows
        },
        "camera_matched_eye_validity": {
            "both_eyes_valid": int(both_export.sum()),
            "exactly_one_eye_valid": int(
                (left_only_export | right_only_export).sum()
            ),
            "both_eyes_invalid": int(neither_export.sum()),
        },
        "method_output_by_eye_validity": method_eye_breakdown,
        "interpretation": (
            "Bike-frame Methods A--C are camera-rate products. A single "
            "valid-eye fallback is allowed, but a frame with both eyes "
            "invalid is never accepted."
        ),
    }
    P9.write_json(
        tables / "gaze_temporal_alignment_summary.json",
        temporal_summary,
    )
    eye_quality_rows = []
    for eye in ("left", "right"):
        export_valid = gaze_ego[f"{eye}_eye_export_valid"].fillna(False).astype(bool)
        consistent = gaze_ego[
            f"{eye}_eye_direction_consistent"
        ].fillna(False).astype(bool)
        accepted = gaze_ego[f"{eye}_eye_valid"].fillna(False).astype(bool)
        residual = pd.to_numeric(
            gaze_ego[f"{eye}_eye_direction_residual_deg"], errors="coerce"
        )
        eye_quality_rows.append({
            "eye": eye,
            "camera_frames_in_window": int(len(gaze_ego)),
            "export_valid": int(export_valid.sum()),
            "export_valid_but_internally_inconsistent": int(
                (export_valid & ~consistent).sum()
            ),
            "accepted_for_direction_ray": int(accepted.sum()),
            "direction_residual_median_deg": float(residual.median()),
            "direction_residual_p95_deg": float(residual.quantile(0.95)),
            "direction_residual_max_deg": float(residual.max()),
        })
    eye_quality = pd.DataFrame(eye_quality_rows)
    eye_quality.to_csv(tables / "tobii_eye_direction_quality.csv", index=False)
    inconsistent_eye_record_count = sum(
        row["export_valid_but_internally_inconsistent"]
        for row in eye_quality_rows
    )
    headpose = pd.read_csv(headpose_path)
    status_summary = headpose["status"].fillna("missing").value_counts().rename_axis(
        "status"
    ).reset_index(name="frame_count")
    status_summary.to_csv(tables / "headpose_status_summary.csv", index=False)
    head_tags = pd.to_numeric(headpose["num_head_tags"], errors="coerce").fillna(0)
    head_rmse = pd.to_numeric(headpose["head_rmse_px"], errors="coerce")
    head_quality = pd.to_numeric(
        headpose["head_quality_ok"], errors="coerce"
    ).eq(1)
    readable = ~headpose["status"].fillna("").isin(
        ["image_not_found", "image_read_failed"]
    )
    head_quality_breakdown = pd.DataFrame([
        {"quality_stage": "camera frame available and readable", "frame_count": int(readable.sum())},
        {"quality_stage": f"at least {args.min_head_tags} helmet tags", "frame_count": int((readable & head_tags.ge(args.min_head_tags)).sum())},
        {"quality_stage": f"finite reprojection RMSE <= {args.max_head_rmse_px:g} px", "frame_count": int((readable & head_tags.ge(args.min_head_tags) & head_rmse.le(args.max_head_rmse_px)).sum())},
        {"quality_stage": "final accepted head pose", "frame_count": int(head_quality.sum())},
    ])
    head_quality_breakdown["percent_of_camera_frames"] = (
        100.0 * head_quality_breakdown["frame_count"] / len(headpose)
    )
    head_quality_breakdown.to_csv(
        tables / "headpose_quality_breakdown.csv", index=False
    )

    rows = {row["method"]: row for row in summary_rows}
    rates = temporal_summary["rates"]
    raw_rate = rates["Raw Tobii gaze"]["effective_rate_hz"]
    camera_rate = rates["Rear camera"]["effective_rate_hz"]
    method_a_rate = rates["Method A valid output"]["effective_rate_hz"]
    method_c_rate = rates["Method C valid output"]["effective_rate_hz"]
    camera_any_eye = int((left_export | right_export).sum())
    camera_single_eye = int(
        (left_only_export | right_only_export).sum()
    )
    text = (
        "Figure X presents the rider inputs and bicycle response during the "
        f"selected {window_duration_s:.0f} s scenario from one recording. "
        "Panels (g) and (h) show the east and north components of the GNSS "
        "RTK position, $x_{\\mathrm{ENU}}(t)$ and $y_{\\mathrm{ENU}}(t)$, "
        "obtained from receiver positions with "
        "RTK fixed or float carrier solutions. "
        "The first valid position is used as the local origin. Both panels use "
        "the same elapsed record-time axis as the other sensor signals. "
        f"The panels contain {len(trajectory):,} valid GNSS position epochs and "
        f"cover {route_summary['covered_duration_s']:.2f} s of the "
        f"{window_duration_s:.2f} s analysis interval. These positions provide "
        "trajectory context and are not treated as ground truth.\n\n"

        "Panel (a) shows horizontal gaze angles in the nominal bicycle ego frame "
        "($x$ forward and $y$ left). Rear-camera frames were processed with the "
        "AprilTag head-pose estimator. A camera pose was retained when the image "
        f"was readable, at least {args.min_head_tags} helmet tags were detected, "
        f"and the head-pose reprojection RMSE was no greater than {args.max_head_rmse_px:g} px. "
        f"This condition was satisfied by {int(pose_valid.sum()):,} of the "
        f"{len(gaze_ego):,} analysed rear-camera frames. The raw Tobii stream "
        f"contained {len(gaze_window):,} gaze samples at {raw_rate:.2f} Hz, "
        f"while the rear camera provided {len(gaze_ego):,} frames at "
        f"{camera_rate:.2f} Hz. Methods A--C were calculated once per camera "
        "frame because the dynamic head pose was obtained from the image. "
        "Their temporal resolution is therefore limited by the camera rate.\n\n"

        f"Each camera timestamp was paired with the nearest raw Tobii gaze "
        f"sample within {args.sync_tolerance_ms:g} ms. No interpolation or "
        f"up-sampling was applied. All {len(paired_gaze):,} camera frames were "
        "paired with different Tobii samples, and no gaze sample was reused. "
        f"The other {len(gaze_window) - paired_gaze.nunique():,} raw Tobii "
        "samples were not converted to bicycle-frame angles because no dynamic "
        "head pose was available between camera frames. "
        "The absolute camera--gaze time difference had a median of "
        f"{gaze_ego['sync_dt_ms'].median():.2f} ms, a 95th percentile of "
        f"{gaze_ego['sync_dt_ms'].quantile(0.95):.2f} ms, and a maximum of "
        f"{gaze_ego['sync_dt_ms'].max():.2f} ms. The lower camera rate does not "
        "indicate a synchronization error, but rapid head or gaze changes "
        "between camera frames cannot be represented. After all checks, "
        f"Methods A and B each provided {rows['Method A']['final_valid']:,} "
        f"valid estimates ({rows['Method A']['final_valid_percent_of_camera_frames']:.2f}\\%; "
        f"{method_a_rate:.2f} Hz). Method C provided "
        f"{rows['Method C']['final_valid']:,} estimates "
        f"({rows['Method C']['final_valid_percent_of_camera_frames']:.2f}\\%; "
        f"{method_c_rate:.2f} Hz).\n\n"

        "Method A uses the Tobii 3-D "
        "gaze point relative to the mean pupil origin of the accepted eye(s). "
        "Method B uses eye-direction rays that pass the Tobii validity flag and "
        f"a ${args.eye_direction_consistency_deg:g}^\\circ$ internal consistency "
        "check against the corresponding 3-D gaze ray. It averages both eyes "
        "when both pass and otherwise uses the accepted eye. Method C "
        "back-projects the normalized 2-D gaze point using the distortion-corrected "
        "P1 scene-camera intrinsics. All three directions were transformed through "
        "the dynamic head pose and fixed rear-camera-to-bicycle rotation before "
        "calculating $\\operatorname{atan2}(d_y,d_x)$. The gaze geometry was "
        "read from the raw \\texttt{gazedata.gz} stream. Eye-validity labels "
        f"and I-VT {annotation_kind} intervals were read from the Tobii tabular export. "
        f"The {annotation_kind} intervals are background annotations and are not used to "
        "calculate the gaze angles.\n\n"

        "The Tobii eye-validity percentages and the end-to-end workflow "
        "availability use different denominators. In the raw eye-tracker "
        f"stream, at least one eye was marked \\texttt{{Valid}} in "
        f"{eye_validity_summary['selected_window_any_eye_valid_count']:,} samples "
        f"({100.0 * eye_validity_summary['selected_window_any_eye_valid_count'] / eye_validity_summary['selected_window_row_count']:.2f}\\%), "
        f"and both eyes were marked \\texttt{{Valid}} in "
        f"{eye_validity_summary['selected_window_both_eyes_valid_count']:,} samples "
        f"({100.0 * eye_validity_summary['selected_window_both_eyes_valid_count'] / eye_validity_summary['selected_window_row_count']:.2f}\\%). "
        f"Among the {len(gaze_ego):,} camera-matched samples, "
        f"{int(both_export.sum()):,} had two valid eyes, "
        f"{camera_single_eye:,} had only one valid eye, and "
        f"{int(neither_export.sum()):,} had no valid eye. The workflow requires "
        "at least one valid eye, not two. "
        f"{method_eye_breakdown['method_a']['single_export_valid_count']:,} of "
        f"the {camera_single_eye:,} monocular samples still produced valid "
        "Method A and Method B estimates: the invalid eye was excluded and the "
        "valid eye was used. No sample with both eyes invalid was accepted. "
        f"For Method B, {method_eye_breakdown['method_b']['two_accepted_direction_rays']:,} "
        "outputs used two accepted eye-direction rays and "
        f"{method_eye_breakdown['method_b']['one_accepted_direction_ray']:,} "
        "used one accepted ray. "
        "This monocular fallback, together with the different denominator, "
        "explains why the Method A/B camera-frame workflow availability "
        f"({rows['Method A']['final_valid_percent_of_camera_frames']:.2f}\\%) "
        "can be higher than the raw binocular-valid fraction "
        f"({100.0 * eye_validity_summary['selected_window_both_eyes_valid_count'] / eye_validity_summary['selected_window_row_count']:.2f}\\%). "
        "The directly "
        f"comparable camera-frame any-eye availability is "
        f"{camera_any_eye:,}/{len(gaze_ego):,} "
        f"({100.0 * camera_any_eye / len(gaze_ego):.2f}\\%); the reduction to "
        "the final workflow availability results from the additional head-pose, "
        "temporal-pairing, geometric, and projection checks. Missing or rejected "
        "values are shown as line breaks and are not interpolated.\n\n"

        f"The selected Method A gaze deflection occurred near "
        f"{route_summary['selected_gaze_event']['camera_elapsed_s']:.1f} s "
        f"and reached ${abs(route_summary['selected_gaze_event']['method_a_angle_deg']):.0f}^\\circ$ "
        f"to the {route_summary['selected_gaze_event']['direction']}. "
        + (
            "The corresponding video frames were checked and confirm a real "
            "head turn associated with the gaze deflection. "
            if args.video_confirmed_event else
            "This event is identified from the quality-approved gaze series; "
            "no manual behavioural label is assigned. "
        )
        + "The vertical dotted line marks the event in all panels. "
        "The corresponding receiver position was approximately "
        f"({route_summary['selected_gaze_event']['relative_east_m']:.1f}, "
        f"{route_summary['selected_gaze_event']['relative_north_m']:.1f}) m in the "
        "relative ENU frame. The Method B "
        f"consistency check rejected {inconsistent_eye_record_count:,} individual "
        "eye records that Tobii marked "
        "as \\texttt{Valid} but that were inconsistent with the 3-D gaze geometry. Shaded "
        f"intervals indicate Tobii I-VT {annotation_kind} and do not alter the gaze-angle "
        "calculations. The back tag was not required because it describes the "
        "rider's torso frame rather than the bicycle frame. "
        "\n\nMethods A--C are derived from the same Tobii recording and share the same "
        "head-pose and mounting transformations. Their agreement therefore "
        "indicates internal consistency, but it is not an independent validation "
        "of gaze accuracy. The calculation reuses an existing Tobii-to-helmet "
        "calibration and a nominal rear-camera-to-bicycle extrinsic calibration. "
        "The result should therefore be interpreted as a nominal bicycle ego-frame "
        "gaze workflow and data-availability assessment rather than an absolute "
        "angular-accuracy evaluation."
    )
    if steering_peak is not None and yaw_peak is not None:
        gaze_to_steering_s = sequence_summary[
            "peak_time_differences_s"
        ]["gaze_to_steering"]
        steering_to_yaw_s = sequence_summary[
            "peak_time_differences_s"
        ]["steering_to_yaw_rate"]
        text += (
            "\n\nIn this selected scenario, the Method A gaze deflection was "
            f"marked at {(gaze_peak_ns - start_ns) / NS_PER_SECOND:.2f} s. "
            "Within the following four seconds, the main relative-steering "
            f"peak occurred at {steering_peak['elapsed_s']:.2f} s "
            f"({steering_peak['smoothed_value']:.2f}$^\\circ$), and the main "
            f"raw-gyroscope $z$ peak occurred at {yaw_peak['elapsed_s']:.2f} s "
            f"({yaw_peak['smoothed_value']:.3f} rad s$^{{-1}}$). The selected "
            f"gaze peak preceded the steering peak by {gaze_to_steering_s:.2f} s. "
            "The steering and yaw-rate peaks were separated by only "
            f"{abs(steering_to_yaw_s):.3f} s and are therefore treated as "
            "near-synchronous input and bicycle-response signatures at the "
            "available sampling rates. "
        )
        if course_change_deg is not None:
            text += (
                "The receiver-derived course over ground changed by "
                f"{course_change_deg:+.1f}$^\\circ$ across the interval, "
                "which provides trajectory context that a turn occurred. "
            )
        text += (
            "The temporal order is descriptive and does not establish a causal "
            "delay between gaze, rider input, and bicycle motion."
        )
    (args.out / "technical_validation_gaze_text.txt").write_text(text + "\n", encoding="utf-8")
    caption = (
        "Rider inputs and bicycle response during a selected "
        f"{window_duration_s:.0f} s dynamic scenario. "
        "(a) Horizontal gaze "
        "angles in the nominal bicycle ego frame obtained "
        f"using Methods A--C; shaded regions indicate Tobii I-VT {annotation_kind}. "
        "(b) Steering angle relative to the estimated neutral position. "
        "(c) Raw gyroscope $z$-axis angular velocity. (d) Reported instantaneous "
        "crank power. (e) Left- and right-brake force with the estimated "
        "zero-input bands. (f) GNSS ground speed and wheel speed. "
        "(g) GNSS RTK position---relative east component, "
        "$x_{\\mathrm{ENU}}(t)$. (h) GNSS RTK position---relative north "
        "component, $y_{\\mathrm{ENU}}(t)$. Both components include RTK fixed "
        "and float carrier solutions. All panels "
        "use the same elapsed record-time axis. The vertical dotted line marks "
        f"the {args.trajectory_event_label.lower()} event. Breaks in the "
        "gaze curves indicate unavailable or quality-rejected estimates and are "
        "not interpolated."
    )
    if steering_peak is not None and yaw_peak is not None:
        caption += (
            f" The selected gaze event occurs at "
            f"{(gaze_peak_ns - start_ns) / NS_PER_SECOND:.2f} s; the main "
            f"relative-steering and raw-yaw-rate peaks follow at "
            f"{steering_peak['elapsed_s']:.2f} and {yaw_peak['elapsed_s']:.2f} s, "
            "respectively. This ordering is descriptive rather than a causal "
            "delay estimate."
        )
    (args.out / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": [sys.executable, *sys.argv],
        "source_workflow_window": {
            "start_ns": source_start_ns,
            "end_ns": source_end_ns,
        },
        "window": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_s": window_duration_s,
            "offset_from_source_start_s": (
                (start_ns - source_start_ns) / NS_PER_SECOND
            ),
            "selection_basis": window_basis,
            "requested_avi_playback_start_s": requested_video_start_s,
            "actual_avi_playback_start_s": actual_video_start_s,
            "actual_avi_playback_end_s": actual_video_end_s,
        },
        "frame_extraction": extraction,
        "headpose_run": headpose_run,
        "quality_gates": {
            "minimum_head_tags": args.min_head_tags,
            "maximum_head_rmse_px": args.max_head_rmse_px,
            "apriltag_detector_threads": args.detector_threads,
            "maximum_gaze_pose_time_difference_ms": args.sync_tolerance_ms,
            "maximum_raw_gaze_to_validity_difference_ms": args.eye_validity_tolerance_ms,
            "tobii_eye_validity_required": "at least one valid eye",
            "method_b_direction_rule": "mean of valid eyes; one-eye fallback",
            "maximum_eye_direction_to_3d_gaze_ray_difference_deg": args.eye_direction_consistency_deg,
            "horizontal_projection_required": True,
            "bike_forward_direction_required": False,
            "scene_camera_distortion_correction": True,
            "back_tag_required": False,
        },
        "coordinate_definition": {
            "frame": "base_link / bicycle ego frame",
            "x": "forward", "y": "left", "z": "up",
            "horizontal_angle": "atan2(direction_y, direction_x), positive left",
            "head_pose_source": "cam_head in rear-camera optical frame",
            "calibration_assumptions": [
                "Tobii-to-helmet calibration is reused from 2026-05-06 and assumes the physical mount did not change.",
                "Rear-camera-to-bike rotation is a nominal measured mounting transform, not a survey-grade orientation calibration."
            ],
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in {
                "video": video,
                "camera_timestamps": timestamps,
                "recording_g3": recording,
                "raw_gaze": gaze_path,
                "fixation_export": args.fixation_table,
                "gnss_pvt_route": args.gnss_route,
                "rear_camera_calibration": args.rear_camera_calibration,
                "head_rig_config": args.head_rig_config,
                "rig_calibration": args.rig_calibration,
                "tobii_head_transforms": args.tobii_head_transforms,
                "bike_extrinsics": args.bike_extrinsics,
                "headpose_script": args.headpose_script,
                "headpose_utils": HEADPOSE_DIR / "pose_utils.py",
                "plotting_helper": SCRIPT_DIR / "p9_speed_timing_closed_loop.py",
                "this_script": Path(__file__).resolve(),
            }.items()
        },
        "validity": summary_rows,
        "headpose_quality_breakdown": head_quality_breakdown.to_dict("records"),
        "eye_direction_quality": eye_quality_rows,
        "fixation": fixation_summary,
        "eye_validity": eye_validity_summary,
        "temporal_alignment": temporal_summary,
        "trajectory": route_summary,
        "figure_layout": (
            "gaze, steering, yaw rate, power, brake, and speed followed by "
            "relative east and north GNSS RTK position components on one "
            "shared record-time axis"
        ),
        "note": "Raw data and previous generated outputs were not modified.",
    }
    P9.write_json(args.out / "run_manifest.json", manifest)
    write_output_checksums(args.out)
    print(args.out)


if __name__ == "__main__":
    apply_paper_style()
    main()
