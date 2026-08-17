#!/usr/bin/env python3
"""Map constant-rate video-player time to per-frame Unix timestamps."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NS_PER_SECOND = 1_000_000_000


def load_video_clock(video_path, timestamps_path):
    """Load the AVI playback rate and the recorded timestamp of each frame."""
    video_path = Path(video_path).resolve()
    timestamps_path = Path(timestamps_path).resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not np.isfinite(nominal_fps) or nominal_fps <= 0:
        raise RuntimeError(f"Invalid nominal video frame rate: {nominal_fps}")

    table = pd.read_csv(timestamps_path, usecols=["frame_idx", "unix_ns"])
    table["frame_idx"] = pd.to_numeric(table["frame_idx"], errors="coerce")
    table["unix_ns"] = pd.to_numeric(table["unix_ns"], errors="coerce")
    table = table.dropna().sort_values("frame_idx").drop_duplicates("frame_idx")
    if len(table) < 2:
        raise RuntimeError(f"Too few valid frame timestamps: {timestamps_path}")
    frames = table["frame_idx"].to_numpy(dtype=float)
    unix_ns = table["unix_ns"].to_numpy(dtype=np.int64)
    if np.any(np.diff(frames) <= 0) or np.any(np.diff(unix_ns) <= 0):
        raise RuntimeError("Camera frame indices and Unix timestamps must increase strictly")

    first_frame = float(frames[0])
    last_frame = float(frames[-1])
    timestamp_duration_s = float((unix_ns[-1] - unix_ns[0]) / NS_PER_SECOND)
    playback_duration_s = float((last_frame - first_frame) / nominal_fps)
    effective_capture_rate_hz = float(
        (len(frames) - 1) / timestamp_duration_s
    )
    return {
        "video_path": str(video_path),
        "timestamps_path": str(timestamps_path),
        "nominal_fps": nominal_fps,
        "frame_count_from_timestamps": int(len(frames)),
        "first_frame_idx": first_frame,
        "last_frame_idx": last_frame,
        "first_unix_ns": int(unix_ns[0]),
        "last_unix_ns": int(unix_ns[-1]),
        "playback_duration_s": playback_duration_s,
        "timestamp_duration_s": timestamp_duration_s,
        "effective_capture_rate_hz": effective_capture_rate_hz,
        "frames": frames,
        "unix_ns": unix_ns,
    }


def playback_seconds_to_unix_ns(clock, playback_seconds):
    """Convert AVI player seconds through frame number to recorded Unix time."""
    playback_seconds = float(playback_seconds)
    target_frame = clock["first_frame_idx"] + playback_seconds * clock["nominal_fps"]
    if target_frame < clock["first_frame_idx"] or target_frame > clock["last_frame_idx"]:
        raise ValueError(
            f"Playback time {playback_seconds:.6f} s maps to frame {target_frame:.3f}, "
            f"outside [{clock['first_frame_idx']:.0f}, {clock['last_frame_idx']:.0f}]"
        )
    # Subtract the large epoch value before interpolation to retain nanosecond
    # precision in float64.
    origin_ns = int(clock["unix_ns"][0])
    offsets_ns = clock["unix_ns"] - origin_ns
    offset_ns = float(np.interp(target_frame, clock["frames"], offsets_ns.astype(float)))
    return origin_ns + int(round(offset_ns)), float(target_frame)


def unix_ns_to_playback_seconds(clock, unix_ns):
    """Convert a recorded Unix timestamp back to the AVI player timeline."""
    unix_ns = int(unix_ns)
    if unix_ns < int(clock["unix_ns"][0]) or unix_ns > int(clock["unix_ns"][-1]):
        raise ValueError("Unix timestamp is outside the recorded camera frame span")
    origin_ns = int(clock["unix_ns"][0])
    offsets_ns = (clock["unix_ns"] - origin_ns).astype(float)
    target_frame = float(
        np.interp(float(unix_ns - origin_ns), offsets_ns, clock["frames"])
    )
    return float(
        (target_frame - clock["first_frame_idx"]) / clock["nominal_fps"]
    )


def map_playback_interval(video_path, timestamps_path, start_s, end_s):
    """Return exact Unix boundaries and an auditable mapping summary."""
    if float(end_s) <= float(start_s):
        raise ValueError("Video interval end must be after its start")
    clock = load_video_clock(video_path, timestamps_path)
    start_ns, start_frame = playback_seconds_to_unix_ns(clock, start_s)
    end_ns, end_frame = playback_seconds_to_unix_ns(clock, end_s)
    summary = {
        key: value for key, value in clock.items()
        if key not in {"frames", "unix_ns"}
    }
    summary.update({
        "mapping_method": "AVI playback seconds -> nominal-fps frame index -> timestamps.csv unix_ns",
        "video_start_s": float(start_s),
        "video_end_s": float(end_s),
        "start_frame_idx": start_frame,
        "end_frame_idx": end_frame,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "record_time_duration_s": (end_ns - start_ns) / NS_PER_SECOND,
    })
    return start_ns, end_ns, summary
