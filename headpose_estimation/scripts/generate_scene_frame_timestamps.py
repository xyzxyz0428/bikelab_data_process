#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import cv2


def iso_to_unix_us(iso_str: str) -> int:
    """
    Convert ISO8601 string like:
      2026-04-23T09:46:53.190440Z
    to unix timestamp in microseconds.
    """
    s = iso_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(round(dt.timestamp() * 1e6))


def parse_recording_g3_start_us(recording_g3_path: str):
    with open(recording_g3_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # First try ISO-style "created"
    created = data.get("created", None)
    if isinstance(created, str) and len(created) > 0:
        try:
            return iso_to_unix_us(created)
        except Exception as e:
            print(f"[WARN] Failed to parse ISO created='{created}': {e}")

    # Fallback candidates if future files differ
    candidates = [
        ("recording", "created"),
        ("recording", "start_time"),
        ("meta-folder", "created"),
        ("start_time",),
    ]

    for keys in candidates:
        cur = data
        ok = True
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if not ok:
            continue

        if isinstance(cur, (int, float)):
            # assume seconds if small, microseconds if large
            if cur < 1e12:
                return int(round(cur * 1e6))
            return int(cur)

        if isinstance(cur, str):
            try:
                return iso_to_unix_us(cur)
            except Exception:
                pass

    print("[WARN] Could not find a usable absolute start time in recording.g3. "
          "Will use video-relative timestamps only.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording-g3", required=True)
    ap.add_argument("--scene-video", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="optional folder to extract frames as frame_000001.png etc.")
    args = ap.parse_args()

    start_us = parse_recording_g3_start_us(args.recording_g3)
    if start_us > 0:
        print(f"[INFO] Parsed recording start time: {start_us} us "
              f"({start_us / 1e6:.6f} s unix)")
    else:
        print("[INFO] Using video-relative timestamps only.")

    cap = cv2.VideoCapture(args.scene_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.scene_video}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # OpenCV returns current position in ms from video start
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_us_rel = int(round(ts_ms * 1000.0))
        unix_ns = int((start_us + ts_us_rel) * 1000)

        filename = f"frame_{frame_idx:06d}.png"
        if out_dir is not None:
            cv2.imwrite(str(out_dir / filename), frame)

        rows.append({
            "frame_idx": frame_idx,
            "unix_ns": unix_ns,
            "filename": filename,
            "video_time_ms": f"{ts_ms:.3f}"
        })

    cap.release()

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "unix_ns", "filename", "video_time_ms"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()