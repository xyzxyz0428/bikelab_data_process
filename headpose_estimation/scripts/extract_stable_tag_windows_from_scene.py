#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from pupil_apriltags import Detector


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def resolve_frame_path(frame_dir, frame_idx):
    frame_idx = int(frame_idx)
    for ext in [".png", ".jpg", ".jpeg"]:
        p = frame_dir / f"frame_{frame_idx:06d}{ext}"
        if p.exists():
            return p
    return None


def tag_center_and_width(corners):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = c.mean(axis=0)

    edge_lengths = [
        np.linalg.norm(c[0] - c[1]),
        np.linalg.norm(c[1] - c[2]),
        np.linalg.norm(c[2] - c[3]),
        np.linalg.norm(c[3] - c[0]),
    ]

    width = float(np.mean(edge_lengths))
    return center, width


def choose_best_detection(dets, target_tag_ids):
    candidates = []

    for d in dets:
        tid = int(d.tag_id)
        if target_tag_ids is not None and tid not in target_tag_ids:
            continue

        margin = float(getattr(d, "decision_margin", 0.0))
        center, width = tag_center_and_width(d.corners)
        candidates.append((margin, width, tid, d, center))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    margin, width, tid, det, center = candidates[0]
    return {
        "tag_id": tid,
        "det": det,
        "center": center,
        "width": width,
        "decision_margin": margin,
    }


def summarize(vals):
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return ""
    return float(median(vals))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scene-timestamps-csv", required=True)
    ap.add_argument("--scene-frame-dir", required=True)
    ap.add_argument("--output-csv", required=True)

    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--target-tag-ids", default="",
                    help="comma-separated tag IDs, e.g. 17 or 30. Empty = all tags.")

    ap.add_argument("--max-center-motion-px", type=float, default=8.0)
    ap.add_argument("--max-center-motion-over-width", type=float, default=0.15)
    ap.add_argument("--max-width-change-ratio", type=float, default=0.15)

    ap.add_argument("--min-window-frames", type=int, default=15)
    ap.add_argument("--max-gap-frames", type=int, default=3)

    args = ap.parse_args()

    target_tag_ids = None
    if args.target_tag_ids.strip():
        target_tag_ids = set(int(x) for x in args.target_tag_ids.split(","))

    rows = read_csv_dicts(args.scene_timestamps_csv)
    frame_dir = Path(args.scene_frame_dir)

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frame_infos = []

    prev = None

    for r in rows:
        frame_idx = to_int(r.get("frame_idx"))
        unix_ns = to_int(r.get("unix_ns"))

        if frame_idx is None or unix_ns is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)

        info = {
            "frame_idx": frame_idx,
            "unix_ns": unix_ns,
            "tag_id": None,
            "center_x": np.nan,
            "center_y": np.nan,
            "tag_width_px": np.nan,
            "center_motion_px": np.nan,
            "center_motion_over_width": np.nan,
            "width_change_ratio": np.nan,
            "stable": False,
        }

        if img_path is None:
            frame_infos.append(info)
            prev = None
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            frame_infos.append(info)
            prev = None
            continue

        dets = detector.detect(img, estimate_tag_pose=False)
        best = choose_best_detection(dets, target_tag_ids)

        if best is None:
            frame_infos.append(info)
            prev = None
            continue

        tid = best["tag_id"]
        center = best["center"]
        width = best["width"]

        info["tag_id"] = tid
        info["center_x"] = float(center[0])
        info["center_y"] = float(center[1])
        info["tag_width_px"] = float(width)

        if prev is not None and prev["tag_id"] == tid:
            prev_center = np.array([prev["center_x"], prev["center_y"]], dtype=np.float64)
            cur_center = np.array([info["center_x"], info["center_y"]], dtype=np.float64)

            center_motion_px = float(np.linalg.norm(cur_center - prev_center))
            mean_width = 0.5 * (prev["tag_width_px"] + info["tag_width_px"])

            center_motion_over_width = center_motion_px / mean_width if mean_width > 1e-6 else np.inf
            width_change_ratio = abs(info["tag_width_px"] - prev["tag_width_px"]) / mean_width if mean_width > 1e-6 else np.inf

            info["center_motion_px"] = center_motion_px
            info["center_motion_over_width"] = center_motion_over_width
            info["width_change_ratio"] = width_change_ratio

            info["stable"] = (
                center_motion_px <= args.max_center_motion_px
                and center_motion_over_width <= args.max_center_motion_over_width
                and width_change_ratio <= args.max_width_change_ratio
            )
        else:
            info["stable"] = False

        frame_infos.append(info)
        prev = info

    # group stable frames into windows
    windows = []
    cur = []

    def flush_window():
        nonlocal cur, windows
        if len(cur) >= args.min_window_frames:
            windows.append(cur)
        cur = []

    last_frame_idx = None

    for info in frame_infos:
        if not info["stable"]:
            flush_window()
            last_frame_idx = None
            continue

        if not cur:
            cur = [info]
            last_frame_idx = info["frame_idx"]
            continue

        same_tag = info["tag_id"] == cur[-1]["tag_id"]
        gap_ok = (info["frame_idx"] - last_frame_idx) <= args.max_gap_frames

        if same_tag and gap_ok:
            cur.append(info)
            last_frame_idx = info["frame_idx"]
        else:
            flush_window()
            cur = [info]
            last_frame_idx = info["frame_idx"]

    flush_window()

    out_rows = []

    for i, win in enumerate(windows, start=1):
        tag_id = win[0]["tag_id"]
        start_frame_idx = win[0]["frame_idx"]
        end_frame_idx = win[-1]["frame_idx"]
        start_unix_ns = win[0]["unix_ns"]
        end_unix_ns = win[-1]["unix_ns"]
        duration_ms = (end_unix_ns - start_unix_ns) / 1e6

        out_rows.append({
            "window_id": i,
            "tag_id": tag_id,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "start_unix_ns": start_unix_ns,
            "end_unix_ns": end_unix_ns,
            "num_rows": len(win),
            "duration_ms": duration_ms,
            "median_center_x": summarize([x["center_x"] for x in win]),
            "median_center_y": summarize([x["center_y"] for x in win]),
            "median_tag_width_px": summarize([x["tag_width_px"] for x in win]),
            "median_center_motion_px": summarize([x["center_motion_px"] for x in win]),
            "median_center_motion_over_width": summarize([x["center_motion_over_width"] for x in win]),
            "median_width_change_ratio": summarize([x["width_change_ratio"] for x in win]),
        })

    fieldnames = [
        "window_id",
        "tag_id",
        "start_frame_idx",
        "end_frame_idx",
        "start_unix_ns",
        "end_unix_ns",
        "num_rows",
        "duration_ms",
        "median_center_x",
        "median_center_y",
        "median_tag_width_px",
        "median_center_motion_px",
        "median_center_motion_over_width",
        "median_width_change_ratio",
    ]

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"saved {len(out_rows)} stable windows to {args.output_csv}")
    print(f"frames checked = {len(frame_infos)}")
    print(f"stable frames = {sum(1 for x in frame_infos if x['stable'])}")

    import os
    os._exit(0)


if __name__ == "__main__":
    main()