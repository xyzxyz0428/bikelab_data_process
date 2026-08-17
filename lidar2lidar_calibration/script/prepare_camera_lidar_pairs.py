#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_camera_lidar_pairs.py

Pair camera frames and LiDAR PCD frames by nearest Unix-ns timestamp.

Camera timestamps CSV:
    frame_idx, unix_ns
or frame_idx, t_unix_ns / timestamp_ns

LiDAR frames CSV:
    pcd_path plus t_unix_ns / unix_ns / timestamp_ns / ns

Output:
    calib_pairs.csv with image_path, pcd_path, dt_ms
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_time_col(df, label):
    for c in ["t_unix_ns", "unix_ns", "timestamp_ns", "ns"]:
        if c in df.columns:
            return c
    raise ValueError(f"No time column found in {label}. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-timestamps", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--lidar-frames-csv", required=True)
    ap.add_argument("--pcd-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-dt-ms", type=float, default=50.0)
    ap.add_argument("--image-pattern", default="frame_{frame_idx:06d}.png")
    args = ap.parse_args()

    cam = normalize_cols(pd.read_csv(args.camera_timestamps))
    lidar = normalize_cols(pd.read_csv(args.lidar_frames_csv))

    if "frame_idx" not in cam.columns:
        raise ValueError(f"Camera CSV must contain frame_idx. Columns: {list(cam.columns)}")
    if "pcd_path" not in lidar.columns:
        raise ValueError(f"LiDAR CSV must contain pcd_path. Columns: {list(lidar.columns)}")

    cam_time = find_time_col(cam, "camera timestamps")
    lidar_time = find_time_col(lidar, "lidar frames")

    cam["t_unix_ns"] = pd.to_numeric(cam[cam_time], errors="coerce")
    lidar["t_unix_ns"] = pd.to_numeric(lidar[lidar_time], errors="coerce")

    cam["frame_idx"] = pd.to_numeric(cam["frame_idx"], errors="coerce")
    cam = cam.dropna(subset=["t_unix_ns", "frame_idx"]).copy()
    lidar = lidar.dropna(subset=["t_unix_ns", "pcd_path"]).copy()

    cam["t_unix_ns"] = cam["t_unix_ns"].astype("int64")
    lidar["t_unix_ns"] = lidar["t_unix_ns"].astype("int64")
    cam["frame_idx"] = cam["frame_idx"].astype("int64")

    frame_dir = Path(args.frame_dir)
    cam["image_path"] = cam["frame_idx"].apply(
        lambda x: str(frame_dir / args.image_pattern.format(frame_idx=int(x)))
    )

    pcd_root = Path(args.pcd_root) if args.pcd_root else None
    if pcd_root is not None:
        def resolve_pcd(p):
            p = Path(str(p))
            if p.is_absolute():
                return str(p)
            return str((pcd_root / p).resolve())
        lidar["pcd_path"] = lidar["pcd_path"].apply(resolve_pcd)
    else:
        lidar["pcd_path"] = lidar["pcd_path"].astype(str)

    cam = cam.sort_values("t_unix_ns").reset_index(drop=True)
    lidar = lidar.sort_values("t_unix_ns").reset_index(drop=True)

    lidar_join = lidar.rename(columns={"t_unix_ns": "lidar_t_unix_ns"}).copy()
    keep = ["lidar_t_unix_ns", "pcd_path"]
    for c in ["frame_idx", "n_points", "frame_id", "topic"]:
        if c in lidar_join.columns:
            if c == "frame_idx":
                lidar_join = lidar_join.rename(columns={"frame_idx": "lidar_frame_idx"})
                keep.append("lidar_frame_idx")
            else:
                keep.append(c)

    pairs = pd.merge_asof(
        cam[["frame_idx", "t_unix_ns", "image_path"]],
        lidar_join[keep],
        left_on="t_unix_ns",
        right_on="lidar_t_unix_ns",
        direction="nearest",
        tolerance=int(args.max_dt_ms * 1e6),
    )

    pairs["dt_ms"] = (pairs["t_unix_ns"] - pairs["lidar_t_unix_ns"]).abs() / 1e6
    pairs = pairs.dropna(subset=["lidar_t_unix_ns", "pcd_path"]).copy()

    pairs["image_exists"] = pairs["image_path"].apply(lambda p: Path(p).exists())
    pairs["pcd_exists"] = pairs["pcd_path"].apply(lambda p: Path(p).exists())
    pairs = pairs[pairs["image_exists"] & pairs["pcd_exists"]].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out, index=False)

    print(f"[OK] wrote: {out}")
    print(f"camera rows: {len(cam)}")
    print(f"lidar rows: {len(lidar)}")
    print(f"valid pairs: {len(pairs)}")
    if len(pairs):
        print(f"dt_ms median = {pairs['dt_ms'].median():.3f}")
        print(f"dt_ms p95    = {np.percentile(pairs['dt_ms'], 95):.3f}")
        print(f"dt_ms max    = {pairs['dt_ms'].max():.3f}")
    else:
        print("[WARN] No valid pairs. Try increasing --max-dt-ms or check timestamps/image pattern.")


if __name__ == "__main__":
    main()
