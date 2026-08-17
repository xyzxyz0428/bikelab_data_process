#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project_lidar_to_image_overlay.py

Project LiDAR PCD to camera image using calibration_result.json.

Input extrinsic JSON must contain:
    T_camera_lidar
or:
    T_lidar_camera
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_camera_json(path):
    cam = read_json(path)
    if all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], dtype=float)
    elif "K" in cam:
        K = np.asarray(cam["K"], dtype=float)
        if K.shape != (3, 3):
            K = K.reshape(3, 3)
    elif "camera_matrix" in cam:
        cm = cam["camera_matrix"]
        data = cm["data"] if isinstance(cm, dict) and "data" in cm else cm
        K = np.asarray(data, dtype=float).reshape(3, 3)
    else:
        raise KeyError(f"Cannot read camera intrinsics. Keys: {list(cam.keys())}")

    dist = np.zeros(5, dtype=float)
    if "dist" in cam:
        dist = np.asarray(cam["dist"], dtype=float).reshape(-1)
    elif "distortion_coefficients" in cam:
        dc = cam["distortion_coefficients"]
        data = dc["data"] if isinstance(dc, dict) and "data" in dc else dc
        dist = np.asarray(data, dtype=float).reshape(-1)
    return K, dist


def load_T_camera_lidar(path):
    data = read_json(path)
    if "T_camera_lidar" in data:
        return np.asarray(data["T_camera_lidar"], dtype=float).reshape(4, 4)
    if "T_lidar_camera" in data:
        return np.linalg.inv(np.asarray(data["T_lidar_camera"], dtype=float).reshape(4, 4))
    raise KeyError("Extrinsic JSON must contain T_camera_lidar or T_lidar_camera")


def color_by_depth(z, z_min, z_max):
    a = np.clip((z - z_min) / max(1e-6, z_max - z_min), 0, 1)
    vals = (a * 255).astype(np.uint8)
    return cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--extrinsic-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--point-size", type=int, default=1)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--z-min", type=float, default=0.3)
    ap.add_argument("--z-max", type=float, default=80.0)
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    K, dist = read_camera_json(Path(args.camera_json))
    T_C_L = load_T_camera_lidar(Path(args.extrinsic_json))

    pcd = o3d.io.read_point_cloud(args.pcd)
    pts_L = np.asarray(pcd.points, dtype=float)
    if args.stride > 1:
        pts_L = pts_L[::args.stride]

    pts_C = (T_C_L @ np.c_[pts_L, np.ones(len(pts_L))].T).T[:, :3]
    mask_z = (pts_C[:, 2] > args.z_min) & (pts_C[:, 2] < args.z_max)
    pts_C = pts_C[mask_z]

    uv, _ = cv2.projectPoints(pts_C.astype(np.float64), np.zeros((3,1)), np.zeros((3,1)), K, dist)
    uv = uv.reshape(-1, 2)

    h, w = img.shape[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[inside]
    z = pts_C[:, 2][inside]
    colors = color_by_depth(z, args.z_min, args.z_max)

    out_img = img.copy()
    for (u, v), c in zip(uv, colors):
        cv2.circle(out_img, (int(round(u)), int(round(v))), args.point_size, tuple(int(x) for x in c), -1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out_img)
    print(f"[OK] wrote {args.out}, projected points={len(uv)}")


if __name__ == "__main__":
    main()
