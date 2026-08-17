#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
project_lidar_to_image_overlay_from_bike_json.py

Project a LiDAR PCD onto a camera image.

This version can directly read bike_extrinsics.json and compute:

    T_camera_lidar = T_camera_frame_lidar_frame

where T_parent_child maps child coordinates to parent coordinates.

Typical for your case:
    --lidar-frame middle_lidar
    --camera-frame camera_optical_frame

Even if lidar_frames.csv frame_id is "rslidar_201", you can still use
--lidar-frame middle_lidar IF the PCD coordinates are already expressed in the
same coordinate system as middle_lidar. If rslidar_201 is a separate frame, add
a transform for rslidar_201 in bike_extrinsics.json or use an alias.

Outputs an image with projected LiDAR points colored by camera-depth.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_camera_json(path: Path):
    cam = read_json(path)

    if all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        K = np.array([
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0],
        ], dtype=float)

    elif "K" in cam:
        K = np.asarray(cam["K"], dtype=float)
        if K.shape != (3, 3):
            K = K.reshape(3, 3)

    elif "camera_matrix" in cam:
        cm = cam["camera_matrix"]
        data = cm["data"] if isinstance(cm, dict) and "data" in cm else cm
        K = np.asarray(data, dtype=float).reshape(3, 3)

    else:
        raise KeyError(
            "Cannot read camera intrinsics. Supported keys: fx/fy/cx/cy, K, camera_matrix. "
            f"Available keys: {list(cam.keys())}"
        )

    dist = np.zeros(5, dtype=float)
    if "dist" in cam:
        dist = np.asarray(cam["dist"], dtype=float).reshape(-1)
    elif "distortion_coefficients" in cam:
        dc = cam["distortion_coefficients"]
        data = dc["data"] if isinstance(dc, dict) and "data" in dc else dc
        dist = np.asarray(data, dtype=float).reshape(-1)

    return K, dist


def rpy_zyx_to_R(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Static bike_extrinsics.json convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)]
    ], dtype=float)

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ], dtype=float)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1]
    ], dtype=float)

    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def build_transform_graph_from_bike_json(path: Path):
    data = read_json(path)
    if "transforms" not in data:
        raise KeyError("bike_extrinsics.json must contain a top-level 'transforms' list.")

    graph = {}

    for tr in data["transforms"]:
        parent = tr["parent"]
        child = tr["child"]
        t = tr["translation"]
        r = tr["rotation_rpy_deg"]

        T_parent_child = make_T(
            rpy_zyx_to_R(float(r["roll"]), float(r["pitch"]), float(r["yaw"])),
            np.array([float(t["x"]), float(t["y"]), float(t["z"])], dtype=float),
        )

        # T_parent_child maps child -> parent
        graph[(parent, child)] = T_parent_child
        graph[(child, parent)] = np.linalg.inv(T_parent_child)

    return graph


def find_transform(graph, parent: str, child: str) -> np.ndarray:
    """
    Return T_parent_child, mapping p_child -> p_parent.
    """
    if (parent, child) in graph:
        return graph[(parent, child)]

    frames = sorted(set([a for a, _ in graph.keys()] + [b for _, b in graph.keys()]))
    neighbors = {f: [] for f in frames}

    for (a, b), T_a_b in graph.items():
        neighbors[a].append((b, T_a_b))

    queue = [(parent, np.eye(4))]
    visited = {parent}

    while queue:
        cur, T_parent_cur = queue.pop(0)

        if cur == child:
            return T_parent_cur

        for nb, T_cur_nb in neighbors.get(cur, []):
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, T_parent_cur @ T_cur_nb))

    raise RuntimeError(
        f"Cannot find transform T_{parent}_{child}. "
        f"Known frames: {frames}"
    )


def load_points(pcd_path: Path, stride: int = 1, max_points: int = 300000) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=float)

    if len(pts) == 0:
        raise RuntimeError(f"No points in PCD: {pcd_path}")

    if stride > 1:
        pts = pts[::stride]

    if len(pts) > max_points:
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]

    return pts


def project_lidar_points(pts_lidar: np.ndarray, T_camera_lidar: np.ndarray, K: np.ndarray, dist: np.ndarray):
    pts_h = np.c_[pts_lidar, np.ones(len(pts_lidar))]
    pts_cam = (T_camera_lidar @ pts_h.T).T[:, :3]

    # In camera optical frame, z is forward.
    rvec = np.zeros((3, 1), dtype=float)
    tvec = np.zeros((3, 1), dtype=float)

    uv, _ = cv2.projectPoints(
        pts_cam.astype(np.float64),
        rvec,
        tvec,
        K,
        dist,
    )

    return pts_cam, uv.reshape(-1, 2)


def color_by_depth(z: np.ndarray, z_min: float, z_max: float):
    a = np.clip((z - z_min) / max(1e-6, z_max - z_min), 0, 1)
    vals = (a * 255).astype(np.uint8)
    colors = cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Camera image path")
    ap.add_argument("--pcd", required=True, help="LiDAR PCD path")
    ap.add_argument("--camera-json", required=True, help="Camera intrinsics JSON")
    ap.add_argument("--bike-extrinsics-json", required=True, help="bike_extrinsics.json")
    ap.add_argument("--lidar-frame", default="middle_lidar", help="Frame of PCD coordinates, as named in bike_extrinsics.json")
    ap.add_argument("--camera-frame", default="camera_optical_frame", help="Camera projection frame, usually camera_optical_frame")
    ap.add_argument("--out", required=True, help="Output overlay image")
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-points", type=int, default=300000)
    ap.add_argument("--z-min", type=float, default=0.3)
    ap.add_argument("--z-max", type=float, default=80.0)
    ap.add_argument("--draw-axis", action="store_true", help="Draw camera optical axes projected from LiDAR frame origin if visible")
    args = ap.parse_args()

    image_path = Path(args.image)
    pcd_path = Path(args.pcd)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    K, dist = read_camera_json(Path(args.camera_json))

    graph = build_transform_graph_from_bike_json(Path(args.bike_extrinsics_json))
    T_camera_lidar = find_transform(graph, args.camera_frame, args.lidar_frame)

    pts_lidar = load_points(pcd_path, stride=args.stride, max_points=args.max_points)
    pts_cam, uv = project_lidar_points(pts_lidar, T_camera_lidar, K, dist)

    h, w = img.shape[:2]

    valid = (
        (pts_cam[:, 2] > args.z_min) &
        (pts_cam[:, 2] < args.z_max) &
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    uv_v = uv[valid]
    z_v = pts_cam[:, 2][valid]
    colors = color_by_depth(z_v, args.z_min, args.z_max)

    overlay = img.copy()

    for (u, v), col in zip(uv_v, colors):
        cv2.circle(
            overlay,
            (int(round(u)), int(round(v))),
            args.point_size,
            tuple(int(c) for c in col),
            -1,
        )

    # Draw text summary
    txt = [
        f"lidar_frame: {args.lidar_frame}",
        f"camera_frame: {args.camera_frame}",
        f"projected: {len(uv_v)} / {len(pts_lidar)}",
    ]
    y = 30
    for line in txt:
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
        cv2.putText(overlay, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        y += 30

    cv2.imwrite(str(out_path), overlay)

    print("[OK] wrote:", out_path)
    print("T_camera_lidar maps lidar -> camera:")
    print(T_camera_lidar)
    print(f"projected points in image: {len(uv_v)} / {len(pts_lidar)}")
    print(f"z range visible: {z_v.min() if len(z_v) else np.nan:.2f} to {z_v.max() if len(z_v) else np.nan:.2f} m")


if __name__ == "__main__":
    main()
