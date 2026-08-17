#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interactive_camera_lidar_rough_tuning_v2.py

Interactive rough camera-LiDAR tuning.

Compared with v1:
1) --draw-mode pixel draws 1-pixel points, smaller than cv2 circle radius=1.
2) --accumulate-pairs-csv can load multiple synchronized PCDs and combine them,
   useful when one LiDAR frame is too sparse.
3) --accumulate-radius chooses neighboring rows around the selected pair index.
4) --sample-max-points limits the combined cloud after accumulation.
5) --crop-lidar can crop the cloud before projection.

Coordinate assumptions:
    LiDAR / camera_link:
        x forward, y left, z up
    camera_optical:
        x right, y down, z forward

Tuned transform:
    T_camera_link_lidar

Then internally:
    T_camera_optical_lidar = T_camera_optical_camera_link @ T_camera_link_lidar
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    import open3d as o3d
except Exception:
    o3d = None


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


def rpy_to_R_zyx(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r),  np.cos(r)],
    ], dtype=float)

    Ry = np.array([
        [ np.cos(p), 0, np.sin(p)],
        [0,          1, 0],
        [-np.sin(p), 0, np.cos(p)],
    ], dtype=float)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1],
    ], dtype=float)

    return Rz @ Ry @ Rx


def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def camera_link_to_optical_T():
    """
    T_camera_optical_camera_link.
    """
    R_link_opt = rpy_to_R_zyx(-90.0, 0.0, -90.0)
    T_link_opt = make_T(R_link_opt, [0.0, 0.0, 0.0])
    return np.linalg.inv(T_link_opt)


def load_one_pcd(pcd_path: Path):
    if o3d is None:
        raise RuntimeError("open3d is required. Install with: python3 -m pip install open3d")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=float)
    if len(pts) == 0:
        raise RuntimeError(f"No points found in PCD: {pcd_path}")
    return pts


def load_points(args):
    if args.accumulate_pairs_csv:
        pairs = pd.read_csv(args.accumulate_pairs_csv)
        pairs.columns = [str(c).strip() for c in pairs.columns]
        if "pcd_path" not in pairs.columns:
            raise ValueError("accumulate pairs csv must contain pcd_path")

        idx = int(args.accumulate_index)
        r = int(args.accumulate_radius)
        i0 = max(0, idx - r)
        i1 = min(len(pairs), idx + r + 1)
        sub = pairs.iloc[i0:i1]

        clouds = []
        print(f"[INFO] accumulating PCDs from rows [{i0}, {i1})")
        for _, row in sub.iterrows():
            p = Path(str(row["pcd_path"]))
            if not p.exists():
                print(f"[WARN] missing pcd: {p}")
                continue
            pts = load_one_pcd(p)
            clouds.append(pts)

        if not clouds:
            raise RuntimeError("No PCDs loaded from accumulate pairs CSV")

        pts = np.vstack(clouds)
        print(f"[INFO] accumulated points before sampling: {len(pts)}")
    else:
        pts = load_one_pcd(Path(args.pcd))

    if args.crop_lidar:
        xmin, xmax, ymin, ymax, zmin, zmax = args.crop_lidar
        mask = (
            (pts[:, 0] >= xmin) & (pts[:, 0] <= xmax) &
            (pts[:, 1] >= ymin) & (pts[:, 1] <= ymax) &
            (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        )
        pts = pts[mask]
        print(f"[INFO] points after crop: {len(pts)}")

    if args.sample_max_points and len(pts) > args.sample_max_points:
        rng = np.random.default_rng(12345)
        sel = rng.choice(len(pts), size=args.sample_max_points, replace=False)
        pts = pts[sel]
        print(f"[INFO] points after sampling: {len(pts)}")

    return pts


def color_by_depth(z, z_min, z_max):
    a = np.clip((z - z_min) / max(1e-6, z_max - z_min), 0, 1)
    vals = (a * 255).astype(np.uint8)
    colors = cv2.applyColorMap(vals.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return colors


def project_points(pts_lidar, T_opt_lidar, K, dist, image_shape, z_min, z_max, stride):
    pts = pts_lidar[::max(1, stride)]
    pts_h = np.c_[pts, np.ones(len(pts))]
    pts_opt = (T_opt_lidar @ pts_h.T).T[:, :3]

    mask_z = (pts_opt[:, 2] > z_min) & (pts_opt[:, 2] < z_max)
    pts_opt = pts_opt[mask_z]

    if len(pts_opt) == 0:
        return np.empty((0, 2)), np.empty((0,)), 0, len(pts)

    uv, _ = cv2.projectPoints(
        pts_opt.astype(np.float64),
        np.zeros((3, 1), dtype=float),
        np.zeros((3, 1), dtype=float),
        K,
        dist,
    )
    uv = uv.reshape(-1, 2)

    h, w = image_shape[:2]
    inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )

    return uv[inside], pts_opt[:, 2][inside], int(np.sum(inside)), len(pts)


def draw_overlay(img, uv, z, z_min, z_max, point_size, draw_mode, alpha):
    overlay = img.copy()
    if len(uv) == 0:
        return overlay

    layer = img.copy()
    colors = color_by_depth(z, z_min, z_max)

    h, w = img.shape[:2]
    ui = np.round(uv[:, 0]).astype(int)
    vi = np.round(uv[:, 1]).astype(int)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, colors = ui[inside], vi[inside], colors[inside]

    if draw_mode == "pixel":
        layer[vi, ui] = colors
    else:
        for u, v, col in zip(ui, vi, colors):
            cv2.circle(
                layer,
                (int(u), int(v)),
                max(1, int(point_size)),
                tuple(int(c) for c in col),
                -1,
            )

    if alpha < 1.0:
        overlay = cv2.addWeighted(layer, alpha, overlay, 1.0 - alpha, 0)
    else:
        overlay = layer

    return overlay


def save_json(path, T_link_lidar, T_opt_lidar, params):
    out = {
        "description": "Manually tuned rough camera-LiDAR extrinsic from visual projection.",
        "coordinate_convention": {
            "camera_link": "x forward, y left, z up",
            "camera_optical": "x right, y down, z forward",
            "lidar": "x forward, y left, z up"
        },
        "transform_convention": {
            "T_camera_link_lidar": "maps LiDAR coordinates to camera_link coordinates",
            "T_camera_optical_lidar": "maps LiDAR coordinates to camera_optical coordinates; use this as T_camera_lidar for cv2 projection",
            "T_lidar_camera_optical": "inverse of T_camera_optical_lidar"
        },
        "translation_camera_link_lidar_m": {
            "x": float(params["x"]),
            "y": float(params["y"]),
            "z": float(params["z"]),
        },
        "rotation_camera_link_lidar_rpy_deg": {
            "roll": float(params["roll"]),
            "pitch": float(params["pitch"]),
            "yaw": float(params["yaw"]),
            "convention": "Rz(yaw) @ Ry(pitch) @ Rx(roll)"
        },
        "T_camera_link_lidar": T_link_lidar.tolist(),
        "T_camera_optical_lidar": T_opt_lidar.tolist(),
        "T_lidar_camera_optical": np.linalg.inv(T_opt_lidar).tolist(),
        "T_camera_lidar": T_opt_lidar.tolist()
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] saved: {path}")
    print("translation camera_link <- lidar:", params["x"], params["y"], params["z"])
    print("rpy deg:", params["roll"], params["pitch"], params["yaw"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pcd", default=None, help="Single PCD. Optional if --accumulate-pairs-csv is used.")
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-image", default=None)

    ap.add_argument("--accumulate-pairs-csv", "--accumulate_pairs_csv", dest="accumulate_pairs_csv", default=None, help="calib pairs csv with pcd_path; use multiple PCDs around --accumulate-index")
    ap.add_argument("--accumulate-index", "--accumulate_index", dest="accumulate_index", type=int, default=0)
    ap.add_argument("--accumulate-radius", "--accumulate_radius", dest="accumulate_radius", type=int, default=2, help="Use rows index-radius ... index+radius")

    # LiDAR is in front/right/lower of camera means lidar origin in camera_link:
    # x forward +, y right = negative left, z lower = negative up.
    ap.add_argument("--init-x", type=float, default=0.92)
    ap.add_argument("--init-y", type=float, default=-0.06)
    ap.add_argument("--init-z", type=float, default=-0.59)
    ap.add_argument("--init-roll", type=float, default=0.0)
    ap.add_argument("--init-pitch", type=float, default=0.0)
    ap.add_argument("--init-yaw", type=float, default=0.0)

    ap.add_argument("--z-min", type=float, default=0.3)
    ap.add_argument("--z-max", type=float, default=80.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sample-max-points", type=int, default=500000)
    ap.add_argument("--point-size", type=int, default=1)
    ap.add_argument("--draw-mode", choices=["pixel", "circle"], default="pixel")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--display-scale", type=float, default=0.6)
    ap.add_argument("--crop-lidar", nargs=6, type=float, metavar=("XMIN","XMAX","YMIN","YMAX","ZMIN","ZMAX"), default=None)
    args = ap.parse_args()

    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    K, dist = read_camera_json(Path(args.camera_json))
    pts_lidar = load_points(args)
    print(f"[INFO] loaded lidar points: {len(pts_lidar)}")

    T_opt_link = camera_link_to_optical_T()

    win = "camera-lidar rough tuning v2"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mm_to_pos(v_m): return int(round(v_m * 1000.0 + 3000))
    def pos_to_m(pos): return (float(pos) - 3000.0) / 1000.0
    def deg_to_pos(v_deg): return int(round(v_deg + 180))
    def pos_to_deg(pos): return float(pos) - 180.0

    cv2.createTrackbar("x_m", win, mm_to_pos(args.init_x), 6000, lambda v: None)
    cv2.createTrackbar("y_m", win, mm_to_pos(args.init_y), 6000, lambda v: None)
    cv2.createTrackbar("z_m", win, mm_to_pos(args.init_z), 6000, lambda v: None)
    cv2.createTrackbar("roll_deg", win, deg_to_pos(args.init_roll), 360, lambda v: None)
    cv2.createTrackbar("pitch_deg", win, deg_to_pos(args.init_pitch), 360, lambda v: None)
    cv2.createTrackbar("yaw_deg", win, deg_to_pos(args.init_yaw), 360, lambda v: None)

    stride = max(1, int(args.stride))
    point_size = max(1, int(args.point_size))
    draw_mode = args.draw_mode

    last_T_link_lidar = None
    last_T_opt_lidar = None
    last_params = None
    last_overlay = None

    print("Controls:")
    print("  s: save JSON and optional image")
    print("  q/ESC: quit")
    print("  r: reset")
    print("  p: toggle pixel/circle draw mode")
    print("  + / -: point size for circle mode")
    print("  [ / ]: increase/decrease stride")

    while True:
        x = pos_to_m(cv2.getTrackbarPos("x_m", win))
        y = pos_to_m(cv2.getTrackbarPos("y_m", win))
        z = pos_to_m(cv2.getTrackbarPos("z_m", win))
        roll = pos_to_deg(cv2.getTrackbarPos("roll_deg", win))
        pitch = pos_to_deg(cv2.getTrackbarPos("pitch_deg", win))
        yaw = pos_to_deg(cv2.getTrackbarPos("yaw_deg", win))

        R_link_lidar = rpy_to_R_zyx(roll, pitch, yaw)
        T_link_lidar = make_T(R_link_lidar, [x, y, z])
        T_opt_lidar = T_opt_link @ T_link_lidar

        uv, depth, n_inside, n_total = project_points(
            pts_lidar,
            T_opt_lidar,
            K,
            dist,
            img.shape,
            args.z_min,
            args.z_max,
            stride,
        )

        overlay = draw_overlay(img, uv, depth, args.z_min, args.z_max, point_size, draw_mode, args.alpha)

        text = [
            f"x={x:.3f} y={y:.3f} z={z:.3f} m",
            f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f} deg",
            f"inside={n_inside}/{n_total} stride={stride} draw={draw_mode} point={point_size}",
            "s save | q quit | p pixel/circle | +/- point | [] stride"
        ]

        yy = 30
        for line in text:
            cv2.putText(overlay, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(overlay, line, (20, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            yy += 30

        show = overlay
        if args.display_scale != 1.0:
            show = cv2.resize(
                overlay,
                None,
                fx=args.display_scale,
                fy=args.display_scale,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(win, show)

        last_T_link_lidar = T_link_lidar
        last_T_opt_lidar = T_opt_lidar
        last_params = {
            "x": x, "y": y, "z": z,
            "roll": roll, "pitch": pitch, "yaw": yaw
        }
        last_overlay = overlay

        key = cv2.waitKey(30) & 0xFF

        if key in [27, ord("q")]:
            break

        if key == ord("s"):
            save_json(args.out_json, last_T_link_lidar, last_T_opt_lidar, last_params)
            if args.out_image:
                cv2.imwrite(str(args.out_image), last_overlay)
                print(f"[OK] saved overlay: {args.out_image}")

        if key == ord("r"):
            cv2.setTrackbarPos("x_m", win, mm_to_pos(args.init_x))
            cv2.setTrackbarPos("y_m", win, mm_to_pos(args.init_y))
            cv2.setTrackbarPos("z_m", win, mm_to_pos(args.init_z))
            cv2.setTrackbarPos("roll_deg", win, deg_to_pos(args.init_roll))
            cv2.setTrackbarPos("pitch_deg", win, deg_to_pos(args.init_pitch))
            cv2.setTrackbarPos("yaw_deg", win, deg_to_pos(args.init_yaw))

        if key == ord("p"):
            draw_mode = "circle" if draw_mode == "pixel" else "pixel"
            print("draw_mode:", draw_mode)

        if key == ord("+") or key == ord("="):
            point_size = min(point_size + 1, 10)
            print("point_size:", point_size)

        if key == ord("-") or key == ord("_"):
            point_size = max(point_size - 1, 1)
            print("point_size:", point_size)

        if key == ord("["):
            stride = min(stride + 1, 100)
            print("stride:", stride)

        if key == ord("]"):
            stride = max(stride - 1, 1)
            print("stride:", stride)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
