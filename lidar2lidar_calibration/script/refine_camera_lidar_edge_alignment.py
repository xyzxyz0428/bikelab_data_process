#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_camera_lidar_edge_alignment.py

Targetless camera-LiDAR local refinement by edge alignment.

This is NOT a global calibration method. It refines an already reasonable
initial extrinsic by aligning projected LiDAR depth-discontinuity edges to
Canny image edges.

Input:
  --pairs-csv with columns: image_path, pcd_path
  --camera-json with K and optional dist
  --init-extrinsic-json:
      either bike_extrinsics.json with "transforms"
      or calibration JSON with T_camera_lidar / T_lidar_camera

Output:
  edge_refined_extrinsic.json
  sample_edge_summary.csv
  debug_edges/
  overlay_before_after/

Transform convention:
  T_camera_lidar maps LiDAR frame to camera optical frame:
      p_camera = T_camera_lidar @ p_lidar
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_camera_json(path):
    cam = read_json(path)

    if all(k in cam for k in ["fx", "fy", "cx", "cy"]):
        K = np.array([[cam["fx"], 0, cam["cx"]],
                      [0, cam["fy"], cam["cy"]],
                      [0, 0, 1]], dtype=float)
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


def make_T(Rm, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def rpy_static_to_R_zyx(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]], dtype=float)
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [0,          1, 0],
                   [-np.sin(p), 0, np.cos(p)]], dtype=float)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]], dtype=float)
    return Rz @ Ry @ Rx


def build_graph_from_bike_json(path):
    data = read_json(path)
    graph = {}

    for tr in data["transforms"]:
        parent = tr["parent"]
        child = tr["child"]
        t = tr["translation"]
        r = tr["rotation_rpy_deg"]
        Rm = rpy_static_to_R_zyx(float(r["roll"]), float(r["pitch"]), float(r["yaw"]))
        T_parent_child = make_T(Rm, [float(t["x"]), float(t["y"]), float(t["z"])])
        graph[(parent, child)] = T_parent_child
        graph[(child, parent)] = np.linalg.inv(T_parent_child)

    return graph


def find_transform(graph, parent, child):
    if (parent, child) in graph:
        return graph[(parent, child)]

    frames = sorted(set([a for a, _ in graph.keys()] + [b for _, b in graph.keys()]))
    neighbors = {f: [] for f in frames}
    for (a, b), T_a_b in graph.items():
        neighbors[a].append((b, T_a_b))

    visited = {parent}
    queue = [(parent, np.eye(4))]
    while queue:
        cur, T_parent_cur = queue.pop(0)
        if cur == child:
            return T_parent_cur
        for nb, T_cur_nb in neighbors.get(cur, []):
            if nb in visited:
                continue
            visited.add(nb)
            queue.append((nb, T_parent_cur @ T_cur_nb))

    raise KeyError(f"Cannot find transform T_{parent}_{child}")


def load_initial_T_camera_lidar(path, init_format, lidar_frame, camera_frame):
    data = read_json(path)

    if init_format == "auto":
        if "T_camera_lidar" in data or "T_lidar_camera" in data:
            init_format = "calibration_json"
        elif "transforms" in data:
            init_format = "bike_json"
        else:
            raise ValueError("Cannot infer init format.")

    if init_format == "calibration_json":
        if "T_camera_lidar" in data:
            return np.asarray(data["T_camera_lidar"], dtype=float).reshape(4, 4)
        if "T_lidar_camera" in data:
            return np.linalg.inv(np.asarray(data["T_lidar_camera"], dtype=float).reshape(4, 4))
        raise KeyError("calibration_json must contain T_camera_lidar or T_lidar_camera")

    if init_format == "bike_json":
        graph = build_graph_from_bike_json(path)
        return find_transform(graph, camera_frame, lidar_frame)

    raise ValueError(f"Unknown init format: {init_format}")


def compose_left_correction(x):
    R_corr = R.from_rotvec(x[:3]).as_matrix()
    t_corr = x[3:6]
    return make_T(R_corr, t_corr)


def load_pcd_points(path, voxel_size=0.0, max_points=200000):
    pcd = o3d.io.read_point_cloud(str(path))
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd.points, dtype=float)
    if len(pts) > max_points:
        idx = np.random.default_rng(12345).choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def transform_points(T_C_L, pts_L):
    pts_h = np.c_[pts_L, np.ones(len(pts_L))]
    return (T_C_L @ pts_h.T).T[:, :3]


def project_camera_points(pts_C, K, dist):
    rvec = np.zeros((3, 1), dtype=float)
    tvec = np.zeros((3, 1), dtype=float)
    uv, _ = cv2.projectPoints(pts_C.astype(np.float64), rvec, tvec, K, dist)
    return uv.reshape(-1, 2)


def valid_projection_mask(pts_C, uv, w, h, z_min, z_max):
    return (
        (pts_C[:, 2] > z_min) &
        (pts_C[:, 2] < z_max) &
        (uv[:, 0] >= 0) & (uv[:, 0] < w) &
        (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )


def make_image_distance_transform(image_path, canny1, canny2, blur_ksize=3, dilate_iter=1, debug_edge_path=None):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, canny1, canny2)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dist_img = cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    if debug_edge_path is not None:
        debug_edge_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_edge_path), edges)

    return img, dist_img


def extract_lidar_depth_edges(pts_L, T_C_L, K, dist, image_shape,
                              z_min, z_max, depth_jump_m,
                              grid_stride=2, max_edge_points=5000):
    h, w = image_shape[:2]
    pts_C = transform_points(T_C_L, pts_L)
    uv = project_camera_points(pts_C, K, dist)
    mask = valid_projection_mask(pts_C, uv, w, h, z_min, z_max)
    if not np.any(mask):
        return np.empty((0, 3), dtype=float)

    pts_L_v = pts_L[mask]
    pts_C_v = pts_C[mask]
    uv_v = uv[mask]

    sw = max(1, w // grid_stride)
    sh = max(1, h // grid_stride)

    u = np.clip((uv_v[:, 0] / grid_stride).astype(int), 0, sw - 1)
    v = np.clip((uv_v[:, 1] / grid_stride).astype(int), 0, sh - 1)
    z = pts_C_v[:, 2]

    depth = np.full((sh, sw), np.inf, dtype=np.float32)
    index = np.full((sh, sw), -1, dtype=np.int64)

    for i in range(len(z)):
        if z[i] < depth[v[i], u[i]]:
            depth[v[i], u[i]] = z[i]
            index[v[i], u[i]] = i

    valid = np.isfinite(depth)
    depth_filled = depth.copy()
    depth_filled[~valid] = 0.0

    gx = np.abs(cv2.Sobel(depth_filled, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(depth_filled, cv2.CV_32F, 0, 1, ksize=3))
    grad = np.maximum(gx, gy)

    edge_pix = (grad > depth_jump_m) & valid
    ys, xs = np.where(edge_pix)
    ids = index[ys, xs]
    ids = ids[ids >= 0]

    if len(ids) == 0:
        return np.empty((0, 3), dtype=float)

    ids = np.unique(ids)
    edge_pts = pts_L_v[ids]

    if len(edge_pts) > max_edge_points:
        sel = np.random.default_rng(12345).choice(len(edge_pts), size=max_edge_points, replace=False)
        edge_pts = edge_pts[sel]

    return edge_pts


def sample_distance_transform(dist_img, uv, max_distance_px):
    """
    Bilinear sampling of the image-edge distance transform.

    Important:
    Do NOT use rounded integer pixels here. Nearest-pixel sampling makes the
    residual piecewise constant, so scipy's finite-difference Jacobian can
    become zero and the optimizer stops after one iteration.
    """
    h, w = dist_img.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)

    x = uv[:, 0]
    y = uv[:, 1]

    inside = (x >= 0) & (x <= w - 2) & (y >= 0) & (y <= h - 2)
    out = np.full(len(uv), max_distance_px, dtype=np.float64)

    if not np.any(inside):
        return out

    xi = x[inside]
    yi = y[inside]

    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = xi - x0
    dy = yi - y0

    Ia = dist_img[y0, x0]
    Ib = dist_img[y0, x1]
    Ic = dist_img[y1, x0]
    Id = dist_img[y1, x1]

    val = (
        Ia * (1.0 - dx) * (1.0 - dy) +
        Ib * dx * (1.0 - dy) +
        Ic * (1.0 - dx) * dy +
        Id * dx * dy
    )

    out[inside] = np.minimum(val, max_distance_px)
    return out


def prepare_samples(pairs, K, dist, T_init, args, outdir):
    debug_dir = outdir / "debug_image_edges"
    debug_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    rows = []

    for idx, row in pairs.head(args.max_pairs).iterrows():
        image_path = Path(row["image_path"])
        pcd_path = Path(row["pcd_path"])

        img, dt = make_image_distance_transform(
            image_path,
            args.canny1,
            args.canny2,
            blur_ksize=args.blur_ksize,
            dilate_iter=args.edge_dilate_iter,
            debug_edge_path=debug_dir / f"image_edges_{idx:04d}.png",
        )

        pts_L = load_pcd_points(pcd_path, voxel_size=args.voxel_size, max_points=args.max_cloud_points)

        edge_pts_L = extract_lidar_depth_edges(
            pts_L,
            T_init,
            K,
            dist,
            img.shape,
            z_min=args.z_min,
            z_max=args.z_max,
            depth_jump_m=args.depth_jump_m,
            grid_stride=args.depth_grid_stride,
            max_edge_points=args.max_lidar_edge_points,
        )

        ok = len(edge_pts_L) >= args.min_lidar_edge_points
        rows.append({
            "sample_index": idx,
            "image_path": str(image_path),
            "pcd_path": str(pcd_path),
            "ok": int(ok),
            "n_cloud_points": len(pts_L),
            "n_lidar_edge_points": len(edge_pts_L),
            "reason": "ok" if ok else "too_few_lidar_edges",
        })

        if ok:
            samples.append({
                "image_path": str(image_path),
                "pcd_path": str(pcd_path),
                "image_shape": img.shape,
                "dt": dt,
                "edge_pts_L": edge_pts_L,
            })

    pd.DataFrame(rows).to_csv(outdir / "sample_edge_summary.csv", index=False)
    return samples


def residuals_edge_alignment(x, samples, T_init, K, dist, args):
    """
    Fixed-length residual vector.

    scipy least_squares requires the residual length to stay constant for all
    parameter values. Therefore we do NOT drop invalid projected points here.
    Points outside the camera image or outside the z range receive the maximum
    distance penalty.
    """
    T_corr = compose_left_correction(x)
    T = T_corr @ T_init

    all_res = []

    for s in samples:
        pts_L = s["edge_pts_L"]
        pts_C = transform_points(T, pts_L)
        uv = project_camera_points(pts_C, K, dist)

        h, w = s["dt"].shape[:2]
        valid = valid_projection_mask(pts_C, uv, w, h, args.z_min, args.z_max)

        d = np.full(len(pts_L), args.max_distance_px, dtype=float)
        if np.any(valid):
            d[valid] = sample_distance_transform(s["dt"], uv[valid], args.max_distance_px)

        all_res.append(d / args.residual_scale_px)

    if not all_res:
        return np.array([args.max_distance_px / args.residual_scale_px], dtype=float)

    return np.concatenate(all_res)


def draw_lidar_edges_overlay(image_path, edge_pts_L, T_C_L, K, dist, out_path, z_min, z_max, point_size=2):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return

    h, w = img.shape[:2]
    pts_C = transform_points(T_C_L, edge_pts_L)
    uv = project_camera_points(pts_C, K, dist)
    mask = valid_projection_mask(pts_C, uv, w, h, z_min, z_max)

    overlay = img.copy()
    for u, v in uv[mask]:
        cv2.circle(overlay, (int(round(u)), int(round(v))), point_size, (0, 0, 255), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--init-extrinsic-json", required=True)
    ap.add_argument("--init-format", default="auto", choices=["auto", "bike_json", "calibration_json"])
    ap.add_argument("--lidar-frame", default="middle_lidar")
    ap.add_argument("--camera-frame", default="camera_optical_frame")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--canny1", type=float, default=80)
    ap.add_argument("--canny2", type=float, default=160)
    ap.add_argument("--blur-ksize", type=int, default=3)
    ap.add_argument("--edge-dilate-iter", type=int, default=1)

    ap.add_argument("--voxel-size", type=float, default=0.05)
    ap.add_argument("--max-cloud-points", type=int, default=200000)
    ap.add_argument("--max-lidar-edge-points", type=int, default=5000)
    ap.add_argument("--min-lidar-edge-points", type=int, default=200)
    ap.add_argument("--depth-jump-m", type=float, default=0.8)
    ap.add_argument("--depth-grid-stride", type=int, default=2)

    ap.add_argument("--z-min", type=float, default=0.5)
    ap.add_argument("--z-max", type=float, default=50.0)
    ap.add_argument("--max-distance-px", type=float, default=30.0)
    ap.add_argument("--residual-scale-px", type=float, default=10.0)
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    ap.add_argument("--f-scale", type=float, default=1.0)

    ap.add_argument("--max-rot-deg", type=float, default=8.0)
    ap.add_argument("--max-trans-m", type=float, default=0.3)
    ap.add_argument("--diff-step", type=float, default=1e-4, help="Finite-difference step for least_squares. Larger helps with pixel-distance residuals.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    K, dist = read_camera_json(Path(args.camera_json))
    T_init = load_initial_T_camera_lidar(
        Path(args.init_extrinsic_json),
        args.init_format,
        args.lidar_frame,
        args.camera_frame,
    )

    pairs = pd.read_csv(args.pairs_csv)
    pairs.columns = [str(c).strip() for c in pairs.columns]
    if "image_path" not in pairs.columns or "pcd_path" not in pairs.columns:
        raise ValueError("pairs.csv must contain image_path and pcd_path")

    print("[INFO] preparing image edges and LiDAR edge points...")
    samples = prepare_samples(pairs, K, dist, T_init, args, outdir)
    print(f"[INFO] valid samples: {len(samples)}")

    if len(samples) < 3:
        raise RuntimeError("Too few valid samples. Need at least 3, preferably 5-20.")

    overlay_dir = outdir / "overlay_before_after"
    for i, s in enumerate(samples[:10]):
        draw_lidar_edges_overlay(
            s["image_path"], s["edge_pts_L"], T_init, K, dist,
            overlay_dir / f"before_{i:04d}.png", args.z_min, args.z_max
        )

    x0 = np.zeros(6, dtype=float)
    max_rot = np.deg2rad(args.max_rot_deg)
    lb = np.array([-max_rot, -max_rot, -max_rot,
                   -args.max_trans_m, -args.max_trans_m, -args.max_trans_m], dtype=float)
    ub = np.array([ max_rot,  max_rot,  max_rot,
                    args.max_trans_m,  args.max_trans_m,  args.max_trans_m], dtype=float)

    print("[INFO] optimizing local correction...")
    result = least_squares(
        residuals_edge_alignment,
        x0,
        bounds=(lb, ub),
        args=(samples, T_init, K, dist, args),
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
        diff_step=args.diff_step,
        verbose=1,
    )

    T_corr = compose_left_correction(result.x)
    T_refined = T_corr @ T_init

    for i, s in enumerate(samples[:10]):
        draw_lidar_edges_overlay(
            s["image_path"], s["edge_pts_L"], T_refined, K, dist,
            overlay_dir / f"after_{i:04d}.png", args.z_min, args.z_max
        )

    before_res = residuals_edge_alignment(np.zeros(6), samples, T_init, K, dist, args) * args.residual_scale_px
    after_res = residuals_edge_alignment(result.x, samples, T_init, K, dist, args) * args.residual_scale_px

    out = {
        "description": "Edge-based local refinement of camera-LiDAR extrinsic. Requires good initial extrinsic.",
        "transform_convention": {
            "T_camera_lidar": "maps LiDAR coordinates to camera optical coordinates: p_camera = T_camera_lidar @ p_lidar",
            "T_lidar_camera": "inverse transform: p_lidar = T_lidar_camera @ p_camera",
        },
        "T_camera_lidar_initial": T_init.tolist(),
        "T_camera_lidar_refined": T_refined.tolist(),
        "T_lidar_camera_refined": np.linalg.inv(T_refined).tolist(),
        "T_correction_left": T_corr.tolist(),
        "correction_rotvec_rad": result.x[:3].tolist(),
        "correction_translation_m": result.x[3:6].tolist(),
        "correction_angle_deg": float(np.linalg.norm(result.x[:3]) * 180.0 / np.pi),
        "n_valid_samples": len(samples),
        "edge_distance_px_before": {
            "median": float(np.median(before_res)),
            "mean": float(np.mean(before_res)),
            "p95": float(np.percentile(before_res, 95)),
        },
        "edge_distance_px_after": {
            "median": float(np.median(after_res)),
            "mean": float(np.mean(after_res)),
            "p95": float(np.percentile(after_res, 95)),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "cost": float(result.cost),
            "nfev": int(result.nfev),
        },
        "parameters": vars(args),
    }

    with open(outdir / "edge_refined_extrinsic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[OK] edge refinement finished")
    print(f"before median edge distance px: {np.median(before_res):.2f}")
    print(f"after  median edge distance px: {np.median(after_res):.2f}")
    print(f"correction angle deg: {out['correction_angle_deg']:.3f}")
    print(f"correction translation m: {result.x[3:6]}")
    print(f"outputs: {outdir}")


if __name__ == "__main__":
    main()
