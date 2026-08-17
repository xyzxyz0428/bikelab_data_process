#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_camera_lidar_manual_2d3d.py

Estimate camera-LiDAR extrinsic from manual 2D-3D correspondences.

Input CSV required columns:
    point_id,u,v,x,y,z

where:
    u,v = image pixel coordinates
    x,y,z = same physical point in LiDAR frame, meters

Output:
    calibration_result.json
    reprojection_errors.csv

Transform convention:
    T_camera_lidar maps LiDAR frame to camera optical frame:
        p_camera = T_camera_lidar @ p_lidar
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
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


def project(x, pts_lidar, K, dist):
    rvec = x[:3].reshape(3, 1)
    tvec = x[3:6].reshape(3, 1)
    uv, _ = cv2.projectPoints(pts_lidar.astype(np.float64), rvec, tvec, K, dist)
    return uv.reshape(-1, 2)


def residuals(x, pts_lidar, pts_img, K, dist):
    pred = project(x, pts_lidar, K, dist)
    return (pred - pts_img).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correspondences", required=True, help="CSV with point_id,u,v,x,y,z")
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--use-ransac-init", action="store_true")
    ap.add_argument("--ransac-threshold-px", type=float, default=8.0)
    ap.add_argument("--loss", default="soft_l1", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    ap.add_argument("--f-scale", type=float, default=5.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    K, dist = read_camera_json(Path(args.camera_json))
    df = pd.read_csv(args.correspondences)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["u", "v", "x", "y", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).copy()

    pts_img = df[["u", "v"]].to_numpy(np.float64)
    pts_lidar = df[["x", "y", "z"]].to_numpy(np.float64)

    if len(df) < 6:
        raise RuntimeError("Need at least 6 correspondences; 15-40 are recommended.")

    if args.use_ransac_init:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_lidar, pts_img, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=args.ransac_threshold_px,
            iterationsCount=1000,
            confidence=0.999,
        )
        if not ok:
            raise RuntimeError("solvePnPRansac failed.")
        print(f"[INFO] RANSAC inliers: {0 if inliers is None else len(inliers)} / {len(df)}")
    else:
        ok, rvec, tvec = cv2.solvePnP(
            pts_lidar, pts_img, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP failed.")

    x0 = np.zeros(6, dtype=float)
    x0[:3] = rvec.reshape(3)
    x0[3:6] = tvec.reshape(3)

    result = least_squares(
        residuals,
        x0,
        args=(pts_lidar, pts_img, K, dist),
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=3000,
    )

    pred = project(result.x, pts_lidar, K, dist)
    err = np.linalg.norm(pred - pts_img, axis=1)

    R_C_L = R.from_rotvec(result.x[:3]).as_matrix()
    t_C_L = result.x[3:6]
    T_C_L = make_T(R_C_L, t_C_L)
    T_L_C = np.linalg.inv(T_C_L)

    err_df = df.copy()
    err_df["u_proj"] = pred[:, 0]
    err_df["v_proj"] = pred[:, 1]
    err_df["reproj_error_px"] = err
    err_df.to_csv(outdir / "reprojection_errors.csv", index=False)

    out = {
        "description": "Camera-LiDAR calibration from manual 2D-3D correspondences.",
        "transform_convention": {
            "T_camera_lidar": "maps LiDAR coordinates to camera optical coordinates: p_camera = T_camera_lidar @ p_lidar",
            "T_lidar_camera": "inverse transform: p_lidar = T_lidar_camera @ p_camera"
        },
        "T_camera_lidar": T_C_L.tolist(),
        "T_lidar_camera": T_L_C.tolist(),
        "translation_camera_lidar_m": t_C_L.tolist(),
        "rotation_matrix_camera_lidar": R_C_L.tolist(),
        "rotvec_camera_lidar": result.x[:3].tolist(),
        "n_correspondences": int(len(df)),
        "reprojection_error_px": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)),
            "max": float(np.max(err)),
        },
        "optimizer": {
            "success": bool(result.success),
            "message": result.message,
            "cost": float(result.cost),
        }
    }

    with open(outdir / "calibration_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("[OK] calibration finished")
    print(f"reprojection error px: median={np.median(err):.2f}, p95={np.percentile(err,95):.2f}, max={np.max(err):.2f}")
    print(f"T_camera_lidar:\n{T_C_L}")
    print(f"outputs: {outdir}")


if __name__ == "__main__":
    main()
