#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import os
import sys
import gc

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector


def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


def mean_transform(T_list):
    mats = np.stack([T[:3, :3] for T in T_list], axis=0)
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)
    R_mean = R.from_matrix(mats).mean().as_matrix()
    t_mean = np.mean(trans, axis=0)
    return rt_to_T(R_mean, t_mean)


def compute_transform_stats(T_list, T_mean):
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_mean = T_mean[:3, 3]
    d = np.linalg.norm(trans - t_mean[None, :], axis=1)
    translation_std_m = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0

    R_mean = R.from_matrix(T_mean[:3, :3])
    angs = []
    for T in T_list:
        Ri = R.from_matrix(T[:3, :3])
        dR = R_mean.inv() * Ri
        angs.append(np.degrees(dR.magnitude()))
    rotation_std_deg = float(np.std(angs, ddof=1)) if len(angs) > 1 else 0.0

    return translation_std_m, rotation_std_deg


def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return K, dist


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


def square_object_points(tag_size_m):
    s = float(tag_size_m)
    return np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)


def reorder_corners(corners):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return c[[3, 2, 1, 0], :]


def solve_single_tag_pose(det, tag_size_m, K, dist):
    obj = square_object_points(tag_size_m)
    img = reorder_corners(det.corners)
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    Rm, _ = cv2.Rodrigues(rvec)
    return rt_to_T(Rm, tvec.reshape(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)
    ap.add_argument("--ref-tag-id", type=int, default=17)
    ap.add_argument("--tag-size-m", type=float, default=0.1)
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--output-json", required=True)

    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--max-translation-std-m", type=float, default=0.01)
    ap.add_argument("--max-rotation-std-deg", type=float, default=2.0)

    args = ap.parse_args()

    K, dist = load_camera_json(args.camera_json)
    rows = read_csv_dicts(args.timestamps_csv)
    frame_dir = Path(args.frame_dir)

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    T_W_C2_list = []

    for r in rows:
        frame_idx = to_int(r.get("frame_idx"))
        if frame_idx is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=False)

        det_ref = None
        for d in dets:
            if int(d.tag_id) == args.ref_tag_id:
                det_ref = d
                break
        if det_ref is None:
            continue

        T_C2_W = solve_single_tag_pose(det_ref, args.tag_size_m, K, dist)
        if T_C2_W is None:
            continue

        T_W_C2 = invert_T(T_C2_W)
        T_W_C2_list.append(T_W_C2)

    if len(T_W_C2_list) == 0:
        raise RuntimeError("No valid ref-tag detections found.")

    T_W_C2 = mean_transform(T_W_C2_list)
    translation_std_m, rotation_std_deg = compute_transform_stats(T_W_C2_list, T_W_C2)

    low_confidence = (
        len(T_W_C2_list) < args.min_samples or
        translation_std_m > args.max_translation_std_m or
        rotation_std_deg > args.max_rotation_std_deg
    )

    result = {
        "num_samples": len(T_W_C2_list),
        "ref_tag_id": args.ref_tag_id,
        "tag_size_m": args.tag_size_m,
        "T_W_C2": T_W_C2.tolist(),
        "translation_std_m": translation_std_m,
        "rotation_std_deg": rotation_std_deg,
        "low_confidence": bool(low_confidence),
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved to {args.output_json}")
    print(f"num_samples = {len(T_W_C2_list)}")
    print(f"translation_std_m = {translation_std_m:.4f}")
    print(f"rotation_std_deg = {rotation_std_deg:.3f}")
    print(f"low_confidence = {low_confidence}")

    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()