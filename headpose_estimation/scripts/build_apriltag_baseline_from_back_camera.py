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
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mean
    T[:3, 3] = t_mean
    return T


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


def square_object_points(tag_size_m):
    s = float(tag_size_m)
    return np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)


def reorder_corners(corners):
    # pupil_apriltags returns lb, rb, rt, lt
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


def resolve_frame_path(frame_dir, frame_idx):
    frame_idx = int(frame_idx)
    candidates = [
        frame_dir / f"frame_{frame_idx:06d}.png",
        frame_dir / f"frame_{frame_idx:06d}.jpg",
        frame_dir / f"frame_{frame_idx:06d}.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--default-size-m", type=float, required=True)
    ap.add_argument("--ref-tag-id", type=int, required=True)
    ap.add_argument("--target-tag-ids", required=True)
    ap.add_argument("--output-json", required=True)

    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-translation-std-m", type=float, default=0.01)
    ap.add_argument("--max-rotation-std-deg", type=float, default=2.0)
    ap.add_argument("--skip-low-confidence-tags", action="store_true")

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

    target_ids = [int(x) for x in args.target_tag_ids.split(",")]
    all_ids = set(target_ids + [args.ref_tag_id])

    samples = {tid: [] for tid in target_ids}
    used_frames = 0

    for r in rows:
        frame_idx = int(float(r["frame_idx"]))
        img_path = resolve_frame_path(frame_dir, frame_idx)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=False)
        det_map = {int(d.tag_id): d for d in dets if int(d.tag_id) in all_ids}

        if args.ref_tag_id not in det_map:
            continue

        T_C_ref = solve_single_tag_pose(det_map[args.ref_tag_id], args.default_size_m, K, dist)
        if T_C_ref is None:
            continue

        T_ref_C = invert_T(T_C_ref)
        used_this_frame = False

        for tid in target_ids:
            if tid not in det_map:
                continue

            T_C_tag = solve_single_tag_pose(det_map[tid], args.default_size_m, K, dist)
            if T_C_tag is None:
                continue

            T_W_tag = T_ref_C @ T_C_tag
            samples[tid].append(T_W_tag)
            used_this_frame = True

        if used_this_frame:
            used_frames += 1

    result = {
        "ref_tag_id": args.ref_tag_id,
        "default_size_m": args.default_size_m,
        "used_frames": used_frames,
        "tags": {}
    }

    for tid, Ts in samples.items():
        if len(Ts) == 0:
            print(f"[WARN] no samples for tag {tid}")
            continue

        T_mean = mean_transform(Ts)
        translation_std_m, rotation_std_deg = compute_transform_stats(Ts, T_mean)

        low_confidence = (
            len(Ts) < args.min_samples or
            translation_std_m > args.max_translation_std_m or
            rotation_std_deg > args.max_rotation_std_deg
        )

        print(
            f"tag {tid}: n={len(Ts)}, "
            f"translation_std_m={translation_std_m:.4f}, "
            f"rotation_std_deg={rotation_std_deg:.3f}, "
            f"low_confidence={low_confidence}"
        )

        if args.skip_low_confidence_tags and low_confidence:
            continue

        result["tags"][str(tid)] = {
            "num_samples": len(Ts),
            "T_W_Tag": T_mean.tolist(),
            "center_W": T_mean[:3, 3].tolist(),
            "normal_W": T_mean[:3, 2].tolist(),
            "translation_std_m": translation_std_m,
            "rotation_std_deg": rotation_std_deg,
            "low_confidence": bool(low_confidence),
        }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"used_frames={used_frames}")
    print(f"saved to {args.output_json}")

    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()