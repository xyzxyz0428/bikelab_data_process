#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import os
import sys
import gc
from collections import defaultdict

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector


def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


def mean_transform(T_list):
    R_stack = np.stack([T[:3, :3] for T in T_list], axis=0)
    t_stack = np.stack([T[:3, 3] for T in T_list], axis=0)
    R_mean = R.from_matrix(R_stack).mean().as_matrix()
    t_mean = np.mean(t_stack, axis=0)
    return rt_to_T(R_mean, t_mean)


def transform_points(T_ab, pts_b):
    return (T_ab[:3, :3] @ pts_b.T).T + T_ab[:3, 3]


def transform_errors(T_list, T_mean):
    trans_err = []
    rot_err = []

    R_mean = R.from_matrix(T_mean[:3, :3])
    t_mean = T_mean[:3, 3]

    for T in T_list:
        trans_err.append(float(np.linalg.norm(T[:3, 3] - t_mean)))
        Ri = R.from_matrix(T[:3, :3])
        dR = R_mean.inv() * Ri
        rot_err.append(float(np.degrees(dR.magnitude())))

    return np.asarray(trans_err), np.asarray(rot_err)


def summarize(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


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


def reorder_corners(corners, mode):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)

    if mode == "pupil_3210":
        # pupil_apriltags often gives lb rb rt lt
        # convert to lt rt rb lb
        return c[[3, 2, 1, 0], :]

    if mode == "as_is_0123":
        return c[[0, 1, 2, 3], :]

    if mode == "shift1_1230":
        return c[[1, 2, 3, 0], :]

    if mode == "shift2_2301":
        return c[[2, 3, 0, 1], :]

    if mode == "shift3_3012":
        return c[[3, 0, 1, 2], :]

    raise ValueError(f"Unsupported corner order: {mode}")


def reprojection_rmse(obj_pts, img_pts, T_C_W, K, dist):
    Rm = T_C_W[:3, :3]
    t = T_C_W[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else np.nan


def solve_single_tag_pose(det, tag_size_m, K, dist, corner_order):
    obj = square_object_points(tag_size_m)
    img = reorder_corners(det.corners, corner_order)

    candidates = []

    for name, flag in [
        ("IPPE_SQUARE", cv2.SOLVEPNP_IPPE_SQUARE),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ]:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj,
                img,
                K,
                dist,
                flags=flag,
            )
        except cv2.error:
            ok = False

        if not ok:
            continue

        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj,
                img,
                K,
                dist,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        Rm, _ = cv2.Rodrigues(rvec)
        T_C_Tag = rt_to_T(Rm, tvec.reshape(3))

        rmse = reprojection_rmse(obj, img, T_C_Tag, K, dist)
        if np.isfinite(rmse):
            candidates.append((rmse, name, T_C_Tag))

    if not candidates:
        return None, np.nan

    candidates.sort(key=lambda x: x[0])
    return candidates[0][2], candidates[0][0]


def deduplicate_detections(dets, allowed_ids):
    allowed = set(allowed_ids)
    best = {}

    for d in dets:
        tid = int(d.tag_id)
        if tid not in allowed:
            continue

        margin = float(getattr(d, "decision_margin", 0.0))
        if tid not in best or margin > best[tid][0]:
            best[tid] = (margin, d)

    return {tid: item[1] for tid, item in best.items()}


def solve_multitag_camera_pose(dets_by_id, tag_map_W, tag_size_m, K, dist, corner_order, min_tags):
    obj_all = []
    img_all = []
    used_ids = []

    tag_local_corners = square_object_points(tag_size_m)

    for tid, det in dets_by_id.items():
        if tid not in tag_map_W:
            continue

        T_W_Tag = tag_map_W[tid]
        obj_W = transform_points(T_W_Tag, tag_local_corners)
        img = reorder_corners(det.corners, corner_order)

        obj_all.append(obj_W)
        img_all.append(img)
        used_ids.append(tid)

    used_unique = sorted(list(set(used_ids)))

    if len(used_unique) < min_tags:
        return None, used_unique, np.nan

    obj_pts = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img_pts = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            dist,
            iterationsCount=300,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        ok = False

    if not ok:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
                K,
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False

    if not ok:
        return None, used_unique, np.nan

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts,
            img_pts,
            K,
            dist,
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    Rm, _ = cv2.Rodrigues(rvec)
    T_C_W = rt_to_T(Rm, tvec.reshape(3))

    rmse = reprojection_rmse(obj_pts, img_pts, T_C_W, K, dist)
    return T_C_W, used_unique, rmse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)
    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--ref-tag-id", type=int, required=True)
    ap.add_argument("--target-tag-ids", required=True)
    ap.add_argument("--tag-size-m", type=float, required=True)

    ap.add_argument("--corner-order", default="pupil_3210",
                    choices=[
                        "pupil_3210",
                        "as_is_0123",
                        "shift1_1230",
                        "shift2_2301",
                        "shift3_3012",
                    ])

    ap.add_argument("--min-ref-samples", type=int, default=20)
    ap.add_argument("--min-tag-samples", type=int, default=20)
    ap.add_argument("--min-tags-per-frame", type=int, default=3)

    ap.add_argument("--max-single-tag-rmse-px", type=float, default=3.0)
    ap.add_argument("--max-multitag-rmse-px", type=float, default=3.0)

    ap.add_argument("--max-translation-std-m", type=float, default=0.01)
    ap.add_argument("--max-rotation-std-deg", type=float, default=2.0)

    ap.add_argument("--output-json", required=True)

    args = ap.parse_args()

    K, dist = load_camera_json(args.camera_json)
    rows = read_csv_dicts(args.timestamps_csv)
    frame_dir = Path(args.frame_dir)

    target_tag_ids = [int(x) for x in args.target_tag_ids.split(",")]
    all_tag_ids = sorted(list(set(target_tag_ids + [args.ref_tag_id])))

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    # ------------------------------------------------------------
    # Pass 1: build initial tag map using ref tag 17 as world frame
    # ------------------------------------------------------------
    T_W_Tag_samples = defaultdict(list)
    T_W_C2_from_ref_samples = []
    single_tag_rmse = defaultdict(list)

    used_frames_pass1 = 0
    ref_seen_frames = 0

    for r in rows:
        frame_idx = to_int(r.get("frame_idx"))
        if frame_idx is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        dets = detector.detect(img, estimate_tag_pose=False)
        dets_by_id = deduplicate_detections(dets, all_tag_ids)

        if args.ref_tag_id not in dets_by_id:
            continue

        T_C_Ref, ref_rmse = solve_single_tag_pose(
            dets_by_id[args.ref_tag_id],
            args.tag_size_m,
            K,
            dist,
            args.corner_order,
        )

        if T_C_Ref is None or ref_rmse > args.max_single_tag_rmse_px:
            continue

        ref_seen_frames += 1

        # Define W frame as ref tag frame:
        # T_W_C = inv(T_C_Ref)
        T_W_C = invert_T(T_C_Ref)
        T_W_C2_from_ref_samples.append(T_W_C)

        for tid, det in dets_by_id.items():
            T_C_Tag, rmse = solve_single_tag_pose(
                det,
                args.tag_size_m,
                K,
                dist,
                args.corner_order,
            )

            if T_C_Tag is None or rmse > args.max_single_tag_rmse_px:
                continue

            T_W_Tag = T_W_C @ T_C_Tag
            T_W_Tag_samples[tid].append(T_W_Tag)
            single_tag_rmse[tid].append(rmse)

        used_frames_pass1 += 1

    if len(T_W_C2_from_ref_samples) < args.min_ref_samples:
        raise RuntimeError(
            f"Too few ref-tag samples: {len(T_W_C2_from_ref_samples)}"
        )

    tag_map_W = {}
    tag_stats = {}

    for tid in all_tag_ids:
        samples = T_W_Tag_samples.get(tid, [])
        if len(samples) < args.min_tag_samples:
            tag_stats[str(tid)] = {
                "num_samples": len(samples),
                "status": "rejected_too_few_samples",
            }
            continue

        T_mean = mean_transform(samples)
        trans_err, rot_err = transform_errors(samples, T_mean)

        translation_std_m = float(np.std(trans_err, ddof=1)) if len(trans_err) > 1 else 0.0
        rotation_std_deg = float(np.std(rot_err, ddof=1)) if len(rot_err) > 1 else 0.0

        low_conf = (
            translation_std_m > args.max_translation_std_m
            or rotation_std_deg > args.max_rotation_std_deg
        )

        tag_map_W[tid] = T_mean

        tag_stats[str(tid)] = {
            "num_samples": len(samples),
            "status": "map_used",
            "T_W_Tag": T_mean.tolist(),
            "center_W": T_mean[:3, 3].tolist(),
            "normal_W": T_mean[:3, 2].tolist(),
            "translation_std_m": translation_std_m,
            "rotation_std_deg": rotation_std_deg,
            "single_tag_rmse_px": summarize(single_tag_rmse[tid]),
            "low_confidence": bool(low_conf),
        }

    if args.ref_tag_id not in tag_map_W:
        raise RuntimeError("Reference tag was not included in tag map.")

    # Force exact ref tag pose as identity in W
    tag_map_W[args.ref_tag_id] = np.eye(4, dtype=np.float64)
    tag_stats[str(args.ref_tag_id)]["T_W_Tag"] = tag_map_W[args.ref_tag_id].tolist()
    tag_stats[str(args.ref_tag_id)]["center_W"] = [0.0, 0.0, 0.0]
    tag_stats[str(args.ref_tag_id)]["normal_W"] = [0.0, 0.0, 1.0]

    # ------------------------------------------------------------
    # Pass 2: estimate camera pose with multiple mapped tags
    # ------------------------------------------------------------
    T_W_C2_multitag_samples = []
    multitag_rmse_list = []
    used_ids_per_frame = []

    frames_checked_pass2 = 0
    frames_multitag_ok = 0
    frames_rejected_rmse = 0

    for r in rows:
        frame_idx = to_int(r.get("frame_idx"))
        if frame_idx is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        frames_checked_pass2 += 1

        dets = detector.detect(img, estimate_tag_pose=False)
        dets_by_id = deduplicate_detections(dets, tag_map_W.keys())

        T_C_W, used_ids, rmse = solve_multitag_camera_pose(
            dets_by_id,
            tag_map_W,
            args.tag_size_m,
            K,
            dist,
            args.corner_order,
            args.min_tags_per_frame,
        )

        if T_C_W is None:
            continue

        if rmse > args.max_multitag_rmse_px:
            frames_rejected_rmse += 1
            continue

        T_W_C2 = invert_T(T_C_W)
        T_W_C2_multitag_samples.append(T_W_C2)
        multitag_rmse_list.append(rmse)
        used_ids_per_frame.append(used_ids)
        frames_multitag_ok += 1

    if len(T_W_C2_multitag_samples) < args.min_ref_samples:
        raise RuntimeError(
            f"Too few multitag camera pose samples: {len(T_W_C2_multitag_samples)}"
        )

    T_W_C2_mean = mean_transform(T_W_C2_multitag_samples)
    trans_err, rot_err = transform_errors(T_W_C2_multitag_samples, T_W_C2_mean)

    translation_std_m = float(np.std(trans_err, ddof=1)) if len(trans_err) > 1 else 0.0
    rotation_std_deg = float(np.std(rot_err, ddof=1)) if len(rot_err) > 1 else 0.0

    low_confidence = (
        translation_std_m > args.max_translation_std_m
        or rotation_std_deg > args.max_rotation_std_deg
    )

    used_tag_count = defaultdict(int)
    for ids in used_ids_per_frame:
        for tid in ids:
            used_tag_count[tid] += 1

    result = {
        "method": "bootstrap_multitag_from_ref_tag_map",
        "ref_tag_id": args.ref_tag_id,
        "tag_size_m": args.tag_size_m,
        "target_tag_ids": target_tag_ids,
        "corner_order": args.corner_order,

        "T_W_C2": T_W_C2_mean.tolist(),
        "num_samples": len(T_W_C2_multitag_samples),
        "translation_std_m": translation_std_m,
        "rotation_std_deg": rotation_std_deg,
        "low_confidence": bool(low_confidence),

        "translation_error_m": summarize(trans_err),
        "rotation_error_deg": summarize(rot_err),
        "multitag_rmse_px": summarize(multitag_rmse_list),

        "debug": {
            "used_frames_pass1": used_frames_pass1,
            "ref_seen_frames": ref_seen_frames,
            "frames_checked_pass2": frames_checked_pass2,
            "frames_multitag_ok": frames_multitag_ok,
            "frames_rejected_rmse": frames_rejected_rmse,
            "used_tag_count_pass2": {str(k): int(v) for k, v in sorted(used_tag_count.items())},
        },

        "tag_map": tag_stats,
        "note": (
            "This is a bootstrap multi-tag estimate. Tag map geometry was estimated "
            "from the same baseline image sequence using the reference tag as world frame. "
            "It reduces single-tag frame noise but is not an independently measured reference board."
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[INFO] Bootstrap multi-tag T_W_C2 result")
    print(f"  num_samples = {len(T_W_C2_multitag_samples)}")
    print(f"  translation_std_m = {translation_std_m:.4f}")
    print(f"  rotation_std_deg = {rotation_std_deg:.3f}")
    print(f"  low_confidence = {low_confidence}")
    print(f"  multitag_rmse_px median = {summarize(multitag_rmse_list)['median']}")
    print("[INFO] used tag count:")
    for tid, n in sorted(used_tag_count.items()):
        print(f"  tag {tid}: {n}")
    print(f"[INFO] saved to {output_path}")

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()