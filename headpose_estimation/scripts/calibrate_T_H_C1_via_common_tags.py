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


# =========================
# Basic transforms
# =========================

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
    R_stack = np.stack([T[:3, :3] for T in T_list], axis=0)
    t_stack = np.stack([T[:3, 3] for T in T_list], axis=0)

    R_mean = R.from_matrix(R_stack).mean().as_matrix()
    t_mean = np.mean(t_stack, axis=0)

    return rt_to_T(R_mean, t_mean)


def compute_transform_stats(T_list, T_mean):
    t_stack = np.stack([T[:3, 3] for T in T_list], axis=0)
    t_mean = T_mean[:3, 3]

    t_err = np.linalg.norm(t_stack - t_mean[None, :], axis=1)
    translation_std_m = float(np.std(t_err, ddof=1)) if len(t_err) > 1 else 0.0

    R_mean = R.from_matrix(T_mean[:3, :3])
    rot_err = []

    for T in T_list:
        Ri = R.from_matrix(T[:3, :3])
        dR = R_mean.inv() * Ri
        rot_err.append(np.degrees(dR.magnitude()))

    rotation_std_deg = float(np.std(rot_err, ddof=1)) if len(rot_err) > 1 else 0.0

    return translation_std_m, rotation_std_deg


def reprojection_rmse(obj_pts, img_pts, T_C_M, K, dist):
    Rm = T_C_M[:3, :3]
    t = T_C_M[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else np.nan


# =========================
# I/O
# =========================

def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)

    return K, dist


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(row, key, default=None):
    try:
        v = row.get(key, "")
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def nearest_row(rows, target_time_ns, time_col="unix_ns"):
    best = None
    best_dt = None

    for r in rows:
        t = to_int(r, time_col)
        if t is None:
            continue

        dt = abs(int(t) - int(target_time_ns))

        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = r

    return best, best_dt


def resolve_frame_path(frame_dir, frame_idx):
    frame_idx = int(frame_idx)

    for ext in [".png", ".jpg", ".jpeg"]:
        p = frame_dir / f"frame_{frame_idx:06d}{ext}"
        if p.exists():
            return p

    return None


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


def print_summary(name, vals):
    s = summarize(vals)
    if s["mean"] is None:
        print(f"  {name}: no finite values")
    else:
        print(
            f"  {name}: mean={s['mean']:.3f}, "
            f"median={s['median']:.3f}, "
            f"p95={s['p95']:.3f}, "
            f"min={s['min']:.3f}, "
            f"max={s['max']:.3f}"
        )


# =========================
# AprilTag geometry
# =========================

def square_object_points(tag_size_m):
    s = float(tag_size_m)

    return np.array([
        [-s / 2,  s / 2, 0.0],  # lt
        [ s / 2,  s / 2, 0.0],  # rt
        [ s / 2, -s / 2, 0.0],  # rb
        [-s / 2, -s / 2, 0.0],  # lb
    ], dtype=np.float64)


def reorder_head_corners(corners):
    """
    Keep your existing head-tag corner order.
    Do not change this because your head pose result was already good.
    """
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return c[[3, 2, 1, 0], :]


def reorder_common_tag_corners(corners, mode="as_is_0123"):
    """
    Common board tag corner order.
    Your debug result showed that as_is_0123 works best for the common board.
    """
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)

    if mode == "as_is_0123":
        return c[[0, 1, 2, 3], :]

    if mode == "current_reorder_3210":
        return c[[3, 2, 1, 0], :]

    if mode == "shift1_1230":
        return c[[1, 2, 3, 0], :]

    if mode == "shift2_2301":
        return c[[2, 3, 0, 1], :]

    if mode == "shift3_3012":
        return c[[3, 0, 1, 2], :]

    if mode == "reverse_3210":
        return c[::-1, :]

    raise ValueError(f"Unsupported common tag corner order: {mode}")


def tag_area_from_corners(corners):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(abs(cv2.contourArea(c.astype(np.float32))))


def deduplicate_detections_by_id(dets, allowed_ids=None):
    allowed = set(allowed_ids) if allowed_ids is not None else None

    best = {}
    counts = defaultdict(int)

    for d in dets:
        tid = int(d.tag_id)

        if allowed is not None and tid not in allowed:
            continue

        counts[tid] += 1

        margin = float(getattr(d, "decision_margin", 0.0))
        area = tag_area_from_corners(d.corners)
        score = (margin, area)

        if tid not in best or score > best[tid][0]:
            best[tid] = (score, d)

    duplicate_ids = sorted([tid for tid, n in counts.items() if n > 1])
    return {tid: v[1] for tid, v in best.items()}, duplicate_ids


def solve_single_common_tag_pose(det, tag_size_m, K, dist, corner_order):
    obj_pts = square_object_points(tag_size_m)
    img_pts = reorder_common_tag_corners(det.corners, corner_order)

    obj_pts = np.ascontiguousarray(obj_pts.astype(np.float64))
    img_pts = np.ascontiguousarray(img_pts.astype(np.float64))

    candidates = []

    for name, flag in [
        ("IPPE_SQUARE", cv2.SOLVEPNP_IPPE_SQUARE),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
    ]:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                img_pts,
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
        T_C_Tag = rt_to_T(Rm, tvec.reshape(3))
        rmse = reprojection_rmse(obj_pts, img_pts, T_C_Tag, K, dist)

        if np.isfinite(rmse):
            candidates.append((rmse, name, T_C_Tag))

    if not candidates:
        return None, None, np.nan

    candidates.sort(key=lambda x: x[0])
    rmse, method, T_C_Tag = candidates[0]

    return T_C_Tag, method, rmse


# =========================
# Head pose
# =========================

def load_rig_calib(path):
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)

    T_H_T = {
        int(k): np.array(v, dtype=np.float64)
        for k, v in rig["T_H_T"].items()
    }

    head_tag_ids = set(int(x) for x in rig["head_tag_ids"])

    head_tag_size_m = {
        int(k): float(v)
        for k, v in rig["head_tag_size_m"].items()
    }

    return T_H_T, head_tag_ids, head_tag_size_m


def collect_head_correspondences(dets, T_H_T, head_tag_size_m):
    obj_all = []
    img_all = []
    used_ids = []

    for d in dets:
        tid = int(d.tag_id)

        if tid not in T_H_T:
            continue

        pts_tag = square_object_points(head_tag_size_m[tid])
        T_H_Tag = T_H_T[tid]

        pts_H = (T_H_Tag[:3, :3] @ pts_tag.T).T + T_H_Tag[:3, 3]
        img = reorder_head_corners(d.corners)

        obj_all.append(pts_H)
        img_all.append(img)
        used_ids.append(tid)

    if not obj_all:
        return None, None, []

    obj = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))

    return obj, img, used_ids


def solve_head_bundle_pose(dets, T_H_T, head_tag_size_m, K, dist, min_head_tags=2):
    obj_pts, img_pts, used_ids = collect_head_correspondences(
        dets,
        T_H_T,
        head_tag_size_m,
    )

    used_unique = sorted(list(set(used_ids)))

    if obj_pts is None or len(used_unique) < min_head_tags:
        return None, used_unique, np.nan

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts,
            img_pts,
            K,
            dist,
            iterationsCount=200,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        ok = False
        inliers = None

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
    T_C2_H = rt_to_T(Rm, tvec.reshape(3))
    rmse = reprojection_rmse(obj_pts, img_pts, T_C2_H, K, dist)

    return T_C2_H, used_unique, rmse


# =========================
# Time windows
# =========================

def get_scene_window(scene_rows):
    vals = []

    for r in scene_rows:
        fi = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")

        if fi is not None and ts is not None:
            vals.append((fi, ts))

    if not vals:
        raise RuntimeError("No valid frame_idx/unix_ns in scene_timestamps.csv")

    return {
        "scene_frame_min": min(v[0] for v in vals),
        "scene_frame_max": max(v[0] for v in vals),
        "scene_ts_min": min(v[1] for v in vals),
        "scene_ts_max": max(v[1] for v in vals),
        "num_rows": len(vals),
    }


def get_back_window(back_rows, ts_min, ts_max):
    vals = []

    for r in back_rows:
        fi = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")

        if fi is not None and ts is not None and ts_min <= ts <= ts_max:
            vals.append((fi, ts))

    if not vals:
        raise RuntimeError("No back-camera rows found inside scene timestamp window")

    return {
        "back_frame_min": min(v[0] for v in vals),
        "back_frame_max": max(v[0] for v in vals),
        "back_ts_min": min(v[1] for v in vals),
        "back_ts_max": max(v[1] for v in vals),
        "num_rows": len(vals),
    }


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--back-camera-json", required=True)
    ap.add_argument("--back-frame-dir", required=True)
    ap.add_argument("--back-timestamps-csv", required=True)

    ap.add_argument("--scene-camera-json", required=True)
    ap.add_argument("--scene-frame-dir", required=True)
    ap.add_argument("--scene-timestamps-csv", required=True)

    ap.add_argument("--rig-calib-json", required=True)

    ap.add_argument("--common-tag-ids", required=True)
    ap.add_argument("--common-tag-size-m", type=float, required=True)

    ap.add_argument(
        "--common-corner-order",
        default="as_is_0123",
        choices=[
            "as_is_0123",
            "current_reorder_3210",
            "shift1_1230",
            "shift2_2301",
            "shift3_3012",
            "reverse_3210",
        ],
    )

    ap.add_argument("--duplicate-policy", default="reject", choices=["reject", "best"])

    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--min-head-tags", type=int, default=2)

    ap.add_argument("--max-pair-dt-ms", type=float, default=30.0)

    ap.add_argument("--max-scene-tag-rmse-px", type=float, default=5.0)
    ap.add_argument("--max-back-tag-rmse-px", type=float, default=5.0)
    ap.add_argument("--max-head-rmse-px", type=float, default=3.0)

    ap.add_argument("--output-json", required=True)

    args = ap.parse_args()

    common_tag_ids = [int(x) for x in args.common_tag_ids.split(",")]
    common_tag_set = set(common_tag_ids)

    K_back, dist_back = load_camera_json(args.back_camera_json)
    K_scene, dist_scene = load_camera_json(args.scene_camera_json)

    T_H_T, head_tag_ids, head_tag_size_m = load_rig_calib(args.rig_calib_json)

    overlap = common_tag_set & head_tag_ids
    if overlap:
        raise RuntimeError(
            f"Common tag IDs overlap with helmet head tag IDs: {sorted(overlap)}"
        )

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    scene_rows = read_csv_dicts(args.scene_timestamps_csv)
    back_rows = read_csv_dicts(args.back_timestamps_csv)

    scene_window = get_scene_window(scene_rows)
    back_window = get_back_window(
        back_rows,
        scene_window["scene_ts_min"],
        scene_window["scene_ts_max"],
    )

    print("[INFO] Scene window:")
    print(scene_window)
    print("[INFO] Back-camera window:")
    print(back_window)

    scene_frame_dir = Path(args.scene_frame_dir)
    back_frame_dir = Path(args.back_frame_dir)

    candidate_back_rows = []

    for br in back_rows:
        fi = to_int(br, "frame_idx")
        ts = to_int(br, "unix_ns")

        if fi is None or ts is None:
            continue

        if back_window["back_frame_min"] <= fi <= back_window["back_frame_max"]:
            candidate_back_rows.append(br)

    valid_scene_rows = []
    seen = set()

    for r in scene_rows:
        fi = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")

        if fi is None or ts is None:
            continue

        if fi in seen:
            continue

        seen.add(fi)
        valid_scene_rows.append(r)

    T_H_C1_list = []

    pair_dt_ms_list = []
    scene_tag_rmse_list = []
    back_tag_rmse_list = []
    head_rmse_list = []

    scene_tag_rmse_all = []
    back_tag_rmse_all = []
    head_rmse_all = []

    per_common_tag_count = defaultdict(int)

    counts = {
        "scene_frames_checked": 0,

        "scene_common_tags_detected_frames": 0,
        "scene_common_tags_duplicate_rejected_frames": 0,
        "scene_common_tag_pose_ok": 0,
        "scene_common_tag_pose_rejected_by_rmse": 0,

        "back_common_tags_detected_frames": 0,
        "back_common_tags_duplicate_rejected_frames": 0,
        "back_common_tag_pose_ok": 0,
        "back_common_tag_pose_rejected_by_rmse": 0,

        "head_detected": 0,
        "head_rejected_by_rmse": 0,
        "head_ok": 0,

        "paired_common_tag_samples": 0,
    }

    for scene_row in valid_scene_rows:
        scene_frame_idx = to_int(scene_row, "frame_idx")
        scene_time_ns = to_int(scene_row, "unix_ns")

        if scene_frame_idx is None or scene_time_ns is None:
            continue

        counts["scene_frames_checked"] += 1

        back_row, dt_ns = nearest_row(candidate_back_rows, scene_time_ns)

        if back_row is None:
            continue

        dt_ms = dt_ns / 1e6
        if dt_ms > args.max_pair_dt_ms:
            continue

        back_frame_idx = to_int(back_row, "frame_idx")

        if back_frame_idx is None:
            continue

        scene_img_path = resolve_frame_path(scene_frame_dir, scene_frame_idx)
        back_img_path = resolve_frame_path(back_frame_dir, back_frame_idx)

        if scene_img_path is None or back_img_path is None:
            continue

        img_scene = cv2.imread(str(scene_img_path), cv2.IMREAD_COLOR)
        img_back = cv2.imread(str(back_img_path), cv2.IMREAD_COLOR)

        if img_scene is None or img_back is None:
            continue

        gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
        gray_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)

        dets_scene = detector.detect(gray_scene, estimate_tag_pose=False)
        dets_back = detector.detect(gray_back, estimate_tag_pose=False)

        scene_common, scene_dup = deduplicate_detections_by_id(
            dets_scene,
            allowed_ids=common_tag_set,
        )

        back_common, back_dup = deduplicate_detections_by_id(
            dets_back,
            allowed_ids=common_tag_set,
        )

        if scene_common:
            counts["scene_common_tags_detected_frames"] += 1

        if back_common:
            counts["back_common_tags_detected_frames"] += 1

        if args.duplicate_policy == "reject" and scene_dup:
            counts["scene_common_tags_duplicate_rejected_frames"] += 1
            continue

        if args.duplicate_policy == "reject" and back_dup:
            counts["back_common_tags_duplicate_rejected_frames"] += 1
            continue

        # Scene camera common tag poses
        scene_tag_poses = {}

        for tid, det in scene_common.items():
            T_C1_Tag, method, rmse = solve_single_common_tag_pose(
                det,
                args.common_tag_size_m,
                K_scene,
                dist_scene,
                args.common_corner_order,
            )

            if T_C1_Tag is None:
                continue

            scene_tag_rmse_all.append(rmse)

            if rmse > args.max_scene_tag_rmse_px:
                counts["scene_common_tag_pose_rejected_by_rmse"] += 1
                continue

            scene_tag_poses[tid] = (T_C1_Tag, rmse)
            counts["scene_common_tag_pose_ok"] += 1

        if not scene_tag_poses:
            continue

        # Back camera common tag poses
        back_tag_poses = {}

        for tid, det in back_common.items():
            T_C2_Tag, method, rmse = solve_single_common_tag_pose(
                det,
                args.common_tag_size_m,
                K_back,
                dist_back,
                args.common_corner_order,
            )

            if T_C2_Tag is None:
                continue

            back_tag_rmse_all.append(rmse)

            if rmse > args.max_back_tag_rmse_px:
                counts["back_common_tag_pose_rejected_by_rmse"] += 1
                continue

            back_tag_poses[tid] = (T_C2_Tag, rmse)
            counts["back_common_tag_pose_ok"] += 1

        if not back_tag_poses:
            continue

        common_visible_ids = sorted(set(scene_tag_poses.keys()) & set(back_tag_poses.keys()))

        if not common_visible_ids:
            continue

        # Back camera head pose
        head_dets = [d for d in dets_back if int(d.tag_id) in head_tag_ids]

        T_C2_H, used_head_ids, head_rmse = solve_head_bundle_pose(
            head_dets,
            T_H_T,
            head_tag_size_m,
            K_back,
            dist_back,
            min_head_tags=args.min_head_tags,
        )

        if T_C2_H is None:
            continue

        counts["head_detected"] += 1
        head_rmse_all.append(head_rmse)

        if head_rmse > args.max_head_rmse_px:
            counts["head_rejected_by_rmse"] += 1
            continue

        counts["head_ok"] += 1

        for tid in common_visible_ids:
            T_C1_Tag, scene_rmse = scene_tag_poses[tid]
            T_C2_Tag, back_rmse = back_tag_poses[tid]

            T_H_C1 = invert_T(T_C2_H) @ T_C2_Tag @ invert_T(T_C1_Tag)

            T_H_C1_list.append(T_H_C1)

            pair_dt_ms_list.append(dt_ms)
            scene_tag_rmse_list.append(scene_rmse)
            back_tag_rmse_list.append(back_rmse)
            head_rmse_list.append(head_rmse)

            per_common_tag_count[tid] += 1
            counts["paired_common_tag_samples"] += 1

    print("[INFO] Debug counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("[INFO] RMSE debug before filtering:")
    print_summary("scene_common_tag_rmse_px_all", scene_tag_rmse_all)
    print_summary("back_common_tag_rmse_px_all", back_tag_rmse_all)
    print_summary("head_rmse_px_all", head_rmse_all)

    print("[INFO] paired samples per common tag:")
    for tid in sorted(per_common_tag_count):
        print(f"  tag {tid}: {per_common_tag_count[tid]}")

    if len(T_H_C1_list) == 0:
        raise RuntimeError(
            "No valid paired common-tag samples found. "
            "Try --duplicate-policy best, or relax RMSE thresholds, or test another common-corner-order."
        )

    T_H_C1_mean = mean_transform(T_H_C1_list)
    translation_std_m, rotation_std_deg = compute_transform_stats(
        T_H_C1_list,
        T_H_C1_mean,
    )

    pair_dt_stats = summarize(pair_dt_ms_list)
    scene_rmse_stats = summarize(scene_tag_rmse_list)
    back_rmse_stats = summarize(back_tag_rmse_list)
    head_rmse_stats = summarize(head_rmse_list)

    result = {
        "num_samples": len(T_H_C1_list),
        "T_H_C1": T_H_C1_mean.tolist(),

        "translation_std_m": translation_std_m,
        "rotation_std_deg": rotation_std_deg,

        "common_tag_ids": common_tag_ids,
        "common_tag_size_m": args.common_tag_size_m,
        "common_corner_order": args.common_corner_order,
        "duplicate_policy": args.duplicate_policy,

        "min_head_tags": args.min_head_tags,
        "max_pair_dt_ms": args.max_pair_dt_ms,
        "max_scene_tag_rmse_px": args.max_scene_tag_rmse_px,
        "max_back_tag_rmse_px": args.max_back_tag_rmse_px,
        "max_head_rmse_px": args.max_head_rmse_px,

        "pair_dt_ms": pair_dt_stats,
        "scene_common_tag_rmse_px": scene_rmse_stats,
        "back_common_tag_rmse_px": back_rmse_stats,
        "head_rmse_px": head_rmse_stats,

        "rmse_debug_before_filtering": {
            "scene_common_tag_rmse_px_all": summarize(scene_tag_rmse_all),
            "back_common_tag_rmse_px_all": summarize(back_tag_rmse_all),
            "head_rmse_px_all": summarize(head_rmse_all),
        },

        "per_common_tag_count": {
            str(k): int(v) for k, v in sorted(per_common_tag_count.items())
        },

        "scene_window": scene_window,
        "back_window": back_window,
        "debug_counts": counts,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] saved to {output_path}")
    print(f"[INFO] num_samples = {len(T_H_C1_list)}")
    print(f"[INFO] translation_std_m = {translation_std_m:.4f}")
    print(f"[INFO] rotation_std_deg = {rotation_std_deg:.3f}")
    print(f"[INFO] pair_dt_ms mean/p95 = {pair_dt_stats['mean']} / {pair_dt_stats['p95']}")
    print(f"[INFO] scene common tag rmse mean/p95 = {scene_rmse_stats['mean']} / {scene_rmse_stats['p95']}")
    print(f"[INFO] back common tag rmse mean/p95 = {back_rmse_stats['mean']} / {back_rmse_stats['p95']}")
    print(f"[INFO] head rmse mean/p95 = {head_rmse_stats['mean']} / {head_rmse_stats['p95']}")

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()