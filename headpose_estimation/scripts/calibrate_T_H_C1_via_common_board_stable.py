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


# ============================================================
# Basic transforms
# ============================================================

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


def pose_delta(T_prev, T_cur):
    if T_prev is None or T_cur is None:
        return None, None

    dt = float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))

    R_prev = R.from_matrix(T_prev[:3, :3])
    R_cur = R.from_matrix(T_cur[:3, :3])
    dR = R_prev.inv() * R_cur
    da = float(np.degrees(dR.magnitude()))

    return dt, da


def transform_errors_to_mean(T_list, T_mean):
    t_mean = T_mean[:3, 3]
    R_mean = R.from_matrix(T_mean[:3, :3])

    trans_err = []
    rot_err = []

    for T in T_list:
        trans_err.append(float(np.linalg.norm(T[:3, 3] - t_mean)))
        Ri = R.from_matrix(T[:3, :3])
        dR = R_mean.inv() * Ri
        rot_err.append(float(np.degrees(dR.magnitude())))

    return np.asarray(trans_err), np.asarray(rot_err)


def robust_filter_transforms(
    T_list,
    max_translation_error_m,
    max_rotation_error_deg,
    max_iterations=5,
):
    if len(T_list) == 0:
        return None, [], {}

    keep_idx = list(range(len(T_list)))

    for _ in range(max_iterations):
        cur_Ts = [T_list[i] for i in keep_idx]
        if len(cur_Ts) == 0:
            break

        T_mean = mean_transform(cur_Ts)
        trans_err, rot_err = transform_errors_to_mean(cur_Ts, T_mean)

        new_keep_local = []
        for i, (te, re) in enumerate(zip(trans_err, rot_err)):
            if te <= max_translation_error_m and re <= max_rotation_error_deg:
                new_keep_local.append(i)

        new_keep_idx = [keep_idx[i] for i in new_keep_local]

        if len(new_keep_idx) == len(keep_idx):
            break

        keep_idx = new_keep_idx

    if len(keep_idx) == 0:
        return None, [], {}

    final_Ts = [T_list[i] for i in keep_idx]
    T_final = mean_transform(final_Ts)
    trans_err, rot_err = transform_errors_to_mean(final_Ts, T_final)

    stats = {
        "num_input": int(len(T_list)),
        "num_kept": int(len(final_Ts)),
        "num_removed": int(len(T_list) - len(final_Ts)),
        "translation_error_m": summarize(trans_err),
        "rotation_error_deg": summarize(rot_err),
    }

    return T_final, keep_idx, stats


# ============================================================
# Stats
# ============================================================

def finite_array(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals[np.isfinite(vals)]


def summarize(vals):
    vals = finite_array(vals)
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
            f"  {name}: "
            f"mean={s['mean']:.4f}, "
            f"median={s['median']:.4f}, "
            f"p95={s['p95']:.4f}, "
            f"min={s['min']:.4f}, "
            f"max={s['max']:.4f}"
        )


# ============================================================
# I/O
# ============================================================

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


# ============================================================
# AprilTag geometry
# ============================================================

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

    if mode == "as_is_0123":
        return c[[0, 1, 2, 3], :]
    if mode == "pupil_3210":
        return c[[3, 2, 1, 0], :]
    if mode == "shift1_1230":
        return c[[1, 2, 3, 0], :]
    if mode == "shift2_2301":
        return c[[2, 3, 0, 1], :]
    if mode == "shift3_3012":
        return c[[3, 0, 1, 2], :]
    if mode == "reverse_3210":
        return c[::-1, :]

    raise ValueError(f"Unsupported corner order: {mode}")


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


def reprojection_rmse(obj_pts, img_pts, T_C_B, K, dist):
    Rm = T_C_B[:3, :3]
    t = T_C_B[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(Rm)

    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else np.nan


def build_board_object_points(
    board_tag_ids,
    board_rows,
    board_cols,
    tag_size_m,
    gap_x_m,
    gap_y_m,
):
    """
    Board frame B:
      x right
      y up
      z out of board plane

    ID order:
      row-major, left-to-right, top-to-bottom

    board_rows = number of tag rows
    board_cols = number of tag cols
    """
    if len(board_tag_ids) != board_rows * board_cols:
        raise ValueError(
            "len(board_tag_ids) must equal board_rows * board_cols"
        )

    total_w = board_cols * tag_size_m + (board_cols - 1) * gap_x_m
    total_h = board_rows * tag_size_m + (board_rows - 1) * gap_y_m

    tag_local = square_object_points(tag_size_m)

    board_obj = {}

    for idx, tid in enumerate(board_tag_ids):
        r = idx // board_cols
        c = idx % board_cols

        cx = -total_w / 2.0 + tag_size_m / 2.0 + c * (tag_size_m + gap_x_m)
        cy =  total_h / 2.0 - tag_size_m / 2.0 - r * (tag_size_m + gap_y_m)

        pts = tag_local.copy()
        pts[:, 0] += cx
        pts[:, 1] += cy

        board_obj[int(tid)] = pts

    return board_obj


def solve_board_pose(
    dets_by_id,
    board_obj_pts,
    K,
    dist,
    corner_order,
    min_board_tags,
):
    obj_all = []
    img_all = []
    used_ids = []

    for tid, det in dets_by_id.items():
        if tid not in board_obj_pts:
            continue

        obj_all.append(board_obj_pts[tid])
        img_all.append(reorder_corners(det.corners, corner_order))
        used_ids.append(tid)

    used_unique = sorted(list(set(used_ids)))

    if len(used_unique) < min_board_tags:
        return None, used_unique, np.nan

    obj_pts = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img_pts = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))

    candidates = []

    # EPNP + refine
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

    if ok:
        try:
            if inliers is not None and len(inliers) >= 4:
                idx = inliers.reshape(-1)
                obj_ref = obj_pts[idx]
                img_ref = img_pts[idx]
            else:
                obj_ref = obj_pts
                img_ref = img_pts

            rvec, tvec = cv2.solvePnPRefineLM(
                obj_ref,
                img_ref,
                K,
                dist,
                rvec,
                tvec,
            )
        except cv2.error:
            pass

        Rm, _ = cv2.Rodrigues(rvec)
        T_C_B = rt_to_T(Rm, tvec.reshape(3))
        rmse = reprojection_rmse(obj_pts, img_pts, T_C_B, K, dist)
        if np.isfinite(rmse):
            candidates.append((rmse, "EPNP_RANSAC", T_C_B))

    # ITERATIVE fallback
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

    if ok:
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
        T_C_B = rt_to_T(Rm, tvec.reshape(3))
        rmse = reprojection_rmse(obj_pts, img_pts, T_C_B, K, dist)
        if np.isfinite(rmse):
            candidates.append((rmse, "ITERATIVE", T_C_B))

    if not candidates:
        return None, used_unique, np.nan

    candidates.sort(key=lambda x: x[0])
    return candidates[0][2], used_unique, candidates[0][0]


# ============================================================
# Head pose
# ============================================================

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
        img = reorder_corners(d.corners, "pupil_3210")

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


# ============================================================
# Time windows
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--back-camera-json", required=True)
    ap.add_argument("--back-frame-dir", required=True)
    ap.add_argument("--back-timestamps-csv", required=True)

    ap.add_argument("--scene-camera-json", required=True)
    ap.add_argument("--scene-frame-dir", required=True)
    ap.add_argument("--scene-timestamps-csv", required=True)

    ap.add_argument("--rig-calib-json", required=True)

    ap.add_argument("--board-tag-ids", required=True)
    ap.add_argument("--board-rows", type=int, required=True)
    ap.add_argument("--board-cols", type=int, required=True)
    ap.add_argument("--board-tag-size-m", type=float, required=True)
    ap.add_argument("--board-gap-x-m", type=float, required=True)
    ap.add_argument("--board-gap-y-m", type=float, required=True)

    ap.add_argument(
        "--board-corner-order",
        default="as_is_0123",
        choices=[
            "as_is_0123",
            "pupil_3210",
            "shift1_1230",
            "shift2_2301",
            "shift3_3012",
            "reverse_3210",
        ],
    )

    ap.add_argument("--duplicate-policy", default="reject", choices=["reject", "best"])
    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--min-board-tags-scene", type=int, default=3)
    ap.add_argument("--min-board-tags-back", type=int, default=3)
    ap.add_argument("--min-head-tags", type=int, default=2)

    ap.add_argument("--max-pair-dt-ms", type=float, default=30.0)

    ap.add_argument("--max-scene-board-rmse-px", type=float, default=5.0)
    ap.add_argument("--max-back-board-rmse-px", type=float, default=5.0)
    ap.add_argument("--max-head-rmse-px", type=float, default=3.0)

    # Stability filter for move-stop-move-stop data
    ap.add_argument("--max-scene-board-motion-m", type=float, default=0.02)
    ap.add_argument("--max-scene-board-rotation-deg", type=float, default=3.0)

    ap.add_argument("--max-back-board-motion-m", type=float, default=0.03)
    ap.add_argument("--max-back-board-rotation-deg", type=float, default=4.0)

    ap.add_argument("--max-head-motion-m", type=float, default=0.02)
    ap.add_argument("--max-head-rotation-deg", type=float, default=3.0)

    # Final robust filter
    ap.add_argument("--final-max-translation-error-m", type=float, default=0.05)
    ap.add_argument("--final-max-rotation-error-deg", type=float, default=8.0)
    ap.add_argument("--min-final-samples", type=int, default=50)

    ap.add_argument("--output-json", required=True)

    args = ap.parse_args()

    board_tag_ids = [int(x) for x in args.board_tag_ids.split(",")]
    board_tag_set = set(board_tag_ids)

    K_back, dist_back = load_camera_json(args.back_camera_json)
    K_scene, dist_scene = load_camera_json(args.scene_camera_json)
    T_H_T, head_tag_ids, head_tag_size_m = load_rig_calib(args.rig_calib_json)

    overlap = board_tag_set & head_tag_ids
    if overlap:
        raise RuntimeError(
            f"Board tag IDs overlap with helmet head tag IDs: {sorted(overlap)}"
        )

    board_obj_pts = build_board_object_points(
        board_tag_ids=board_tag_ids,
        board_rows=args.board_rows,
        board_cols=args.board_cols,
        tag_size_m=args.board_tag_size_m,
        gap_x_m=args.board_gap_x_m,
        gap_y_m=args.board_gap_y_m,
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

    T_H_C1_samples = []

    scene_board_rmse_all = []
    back_board_rmse_all = []
    head_rmse_all = []
    pair_dt_ms_all = []

    prev_scene_board_pose = None
    prev_back_board_pose = None
    prev_head_pose = None

    sample_meta = []

    counts = {
        "scene_frames_checked": 0,
        "rejected_by_pair_dt": 0,

        "scene_board_detected": 0,
        "scene_board_duplicate_rejected": 0,
        "scene_board_rejected_by_rmse": 0,
        "scene_board_ok": 0,

        "back_board_detected": 0,
        "back_board_duplicate_rejected": 0,
        "back_board_rejected_by_rmse": 0,
        "back_board_ok": 0,

        "head_detected": 0,
        "head_rejected_by_rmse": 0,
        "head_ok": 0,

        "rejected_by_scene_board_motion": 0,
        "rejected_by_back_board_motion": 0,
        "rejected_by_head_motion": 0,

        "used_samples": 0,
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
            counts["rejected_by_pair_dt"] += 1
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

        scene_board_dets, scene_dup = deduplicate_detections_by_id(
            dets_scene,
            allowed_ids=board_tag_set,
        )

        back_board_dets, back_dup = deduplicate_detections_by_id(
            dets_back,
            allowed_ids=board_tag_set,
        )

        if len(scene_board_dets) >= args.min_board_tags_scene:
            counts["scene_board_detected"] += 1

        if len(back_board_dets) >= args.min_board_tags_back:
            counts["back_board_detected"] += 1

        if args.duplicate_policy == "reject" and scene_dup:
            counts["scene_board_duplicate_rejected"] += 1
            continue

        if args.duplicate_policy == "reject" and back_dup:
            counts["back_board_duplicate_rejected"] += 1
            continue

        T_C1_B, scene_used_ids, scene_rmse = solve_board_pose(
            scene_board_dets,
            board_obj_pts,
            K_scene,
            dist_scene,
            args.board_corner_order,
            args.min_board_tags_scene,
        )

        if T_C1_B is None:
            continue

        scene_board_rmse_all.append(scene_rmse)

        if scene_rmse > args.max_scene_board_rmse_px:
            counts["scene_board_rejected_by_rmse"] += 1
            continue

        counts["scene_board_ok"] += 1

        T_C2_B, back_used_ids, back_rmse = solve_board_pose(
            back_board_dets,
            board_obj_pts,
            K_back,
            dist_back,
            args.board_corner_order,
            args.min_board_tags_back,
        )

        if T_C2_B is None:
            continue

        back_board_rmse_all.append(back_rmse)

        if back_rmse > args.max_back_board_rmse_px:
            counts["back_board_rejected_by_rmse"] += 1
            continue

        counts["back_board_ok"] += 1

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

        # Stability filters
        if prev_scene_board_pose is not None:
            s_dt, s_da = pose_delta(prev_scene_board_pose, T_C1_B)
            if (
                s_dt is not None
                and (
                    s_dt > args.max_scene_board_motion_m
                    or s_da > args.max_scene_board_rotation_deg
                )
            ):
                counts["rejected_by_scene_board_motion"] += 1
                prev_scene_board_pose = T_C1_B
                prev_back_board_pose = T_C2_B
                prev_head_pose = T_C2_H
                continue

        if prev_back_board_pose is not None:
            b_dt, b_da = pose_delta(prev_back_board_pose, T_C2_B)
            if (
                b_dt is not None
                and (
                    b_dt > args.max_back_board_motion_m
                    or b_da > args.max_back_board_rotation_deg
                )
            ):
                counts["rejected_by_back_board_motion"] += 1
                prev_scene_board_pose = T_C1_B
                prev_back_board_pose = T_C2_B
                prev_head_pose = T_C2_H
                continue

        if prev_head_pose is not None:
            h_dt, h_da = pose_delta(prev_head_pose, T_C2_H)
            if (
                h_dt is not None
                and (
                    h_dt > args.max_head_motion_m
                    or h_da > args.max_head_rotation_deg
                )
            ):
                counts["rejected_by_head_motion"] += 1
                prev_scene_board_pose = T_C1_B
                prev_back_board_pose = T_C2_B
                prev_head_pose = T_C2_H
                continue

        prev_scene_board_pose = T_C1_B
        prev_back_board_pose = T_C2_B
        prev_head_pose = T_C2_H

        # Core equation:
        # T_H_C1 = inv(T_C2_H) @ T_C2_B @ inv(T_C1_B)
        T_H_C1 = invert_T(T_C2_H) @ T_C2_B @ invert_T(T_C1_B)

        T_H_C1_samples.append(T_H_C1)
        pair_dt_ms_all.append(dt_ms)

        sample_meta.append({
            "scene_frame_idx": scene_frame_idx,
            "back_frame_idx": back_frame_idx,
            "time_ns": scene_time_ns,
            "pair_dt_ms": dt_ms,
            "scene_rmse": scene_rmse,
            "back_rmse": back_rmse,
            "head_rmse": head_rmse,
            "scene_board_tag_ids": scene_used_ids,
            "back_board_tag_ids": back_used_ids,
            "head_tag_ids": used_head_ids,
        })

        counts["used_samples"] += 1

    print("[INFO] Debug counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("[INFO] RMSE / timing debug:")
    print_summary("scene_board_rmse_px_all", scene_board_rmse_all)
    print_summary("back_board_rmse_px_all", back_board_rmse_all)
    print_summary("head_rmse_px_all", head_rmse_all)
    print_summary("pair_dt_ms_all", pair_dt_ms_all)

    if len(T_H_C1_samples) < args.min_final_samples:
        raise RuntimeError(
            f"Too few valid T_H_C1 samples: {len(T_H_C1_samples)} "
            f"< min_final_samples={args.min_final_samples}"
        )

    T_final, keep_idx, robust_stats = robust_filter_transforms(
        T_H_C1_samples,
        max_translation_error_m=args.final_max_translation_error_m,
        max_rotation_error_deg=args.final_max_rotation_error_deg,
    )

    if T_final is None or len(keep_idx) < args.min_final_samples:
        raise RuntimeError(
            "Too few samples survived final robust filtering."
        )

    final_samples = [T_H_C1_samples[i] for i in keep_idx]
    final_meta = [sample_meta[i] for i in keep_idx]

    final_trans_err, final_rot_err = transform_errors_to_mean(
        final_samples,
        T_final,
    )

    result = {
        "T_H_C1": T_final.tolist(),
        "num_samples": len(final_samples),

        "method": "multi_tag_common_board_stable",
        "board_tag_ids": board_tag_ids,
        "board_rows": args.board_rows,
        "board_cols": args.board_cols,
        "board_tag_size_m": args.board_tag_size_m,
        "board_gap_x_m": args.board_gap_x_m,
        "board_gap_y_m": args.board_gap_y_m,
        "board_corner_order": args.board_corner_order,
        "duplicate_policy": args.duplicate_policy,

        "min_board_tags_scene": args.min_board_tags_scene,
        "min_board_tags_back": args.min_board_tags_back,
        "min_head_tags": args.min_head_tags,

        "max_pair_dt_ms": args.max_pair_dt_ms,
        "max_scene_board_rmse_px": args.max_scene_board_rmse_px,
        "max_back_board_rmse_px": args.max_back_board_rmse_px,
        "max_head_rmse_px": args.max_head_rmse_px,

        "max_scene_board_motion_m": args.max_scene_board_motion_m,
        "max_scene_board_rotation_deg": args.max_scene_board_rotation_deg,
        "max_back_board_motion_m": args.max_back_board_motion_m,
        "max_back_board_rotation_deg": args.max_back_board_rotation_deg,
        "max_head_motion_m": args.max_head_motion_m,
        "max_head_rotation_deg": args.max_head_rotation_deg,

        "final_max_translation_error_m": args.final_max_translation_error_m,
        "final_max_rotation_error_deg": args.final_max_rotation_error_deg,

        "scene_window": scene_window,
        "back_window": back_window,

        "debug_counts": counts,
        "raw_debug": {
            "scene_board_rmse_px_all": summarize(scene_board_rmse_all),
            "back_board_rmse_px_all": summarize(back_board_rmse_all),
            "head_rmse_px_all": summarize(head_rmse_all),
            "pair_dt_ms_all": summarize(pair_dt_ms_all),
        },

        "robust_filter_stats": robust_stats,

        "final_stats": {
            "num_final_samples": len(final_samples),
            "translation_error_m": summarize(final_trans_err),
            "rotation_error_deg": summarize(final_rot_err),
            "pair_dt_ms": summarize([m["pair_dt_ms"] for m in final_meta]),
            "scene_rmse_px": summarize([m["scene_rmse"] for m in final_meta]),
            "back_rmse_px": summarize([m["back_rmse"] for m in final_meta]),
            "head_rmse_px": summarize([m["head_rmse"] for m in final_meta]),
        },

        "note": (
            "T_H_C1 is estimated from multi-tag common-board poses. "
            "For each valid paired frame: T_H_C1 = inv(T_C2_H) @ T_C2_B @ inv(T_C1_B). "
            "The board geometry must match the printed board."
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("[INFO] Final result:")
    print(f"  num_raw_samples = {len(T_H_C1_samples)}")
    print(f"  num_final_samples = {len(final_samples)}")
    print_summary("final_translation_error_m", final_trans_err)
    print_summary("final_rotation_error_deg", final_rot_err)
    print(f"[INFO] saved to {output_path}")

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()