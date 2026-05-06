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


# =========================
# Basic transforms
# =========================

def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
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


def transform_errors(T_list, T_mean):
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


def compute_transform_stats(T_list, T_mean):
    if len(T_list) == 0:
        return {
            "translation_std_m": None,
            "rotation_std_deg": None,
            "translation_error_m": {},
            "rotation_error_deg": {},
        }

    trans_err, rot_err = transform_errors(T_list, T_mean)

    return {
        "translation_std_m": float(np.std(trans_err, ddof=1)) if len(trans_err) > 1 else 0.0,
        "rotation_std_deg": float(np.std(rot_err, ddof=1)) if len(rot_err) > 1 else 0.0,
        "translation_error_m": summarize(trans_err),
        "rotation_error_deg": summarize(rot_err),
    }


def robust_filter_transforms(
    T_list,
    max_translation_error_m,
    max_rotation_error_deg,
    max_iterations,
):
    keep_idx = list(range(len(T_list)))
    history = []

    if len(T_list) == 0:
        return keep_idx, history

    for it in range(max_iterations):
        cur = [T_list[i] for i in keep_idx]
        if len(cur) == 0:
            break

        T_mean = mean_transform(cur)
        trans_err, rot_err = transform_errors(cur, T_mean)

        new_keep_local = []
        removed = 0

        for i, (te, re) in enumerate(zip(trans_err, rot_err)):
            if te <= max_translation_error_m and re <= max_rotation_error_deg:
                new_keep_local.append(i)
            else:
                removed += 1

        new_keep_idx = [keep_idx[i] for i in new_keep_local]

        history.append({
            "iteration": it + 1,
            "num_input": len(keep_idx),
            "num_kept": len(new_keep_idx),
            "num_removed": removed,
            "translation_error_m_mean": safe_mean(trans_err),
            "translation_error_m_median": safe_median(trans_err),
            "translation_error_m_p95": safe_p95(trans_err),
            "translation_error_m_max": safe_max(trans_err),
            "rotation_error_deg_mean": safe_mean(rot_err),
            "rotation_error_deg_median": safe_median(rot_err),
            "rotation_error_deg_p95": safe_p95(rot_err),
            "rotation_error_deg_max": safe_max(rot_err),
        })

        if len(new_keep_idx) == len(keep_idx):
            break

        keep_idx = new_keep_idx

        if len(keep_idx) == 0:
            break

    return keep_idx, history


# =========================
# Stats
# =========================

def finite_array(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals[np.isfinite(vals)]


def safe_mean(vals):
    vals = finite_array(vals)
    return float(np.mean(vals)) if len(vals) else None


def safe_median(vals):
    vals = finite_array(vals)
    return float(np.median(vals)) if len(vals) else None


def safe_p95(vals):
    vals = finite_array(vals)
    return float(np.quantile(vals, 0.95)) if len(vals) else None


def safe_min(vals):
    vals = finite_array(vals)
    return float(np.min(vals)) if len(vals) else None


def safe_max(vals):
    vals = finite_array(vals)
    return float(np.max(vals)) if len(vals) else None


def summarize(vals):
    return {
        "mean": safe_mean(vals),
        "median": safe_median(vals),
        "p95": safe_p95(vals),
        "min": safe_min(vals),
        "max": safe_max(vals),
    }


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


def resolve_frame_path(frame_dir, frame_idx):
    frame_idx = int(frame_idx)

    for ext in [".png", ".jpg", ".jpeg"]:
        p = frame_dir / f"frame_{frame_idx:06d}{ext}"
        if p.exists():
            return p

    return None


# =========================
# AprilTag pose
# =========================

def square_object_points(tag_size_m):
    s = float(tag_size_m)

    return np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)


def reorder_corners(corners, mode="as_is_0123"):
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


def reprojection_rmse(obj_pts, img_pts, T_C_Tag, K, dist):
    Rm = T_C_Tag[:3, :3]
    t = T_C_Tag[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(Rm)
    proj, _ = cv2.projectPoints(obj_pts, rvec, t, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan


def solve_single_tag_pose(det, tag_size_m, K, dist, corner_order):
    obj_pts = square_object_points(tag_size_m)
    img_pts = reorder_corners(det.corners, corner_order)

    obj_pts = np.ascontiguousarray(obj_pts.astype(np.float64))
    img_pts = np.ascontiguousarray(img_pts.astype(np.float64))

    candidates = []

    for method_name, flag in [
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
            candidates.append((rmse, method_name, T_C_Tag))

    if not candidates:
        return None, None, np.nan

    candidates.sort(key=lambda x: x[0])
    rmse, method_name, T_C_Tag = candidates[0]

    return T_C_Tag, method_name, rmse


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)

    ap.add_argument("--ref-tag-id", type=int, required=True)
    ap.add_argument("--tag-size-m", type=float, required=True)

    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument(
        "--corner-order",
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

    ap.add_argument("--max-tag-rmse-px", type=float, default=5.0)

    # Final confidence thresholds
    ap.add_argument("--max-translation-std-m", type=float, default=0.01)
    ap.add_argument("--max-rotation-std-deg", type=float, default=2.0)

    # Robust filtering
    ap.add_argument("--enable-robust-filter", action="store_true")
    ap.add_argument("--max-translation-error-m", type=float, default=0.02)
    ap.add_argument("--max-rotation-error-deg", type=float, default=2.0)
    ap.add_argument("--min-samples-after-filter", type=int, default=50)
    ap.add_argument("--filter-iterations", type=int, default=5)

    ap.add_argument("--output-json", required=True)

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

    T_W_C2_samples = []
    sample_info = []

    debug_counts = {
        "timestamp_rows": len(rows),
        "frames_checked": 0,
        "image_not_found": 0,
        "ref_tag_detected": 0,
        "pose_ok": 0,
        "rejected_by_rmse": 0,
    }

    raw_rmse_list = []

    for r in rows:
        frame_idx = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")

        if frame_idx is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)

        if img_path is None:
            debug_counts["image_not_found"] += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            debug_counts["image_not_found"] += 1
            continue

        debug_counts["frames_checked"] += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=False)

        ref_dets = [d for d in dets if int(d.tag_id) == args.ref_tag_id]

        if not ref_dets:
            continue

        debug_counts["ref_tag_detected"] += 1

        # If duplicated, use the largest area / strongest detection
        def det_score(d):
            corners = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
            area = abs(cv2.contourArea(corners.astype(np.float32)))
            margin = float(getattr(d, "decision_margin", 0.0))
            return (margin, area)

        det = sorted(ref_dets, key=det_score, reverse=True)[0]

        T_C2_Tag, method_name, rmse = solve_single_tag_pose(
            det,
            args.tag_size_m,
            K,
            dist,
            args.corner_order,
        )

        if T_C2_Tag is None:
            continue

        raw_rmse_list.append(rmse)

        if rmse > args.max_tag_rmse_px:
            debug_counts["rejected_by_rmse"] += 1
            continue

        debug_counts["pose_ok"] += 1

        # Define world frame W as the reference tag frame:
        # T_W_Tag = I
        # T_C2_Tag known
        # therefore T_W_C2 = inv(T_C2_Tag)
        T_W_C2 = invert_T(T_C2_Tag)

        T_W_C2_samples.append(T_W_C2)
        sample_info.append({
            "frame_idx": frame_idx,
            "timestamp_ns": ts,
            "tag_rmse_px": float(rmse),
            "pnp_method": method_name,
        })

    if len(T_W_C2_samples) == 0:
        raise RuntimeError("No valid T_W_C2 samples found.")

    raw_T_mean = mean_transform(T_W_C2_samples)
    raw_stats = compute_transform_stats(T_W_C2_samples, raw_T_mean)

    kept_idx = list(range(len(T_W_C2_samples)))
    filter_history = []

    if args.enable_robust_filter:
        kept_idx, filter_history = robust_filter_transforms(
            T_W_C2_samples,
            max_translation_error_m=args.max_translation_error_m,
            max_rotation_error_deg=args.max_rotation_error_deg,
            max_iterations=args.filter_iterations,
        )

        if len(kept_idx) < args.min_samples_after_filter:
            raise RuntimeError(
                f"Too few samples after robust filtering: "
                f"{len(kept_idx)} < {args.min_samples_after_filter}"
            )

    filtered_samples = [T_W_C2_samples[i] for i in kept_idx]
    filtered_info = [sample_info[i] for i in kept_idx]

    T_final = mean_transform(filtered_samples)
    filtered_stats = compute_transform_stats(filtered_samples, T_final)

    translation_std_m = filtered_stats["translation_std_m"]
    rotation_std_deg = filtered_stats["rotation_std_deg"]

    low_confidence = (
        translation_std_m is None
        or rotation_std_deg is None
        or translation_std_m > args.max_translation_std_m
        or rotation_std_deg > args.max_rotation_std_deg
    )

    result = {
        "ref_tag_id": args.ref_tag_id,
        "tag_size_m": args.tag_size_m,
        "tag_family": args.tag_family,
        "corner_order": args.corner_order,

        "num_samples_raw": len(T_W_C2_samples),
        "num_samples": len(filtered_samples),
        "num_samples_after_filter": len(filtered_samples),

        "T_W_C2": T_final.tolist(),

        "translation_std_m": translation_std_m,
        "rotation_std_deg": rotation_std_deg,
        "low_confidence": bool(low_confidence),

        "confidence_thresholds": {
            "max_translation_std_m": args.max_translation_std_m,
            "max_rotation_std_deg": args.max_rotation_std_deg,
        },

        "raw_stats": raw_stats,
        "filtered_stats": filtered_stats,

        "tag_rmse_px_all": summarize(raw_rmse_list),
        "debug_counts": debug_counts,

        "robust_filter": {
            "enabled": bool(args.enable_robust_filter),
            "max_translation_error_m": args.max_translation_error_m,
            "max_rotation_error_deg": args.max_rotation_error_deg,
            "min_samples_after_filter": args.min_samples_after_filter,
            "filter_iterations": args.filter_iterations,
            "history": filter_history,
        },

        "used_samples": filtered_info,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved to {output_path}")
    print(f"num_samples_raw = {len(T_W_C2_samples)}")
    print(f"num_samples = {len(filtered_samples)}")
    print(f"translation_std_m = {translation_std_m:.4f}")
    print(f"rotation_std_deg = {rotation_std_deg:.3f}")
    print(f"low_confidence = {bool(low_confidence)}")

    print("[INFO] raw stats:")
    print(f"  translation_std_m = {raw_stats['translation_std_m']}")
    print(f"  rotation_std_deg = {raw_stats['rotation_std_deg']}")

    print("[INFO] tag rmse:")
    print(
        f"  mean={result['tag_rmse_px_all']['mean']}, "
        f"median={result['tag_rmse_px_all']['median']}, "
        f"p95={result['tag_rmse_px_all']['p95']}"
    )

    if args.enable_robust_filter:
        print("[INFO] robust filter history:")
        for h in filter_history:
            print(
                f"  iter {h['iteration']}: "
                f"n={h['num_input']} -> {h['num_kept']}, "
                f"trans_p95={h['translation_error_m_p95']}, "
                f"rot_p95={h['rotation_error_deg_p95']}"
            )

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()