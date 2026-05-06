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
    mats = np.stack([T[:3, :3] for T in T_list], axis=0)
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)

    R_mean = R.from_matrix(mats).mean().as_matrix()
    t_mean = np.mean(trans, axis=0)

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

    return np.asarray(trans_err, dtype=np.float64), np.asarray(rot_err, dtype=np.float64)


def compute_transform_stats(T_list, T_mean):
    trans_err, rot_err = transform_errors(T_list, T_mean)

    translation_std_m = float(np.std(trans_err, ddof=1)) if len(trans_err) > 1 else 0.0
    rotation_std_deg = float(np.std(rot_err, ddof=1)) if len(rot_err) > 1 else 0.0

    return translation_std_m, rotation_std_deg


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


# =========================
# AprilTag geometry
# =========================

def square_object_points(tag_size_m):
    s = float(tag_size_m)

    # Object point order:
    # lt, rt, rb, lb
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
    return float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else np.nan


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


def detection_score(det):
    corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
    area = abs(cv2.contourArea(corners.astype(np.float32)))
    margin = float(getattr(det, "decision_margin", 0.0))
    return margin, area


def build_detection_map(dets, wanted_ids, duplicate_policy="best"):
    det_map = {}
    duplicate_counts = {}

    for d in dets:
        tid = int(d.tag_id)
        if tid not in wanted_ids:
            continue

        duplicate_counts[tid] = duplicate_counts.get(tid, 0) + 1

        if tid not in det_map:
            det_map[tid] = d
        else:
            if duplicate_policy == "best":
                if detection_score(d) > detection_score(det_map[tid]):
                    det_map[tid] = d
            elif duplicate_policy == "reject":
                det_map[tid] = None
            else:
                raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")

    det_map = {tid: d for tid, d in det_map.items() if d is not None}
    return det_map, duplicate_counts


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)

    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--default-size-m", type=float, required=True)
    ap.add_argument("--ref-tag-id", type=int, required=True)
    ap.add_argument("--target-tag-ids", required=True)

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

    ap.add_argument(
        "--duplicate-policy",
        default="best",
        choices=["best", "reject"],
    )

    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--max-translation-std-m", type=float, default=0.01)
    ap.add_argument("--max-rotation-std-deg", type=float, default=2.0)
    ap.add_argument("--skip-low-confidence-tags", action="store_true")

    ap.add_argument("--output-json", required=True)

    args = ap.parse_args()

    K, dist = load_camera_json(args.camera_json)
    rows = read_csv_dicts(args.timestamps_csv)
    frame_dir = Path(args.frame_dir)

    target_ids = [int(x) for x in args.target_tag_ids.split(",") if x.strip()]
    all_ids = set(target_ids + [args.ref_tag_id])

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    samples = {tid: [] for tid in target_ids}
    sample_info = {tid: [] for tid in target_ids}
    rmse_info = {tid: [] for tid in target_ids}

    used_frames = 0

    debug_counts = {
        "timestamp_rows": len(rows),
        "frames_checked": 0,
        "image_missing": 0,
        "image_read_failed": 0,
        "frames_with_ref_tag": 0,
        "frames_used": 0,
        "ref_pose_failed": 0,
        "target_pose_failed": 0,
        "duplicate_tag_frames": 0,
    }

    for r in rows:
        frame_idx = to_int(r.get("frame_idx"))
        ts = to_int(r.get("unix_ns"))

        if frame_idx is None:
            continue

        img_path = resolve_frame_path(frame_dir, frame_idx)
        if img_path is None:
            debug_counts["image_missing"] += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            debug_counts["image_read_failed"] += 1
            continue

        debug_counts["frames_checked"] += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=False)

        det_map, duplicate_counts = build_detection_map(
            dets,
            wanted_ids=all_ids,
            duplicate_policy=args.duplicate_policy,
        )

        if any(v > 1 for v in duplicate_counts.values()):
            debug_counts["duplicate_tag_frames"] += 1

        if args.ref_tag_id not in det_map:
            continue

        debug_counts["frames_with_ref_tag"] += 1

        T_C_ref, ref_method, ref_rmse = solve_single_tag_pose(
            det_map[args.ref_tag_id],
            args.default_size_m,
            K,
            dist,
            args.corner_order,
        )

        if T_C_ref is None:
            debug_counts["ref_pose_failed"] += 1
            continue

        T_W_C = invert_T(T_C_ref)

        used_this_frame = False

        for tid in target_ids:
            if tid not in det_map:
                continue

            T_C_tag, method_name, tag_rmse = solve_single_tag_pose(
                det_map[tid],
                args.default_size_m,
                K,
                dist,
                args.corner_order,
            )

            if T_C_tag is None:
                debug_counts["target_pose_failed"] += 1
                continue

            # W is defined as the reference tag frame.
            # T_W_Tag = T_W_C @ T_C_Tag
            T_W_tag = T_W_C @ T_C_tag

            samples[tid].append(T_W_tag)
            rmse_info[tid].append(tag_rmse)
            sample_info[tid].append({
                "frame_idx": frame_idx,
                "timestamp_ns": ts,
                "ref_rmse_px": float(ref_rmse),
                "tag_rmse_px": float(tag_rmse),
                "ref_method": ref_method,
                "tag_method": method_name,
            })

            used_this_frame = True

        if used_this_frame:
            used_frames += 1
            debug_counts["frames_used"] += 1

    result = {
        "ref_tag_id": args.ref_tag_id,
        "default_size_m": args.default_size_m,
        "tag_family": args.tag_family,
        "corner_order": args.corner_order,
        "duplicate_policy": args.duplicate_policy,
        "used_frames": used_frames,
        "confidence_thresholds": {
            "min_samples": args.min_samples,
            "max_translation_std_m": args.max_translation_std_m,
            "max_rotation_std_deg": args.max_rotation_std_deg,
        },
        "debug_counts": debug_counts,
        "tags": {},
    }

    for tid, Ts in samples.items():
        if len(Ts) == 0:
            print(f"[WARN] no samples for tag {tid}")
            continue

        T_mean = mean_transform(Ts)
        translation_std_m, rotation_std_deg = compute_transform_stats(Ts, T_mean)

        trans_err, rot_err = transform_errors(Ts, T_mean)
        rmse_stats = summarize(rmse_info[tid])

        low_confidence = (
            len(Ts) < args.min_samples
            or translation_std_m > args.max_translation_std_m
            or rotation_std_deg > args.max_rotation_std_deg
        )

        print(
            f"tag {tid}: n={len(Ts)}, "
            f"translation_std_m={translation_std_m:.4f}, "
            f"rotation_std_deg={rotation_std_deg:.3f}, "
            f"tag_rmse_median={rmse_stats['median']}, "
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
            "translation_error_m": summarize(trans_err),
            "rotation_error_deg": summarize(rot_err),
            "tag_rmse_px": rmse_stats,
            "samples": sample_info[tid],
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"used_frames={used_frames}")
    print(f"saved to {output_path}")

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()