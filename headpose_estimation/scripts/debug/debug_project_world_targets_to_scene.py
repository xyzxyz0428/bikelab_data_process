#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import os
import sys
import gc

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pupil_apriltags import Detector


# =========================
# Basic helpers
# =========================

def to_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def to_float(v, default=np.nan):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_json(path):
    data = load_json(path)
    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def invert_T(T):
    T = np.asarray(T, dtype=np.float64)
    Rm = T[:3, :3]
    t = T[:3, 3]

    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(Rm, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def apply_T(T, p3):
    p = np.ones(4, dtype=np.float64)
    p[:3] = np.asarray(p3, dtype=np.float64).reshape(3)
    q = T @ p
    return q[:3]


def summarize(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def resolve_frame_path(frame_dir, frame_idx):
    frame_idx = int(frame_idx)
    for ext in [".png", ".jpg", ".jpeg"]:
        p = frame_dir / f"frame_{frame_idx:06d}{ext}"
        if p.exists():
            return p
    return None


def nearest_row(rows, target_ns, time_key):
    best = None
    best_dt = None

    for r in rows:
        t = to_int(r.get(time_key))
        if t is None:
            continue
        dt = abs(t - target_ns)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = r

    return best, best_dt


# =========================
# Transforms
# =========================

def extract_matrix_from_json(data, key):
    """
    Accept:
      data[key]
      data["transforms"][key]
    """
    if key in data:
        return np.array(data[key], dtype=np.float64)

    if "transforms" in data and key in data["transforms"]:
        return np.array(data["transforms"][key], dtype=np.float64)

    raise KeyError(f"Cannot find {key} in transforms json.")


def load_transforms(path):
    data = load_json(path)

    T_W_C2 = extract_matrix_from_json(data, "T_W_C2")
    T_H_C1 = extract_matrix_from_json(data, "T_H_C1")

    return T_W_C2, T_H_C1


def load_baseline_tags(path, exclude_low_confidence, max_tag_translation_std_m, max_tag_rotation_std_deg):
    data = load_json(path)

    tags = {}

    for sid, item in data["tags"].items():
        tid = int(sid)

        if exclude_low_confidence and item.get("low_confidence", False):
            continue

        if max_tag_translation_std_m is not None:
            if item.get("translation_std_m", 0.0) > max_tag_translation_std_m:
                continue

        if max_tag_rotation_std_deg is not None:
            if item.get("rotation_std_deg", 0.0) > max_tag_rotation_std_deg:
                continue

        if "center_W" not in item:
            continue

        tags[tid] = {
            "center_W": np.array(item["center_W"], dtype=np.float64),
            "item": item,
        }

    return tags


# =========================
# Headpose CSV parser
# =========================

def find_time_col(row_keys):
    candidates = [
        "unix_ns",
        "timestamp_ns",
        "time_ns",
        "frame_unix_ns",
        "bag_time_ns",
        "camera_unix_ns",
    ]

    for c in candidates:
        if c in row_keys:
            return c

    raise KeyError(f"Cannot find timestamp column. Available columns include: {list(row_keys)[:30]}")


def find_triplet(row_keys, candidates):
    for triplet in candidates:
        if all(k in row_keys for k in triplet):
            return triplet
    return None


def find_quat(row_keys, prefix):
    candidates = [
        [f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw"],
        [f"{prefix}_ori_x", f"{prefix}_ori_y", f"{prefix}_ori_z", f"{prefix}_ori_w"],
        [f"{prefix}_quat_x", f"{prefix}_quat_y", f"{prefix}_quat_z", f"{prefix}_quat_w"],
        [f"{prefix}_orientation_x", f"{prefix}_orientation_y", f"{prefix}_orientation_z", f"{prefix}_orientation_w"],
    ]

    for q in candidates:
        if all(k in row_keys for k in q):
            return q

    return None


def find_matrix_cols(row_keys, prefix):
    patterns = [
        [[f"{prefix}_R_{i}{j}" for j in range(3)] for i in range(3)],
        [[f"{prefix}_R{i}{j}" for j in range(3)] for i in range(3)],
        [[f"{prefix}_r{i}{j}" for j in range(3)] for i in range(3)],
    ]

    for mat_cols in patterns:
        flat = [x for row in mat_cols for x in row]
        if all(k in row_keys for k in flat):
            return mat_cols

    return None


def detect_headpose_schema(headpose_rows, pose_prefix):
    if len(headpose_rows) == 0:
        raise RuntimeError("Headpose CSV is empty.")

    row_keys = set(headpose_rows[0].keys())
    time_col = find_time_col(row_keys)

    trans_candidates = [
        [f"{pose_prefix}_tx", f"{pose_prefix}_ty", f"{pose_prefix}_tz"],
        [f"{pose_prefix}_t_x", f"{pose_prefix}_t_y", f"{pose_prefix}_t_z"],
        [f"{pose_prefix}_x", f"{pose_prefix}_y", f"{pose_prefix}_z"],
        [f"{pose_prefix}_pos_x", f"{pose_prefix}_pos_y", f"{pose_prefix}_pos_z"],
        [f"{pose_prefix}_position_x", f"{pose_prefix}_position_y", f"{pose_prefix}_position_z"],
    ]

    trans_cols = find_triplet(row_keys, trans_candidates)
    if trans_cols is None:
        raise KeyError(
            f"Cannot find translation columns for prefix '{pose_prefix}'. "
            f"Expected e.g. {pose_prefix}_tx/ty/tz or {pose_prefix}_x/y/z."
        )

    quat_cols = find_quat(row_keys, pose_prefix)
    matrix_cols = find_matrix_cols(row_keys, pose_prefix)

    euler_cols = None
    euler_candidates = [
        [f"{pose_prefix}_roll_deg", f"{pose_prefix}_pitch_deg", f"{pose_prefix}_yaw_deg"],
        [f"{pose_prefix}_roll", f"{pose_prefix}_pitch", f"{pose_prefix}_yaw"],
    ]

    euler_cols = find_triplet(row_keys, euler_candidates)

    if quat_cols is None and matrix_cols is None and euler_cols is None:
        raise KeyError(
            f"Cannot find rotation columns for prefix '{pose_prefix}'. "
            f"Expected quaternion, rotation matrix, or roll/pitch/yaw columns."
        )

    return {
        "time_col": time_col,
        "trans_cols": trans_cols,
        "quat_cols": quat_cols,
        "matrix_cols": matrix_cols,
        "euler_cols": euler_cols,
    }


def T_C2_H_from_headpose_row(row, schema, euler_order="xyz"):
    tx, ty, tz = [to_float(row[c]) for c in schema["trans_cols"]]
    if not all(np.isfinite(v) for v in [tx, ty, tz]):
        return None

    t = np.array([tx, ty, tz], dtype=np.float64)

    if schema["matrix_cols"] is not None:
        Rm = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                Rm[i, j] = to_float(row[schema["matrix_cols"][i][j]])

        if not np.all(np.isfinite(Rm)):
            return None

        return rt_to_T(Rm, t)

    if schema["quat_cols"] is not None:
        qx, qy, qz, qw = [to_float(row[c]) for c in schema["quat_cols"]]
        if not all(np.isfinite(v) for v in [qx, qy, qz, qw]):
            return None

        Rm = R.from_quat([qx, qy, qz, qw]).as_matrix()
        return rt_to_T(Rm, t)

    if schema["euler_cols"] is not None:
        roll, pitch, yaw = [to_float(row[c]) for c in schema["euler_cols"]]
        if not all(np.isfinite(v) for v in [roll, pitch, yaw]):
            return None

        # Most headpose exports use degree columns.
        Rm = R.from_euler(euler_order, [roll, pitch, yaw], degrees=True).as_matrix()
        return rt_to_T(Rm, t)

    return None


# =========================
# AprilTag detection in scene image
# =========================

def reorder_corners_for_polygon(corners):
    # For polygon center, order does not affect mean.
    # Use as-is to avoid hidden convention changes.
    return np.asarray(corners, dtype=np.float64).reshape(4, 2)


def detect_tag_center_in_scene(img, tag_id, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector.detect(gray, estimate_tag_pose=False)

    target = None

    for d in dets:
        if int(d.tag_id) == int(tag_id):
            target = d
            break

    if target is None:
        return None, None

    poly = reorder_corners_for_polygon(target.corners)
    center = np.mean(poly, axis=0)

    return center, poly


# =========================
# Projection
# =========================

def project_point_C1_to_pixel(p_C1, K, dist):
    p_C1 = np.asarray(p_C1, dtype=np.float64).reshape(3)

    if p_C1[2] <= 1e-9:
        return None

    obj = p_C1.reshape(1, 1, 3)

    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)

    img_pts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    uv = img_pts.reshape(2)

    return uv


def draw_cross(img, xy, color, label):
    xy = np.asarray(xy, dtype=np.float64).reshape(-1)

    if xy.size < 2:
        return False

    x_f = float(xy[0])
    y_f = float(xy[1])

    if not np.isfinite(x_f) or not np.isfinite(y_f):
        return False

    # Avoid OpenCV crash for extremely large projected points
    if abs(x_f) > 1e6 or abs(y_f) > 1e6:
        return False

    x = int(round(x_f))
    y = int(round(y_f))

    cv2.drawMarker(
        img,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    cv2.putText(
        img,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    return True


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tag-windows-csv", required=True)
    ap.add_argument("--scene-timestamps-csv", required=True)
    ap.add_argument("--scene-frame-dir", required=True)

    ap.add_argument("--apriltag-baseline-json", required=True)
    ap.add_argument("--headpose-csv", required=True)
    ap.add_argument("--scene-camera-json", required=True)
    ap.add_argument("--transforms-json", required=True)

    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--pose-prefix", default="cam_head",
                    help="Prefix of T_C2_H pose columns in headpose CSV, usually cam_head.")

    ap.add_argument("--euler-order", default="xyz",
                    help="Euler order if headpose CSV only contains roll/pitch/yaw columns.")

    ap.add_argument("--window-fraction-start", type=float, default=0.3)
    ap.add_argument("--window-fraction-end", type=float, default=0.7)
    ap.add_argument("--max-headpose-dt-ms", type=float, default=20.0)

    ap.add_argument("--exclude-low-confidence-tags", action="store_true")
    ap.add_argument("--max-tag-translation-std-m", type=float, default=None)
    ap.add_argument("--max-tag-rotation-std-deg", type=float, default=None)

    ap.add_argument("--visualize", action="store_true")
    ap.add_argument("--max-visualizations-per-tag", type=int, default=3)
    ap.add_argument("--visualization-dir", default=None)

    ap.add_argument("--output-csv", required=True)

    args = ap.parse_args()

    windows = read_csv_dicts(args.tag_windows_csv)
    scene_rows = read_csv_dicts(args.scene_timestamps_csv)
    headpose_rows = read_csv_dicts(args.headpose_csv)

    K_scene, dist_scene = load_camera_json(args.scene_camera_json)
    T_W_C2, T_H_C1 = load_transforms(args.transforms_json)

    T_C2_W = invert_T(T_W_C2)
    T_C1_H = invert_T(T_H_C1)

    baseline_tags = load_baseline_tags(
        args.apriltag_baseline_json,
        exclude_low_confidence=args.exclude_low_confidence_tags,
        max_tag_translation_std_m=args.max_tag_translation_std_m,
        max_tag_rotation_std_deg=args.max_tag_rotation_std_deg,
    )

    schema = detect_headpose_schema(headpose_rows, args.pose_prefix)

    print("[INFO] Headpose schema:")
    print(schema)

    scene_by_frame = {}
    for r in scene_rows:
        fi = to_int(r.get("frame_idx"))
        if fi is not None:
            scene_by_frame[fi] = r

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frame_dir = Path(args.scene_frame_dir)

    vis_dir = None
    vis_counts = defaultdict(int)
    if args.visualize:
        if args.visualization_dir is None:
            vis_dir = Path(args.output_csv).with_suffix("").with_name(Path(args.output_csv).stem + "_vis")
        else:
            vis_dir = Path(args.visualization_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []

    for w in windows:
        window_id = to_int(w.get("window_id"))
        tag_id = to_int(w.get("tag_id"))
        start_f = to_int(w.get("start_frame_idx"))
        end_f = to_int(w.get("end_frame_idx"))

        if None in [window_id, tag_id, start_f, end_f]:
            continue

        if tag_id not in baseline_tags:
            continue

        target_W = baseline_tags[tag_id]["center_W"]

        n_frames = end_f - start_f + 1
        sub_start = start_f + int(np.floor(n_frames * args.window_fraction_start))
        sub_end = start_f + int(np.ceil(n_frames * args.window_fraction_end)) - 1
        sub_start = max(sub_start, start_f)
        sub_end = min(sub_end, end_f)

        if sub_end < sub_start:
            continue

        for frame_idx in range(sub_start, sub_end + 1):
            if frame_idx not in scene_by_frame:
                continue

            sr = scene_by_frame[frame_idx]
            scene_ns = to_int(sr.get("unix_ns"))
            if scene_ns is None:
                continue

            head_row, dt_ns = nearest_row(headpose_rows, scene_ns, schema["time_col"])
            if head_row is None:
                continue

            head_dt_ms = dt_ns / 1e6
            if head_dt_ms > args.max_headpose_dt_ms:
                continue

            T_C2_H = T_C2_H_from_headpose_row(
                head_row,
                schema,
                euler_order=args.euler_order,
            )

            if T_C2_H is None:
                continue

            # target_W -> C2 -> H -> C1
            p_C2 = apply_T(T_C2_W, target_W)
            p_H = apply_T(invert_T(T_C2_H), p_C2)
            p_C1 = apply_T(T_C1_H, p_H)

            uv_proj = project_point_C1_to_pixel(p_C1, K_scene, dist_scene)

            if uv_proj is None:
                out_rows.append({
                    "window_id": window_id,
                    "tag_id": tag_id,
                    "frame_idx": frame_idx,
                    "status": "projected_behind_camera",
                    "p_C1_z": float(p_C1[2]),
                    "headpose_dt_ms": head_dt_ms,
                })
                continue

            img_path = resolve_frame_path(frame_dir, frame_idx)
            if img_path is None:
                out_rows.append({
                    "window_id": window_id,
                    "tag_id": tag_id,
                    "frame_idx": frame_idx,
                    "status": "scene_image_missing",
                    "projected_u": float(uv_proj[0]),
                    "projected_v": float(uv_proj[1]),
                    "p_C1_z": float(p_C1[2]),
                    "headpose_dt_ms": head_dt_ms,
                })
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            tag_center_px, tag_poly = detect_tag_center_in_scene(img, tag_id, detector)

            if tag_center_px is None:
                out_rows.append({
                    "window_id": window_id,
                    "tag_id": tag_id,
                    "frame_idx": frame_idx,
                    "status": "tag_not_detected_in_scene",
                    "projected_u": float(uv_proj[0]),
                    "projected_v": float(uv_proj[1]),
                    "p_C1_z": float(p_C1[2]),
                    "headpose_dt_ms": head_dt_ms,
                })
                continue

            err_px = float(np.linalg.norm(uv_proj - tag_center_px))

            out = {
                "window_id": window_id,
                "tag_id": tag_id,
                "frame_idx": frame_idx,
                "scene_unix_ns": scene_ns,
                "status": "ok",
                "headpose_dt_ms": float(head_dt_ms),

                "target_W_x": float(target_W[0]),
                "target_W_y": float(target_W[1]),
                "target_W_z": float(target_W[2]),

                "p_C1_x": float(p_C1[0]),
                "p_C1_y": float(p_C1[1]),
                "p_C1_z": float(p_C1[2]),

                "projected_u": float(uv_proj[0]),
                "projected_v": float(uv_proj[1]),

                "detected_tag_center_u": float(tag_center_px[0]),
                "detected_tag_center_v": float(tag_center_px[1]),

                "reprojection_error_px": err_px,
            }

            out_rows.append(out)

            if args.visualize and vis_counts[tag_id] < args.max_visualizations_per_tag:
                vis = img.copy()

                if tag_poly is not None:
                    cv2.polylines(
                        vis,
                        [tag_poly.astype(np.int32)],
                        isClosed=True,
                        color=(255, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

                ok1 = draw_cross(vis, tag_center_px, (0, 255, 255), "detected tag center")
                ok2 = draw_cross(vis, uv_proj, (0, 0, 255), "projected target_W")

                if not (ok1 and ok2):
                    continue

                tc = np.asarray(tag_center_px, dtype=np.float64).reshape(-1)
                up = np.asarray(uv_proj, dtype=np.float64).reshape(-1)

                if (
                    tc.size >= 2 and up.size >= 2
                    and np.all(np.isfinite(tc[:2]))
                    and np.all(np.isfinite(up[:2]))
                    and abs(float(up[0])) < 1e6
                    and abs(float(up[1])) < 1e6
                ):
                    cv2.line(
                        vis,
                        (int(round(float(tc[0]))), int(round(float(tc[1])))),
                        (int(round(float(up[0]))), int(round(float(up[1])))),
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                header = (
                    f"tag={tag_id} frame={frame_idx} "
                    f"err={err_px:.1f}px head_dt={head_dt_ms:.1f}ms"
                )

                cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), (0, 0, 0), -1)
                cv2.putText(
                    vis,
                    header,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                out_img = vis_dir / f"tag{tag_id}_frame{frame_idx:06d}_err{err_px:.1f}px.png"
                cv2.imwrite(str(out_img), vis)

                vis_counts[tag_id] += 1

    if len(out_rows) == 0:
        print("[WARN] no output rows.")
        return

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    seen = set()
    for r in out_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    ok_rows = [r for r in out_rows if r.get("status") == "ok"]

    summary_rows = []

    grouped = defaultdict(list)
    for r in ok_rows:
        grouped[int(r["tag_id"])].append(r)

    for tag_id, rs in sorted(grouped.items()):
        errs = [float(r["reprojection_error_px"]) for r in rs]
        dts = [float(r["headpose_dt_ms"]) for r in rs]
        s_err = summarize(errs)
        s_dt = summarize(dts)

        summary_rows.append({
            "tag_id": tag_id,
            "n": len(rs),
            "reprojection_error_px_mean": s_err["mean"],
            "reprojection_error_px_median": s_err["median"],
            "reprojection_error_px_p95": s_err["p95"],
            "reprojection_error_px_min": s_err["min"],
            "reprojection_error_px_max": s_err["max"],
            "headpose_dt_ms_mean": s_dt["mean"],
            "headpose_dt_ms_median": s_dt["median"],
            "headpose_dt_ms_p95": s_dt["p95"],
        })

    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")

    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    all_errs = [float(r["reprojection_error_px"]) for r in ok_rows]
    all_dts = [float(r["headpose_dt_ms"]) for r in ok_rows]

    s_all = summarize(all_errs)
    s_dt = summarize(all_dts)

    print(f"[INFO] saved to {output_csv}")
    print(f"[INFO] saved summary to {summary_csv}")
    print(f"[INFO] ok rows = {len(ok_rows)}")
    print(
        f"[INFO] reprojection_error_px: "
        f"mean={s_all['mean']:.2f}, "
        f"median={s_all['median']:.2f}, "
        f"p95={s_all['p95']:.2f}, "
        f"max={s_all['max']:.2f}"
    )
    print(
        f"[INFO] headpose_dt_ms: "
        f"mean={s_dt['mean']:.2f}, "
        f"median={s_dt['median']:.2f}, "
        f"p95={s_dt['p95']:.2f}"
    )

    if args.visualize:
        print(f"[INFO] saved visualizations to {vis_dir}")

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()