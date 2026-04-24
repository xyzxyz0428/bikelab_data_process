#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

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
    mats = np.stack([T[:3, :3] for T in T_list], axis=0)
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)
    R_mean = R.from_matrix(mats).mean().as_matrix()
    t_mean = np.mean(trans, axis=0)
    return rt_to_T(R_mean, t_mean)


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


def nearest_row(rows, target_time_ns, time_col="unix_ns", unit="ns"):
    best = None
    best_dt = None
    for r in rows:
        t = to_int(r, time_col)
        if t is None:
            continue
        if unit == "us":
            t_ns = int(t) * 1000
        else:
            t_ns = int(t)
        dt = abs(t_ns - target_time_ns)
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = r
    return best, best_dt


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


# =========================
# AprilTag helpers
# =========================

def square_object_points(tag_size_m):
    s = float(tag_size_m)
    return np.array([
        [-s/2,  s/2, 0.0],  # lt
        [ s/2,  s/2, 0.0],  # rt
        [ s/2, -s/2, 0.0],  # rb
        [-s/2, -s/2, 0.0],  # lb
    ], dtype=np.float64)


def reorder_corners(corners):
    # pupil_apriltags order: lb rb rt lt -> lt rt rb lb
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


# =========================
# Head pose from head tags
# =========================

def load_rig_calib(path):
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)
    T_H_T = {int(k): np.array(v, dtype=np.float64) for k, v in rig["T_H_T"].items()}
    head_tag_ids = [int(x) for x in rig["head_tag_ids"]]
    head_tag_size_m = {int(k): float(v) for k, v in rig["head_tag_size_m"].items()}
    return T_H_T, set(head_tag_ids), head_tag_size_m


def collect_head_correspondences(dets, T_H_T, head_tag_size_m):
    obj_all = []
    img_all = []
    used_tag_ids = []

    for d in dets:
        tid = int(d.tag_id)
        if tid not in T_H_T:
            continue

        pts_tag = square_object_points(head_tag_size_m[tid])
        T_H_Tag = T_H_T[tid]
        pts_H = (T_H_Tag[:3, :3] @ pts_tag.T).T + T_H_Tag[:3, 3]
        img = reorder_corners(d.corners)

        obj_all.append(pts_H)
        img_all.append(img)
        used_tag_ids.append(tid)

    if len(obj_all) == 0:
        return None, None, []

    obj = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))
    return obj, img, used_tag_ids


def solve_head_bundle_pose(dets, T_H_T, head_tag_size_m, K, dist):
    obj_pts, img_pts, used_ids = collect_head_correspondences(dets, T_H_T, head_tag_size_m)
    if obj_pts is None:
        return None, []

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None, used_ids

    try:
        if inliers is not None and len(inliers) >= 4:
            idx = inliers.reshape(-1)
            obj_ref = obj_pts[idx]
            img_ref = img_pts[idx]
        else:
            obj_ref = obj_pts
            img_ref = img_pts
        rvec, tvec = cv2.solvePnPRefineLM(obj_ref, img_ref, K, dist, rvec, tvec)
    except cv2.error:
        pass

    Rm, _ = cv2.Rodrigues(rvec)
    T_C2_H = rt_to_T(Rm, tvec.reshape(3))
    return T_C2_H, used_ids


# =========================
# New: get time/frame window from scene_timestamps.csv tag_id==17
# =========================

def get_scene_window_from_tag(scene_rows, board_tag_id):
    filtered = []
    for r in scene_rows:
        tid = to_int(r, "tag_id")
        if tid == board_tag_id:
            fi = to_int(r, "frame_idx")
            ts = to_int(r, "unix_ns")
            if fi is not None and ts is not None:
                filtered.append((fi, ts))

    if len(filtered) == 0:
        raise RuntimeError(f"No rows with tag_id == {board_tag_id} found in scene_timestamps.csv")

    frame_idxs = [x[0] for x in filtered]
    timestamps = [x[1] for x in filtered]

    return {
        "scene_frame_min": min(frame_idxs),
        "scene_frame_max": max(frame_idxs),
        "scene_ts_min": min(timestamps),
        "scene_ts_max": max(timestamps),
        "num_rows": len(filtered),
    }


def get_back_window_from_time(back_rows, ts_min, ts_max):
    filtered = []
    for r in back_rows:
        fi = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")
        if fi is None or ts is None:
            continue
        if ts_min <= ts <= ts_max:
            filtered.append((fi, ts))

    if len(filtered) == 0:
        raise RuntimeError("No back-camera rows found inside scene timestamp window.")

    frame_idxs = [x[0] for x in filtered]
    timestamps = [x[1] for x in filtered]

    return {
        "back_frame_min": min(frame_idxs),
        "back_frame_max": max(frame_idxs),
        "back_ts_min": min(timestamps),
        "back_ts_max": max(timestamps),
        "num_rows": len(filtered),
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

    ap.add_argument("--head-rig-config", required=True)
    ap.add_argument("--rig-calib-json", required=True)

    ap.add_argument("--board-tag-id", type=int, default=17)
    ap.add_argument("--board-tag-size-m", type=float, default=0.1)
    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    K_back, dist_back = load_camera_json(args.back_camera_json)
    K_scene, dist_scene = load_camera_json(args.scene_camera_json)

    T_H_T, head_tag_ids, head_tag_size_m = load_rig_calib(args.rig_calib_json)

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

    # -------------------------------------------------
    # 1) find scene window from tag_id == board_tag_id
    # -------------------------------------------------
    scene_window = get_scene_window_from_tag(scene_rows, args.board_tag_id)
    print("[INFO] Scene window from tag rows:")
    print(scene_window)

    # -------------------------------------------------
    # 2) use timestamp range to find back camera window
    # -------------------------------------------------
    back_window = get_back_window_from_time(
        back_rows,
        scene_window["scene_ts_min"],
        scene_window["scene_ts_max"]
    )
    print("[INFO] Back-camera window from timestamp overlap:")
    print(back_window)

    scene_frame_dir = Path(args.scene_frame_dir)
    back_frame_dir = Path(args.back_frame_dir)

    # only use scene rows within scene frame range and matching tag_id == board_tag_id
    valid_scene_rows = []
    for r in scene_rows:
        fi = to_int(r, "frame_idx")
        ts = to_int(r, "unix_ns")
        tid = to_int(r, "tag_id")
        if fi is None or ts is None:
            continue
        if tid != args.board_tag_id:
            continue
        if scene_window["scene_frame_min"] <= fi <= scene_window["scene_frame_max"]:
            valid_scene_rows.append(r)

    T_H_C1_list = []
    used = 0

    for scene_row in valid_scene_rows:
        scene_frame_idx = to_int(scene_row, "frame_idx")
        scene_time_ns = to_int(scene_row, "unix_ns")
        if scene_frame_idx is None or scene_time_ns is None:
            continue

        # nearest back row but constrained to back window
        candidate_back_rows = []
        for br in back_rows:
            bfi = to_int(br, "frame_idx")
            bts = to_int(br, "unix_ns")
            if bfi is None or bts is None:
                continue
            if back_window["back_frame_min"] <= bfi <= back_window["back_frame_max"]:
                candidate_back_rows.append(br)

        back_row, dt_ns = nearest_row(candidate_back_rows, scene_time_ns, time_col="unix_ns", unit="ns")
        if back_row is None:
            continue

        back_frame_idx = to_int(back_row, "frame_idx")
        if back_frame_idx is None:
            continue

        scene_img_path = resolve_frame_path(scene_frame_dir, scene_frame_idx)
        back_img_path = resolve_frame_path(back_frame_dir, back_frame_idx)

        if scene_img_path is None:
            print(f"[WARN] scene image not found for frame_idx={scene_frame_idx}")
            continue
        if back_img_path is None:
            print(f"[WARN] back image not found for frame_idx={back_frame_idx}")
            continue

        img_scene = cv2.imread(str(scene_img_path), cv2.IMREAD_COLOR)
        img_back = cv2.imread(str(back_img_path), cv2.IMREAD_COLOR)
        if img_scene is None or img_back is None:
            continue

        # scene camera sees board tag 17
        gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
        dets_scene = detector.detect(gray_scene, estimate_tag_pose=False)
        det_scene_board = None
        for d in dets_scene:
            if int(d.tag_id) == args.board_tag_id:
                det_scene_board = d
                break
        if det_scene_board is None:
            continue

        T_C1_M = solve_single_tag_pose(det_scene_board, args.board_tag_size_m, K_scene, dist_scene)
        if T_C1_M is None:
            continue

        # back camera sees board tag 17 and head rig
        gray_back = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)
        dets_back = detector.detect(gray_back, estimate_tag_pose=False)

        det_back_board = None
        head_dets = []
        for d in dets_back:
            tid = int(d.tag_id)
            if tid == args.board_tag_id:
                det_back_board = d
            if tid in head_tag_ids:
                head_dets.append(d)

        if det_back_board is None or len(head_dets) < 2:
            continue

        T_C2_M = solve_single_tag_pose(det_back_board, args.board_tag_size_m, K_back, dist_back)
        if T_C2_M is None:
            continue

        T_C2_H, used_head_ids = solve_head_bundle_pose(
            head_dets, T_H_T, head_tag_size_m, K_back, dist_back
        )
        if T_C2_H is None:
            continue

        # T_H_C1 = inv(T_C2_H) * T_C2_M * inv(T_C1_M)
        T_H_C1 = invert_T(T_C2_H) @ T_C2_M @ invert_T(T_C1_M)
        T_H_C1_list.append(T_H_C1)
        used += 1

    if len(T_H_C1_list) == 0:
        raise RuntimeError(
            "No valid paired samples found.\n"
            "Need:\n"
            "  - scene rows with tag_id == 17\n"
            "  - scene frame can solve board tag 17\n"
            "  - back frame in same time window can solve board tag 17 and head rig"
        )

    T_H_C1_mean = mean_transform(T_H_C1_list)

    result = {
        "num_samples": len(T_H_C1_list),
        "board_tag_id": args.board_tag_id,
        "board_tag_size_m": args.board_tag_size_m,
        "scene_window": scene_window,
        "back_window": back_window,
        "T_H_C1": T_H_C1_mean.tolist()
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] used_samples = {used}")
    print(f"[INFO] saved to {args.output_json}")

    # avoid occasional exit-time segfault from native libs
    detector = None
    import gc, sys, os
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()