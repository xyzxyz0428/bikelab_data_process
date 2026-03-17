import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pose_utils import (
    load_camera_json, make_detector, detect_tags, reorder_pupil_corners,
    square_object_points, solve_single_tag_pose_ippe, invert_T, rt_to_T,
    T_to_rt, rot_to_rpy_deg, pose_rmse
)


def transform_points(T_ab, pts_b):
    return (T_ab[:3, :3] @ pts_b.T).T + T_ab[:3, 3]


def load_rig_calib(path):
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)
    T_H_T = {int(k): np.array(v, dtype=np.float64) for k, v in rig["T_H_T"].items()}
    head_tag_ids = [int(x) for x in rig["head_tag_ids"]]
    head_tag_size_m = {int(k): float(v) for k, v in rig["head_tag_size_m"].items()}
    return T_H_T, head_tag_ids, head_tag_size_m


def collect_head_correspondences(dets, T_H_T, head_tag_size_m):
    obj_all = []
    img_all = []
    used_tag_ids = []

    for d in dets:
        tid = int(d.tag_id)
        if tid not in T_H_T:
            continue

        pts_tag = square_object_points(head_tag_size_m[tid])
        pts_H = transform_points(T_H_T[tid], pts_tag)
        img = reorder_pupil_corners(d.corners)

        obj_all.append(pts_H)
        img_all.append(img)
        used_tag_ids.append(tid)

    if len(obj_all) == 0:
        return None, None, []

    obj = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))
    return obj, img, used_tag_ids


def solve_head_bundle_pnp(obj_pts, img_pts, K, dist, T_init=None):
    use_guess = T_init is not None
    if use_guess:
        R0, t0 = T_to_rt(T_init)
        rvec0, _ = cv2.Rodrigues(R0)
        tvec0 = t0.reshape(3, 1)
    else:
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.zeros((3, 1), dtype=np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        rvec=rvec0,
        tvec=tvec0,
        useExtrinsicGuess=use_guess,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist,
            rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        inliers = None
        if not ok:
            return None, None, None

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
    T_cam_head = rt_to_T(Rm, tvec.reshape(3))
    rmse = pose_rmse(obj_pts, img_pts, K, dist, T_cam_head)
    return T_cam_head, rmse, inliers


def load_timestamps_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def back_tag_to_body_transform(back_tag_mount: str):
    """
    Build T_backtag_body where body frame B is:
      x = forward
      y = left
      z = up

    For a tag mounted upright on the person's back and facing rear camera:
      x_body = -z_tag
      y_body = -x_tag
      z_body = +y_tag
    """
    if back_tag_mount == "upright_on_back":
        x_body_in_tag = np.array([0.0, 0.0, -1.0])   # -z_tag
        y_body_in_tag = np.array([-1.0, 0.0, 0.0])   # -x_tag
        z_body_in_tag = np.array([0.0, 1.0, 0.0])    # +y_tag
        R_tag_body = np.column_stack([x_body_in_tag, y_body_in_tag, z_body_in_tag])
        return rt_to_T(R_tag_body, np.zeros(3, dtype=np.float64))

    raise ValueError(f"Unsupported back_tag_mount: {back_tag_mount}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rig-calib", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--timestamps-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--neutral-frame", default=None,
                    help="optional frame filename, e.g. frame_006874.png")
    args = ap.parse_args()

    K, dist = load_camera_json(args.camera)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    back_tag_id = int(cfg["back_tag_id"])
    back_tag_size_m = float(cfg["back_tag_size_m"])
    tag_family = cfg.get("tag_family", "tag36h11")
    back_tag_mount = cfg.get("back_tag_mount", "upright_on_back")
    T_backtag_body = back_tag_to_body_transform(back_tag_mount)

    T_H_T, head_tag_ids, head_tag_size_m = load_rig_calib(args.rig_calib)
    head_tag_ids = set(head_tag_ids)

    detector = make_detector(tag_family=tag_family, quad_decimate=1.0)

    rows = load_timestamps_csv(args.timestamps_csv)

    frame_dir = Path(args.frame_dir)

    last_T_cam_head = None
    last_T_cam_back = None
    neutral_T_B_H = None

    outputs = []

    for r in rows:
        frame_idx = int(r["frame_idx"])
        timestamp_ns = r["unix_ns"]

        frame_name = f"frame_{frame_idx:06d}.png"
        img_path = frame_dir / frame_name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            outputs.append({
                "frame_idx": frame_idx,
                "frame": frame_name,
                "timestamp_ns": timestamp_ns,
                "ok": 0,
                "status": "image_not_found"
            })
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detect_tags(detector, gray)

        # head bundle
        head_dets = [d for d in dets if int(d.tag_id) in head_tag_ids]
        obj_pts, img_pts, used_head_tag_ids = collect_head_correspondences(
            head_dets, T_H_T, head_tag_size_m
        )

        T_cam_head = None
        head_rmse = None
        if obj_pts is not None and len(used_head_tag_ids) >= 1:
            T_cam_head, head_rmse, _ = solve_head_bundle_pnp(
                obj_pts, img_pts, K, dist, T_init=last_T_cam_head
            )
            if T_cam_head is not None:
                last_T_cam_head = T_cam_head

        # back tag
        back_dets = [d for d in dets if int(d.tag_id) == back_tag_id]
        T_cam_back = None
        if len(back_dets) > 0:
            T_cam_back_tag = solve_single_tag_pose_ippe(
                back_dets[0], back_tag_size_m, K, dist
            )
            if T_cam_back_tag is not None:
                # convert from back-tag frame to canonical body frame
                T_cam_back = T_cam_back_tag @ T_backtag_body
                last_T_cam_back = T_cam_back

                if args.neutral_frame is not None and frame_name == args.neutral_frame:
                    if T_cam_head is not None and T_cam_back is not None:
                        neutral_T_B_H = invert_T(T_cam_back) @ T_cam_head
                        print(f"[INFO] neutral frame set from {frame_name}")

        out = {
            "frame_idx": frame_idx,
            "frame": frame_name,
            "timestamp_ns": timestamp_ns,
            "ok": 0,
            "num_head_tags": len(set(used_head_tag_ids)) if used_head_tag_ids else 0,
            "head_rmse_px": head_rmse if head_rmse is not None else ""
        }

        if used_head_tag_ids:
            unique_ids = sorted(list(set(used_head_tag_ids)))
            out["visible_head_tag_ids"] = " ".join(map(str, unique_ids))
        else:
            out["visible_head_tag_ids"] = ""

        if T_cam_head is not None:
            Rh, th = T_to_rt(T_cam_head)
            roll_h, pitch_h, yaw_h = rot_to_rpy_deg(Rh)
            out.update({
                "cam_head_tx": th[0], "cam_head_ty": th[1], "cam_head_tz": th[2],
                "cam_head_roll_deg": roll_h,
                "cam_head_pitch_deg": pitch_h,
                "cam_head_yaw_deg": yaw_h,
            })

        if T_cam_back is not None:
            Rb, tb = T_to_rt(T_cam_back)
            roll_b, pitch_b, yaw_b = rot_to_rpy_deg(Rb)
            out.update({
                "cam_back_tx": tb[0], "cam_back_ty": tb[1], "cam_back_tz": tb[2],
                "cam_back_roll_deg": roll_b,
                "cam_back_pitch_deg": pitch_b,
                "cam_back_yaw_deg": yaw_b,
            })

        if T_cam_head is not None and T_cam_back is not None:
            T_B_H = invert_T(T_cam_back) @ T_cam_head
            Rbh, tbh = T_to_rt(T_B_H)
            roll_bh, pitch_bh, yaw_bh = rot_to_rpy_deg(Rbh)

            out.update({
                "ok": 1,
                "status": "ok",
                "back_head_tx": tbh[0], "back_head_ty": tbh[1], "back_head_tz": tbh[2],
                "back_head_roll_deg": roll_bh,
                "back_head_pitch_deg": pitch_bh,
                "back_head_yaw_deg": yaw_bh,
            })

            if neutral_T_B_H is not None:
                T_rel = invert_T(neutral_T_B_H) @ T_B_H
                Rrel, trel = T_to_rt(T_rel)
                roll_rel, pitch_rel, yaw_rel = rot_to_rpy_deg(Rrel)
                out.update({
                    "neutral_rel_tx": trel[0], "neutral_rel_ty": trel[1], "neutral_rel_tz": trel[2],
                    "neutral_rel_roll_deg": roll_rel,
                    "neutral_rel_pitch_deg": pitch_rel,
                    "neutral_rel_yaw_deg": yaw_rel,
                })
        else:
            if T_cam_head is None and T_cam_back is None:
                out["status"] = "head_and_back_missing"
            elif T_cam_head is None:
                out["status"] = "head_missing"
            else:
                out["status"] = "back_missing"

        outputs.append(out)

    if len(outputs) == 0:
        print("No outputs.")
        return

    all_keys = []
    key_set = set()
    for row in outputs:
        for k in row.keys():
            if k not in key_set:
                key_set.add(k)
                all_keys.append(k)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(outputs)

    print(f"saved to {args.output_csv}")


if __name__ == "__main__":
    main()