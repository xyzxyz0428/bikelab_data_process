#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from pose_utils import (
    load_camera_json,
    make_detector,
    detect_tags,
    solve_single_tag_pose_ippe,
    invert_T,
)


def rt_to_T(Rm, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def mean_transform(T_list):
    if len(T_list) == 0:
        raise ValueError("empty transform list")

    mats = np.stack([T[:3, :3] for T in T_list], axis=0)
    trans = np.stack([T[:3, 3] for T in T_list], axis=0)

    R_mean = R.from_matrix(mats).mean().as_matrix()
    t_mean = np.mean(trans, axis=0)

    return rt_to_T(R_mean, t_mean)


def resolve_T_root_T(ref_tag_id, parent_map, edge_mean):
    """
    edge_mean[(parent, child)] = T_parent_child
    return T_root_T where root = ref_tag_id
    """
    T_root_T = {ref_tag_id: np.eye(4, dtype=np.float64)}

    unresolved = set(parent_map.keys())
    progress = True
    while unresolved and progress:
        progress = False
        for child in list(unresolved):
            parent = parent_map[child]
            if parent in T_root_T and (parent, child) in edge_mean:
                T_root_T[child] = T_root_T[parent] @ edge_mean[(parent, child)]
                unresolved.remove(child)
                progress = True

    if unresolved:
        raise RuntimeError(
            f"Could not resolve transforms for tags: {sorted(list(unresolved))}. "
            f"Missing parent-child edge samples?"
        )

    return T_root_T


def build_anatomical_head_frame(T_root_T, head_order_left_to_right, head_origin_tag_id):
    """
    Construct a canonical head frame H:
      x = forward
      y = left
      z = up

    Strategy:
      - y_left from right-group center to left-group center
      - x_forward from negative average z-axis of middle tags
      - z_up = x cross y
    """
    ordered = head_order_left_to_right
    n = len(ordered)
    centers = {tid: T_root_T[tid][:3, 3] for tid in ordered}
    z_axes = {tid: T_root_T[tid][:3, 2] for tid in ordered}

    # split left / center / right
    mid = n // 2
    if n % 2 == 1:
        left_ids = ordered[:mid]
        center_ids = ordered[max(0, mid - 1): min(n, mid + 2)]   # center trio if possible
        right_ids = ordered[mid + 1:]
    else:
        left_ids = ordered[:mid]
        center_ids = ordered[max(0, mid - 1): min(n, mid + 1)]
        right_ids = ordered[mid:]

    if len(left_ids) == 0 or len(right_ids) == 0:
        raise RuntimeError("Need tags on both left and right sides to build head frame.")

    left_center = np.mean([centers[tid] for tid in left_ids], axis=0)
    right_center = np.mean([centers[tid] for tid in right_ids], axis=0)

    # y = left
    y_left = normalize(left_center - right_center)

    # x = forward
    # tags face backward toward camera, so head forward is opposite average tag z
    avg_tag_normal = np.mean([z_axes[tid] for tid in center_ids], axis=0)
    x_forward_raw = -normalize(avg_tag_normal)

    # orthogonalize x against y
    x_forward = x_forward_raw - np.dot(x_forward_raw, y_left) * y_left
    x_forward = normalize(x_forward)

    # z = up
    z_up = normalize(np.cross(x_forward, y_left))

    # re-orthogonalize y
    y_left = normalize(np.cross(z_up, x_forward))

    R_root_H = np.column_stack([x_forward, y_left, z_up])

    origin = centers[head_origin_tag_id]
    T_root_H = rt_to_T(R_root_H, origin)
    return T_root_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    K, dist = load_camera_json(args.camera)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    head_tag_ids = [int(x) for x in cfg["head_tag_ids"]]
    head_tag_ids_set = set(head_tag_ids)
    head_tag_size_m = {int(k): float(v) for k, v in cfg["head_tag_size_m"].items()}
    ref_tag_id = int(cfg["ref_tag_id"])
    tag_family = cfg.get("tag_family", "tag36h11")
    head_order_left_to_right = [int(x) for x in cfg["head_order_left_to_right"]]
    head_origin_tag_id = int(cfg.get("head_origin_tag_id", ref_tag_id))

    # child -> parent
    raw_parent_map = cfg.get("parent_map", {})
    parent_map = {int(child): int(parent) for child, parent in raw_parent_map.items()}

    detector = make_detector(tag_family=tag_family, quad_decimate=1.0)

    image_paths = sorted(
        [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    edge_samples = {(parent, child): [] for child, parent in parent_map.items()}
    used_images = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detect_tags(detector, gray)

        det_map = {int(d.tag_id): d for d in dets if int(d.tag_id) in head_tag_ids_set}

        any_used_this_image = False

        for child, parent in parent_map.items():
            if parent not in det_map or child not in det_map:
                continue

            T_cam_parent = solve_single_tag_pose_ippe(
                det_map[parent], head_tag_size_m[parent], K, dist
            )
            T_cam_child = solve_single_tag_pose_ippe(
                det_map[child], head_tag_size_m[child], K, dist
            )

            if T_cam_parent is None or T_cam_child is None:
                continue

            T_parent_child = invert_T(T_cam_parent) @ T_cam_child
            edge_samples[(parent, child)].append(T_parent_child)
            any_used_this_image = True

        if any_used_this_image:
            used_images += 1

    edge_mean = {}
    edge_count = {}

    for edge, Ts in edge_samples.items():
        parent, child = edge
        edge_count[edge] = len(Ts)
        if len(Ts) == 0:
            print(f"[WARN] edge {parent}->{child}: 0 samples")
            continue
        edge_mean[edge] = mean_transform(Ts)
        print(f"edge {parent}->{child}: {len(Ts)} samples")

    # raw root frame = ref tag frame
    T_root_T = resolve_T_root_T(ref_tag_id, parent_map, edge_mean)

    # build canonical anatomical head frame
    T_root_H = build_anatomical_head_frame(
        T_root_T,
        head_order_left_to_right=head_order_left_to_right,
        head_origin_tag_id=head_origin_tag_id,
    )
    T_H_root = invert_T(T_root_H)

    # convert all tags into canonical H frame
    T_H_T = {}
    for tid in sorted(T_root_T.keys()):
        T_H_T[tid] = T_H_root @ T_root_T[tid]

    rig = {
        "rig_frame": "head_anatomical_frame",
        "ref_tag_id": ref_tag_id,
        "head_tag_ids": sorted(head_tag_ids),
        "head_tag_size_m": {str(k): float(v) for k, v in head_tag_size_m.items()},
        "parent_map": {str(child): int(parent) for child, parent in parent_map.items()},
        "head_order_left_to_right": head_order_left_to_right,
        "head_origin_tag_id": head_origin_tag_id,
        "edge_sample_count": {
            f"{parent}->{child}": int(edge_count[(parent, child)])
            for (parent, child) in edge_samples.keys()
        },
        "T_H_T": {
            str(tag_id): T_H_T[tag_id].tolist()
            for tag_id in sorted(T_H_T.keys())
        },
        "T_root_H": T_root_H.tolist(),
        "axis_definition": {
            "x": "forward",
            "y": "left",
            "z": "up"
        }
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rig, f, indent=2)

    print(f"used_images = {used_images}")
    print(f"saved rig calibration to {args.output}")


if __name__ == "__main__":
    main()