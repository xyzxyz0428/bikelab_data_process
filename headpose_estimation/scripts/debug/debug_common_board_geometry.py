#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import gc
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from pupil_apriltags import Detector


# ============================================================
# I/O
# ============================================================

def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)
    return K, dist, data


def sample_files(files, max_frames):
    if max_frames <= 0 or max_frames >= len(files):
        return files

    idx = np.linspace(0, len(files) - 1, max_frames).astype(int)
    return [files[i] for i in idx]


# ============================================================
# Geometry
# ============================================================

def square_object_points(tag_size_m):
    s = float(tag_size_m)

    # object corner order:
    # lt, rt, rb, lb
    return np.array([
        [-s / 2,  s / 2, 0.0],
        [ s / 2,  s / 2, 0.0],
        [ s / 2, -s / 2, 0.0],
        [-s / 2, -s / 2, 0.0],
    ], dtype=np.float64)


def build_board_object_points(board_tag_ids, rows, cols, tag_size_m, gap_x_m, gap_y_m):
    """
    Board frame:
      x right
      y up
      z out of board plane

    rows, cols = tag grid rows and columns
    board_tag_ids = ID order according to selected layout
    """
    if len(board_tag_ids) != rows * cols:
        raise ValueError(
            f"len(board_tag_ids)={len(board_tag_ids)} != rows*cols={rows*cols}"
        )

    total_w = cols * tag_size_m + (cols - 1) * gap_x_m
    total_h = rows * tag_size_m + (rows - 1) * gap_y_m

    tag_obj = {}
    tag_local = square_object_points(tag_size_m)

    for idx, tid in enumerate(board_tag_ids):
        r = idx // cols
        c = idx % cols

        cx = -total_w / 2.0 + tag_size_m / 2.0 + c * (tag_size_m + gap_x_m)
        cy =  total_h / 2.0 - tag_size_m / 2.0 - r * (tag_size_m + gap_y_m)

        pts = tag_local.copy()
        pts[:, 0] += cx
        pts[:, 1] += cy

        tag_obj[int(tid)] = pts

    return tag_obj


def corner_order(corners, mode):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)

    if mode == "as_is_0123":
        return c[[0, 1, 2, 3], :]

    if mode == "pupil_3210":
        # common assumption for pupil_apriltags:
        # detected: lb, rb, rt, lt
        # target:   lt, rt, rb, lb
        return c[[3, 2, 1, 0], :]

    if mode == "reverse_3210":
        return c[::-1, :]

    if mode == "shift1_1230":
        return c[[1, 2, 3, 0], :]

    if mode == "shift2_2301":
        return c[[2, 3, 0, 1], :]

    if mode == "shift3_3012":
        return c[[3, 0, 1, 2], :]

    if mode == "swap02_2103":
        return c[[2, 1, 0, 3], :]

    if mode == "swap13_0321":
        return c[[0, 3, 2, 1], :]

    raise ValueError(f"Unsupported corner mode: {mode}")


def tag_area_from_corners(corners):
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return float(abs(cv2.contourArea(c.astype(np.float32))))


def deduplicate_detections(dets, allowed_ids, duplicate_policy="best"):
    """
    Returns:
      dets_by_id, duplicate_ids
    """
    allowed = set(allowed_ids)
    grouped = defaultdict(list)

    for d in dets:
        tid = int(d.tag_id)
        if tid in allowed:
            grouped[tid].append(d)

    duplicate_ids = sorted([tid for tid, xs in grouped.items() if len(xs) > 1])

    if duplicate_policy == "reject" and duplicate_ids:
        return {}, duplicate_ids

    dets_by_id = {}

    for tid, xs in grouped.items():
        if len(xs) == 1:
            dets_by_id[tid] = xs[0]
            continue

        # choose best by decision_margin, then area
        best = None
        best_score = None

        for d in xs:
            margin = float(getattr(d, "decision_margin", 0.0))
            area = tag_area_from_corners(d.corners)
            score = (margin, area)

            if best is None or score > best_score:
                best = d
                best_score = score

        dets_by_id[tid] = best

    return dets_by_id, duplicate_ids


def reprojection_rmse(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)

    err = np.linalg.norm(proj - img_pts, axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def solve_and_rmse(obj_pts, img_pts, K, dist):
    best = None

    methods = [
        ("EPNP", cv2.SOLVEPNP_EPNP),
        ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE),
        ("IPPE", cv2.SOLVEPNP_IPPE),
    ]

    for name, flag in methods:
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

        rmse = reprojection_rmse(obj_pts, img_pts, rvec, tvec, K, dist)

        if np.isfinite(rmse):
            if best is None or rmse < best["rmse"]:
                best = {
                    "rmse": rmse,
                    "method": name,
                    "rvec": rvec,
                    "tvec": tvec,
                }

    return best


def collect_points(dets_by_id, board_obj_pts, corner_mode, min_tags):
    obj_all = []
    img_all = []
    used = []

    for tid, det in dets_by_id.items():
        if tid not in board_obj_pts:
            continue

        obj_all.append(board_obj_pts[tid])
        img_all.append(corner_order(det.corners, corner_mode))
        used.append(tid)

    used_unique = sorted(set(used))

    if len(used_unique) < min_tags:
        return None, None, []

    obj_pts = np.ascontiguousarray(np.vstack(obj_all).astype(np.float64))
    img_pts = np.ascontiguousarray(np.vstack(img_all).astype(np.float64))

    return obj_pts, img_pts, used_unique


# ============================================================
# Layout candidates
# ============================================================

def make_layout_candidates(board_ids_input, enabled_layouts):
    candidates = []

    n = len(board_ids_input)

    def add(name, rows, cols, ids):
        if rows * cols == n:
            if enabled_layouts is None or name in enabled_layouts:
                candidates.append((name, rows, cols, ids))

    # common layouts for 12 tags
    if n == 12:
        ids = board_ids_input

        add("4x3_row_major", 4, 3, ids)
        add("3x4_row_major", 3, 4, ids)

        grid_4x3 = np.array(ids).reshape(4, 3)
        ids_4x3_col = list(grid_4x3.T.reshape(-1))
        add("4x3_col_major_equiv", 4, 3, ids_4x3_col)

        grid_3x4 = np.array(ids).reshape(3, 4)
        ids_3x4_col = list(grid_3x4.T.reshape(-1))
        add("3x4_col_major_equiv", 3, 4, ids_3x4_col)

        # reversed physical orientation candidates
        add("4x3_row_major_reversed", 4, 3, list(reversed(ids)))
        add("3x4_row_major_reversed", 3, 4, list(reversed(ids)))

        # flip left-right / top-bottom for 4x3
        g = np.array(ids).reshape(4, 3)
        add("4x3_flip_lr", 4, 3, list(np.fliplr(g).reshape(-1)))
        add("4x3_flip_ud", 4, 3, list(np.flipud(g).reshape(-1)))
        add("4x3_flip_lr_ud", 4, 3, list(np.flipud(np.fliplr(g)).reshape(-1)))

        # flip left-right / top-bottom for 3x4
        g = np.array(ids).reshape(3, 4)
        add("3x4_flip_lr", 3, 4, list(np.fliplr(g).reshape(-1)))
        add("3x4_flip_ud", 3, 4, list(np.flipud(g).reshape(-1)))
        add("3x4_flip_lr_ud", 3, 4, list(np.flipud(np.fliplr(g)).reshape(-1)))

    else:
        raise ValueError(
            "This debug script currently auto-generates layout candidates mainly for 12 tags. "
            "Please extend make_layout_candidates() if needed."
        )

    return candidates


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--tag-family", default="tag36h11")

    ap.add_argument("--board-tag-ids", required=True)
    ap.add_argument("--tag-size-m", type=float, required=True)
    ap.add_argument("--gap-x-m", type=float, required=True)
    ap.add_argument("--gap-y-m", type=float, required=True)

    ap.add_argument("--max-frames", type=int, default=500)
    ap.add_argument("--min-tags", type=int, default=2)

    ap.add_argument(
        "--duplicate-policy",
        default="best",
        choices=["best", "reject"],
    )

    ap.add_argument(
        "--corner-modes",
        default="all",
        help="comma-separated corner modes or 'all'",
    )

    ap.add_argument(
        "--layouts",
        default="all",
        help="comma-separated layout names or 'all'",
    )

    ap.add_argument(
        "--max-print-frames",
        type=int,
        default=20,
        help="number of frame hit examples to print",
    )

    ap.add_argument(
        "--output-csv",
        default=None,
        help="optional CSV output with all tested results",
    )

    args = ap.parse_args()

    board_ids_input = [int(x) for x in args.board_tag_ids.split(",")]
    board_set = set(board_ids_input)

    if args.corner_modes == "all":
        corner_modes = [
            "as_is_0123",
            "pupil_3210",
            "reverse_3210",
            "shift1_1230",
            "shift2_2301",
            "shift3_3012",
            "swap02_2103",
            "swap13_0321",
        ]
    else:
        corner_modes = [x.strip() for x in args.corner_modes.split(",") if x.strip()]

    enabled_layouts = None if args.layouts == "all" else {
        x.strip() for x in args.layouts.split(",") if x.strip()
    }

    K, dist, cam_data = load_camera_json(args.camera_json)

    detector = Detector(
        families=args.tag_family,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    frame_dir = Path(args.frame_dir)
    all_files = sorted(
        list(frame_dir.glob("frame_*.png"))
        + list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.jpeg"))
    )

    files = sample_files(all_files, args.max_frames)

    if not files:
        raise RuntimeError("No frame files found.")

    img0 = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)

    if img0 is not None:
        print("[INFO] first sampled image:", files[0].name)
        print("[INFO] first sampled image size:", img0.shape[1], "x", img0.shape[0])

    print("[INFO] total files:", len(all_files))
    print("[INFO] sampled files:", len(files))
    print("[INFO] min_tags:", args.min_tags)
    print("[INFO] duplicate_policy:", args.duplicate_policy)
    print("[INFO] K:")
    print(K)
    print("[INFO] dist:", dist.reshape(-1).tolist())

    layout_candidates = make_layout_candidates(board_ids_input, enabled_layouts)

    print("\n[INFO] testing layouts:")
    for name, rows, cols, ids in layout_candidates:
        print(f"  {name}: rows={rows}, cols={cols}, ids={ids}")

    print("\n[INFO] testing corner modes:")
    for cm in corner_modes:
        print(f"  {cm}")

    results = []
    frame_hits = []
    duplicate_hit_count = 0

    board_obj_cache = {}

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=False)

        raw_detected_ids = sorted([
            int(d.tag_id) for d in dets
            if int(d.tag_id) in board_set
        ])

        if raw_detected_ids:
            frame_hits.append((p.name, raw_detected_ids))

        dets_by_id, duplicate_ids = deduplicate_detections(
            dets,
            board_set,
            duplicate_policy=args.duplicate_policy,
        )

        if duplicate_ids:
            duplicate_hit_count += 1

        if len(dets_by_id) < args.min_tags:
            continue

        for layout_name, rows, cols, ids_order in layout_candidates:
            cache_key = (layout_name, args.tag_size_m, args.gap_x_m, args.gap_y_m)

            if cache_key not in board_obj_cache:
                board_obj_cache[cache_key] = build_board_object_points(
                    ids_order,
                    rows,
                    cols,
                    args.tag_size_m,
                    args.gap_x_m,
                    args.gap_y_m,
                )

            board_obj_pts = board_obj_cache[cache_key]

            for cm in corner_modes:
                obj_pts, img_pts, used = collect_points(
                    dets_by_id,
                    board_obj_pts,
                    cm,
                    min_tags=args.min_tags,
                )

                if obj_pts is None:
                    continue

                best = solve_and_rmse(obj_pts, img_pts, K, dist)

                if best is None:
                    continue

                results.append({
                    "frame": p.name,
                    "layout": layout_name,
                    "rows": rows,
                    "cols": cols,
                    "corner_mode": cm,
                    "method": best["method"],
                    "rmse": best["rmse"],
                    "used_ids": " ".join(map(str, used)),
                    "num_tags": len(used),
                    "duplicate_ids": " ".join(map(str, duplicate_ids)),
                })

    print("\n[INFO] frames with board tags in sampled frames:", len(frame_hits))
    print("[INFO] frames with duplicate detected board IDs:", duplicate_hit_count)
    print(f"[INFO] first {args.max_print_frames} sampled hits:")
    for name, ids in frame_hits[:args.max_print_frames]:
        print(f"  {name}: {ids}")

    if not results:
        print("\n[ERROR] no valid board solve results")
        detector = None
        gc.collect()
        sys.stdout.flush()
        os._exit(0)

    # Optional CSV
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame",
            "layout",
            "rows",
            "cols",
            "corner_mode",
            "method",
            "rmse",
            "num_tags",
            "used_ids",
            "duplicate_ids",
        ]

        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)

        print(f"\n[INFO] saved all results to {out}")

    results_sorted = sorted(results, key=lambda x: x["rmse"])

    print("\n=== Best 30 candidates ===")
    for r in results_sorted[:30]:
        print(
            f"rmse={r['rmse']:.3f} px | "
            f"{r['layout']} | {r['corner_mode']} | {r['method']} | "
            f"tags={r['num_tags']} ids=[{r['used_ids']}] | frame={r['frame']}"
        )

    print("\n=== Median RMSE by layout / corner mode ===")
    groups = defaultdict(list)

    for r in results:
        key = (r["layout"], r["corner_mode"])
        groups[key].append(float(r["rmse"]))

    table = []

    for key, vals in groups.items():
        vals = np.asarray(vals, dtype=np.float64)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            continue

        table.append({
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "p95": float(np.quantile(vals, 0.95)),
            "min": float(np.min(vals)),
            "n": int(len(vals)),
            "layout": key[0],
            "corner_mode": key[1],
        })

    table_sorted = sorted(table, key=lambda x: x["median"])

    for r in table_sorted[:40]:
        print(
            f"median={r['median']:.3f} "
            f"mean={r['mean']:.3f} "
            f"p95={r['p95']:.3f} "
            f"min={r['min']:.3f} "
            f"n={r['n']} | "
            f"{r['layout']} | {r['corner_mode']}"
        )

    print("\n=== Recommendation ===")
    best_group = table_sorted[0]
    print(
        f"Best median group: "
        f"{best_group['layout']} + {best_group['corner_mode']} "
        f"(median={best_group['median']:.3f}px, "
        f"p95={best_group['p95']:.3f}px, "
        f"min={best_group['min']:.3f}px, n={best_group['n']})"
    )

    if best_group["median"] > 20:
        print(
            "[WARN] Best median RMSE is still > 20 px. "
            "This strongly suggests wrong board geometry: tag size, gap, rows/cols, ID order, or printed board deformation."
        )
    elif best_group["median"] > 5:
        print(
            "[WARN] Best median RMSE is > 5 px. "
            "The layout may be close but not accurate enough for reliable multi-tag board calibration."
        )
    else:
        print(
            "[OK] Best median RMSE is <= 5 px. "
            "This layout/corner mode is a reasonable candidate for the common-board calibration script."
        )

    detector = None
    gc.collect()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()