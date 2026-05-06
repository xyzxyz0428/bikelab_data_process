#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def to_float(v, default=np.nan):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def to_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


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
            "std": np.nan,
        }

    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
    }


def find_col(keys, candidates, required=True):
    for c in candidates:
        if c in keys:
            return c
    if required:
        raise KeyError(f"Cannot find any of columns: {candidates}")
    return None


def get_columns(rows):
    keys = rows[0].keys()

    return {
        "window_id": find_col(keys, ["window_id"]),
        "tag_id": find_col(keys, ["tag_id"]),
        "gaze_x": find_col(keys, ["gaze_x", "tobii_x", "Gaze point X", "Fixation point X"]),
        "gaze_y": find_col(keys, ["gaze_y", "tobii_y", "Gaze point Y", "Fixation point Y"]),
        "center_x": find_col(keys, ["tag_center_x", "center_x", "tag_center_u", "tag_u"]),
        "center_y": find_col(keys, ["tag_center_y", "center_y", "tag_center_v", "tag_v"]),
        "frame_idx": find_col(keys, ["frame_idx", "scene_frame_idx", "nearest_scene_frame_idx"], required=False),
        "timestamp_ns": find_col(keys, ["timestamp_ns", "scene_timestamp_ns", "scene_unix_ns", "tobii_unix_ns"], required=False),
    }


def build_valid_rows(rows, cols):
    valid = []

    for r in rows:
        gx = to_float(r.get(cols["gaze_x"]))
        gy = to_float(r.get(cols["gaze_y"]))
        cx = to_float(r.get(cols["center_x"]))
        cy = to_float(r.get(cols["center_y"]))

        if not all(np.isfinite(v) for v in [gx, gy, cx, cy]):
            continue

        rr = dict(r)
        rr["_gaze_x"] = gx
        rr["_gaze_y"] = gy
        rr["_center_x"] = cx
        rr["_center_y"] = cy
        rr["_dx"] = gx - cx
        rr["_dy"] = gy - cy
        rr["_center_error"] = float(np.hypot(gx - cx, gy - cy))
        valid.append(rr)

    return valid


def fit_affine(gaze_pts, target_pts):
    gaze_pts = np.asarray(gaze_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)

    if len(gaze_pts) < 3:
        raise RuntimeError("Need at least 3 points to fit affine transform.")

    X = np.column_stack([
        gaze_pts[:, 0],
        gaze_pts[:, 1],
        np.ones(len(gaze_pts)),
    ])

    ax, *_ = np.linalg.lstsq(X, target_pts[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(X, target_pts[:, 1], rcond=None)

    A = np.vstack([ax, ay])
    return A


def apply_affine(A, pts):
    pts = np.asarray(pts, dtype=np.float64)
    X = np.column_stack([
        pts[:, 0],
        pts[:, 1],
        np.ones(len(pts)),
    ])
    return X @ A.T


def center_error_xy(xy, target_xy):
    xy = np.asarray(xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    return np.linalg.norm(xy - target_xy, axis=1)


def group_key(row, mode):
    if mode == "tag":
        return str(row["tag_id"])
    if mode == "window":
        return str(row["window_id"])
    if mode == "window_tag":
        return f"{row['window_id']}_{row['tag_id']}"
    raise ValueError(f"Unsupported group mode: {mode}")


def run_leave_one_group_out(valid_rows, group_mode):
    groups = defaultdict(list)
    for r in valid_rows:
        groups[group_key(r, group_mode)].append(r)

    result_rows = []

    for key, test_rows in sorted(groups.items(), key=lambda kv: kv[0]):
        train_rows = [r for r in valid_rows if group_key(r, group_mode) != key]

        if len(train_rows) < 3 or len(test_rows) == 0:
            continue

        train_gaze = np.array([[r["_gaze_x"], r["_gaze_y"]] for r in train_rows], dtype=np.float64)
        train_target = np.array([[r["_center_x"], r["_center_y"]] for r in train_rows], dtype=np.float64)

        test_gaze = np.array([[r["_gaze_x"], r["_gaze_y"]] for r in test_rows], dtype=np.float64)
        test_target = np.array([[r["_center_x"], r["_center_y"]] for r in test_rows], dtype=np.float64)

        A = fit_affine(train_gaze, train_target)
        pred = apply_affine(A, test_gaze)

        original_err = center_error_xy(test_gaze, test_target)
        affine_err = center_error_xy(pred, test_target)

        dx = test_gaze[:, 0] - test_target[:, 0]
        dy = test_gaze[:, 1] - test_target[:, 1]

        s0 = summarize(original_err)
        s1 = summarize(affine_err)
        sdx = summarize(dx)
        sdy = summarize(dy)

        tag_ids = sorted(set(str(r["tag_id"]) for r in test_rows))
        window_ids = sorted(set(str(r["window_id"]) for r in test_rows))

        result_rows.append({
            "group_mode": group_mode,
            "held_out_group": key,
            "tag_ids": " ".join(tag_ids),
            "window_ids": " ".join(window_ids),
            "n_test": len(test_rows),
            "n_train": len(train_rows),

            "original_center_error_mean_px": s0["mean"],
            "original_center_error_median_px": s0["median"],
            "original_center_error_p95_px": s0["p95"],

            "loto_affine_center_error_mean_px": s1["mean"],
            "loto_affine_center_error_median_px": s1["median"],
            "loto_affine_center_error_p95_px": s1["p95"],

            "improvement_median_px": s0["median"] - s1["median"],
            "improvement_ratio_median": s1["median"] / s0["median"] if s0["median"] and np.isfinite(s0["median"]) else np.nan,

            "dx_median_px": sdx["median"],
            "dy_median_px": sdy["median"],

            "affine_matrix": json.dumps(A.tolist()),
        })

    return result_rows


def resolve_frame_path(scene_frame_dir, frame_idx):
    frame_idx = int(frame_idx)
    for ext in [".png", ".jpg", ".jpeg"]:
        p = scene_frame_dir / f"frame_{frame_idx:06d}{ext}"
        if p.exists():
            return p
    return None


def infer_frame_idx(row, cols):
    if cols["frame_idx"] is not None:
        return to_int(row.get(cols["frame_idx"]))

    # fallback: common possible columns not detected earlier
    for c in ["nearest_scene_frame_idx", "scene_frame", "frame"]:
        if c in row:
            return to_int(row.get(c))

    return None


def draw_cross(img, xy, color, label):
    x, y = int(round(xy[0])), int(round(xy[1]))

    cv2.drawMarker(
        img,
        (x, y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
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


def visualize_samples(
    valid_rows,
    cols,
    scene_frame_dir,
    output_dir,
    samples_per_group,
    group_mode,
    global_dx_med,
    global_dy_med,
    A_all,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in valid_rows:
        groups[group_key(r, group_mode)].append(r)

    saved = []

    for key, rs in sorted(groups.items(), key=lambda kv: kv[0]):
        # Prefer rows near the window gaze median, not near tag center.
        pts = np.array([[r["_gaze_x"], r["_gaze_y"]] for r in rs], dtype=np.float64)
        med = np.median(pts, axis=0)
        d = np.linalg.norm(pts - med[None, :], axis=1)
        order = np.argsort(d)

        selected = []
        for idx in order:
            r = rs[idx]
            frame_idx = infer_frame_idx(r, cols)
            if frame_idx is None:
                continue
            frame_path = resolve_frame_path(scene_frame_dir, frame_idx)
            if frame_path is None:
                continue
            selected.append((r, frame_idx, frame_path))
            if len(selected) >= samples_per_group:
                break

        for i, (r, frame_idx, frame_path) in enumerate(selected):
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            gx = r["_gaze_x"]
            gy = r["_gaze_y"]
            cx = r["_center_x"]
            cy = r["_center_y"]

            original = np.array([gx, gy], dtype=np.float64)
            center = np.array([cx, cy], dtype=np.float64)

            global_corr = np.array([
                gx - global_dx_med,
                gy - global_dy_med,
            ], dtype=np.float64)

            affine_corr = apply_affine(A_all, np.array([[gx, gy]], dtype=np.float64))[0]

            # Draw points
            draw_cross(img, center, (0, 255, 255), "tag center")
            draw_cross(img, original, (0, 0, 255), "original gaze")
            draw_cross(img, global_corr, (255, 0, 0), "global corrected")
            draw_cross(img, affine_corr, (0, 255, 0), "affine corrected")

            # Draw lines to tag center
            cv2.line(
                img,
                (int(round(original[0])), int(round(original[1]))),
                (int(round(center[0])), int(round(center[1]))),
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                (int(round(global_corr[0])), int(round(global_corr[1]))),
                (int(round(center[0])), int(round(center[1]))),
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                img,
                (int(round(affine_corr[0])), int(round(affine_corr[1]))),
                (int(round(center[0])), int(round(center[1]))),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            err_original = float(np.linalg.norm(original - center))
            err_global = float(np.linalg.norm(global_corr - center))
            err_affine = float(np.linalg.norm(affine_corr - center))

            header = (
                f"group={key} tag={r['tag_id']} window={r['window_id']} frame={frame_idx} | "
                f"err original={err_original:.1f}px global={err_global:.1f}px affine={err_affine:.1f}px"
            )

            cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(
                img,
                header,
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out_name = f"{group_mode}_{key}_sample{i+1}_tag{r['tag_id']}_win{r['window_id']}_frame{frame_idx:06d}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), img)

            saved.append({
                "group_mode": group_mode,
                "group": key,
                "tag_id": r["tag_id"],
                "window_id": r["window_id"],
                "frame_idx": frame_idx,
                "image": str(out_path),
                "original_error_px": err_original,
                "global_corrected_error_px": err_global,
                "affine_corrected_error_px": err_affine,
            })

    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--scene-frame-dir", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument(
        "--leave-one-mode",
        default="tag",
        choices=["tag", "window", "window_tag"],
        help="group to leave out for cross validation",
    )

    ap.add_argument(
        "--visualize-group-mode",
        default="tag",
        choices=["tag", "window", "window_tag"],
    )

    ap.add_argument("--samples-per-group", type=int, default=2)

    args = ap.parse_args()

    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError("Input CSV is empty.")

    cols = get_columns(rows)
    valid_rows = build_valid_rows(rows, cols)

    if len(valid_rows) < 10:
        raise RuntimeError(f"Too few valid rows: {len(valid_rows)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaze = np.array([[r["_gaze_x"], r["_gaze_y"]] for r in valid_rows], dtype=np.float64)
    target = np.array([[r["_center_x"], r["_center_y"]] for r in valid_rows], dtype=np.float64)

    dxs = gaze[:, 0] - target[:, 0]
    dys = gaze[:, 1] - target[:, 1]

    dx_med = float(np.median(dxs))
    dy_med = float(np.median(dys))

    A_all = fit_affine(gaze, target)

    original_err = center_error_xy(gaze, target)

    global_corr = gaze.copy()
    global_corr[:, 0] -= dx_med
    global_corr[:, 1] -= dy_med
    global_err = center_error_xy(global_corr, target)

    affine_corr = apply_affine(A_all, gaze)
    affine_err = center_error_xy(affine_corr, target)

    summary_rows = []

    for name, err in [
        ("original_all", original_err),
        ("global_offset_corrected_all", global_err),
        ("affine_corrected_all_in_sample", affine_err),
    ]:
        s = summarize(err)
        summary_rows.append({
            "method": name,
            "n": s["n"],
            "center_error_mean_px": s["mean"],
            "center_error_median_px": s["median"],
            "center_error_p95_px": s["p95"],
            "center_error_min_px": s["min"],
            "center_error_max_px": s["max"],
            "center_error_std_px": s["std"],
        })

    loto_rows = run_leave_one_group_out(valid_rows, args.leave_one_mode)

    if loto_rows:
        loto_err_original = [to_float(r["original_center_error_median_px"]) for r in loto_rows]
        loto_err_affine = [to_float(r["loto_affine_center_error_median_px"]) for r in loto_rows]

        s0 = summarize(loto_err_original)
        s1 = summarize(loto_err_affine)

        summary_rows.append({
            "method": f"leave_one_{args.leave_one_mode}_original_group_medians",
            "n": s0["n"],
            "center_error_mean_px": s0["mean"],
            "center_error_median_px": s0["median"],
            "center_error_p95_px": s0["p95"],
            "center_error_min_px": s0["min"],
            "center_error_max_px": s0["max"],
            "center_error_std_px": s0["std"],
        })

        summary_rows.append({
            "method": f"leave_one_{args.leave_one_mode}_affine_group_medians",
            "n": s1["n"],
            "center_error_mean_px": s1["mean"],
            "center_error_median_px": s1["median"],
            "center_error_p95_px": s1["p95"],
            "center_error_min_px": s1["min"],
            "center_error_max_px": s1["max"],
            "center_error_std_px": s1["std"],
        })

    write_csv(output_dir / "loto_affine_summary.csv", loto_rows)
    write_csv(output_dir / "overall_summary.csv", summary_rows)

    scene_frame_dir = Path(args.scene_frame_dir)

    saved_visuals = visualize_samples(
        valid_rows=valid_rows,
        cols=cols,
        scene_frame_dir=scene_frame_dir,
        output_dir=output_dir / "visualization",
        samples_per_group=args.samples_per_group,
        group_mode=args.visualize_group_mode,
        global_dx_med=dx_med,
        global_dy_med=dy_med,
        A_all=A_all,
    )

    write_csv(output_dir / "visualization_index.csv", saved_visuals)

    info = {
        "input_csv": args.input_csv,
        "scene_frame_dir": args.scene_frame_dir,
        "num_valid_rows": len(valid_rows),
        "global_dx_median_px": dx_med,
        "global_dy_median_px": dy_med,
        "affine_matrix_all_target_from_gaze": A_all.tolist(),
        "leave_one_mode": args.leave_one_mode,
        "visualize_group_mode": args.visualize_group_mode,
        "samples_per_group": args.samples_per_group,
    }

    with open(output_dir / "diagnostic_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print("[INFO] Valid rows:", len(valid_rows))
    print("[INFO] Global offset:")
    print(f"  dx_median = {dx_med:.2f} px")
    print(f"  dy_median = {dy_med:.2f} px")

    print("[INFO] In-sample summary:")
    for r in summary_rows:
        print(
            f"  {r['method']}: "
            f"n={r['n']}, "
            f"median={r['center_error_median_px']:.2f}px, "
            f"p95={r['center_error_p95_px']:.2f}px"
        )

    print("[INFO] saved:")
    print(" ", output_dir / "overall_summary.csv")
    print(" ", output_dir / "loto_affine_summary.csv")
    print(" ", output_dir / "visualization_index.csv")
    print(" ", output_dir / "visualization")


if __name__ == "__main__":
    main()