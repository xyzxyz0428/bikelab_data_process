#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import numpy as np


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


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
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


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "std": np.nan,
        }
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.quantile(vals, 0.95)),
        "std": float(np.std(vals)),
    }


def point_to_polygon_distance(px, py, poly):
    """
    poly: Nx2 polygon.
    Return 0 if inside, otherwise shortest distance to polygon edges.
    """
    p = np.array([px, py], dtype=float)

    # inside test: ray casting
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i

    if inside:
        return 0.0, True

    dmin = np.inf
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
        t = max(0.0, min(1.0, t))
        q = a + t * ab
        d = np.linalg.norm(p - q)
        dmin = min(dmin, d)

    return float(dmin), False


def get_columns(rows):
    keys = rows[0].keys()

    def find(candidates):
        for c in candidates:
            if c in keys:
                return c
        raise KeyError(f"Cannot find any of columns: {candidates}")

    return {
        "window_id": find(["window_id"]),
        "tag_id": find(["tag_id"]),
        "gaze_x": find(["gaze_x", "tobii_x", "Gaze point X", "Fixation point X"]),
        "gaze_y": find(["gaze_y", "tobii_y", "Gaze point Y", "Fixation point Y"]),
        "center_x": find(["tag_center_x", "center_x", "tag_center_u", "tag_u"]),
        "center_y": find(["tag_center_y", "center_y", "tag_center_v", "tag_v"]),
    }


def try_get_polygon(row):
    """
    Accept common polygon column names:
      tag_corner_0_x, tag_corner_0_y ...
      corner0_x, corner0_y ...
      poly_x0, poly_y0 ...
    """
    candidates = [
        ("tag_corner_0_x", "tag_corner_0_y"),
        ("corner0_x", "corner0_y"),
        ("poly_x0", "poly_y0"),
    ]

    for x0, y0 in candidates:
        if x0 in row and y0 in row:
            prefix_type = (x0, y0)
            break
    else:
        return None

    poly = []
    if prefix_type[0].startswith("tag_corner"):
        for i in range(4):
            x = to_float(row.get(f"tag_corner_{i}_x"))
            y = to_float(row.get(f"tag_corner_{i}_y"))
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            poly.append([x, y])
    elif prefix_type[0].startswith("corner"):
        for i in range(4):
            x = to_float(row.get(f"corner{i}_x"))
            y = to_float(row.get(f"corner{i}_y"))
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            poly.append([x, y])
    else:
        for i in range(4):
            x = to_float(row.get(f"poly_x{i}"))
            y = to_float(row.get(f"poly_y{i}"))
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            poly.append([x, y])

    return np.asarray(poly, dtype=float)


def center_error(gx, gy, cx, cy):
    return float(np.hypot(gx - cx, gy - cy))


def fit_affine(gaze_pts, target_pts):
    """
    Fit affine transform:
      [x_target, y_target] = A * [x_gaze, y_gaze, 1]
    """
    G = np.asarray(gaze_pts, dtype=float)
    T = np.asarray(target_pts, dtype=float)

    X = np.column_stack([G[:, 0], G[:, 1], np.ones(len(G))])
    ax, *_ = np.linalg.lstsq(X, T[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(X, T[:, 1], rcond=None)

    A = np.vstack([ax, ay])
    return A


def apply_affine(A, gaze_pts):
    G = np.asarray(gaze_pts, dtype=float)
    X = np.column_stack([G[:, 0], G[:, 1], np.ones(len(G))])
    return X @ A.T


def compute_metrics(rows, cols, correction_name, corrected_xy):
    out = []
    center_errs = []
    poly_dists = []
    inside = []

    for r, (gx, gy) in zip(rows, corrected_xy):
        cx = to_float(r[cols["center_x"]])
        cy = to_float(r[cols["center_y"]])

        if not all(np.isfinite(v) for v in [gx, gy, cx, cy]):
            continue

        ce = center_error(gx, gy, cx, cy)
        center_errs.append(ce)

        poly = try_get_polygon(r)
        if poly is not None:
            pd, ins = point_to_polygon_distance(gx, gy, poly)
            poly_dists.append(pd)
            inside.append(ins)

        rr = dict(r)
        rr[f"{correction_name}_gaze_x"] = gx
        rr[f"{correction_name}_gaze_y"] = gy
        rr[f"{correction_name}_center_error_px"] = ce

        if poly is not None:
            rr[f"{correction_name}_distance_to_polygon_px"] = pd
            rr[f"{correction_name}_inside_tag_polygon"] = int(ins)

        out.append(rr)

    s_center = summarize(center_errs)
    s_poly = summarize(poly_dists)
    inside_rate = float(np.mean(inside)) if inside else np.nan

    return out, {
        "method": correction_name,
        "center_error_px_mean": s_center["mean"],
        "center_error_px_median": s_center["median"],
        "center_error_px_p95": s_center["p95"],
        "distance_to_polygon_px_mean": s_poly["mean"],
        "distance_to_polygon_px_median": s_poly["median"],
        "distance_to_polygon_px_p95": s_poly["p95"],
        "inside_rate": inside_rate,
        "n": s_center["n"],
    }


def select_stable_subset(rows, cols, keep_fraction):
    """
    For each window, keep rows whose gaze point is closest to that window's median gaze.
    This avoids selecting based on tag-center error.
    """
    groups = {}
    for r in rows:
        wid = r[cols["window_id"]]
        groups.setdefault(wid, []).append(r)

    selected = []
    for wid, rs in groups.items():
        pts = []
        valid_rs = []
        for r in rs:
            gx = to_float(r[cols["gaze_x"]])
            gy = to_float(r[cols["gaze_y"]])
            if np.isfinite(gx) and np.isfinite(gy):
                pts.append([gx, gy])
                valid_rs.append(r)

        if len(valid_rs) == 0:
            continue

        pts = np.asarray(pts)
        med = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - med[None, :], axis=1)

        k = max(1, int(round(len(valid_rs) * keep_fraction)))
        idx = np.argsort(dist)[:k]

        selected.extend([valid_rs[i] for i in idx])

    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--stable-keep-fraction", type=float, default=0.3)
    args = ap.parse_args()

    rows = read_csv(args.input_csv)
    if not rows:
        raise RuntimeError("Empty input CSV.")

    cols = get_columns(rows)

    valid_rows = []
    gaze_pts = []
    target_pts = []
    dxs = []
    dys = []

    for r in rows:
        gx = to_float(r[cols["gaze_x"]])
        gy = to_float(r[cols["gaze_y"]])
        cx = to_float(r[cols["center_x"]])
        cy = to_float(r[cols["center_y"]])

        if all(np.isfinite(v) for v in [gx, gy, cx, cy]):
            valid_rows.append(r)
            gaze_pts.append([gx, gy])
            target_pts.append([cx, cy])
            dxs.append(gx - cx)
            dys.append(gy - cy)

    gaze_pts = np.asarray(gaze_pts, dtype=float)
    target_pts = np.asarray(target_pts, dtype=float)

    dx_med = float(np.median(dxs))
    dy_med = float(np.median(dys))

    original_xy = gaze_pts.copy()
    global_xy = gaze_pts.copy()
    global_xy[:, 0] -= dx_med
    global_xy[:, 1] -= dy_med

    A = fit_affine(gaze_pts, target_pts)
    affine_xy = apply_affine(A, gaze_pts)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_summary = []

    original_rows, s0 = compute_metrics(valid_rows, cols, "original", original_xy)
    global_rows, s1 = compute_metrics(valid_rows, cols, "global_offset_corrected", global_xy)
    affine_rows, s2 = compute_metrics(valid_rows, cols, "affine_corrected", affine_xy)

    all_summary.extend([s0, s1, s2])

    stable_rows = select_stable_subset(valid_rows, cols, args.stable_keep_fraction)

    stable_gaze = []
    stable_target = []
    for r in stable_rows:
        stable_gaze.append([
            to_float(r[cols["gaze_x"]]),
            to_float(r[cols["gaze_y"]]),
        ])
        stable_target.append([
            to_float(r[cols["center_x"]]),
            to_float(r[cols["center_y"]]),
        ])

    stable_gaze = np.asarray(stable_gaze, dtype=float)

    _, s3 = compute_metrics(stable_rows, cols, "stable_subset_original", stable_gaze)
    all_summary.append(s3)

    out_rows = []
    for r0, r1, r2 in zip(original_rows, global_rows, affine_rows):
        rr = dict(r0)
        for k, v in r1.items():
            if k not in rr:
                rr[k] = v
        for k, v in r2.items():
            if k not in rr:
                rr[k] = v
        out_rows.append(rr)

    write_csv(output_prefix.with_name(output_prefix.name + "_diagnostic_rows.csv"), out_rows)
    write_csv(output_prefix.with_name(output_prefix.name + "_summary.csv"), all_summary)

    info = {
        "input_csv": args.input_csv,
        "num_valid_rows": len(valid_rows),
        "global_dx_median_px": dx_med,
        "global_dy_median_px": dy_med,
        "affine_matrix_target_from_gaze": A.tolist(),
        "stable_keep_fraction": args.stable_keep_fraction,
        "num_stable_rows": len(stable_rows),
    }

    with open(output_prefix.with_name(output_prefix.name + "_info.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(info, f, indent=2)

    print("[INFO] Global offset:")
    print(f"  dx_median = {dx_med:.2f} px")
    print(f"  dy_median = {dy_med:.2f} px")
    print("[INFO] Affine matrix target_from_gaze:")
    print(A)
    print("[INFO] Summary:")
    for s in all_summary:
        print(
            f"  {s['method']}: "
            f"n={s['n']}, "
            f"center_median={s['center_error_px_median']:.2f}px, "
            f"center_p95={s['center_error_px_p95']:.2f}px, "
            f"poly_median={s['distance_to_polygon_px_median']:.2f}px, "
            f"inside={s['inside_rate']}"
        )

    print("[INFO] saved:")
    print(" ", output_prefix.with_name(output_prefix.name + "_diagnostic_rows.csv"))
    print(" ", output_prefix.with_name(output_prefix.name + "_summary.csv"))
    print(" ", output_prefix.with_name(output_prefix.name + "_info.json"))


if __name__ == "__main__":
    main()