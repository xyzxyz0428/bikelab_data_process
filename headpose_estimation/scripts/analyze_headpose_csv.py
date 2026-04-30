#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


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


def to_float(v, default=np.nan):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def summarize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "peak_to_peak": float(np.max(arr) - np.min(arr)),
    }


def print_stats(name, vals):
    s = summarize(vals)
    if s is None:
        print(f"{name}: no valid values")
        return
    print(
        f"{name}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
        f"min={s['min']:.3f}, max={s['max']:.3f}, "
        f"peak_to_peak={s['peak_to_peak']:.3f}, p95={s['p95']:.3f}"
    )


def save_filtered_csv(rows, out_csv):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--only-ok", action="store_true")
    ap.add_argument("--only-head-quality-ok", action="store_true")
    ap.add_argument("--min-head-tags", type=int, default=0)
    ap.add_argument("--max-rmse", type=float, default=None)
    args = ap.parse_args()

    rows = read_csv_dicts(args.csv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.csv).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered = []
    for r in rows:
        if args.only_ok and to_int(r.get("ok"), 0) != 1:
            continue

        if args.only_head_quality_ok and to_int(r.get("head_quality_ok"), 0) != 1:
            continue

        num_head_tags = to_int(r.get("num_head_tags"), 0)
        if num_head_tags < args.min_head_tags:
            continue

        if args.max_rmse is not None:
            rmse = to_float(r.get("head_rmse_px"))
            if np.isnan(rmse) or rmse > args.max_rmse:
                continue

        filtered.append(r)

    print("=== Basic summary ===")
    print(f"rows: {len(filtered)}")

    # status counts
    status_counts = {}
    for r in filtered:
        s = r.get("status", "")
        status_counts[s] = status_counts.get(s, 0) + 1
    print("\nstatus counts:")
    for k, v in status_counts.items():
        print(f"{k}: {v}")

    # num_head_tags counts
    nh_counts = {}
    for r in filtered:
        nh = to_int(r.get("num_head_tags"), 0)
        nh_counts[nh] = nh_counts.get(nh, 0) + 1
    print("\nnum_head_tags counts:")
    for k, v in sorted(nh_counts.items()):
        print(f"{k}: {v}")

    # visible_head_tag_ids counts
    vis_counts = {}
    for r in filtered:
        vis = r.get("visible_head_tag_ids", "")
        vis_counts[vis] = vis_counts.get(vis, 0) + 1
    print("\nvisible_head_tag_ids counts:")
    for k, v in sorted(vis_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"{k}: {v}")

    # angle stats
    roll = [to_float(r.get("back_head_roll_deg")) for r in filtered]
    pitch = [to_float(r.get("back_head_pitch_deg")) for r in filtered]
    yaw = [to_float(r.get("back_head_yaw_deg")) for r in filtered]
    rmse = [to_float(r.get("head_rmse_px")) for r in filtered]

    print("\n=== Angle stats (back_head) ===")
    print_stats("back_head_roll_deg", roll)
    print_stats("back_head_pitch_deg", pitch)
    print_stats("back_head_yaw_deg", yaw)

    print("\n=== head_rmse_px stats ===")
    print_stats("head_rmse_px", rmse)

    # save filtered csv
    filtered_csv = out_dir / "filtered_data.csv"
    save_filtered_csv(filtered, filtered_csv)

    # plots
    x = np.arange(len(filtered))

    # angles plot
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, roll, label="roll")
    plt.plot(x, pitch, label="pitch")
    plt.plot(x, yaw, label="yaw")
    plt.xlabel("filtered frame index")
    plt.ylabel("deg")
    plt.title("Head pose angles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "angles.png", dpi=150)
    plt.close(fig)

    # rmse and num tags
    fig = plt.figure(figsize=(12, 6))
    plt.plot(x, rmse, label="head_rmse_px")
    plt.plot(x, [to_int(r.get("num_head_tags"), 0) for r in filtered], label="num_head_tags")
    plt.xlabel("filtered frame index")
    plt.title("RMSE and number of visible head tags")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_and_tags.png", dpi=150)
    plt.close(fig)

    # histograms
    for name, vals in [
        ("roll", roll),
        ("pitch", pitch),
        ("yaw", yaw),
        ("rmse", rmse),
    ]:
        fig = plt.figure(figsize=(8, 5))
        vals_arr = np.asarray(vals, dtype=np.float64)
        vals_arr = vals_arr[np.isfinite(vals_arr)]
        if len(vals_arr) > 0:
            plt.hist(vals_arr, bins=30)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{name}.png", dpi=150)
        plt.close(fig)

    print("\nSaved:")
    print(f"  {out_dir / 'angles.png'}")
    print(f"  {out_dir / 'rmse_and_tags.png'}")
    print(f"  {out_dir / 'filtered_data.csv'}")


if __name__ == "__main__":
    main()