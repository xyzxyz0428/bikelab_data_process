#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_angle_columns(df: pd.DataFrame):
    neutral_cols = [
        "neutral_rel_roll_deg",
        "neutral_rel_pitch_deg",
        "neutral_rel_yaw_deg",
    ]
    back_cols = [
        "back_head_roll_deg",
        "back_head_pitch_deg",
        "back_head_yaw_deg",
    ]

    if all(c in df.columns for c in neutral_cols):
        return neutral_cols, "neutral_rel"
    if all(c in df.columns for c in back_cols):
        return back_cols, "back_head"

    raise ValueError(
        "No angle columns found. Need either neutral_rel_* or back_head_*."
    )


def to_numeric_if_exists(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def print_stats(df: pd.DataFrame, angle_cols, mode_name: str):
    print("\n=== Basic summary ===")
    print(f"rows: {len(df)}")

    if "status" in df.columns:
        print("\nstatus counts:")
        print(df["status"].value_counts(dropna=False).to_string())

    if "num_head_tags" in df.columns:
        print("\nnum_head_tags counts:")
        print(df["num_head_tags"].value_counts(dropna=False).sort_index().to_string())

    if "visible_head_tag_ids" in df.columns:
        print("\nvisible_head_tag_ids counts:")
        print(df["visible_head_tag_ids"].value_counts(dropna=False).to_string())

    print(f"\n=== Angle stats ({mode_name}) ===")
    for c in angle_cols:
        s = df[c].dropna()
        if len(s) == 0:
            print(f"{c}: no valid data")
            continue
        print(
            f"{c}: mean={s.mean():.3f}, std={s.std(ddof=1):.3f}, "
            f"min={s.min():.3f}, max={s.max():.3f}, peak_to_peak={s.max()-s.min():.3f}"
        )

    if "head_rmse_px" in df.columns:
        s = df["head_rmse_px"].dropna()
        if len(s) > 0:
            print("\n=== head_rmse_px stats ===")
            print(
                f"mean={s.mean():.3f}, median={s.median():.3f}, std={s.std(ddof=1):.3f}, "
                f"min={s.min():.3f}, max={s.max():.3f}, p95={s.quantile(0.95):.3f}"
            )


def choose_x_axis(df: pd.DataFrame):
    if "frame_idx" in df.columns:
        return df["frame_idx"].to_numpy(), "frame_idx"
    if "timestamp_ns" in df.columns:
        t = pd.to_numeric(df["timestamp_ns"], errors="coerce").to_numpy()
        t = (t - t[0]) * 1e-9
        return t, "time_s"
    return np.arange(len(df)), "index"


def plot_results(df: pd.DataFrame, angle_cols, mode_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    x, x_label = choose_x_axis(df)

    # -------- figure 1: angles --------
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(3, 1, 1)
    ax2 = fig1.add_subplot(3, 1, 2)
    ax3 = fig1.add_subplot(3, 1, 3)

    axes = [ax1, ax2, ax3]
    names = ["roll", "pitch", "yaw"]

    for ax, col, n in zip(axes, angle_cols, names):
        ax.plot(x, df[col].to_numpy(), linewidth=1.0)
        s = df[col].dropna()
        if len(s) > 0:
            mean = s.mean()
            std = s.std(ddof=1)
            ax.axhline(mean, linestyle="--", linewidth=1.0)
            ax.set_title(f"{mode_name} {n}: mean={mean:.2f}, std={std:.2f}")
        else:
            ax.set_title(f"{mode_name} {n}: no valid data")
        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)

    ax3.set_xlabel(x_label)
    fig1.tight_layout()
    fig1.savefig(out_dir / "angles.png", dpi=150)
    plt.close(fig1)

    # -------- figure 2: rmse + num tags --------
    fig2 = plt.figure(figsize=(12, 6))

    nrows = 1
    if "num_head_tags" in df.columns:
        nrows = 2

    ax_rmse = fig2.add_subplot(nrows, 1, 1)
    if "head_rmse_px" in df.columns:
        ax_rmse.plot(x, df["head_rmse_px"].to_numpy(), linewidth=1.0)
        s = df["head_rmse_px"].dropna()
        if len(s) > 0:
            ax_rmse.axhline(s.mean(), linestyle="--", linewidth=1.0)
            ax_rmse.set_title(
                f"head_rmse_px: mean={s.mean():.2f}, median={s.median():.2f}, p95={s.quantile(0.95):.2f}"
            )
        else:
            ax_rmse.set_title("head_rmse_px: no valid data")
        ax_rmse.set_ylabel("px")
        ax_rmse.grid(True, alpha=0.3)

    if "num_head_tags" in df.columns:
        ax_tags = fig2.add_subplot(nrows, 1, 2)
        ax_tags.plot(x, df["num_head_tags"].to_numpy(), linewidth=1.0)
        ax_tags.set_title("num_head_tags")
        ax_tags.set_ylabel("count")
        ax_tags.set_xlabel(x_label)
        ax_tags.grid(True, alpha=0.3)
    else:
        ax_rmse.set_xlabel(x_label)

    fig2.tight_layout()
    fig2.savefig(out_dir / "rmse_and_tags.png", dpi=150)
    plt.close(fig2)

    # -------- figure 3: histograms --------
    fig3 = plt.figure(figsize=(12, 8))
    ax1 = fig3.add_subplot(2, 2, 1)
    ax2 = fig3.add_subplot(2, 2, 2)
    ax3 = fig3.add_subplot(2, 2, 3)
    ax4 = fig3.add_subplot(2, 2, 4)

    for ax, col, n in zip([ax1, ax2, ax3], angle_cols, ["roll", "pitch", "yaw"]):
        s = df[col].dropna()
        if len(s) > 0:
            ax.hist(s.to_numpy(), bins=40)
            ax.set_title(f"{mode_name} {n} histogram")
            ax.set_xlabel("deg")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)

    if "head_rmse_px" in df.columns:
        s = df["head_rmse_px"].dropna()
        if len(s) > 0:
            ax4.hist(s.to_numpy(), bins=40)
            ax4.set_title("head_rmse_px histogram")
            ax4.set_xlabel("px")
            ax4.set_ylabel("count")
            ax4.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(out_dir / "histograms.png", dpi=150)
    plt.close(fig3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to headpose_output.csv")
    parser.add_argument("--out-dir", default="headpose_analysis", help="output folder")
    parser.add_argument("--only-ok", action="store_true", help="keep only ok==1")
    parser.add_argument("--min-head-tags", type=int, default=0, help="minimum num_head_tags")
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=None,
        help="keep only rows with head_rmse_px <= this threshold",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    df = pd.read_csv(csv_path)

    numeric_cols = [
        "frame_idx",
        "timestamp_ns",
        "ok",
        "num_head_tags",
        "head_rmse_px",
        "cam_head_roll_deg",
        "cam_head_pitch_deg",
        "cam_head_yaw_deg",
        "back_head_roll_deg",
        "back_head_pitch_deg",
        "back_head_yaw_deg",
        "neutral_rel_roll_deg",
        "neutral_rel_pitch_deg",
        "neutral_rel_yaw_deg",
    ]
    df = to_numeric_if_exists(df, numeric_cols)

    if args.only_ok and "ok" in df.columns:
        df = df[df["ok"] == 1].copy()

    if args.min_head_tags > 0 and "num_head_tags" in df.columns:
        df = df[df["num_head_tags"] >= args.min_head_tags].copy()

    if args.max_rmse is not None and "head_rmse_px" in df.columns:
        df = df[df["head_rmse_px"] <= args.max_rmse].copy()

    if len(df) == 0:
        raise RuntimeError("No rows left after filtering.")

    angle_cols, mode_name = pick_angle_columns(df)

    print_stats(df, angle_cols, mode_name)
    plot_results(df, angle_cols, mode_name, out_dir)

    # save filtered csv
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_csv = out_dir / "filtered_data.csv"
    df.to_csv(filtered_csv, index=False)

    print("\nSaved:")
    print(f"  {out_dir / 'angles.png'}")
    print(f"  {out_dir / 'rmse_and_tags.png'}")
    print(f"  {out_dir / 'histograms.png'}")
    print(f"  {filtered_csv}")


if __name__ == "__main__":
    main()