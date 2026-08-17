#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_CLASSES = ["CONE", "PED", "BIC", "CAR", "TRUCK_BUS", "ULTRA_VEHICLE", "UNKNOWN"]


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        return normalize_cols(pd.read_excel(path))
    return normalize_cols(pd.read_csv(path))

def read_detection_parts(input_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No detection part files found in {input_dir} with pattern: {pattern}")

    dfs = []
    for f in files:
        print(f"[INFO] reading detection part: {f.name}")
        dfs.append(read_table(f))

    out = pd.concat(dfs, ignore_index=True)
    out = normalize_cols(out)
    print(f"[INFO] merged detection parts: {len(files)} files, total rows={len(out)}")
    return out
def to_unix_ns_scalar(x) -> Optional[int]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    if abs(v) > 1e17:
        return int(round(v))
    if abs(v) > 1e14:
        return int(round(v * 1e3))
    if abs(v) > 1e11:
        return int(round(v * 1e6))
    if abs(v) > 1e8:
        return int(round(v * 1e9))
    return None


def load_scenarios_relative(path: Path, wanted_ids=None) -> pd.DataFrame:
    df = normalize_cols(pd.read_csv(path))

    colmap = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "scenario_id":
            colmap[c] = "scenario_id"
        elif cl == "scenario_type":
            colmap[c] = "scenario_type"
        elif cl.startswith("initial time"):
            colmap[c] = "initial_time"
        elif cl == "start":
            colmap[c] = "start_rel_s"
        elif cl == "end":
            colmap[c] = "end_rel_s"
        elif cl == "note":
            colmap[c] = "note"
    df = df.rename(columns=colmap)

    required = ["scenario_id", "initial_time", "start_rel_s", "end_rel_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"critical_scenarios.csv missing required columns: {missing}")

    df["scenario_id"] = df["scenario_id"].astype(str)
    df["initial_time"] = pd.to_numeric(df["initial_time"], errors="coerce")
    df["start_rel_s"] = pd.to_numeric(df["start_rel_s"], errors="coerce")
    df["end_rel_s"] = pd.to_numeric(df["end_rel_s"], errors="coerce")
    df = df.dropna(subset=["initial_time", "start_rel_s", "end_rel_s"]).copy()

    df["start_ns"] = ((df["initial_time"] + df["start_rel_s"]) * 1e9).round().astype(np.int64)
    df["end_ns"] = ((df["initial_time"] + df["end_rel_s"]) * 1e9).round().astype(np.int64)

    if wanted_ids:
        wanted = set(str(x) for x in wanted_ids)
        df = df[df["scenario_id"].isin(wanted)].copy()

    return df


def infer_detection_time_col(df: pd.DataFrame) -> str:
    for c in ["bag_time", "header_stamp", "ns"]:
        if c in df.columns:
            return c
    raise ValueError("detections table must contain one of: bag_time, header_stamp, ns")


def extract_obj_class(df: pd.DataFrame) -> pd.Series:
    if "type_name" in df.columns:
        cls = df["type_name"].astype(str).str.strip().str.upper()
        return cls.where(cls.isin(VALID_CLASSES), "UNKNOWN")
    if "text" in df.columns:
        cls = df["text"].astype(str).str.extract(
            r"(CONE|PED|BIC|CAR|TRUCK_BUS|ULTRA_VEHICLE|UNKNOWN)",
            expand=False
        ).fillna("UNKNOWN").str.upper()
        return cls.where(cls.isin(VALID_CLASSES), "UNKNOWN")
    return pd.Series(["UNKNOWN"] * len(df), index=df.index)


def prepare_detections(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_cols(df).copy()

    time_col = infer_detection_time_col(df)
    df["t_unix_ns"] = df[time_col].apply(to_unix_ns_scalar)
    df["obj_class"] = extract_obj_class(df)

    numeric_cols = [
        "pose_x", "pose_y", "pose_z",
        "scale_x", "scale_y", "scale_z",
        "id", "marker_index_in_array", "bag_msg_index",
        "points_count", "colors_count"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if all(c in df.columns for c in ["pose_x", "pose_y", "pose_z"]):
        df["distance_m"] = np.sqrt(df["pose_x"]**2 + df["pose_y"]**2 + df["pose_z"]**2)
    else:
        df["distance_m"] = np.nan

    return df.dropna(subset=["t_unix_ns"]).copy()


def save_class_count_bar(df: pd.DataFrame, out_png: Path, title: str):
    counts = df["obj_class"].value_counts().reindex(VALID_CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Detection count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_class_distance_box(df: pd.DataFrame, out_png: Path, title: str):
    data, labels = [], []
    for cls in VALID_CLASSES:
        vals = pd.to_numeric(df.loc[df["obj_class"] == cls, "distance_m"], errors="coerce").dropna()
        if len(vals) > 0:
            data.append(vals.to_numpy())
            labels.append(cls)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Distance to detection frame origin (m)")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_class_distance_hist(df: pd.DataFrame, out_png: Path, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for cls in VALID_CLASSES:
        vals = pd.to_numeric(df.loc[df["obj_class"] == cls, "distance_m"], errors="coerce").dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.45, label=cls)
            plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_class_timeline(df: pd.DataFrame, out_png: Path, title: str):
    if len(df) == 0:
        return

    t0 = df["t_unix_ns"].min()
    fig, ax = plt.subplots(figsize=(10, 4.5))

    ymap = {cls: i for i, cls in enumerate(VALID_CLASSES)}
    for cls in VALID_CLASSES:
        sub = df[df["obj_class"] == cls]
        if len(sub) == 0:
            continue
        t = (sub["t_unix_ns"] - t0) / 1e9
        y = np.full(len(sub), ymap[cls])
        ax.scatter(t, y, s=8)

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(list(ymap.keys()))
    ax.set_xlabel("Time since scenario start (s)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_spatial_scatter(df: pd.DataFrame, out_png: Path, title: str):
    if not all(c in df.columns for c in ["pose_x", "pose_y"]):
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    plotted = False
    for cls in VALID_CLASSES:
        sub = df[df["obj_class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["pose_x"], sub["pose_y"], s=8, alpha=0.5, label=cls)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("pose_x")
    ax.set_ylabel("pose_y")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def build_summary_tables(df: pd.DataFrame):
    n_time = max(df["t_unix_ns"].nunique(), 1)

    class_summary = (
        df.groupby("obj_class")
        .agg(
            n_detections=("obj_class", "size"),
            n_unique_ids=("id", pd.Series.nunique) if "id" in df.columns else ("obj_class", "size"),
            n_unique_timestamps=("t_unix_ns", pd.Series.nunique),
            median_distance_m=("distance_m", "median"),
            min_distance_m=("distance_m", "min"),
            p95_distance_m=("distance_m", lambda x: np.nanpercentile(pd.to_numeric(x, errors="coerce").dropna(), 95)
                            if pd.to_numeric(x, errors="coerce").dropna().size else np.nan),
        )
        .reset_index()
    )
    class_summary["presence_ratio"] = class_summary["n_unique_timestamps"] / n_time

    top_objects = None
    if "id" in df.columns:
        top_objects = (
            df.groupby(["id", "obj_class"])
            .agg(
                n_detections=("obj_class", "size"),
                n_unique_timestamps=("t_unix_ns", pd.Series.nunique),
                median_distance_m=("distance_m", "median"),
                min_distance_m=("distance_m", "min"),
                first_time_ns=("t_unix_ns", "min"),
                last_time_ns=("t_unix_ns", "max"),
            )
            .reset_index()
            .sort_values(["n_unique_timestamps", "n_detections"], ascending=False)
        )

    return class_summary, top_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=None, help="single detection xlsx/csv")
    ap.add_argument("--detections-dir", default=None, help="folder containing detection part csv files")
    ap.add_argument("--detections-glob", default="perception_info_rviz.xlsx_part*.csv")
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario-ids", nargs="*", default=[])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.detections_dir:
        raw_det = read_detection_parts(Path(args.detections_dir), args.detections_glob)
    elif args.detections:
        raw_det = read_table(Path(args.detections))
    else:
        raise ValueError("Please provide either --detections or --detections-dir")

    det = prepare_detections(raw_det)
    scenarios = load_scenarios_relative(
        Path(args.scenarios),
        wanted_ids=args.scenario_ids if args.scenario_ids else None
    )

    if len(scenarios) == 0:
        raise RuntimeError("No scenarios found after filtering.")

    all_summary_rows = []

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        sname = str(srow["scenario_type"]) if "scenario_type" in srow.index else ""
        start_ns, end_ns = int(srow["start_ns"]), int(srow["end_ns"])

        sub = det[(det["t_unix_ns"] >= start_ns) & (det["t_unix_ns"] <= end_ns)].copy()
        scenario_dir = outdir / f"scenario_{sid}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        sub.to_csv(scenario_dir / f"scenario_{sid}_detections_filtered.csv", index=False)

        if len(sub) == 0:
            print(f"[WARN] scenario {sid}: no detections in range")
            all_summary_rows.append({
                "scenario_id": sid,
                "scenario_type": sname,
                "n_detections": 0,
                "n_classes_present": 0,
                "median_distance_m": np.nan,
            })
            continue

        class_summary, top_objects = build_summary_tables(sub)
        class_summary.to_csv(scenario_dir / f"scenario_{sid}_class_summary.csv", index=False)
        if top_objects is not None:
            top_objects.to_csv(scenario_dir / f"scenario_{sid}_top_objects.csv", index=False)

        save_class_count_bar(
            sub,
            scenario_dir / f"scenario_{sid}_class_count_bar.png",
            f"Scenario {sid} ({sname}): detection counts by class"
        )

        save_class_distance_box(
            sub,
            scenario_dir / f"scenario_{sid}_class_distance_box.png",
            f"Scenario {sid} ({sname}): distance distribution by class"
        )

        save_class_distance_hist(
            sub,
            scenario_dir / f"scenario_{sid}_class_distance_hist.png",
            f"Scenario {sid} ({sname}): distance histogram by class"
        )

        save_class_timeline(
            sub,
            scenario_dir / f"scenario_{sid}_class_timeline.png",
            f"Scenario {sid} ({sname}): detections over time"
        )

        save_spatial_scatter(
            sub,
            scenario_dir / f"scenario_{sid}_spatial_scatter_xy.png",
            f"Scenario {sid} ({sname}): spatial distribution in detection frame"
        )

        all_summary_rows.append({
            "scenario_id": sid,
            "scenario_type": sname,
            "n_detections": len(sub),
            "n_classes_present": int(sub["obj_class"].nunique()),
            "median_distance_m": float(pd.to_numeric(sub["distance_m"], errors="coerce").median()) if "distance_m" in sub.columns else np.nan,
            "start_ns": start_ns,
            "end_ns": end_ns,
        })

        print(f"[OK] scenario {sid} -> {scenario_dir}")

    pd.DataFrame(all_summary_rows).to_csv(outdir / "scenario_detection_overview.csv", index=False)
    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()