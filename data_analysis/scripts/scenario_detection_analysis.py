#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_detection_analysis.py

Scenario-level LiDAR detection / traffic interaction analysis.

Inputs
------
- detections CSV/XLSX exported from your detection table
- critical_scenarios.csv
- optional scenario ids

Expected detection columns
--------------------------
bag_time, header_stamp, ns, id, marker_index_in_array, bag_msg_index, topic,
frame_id, type, type_name, action, action_name, text, mesh_resource,
pose_x, pose_y, pose_z, ori_x, ori_y, ori_z, ori_w,
scale_x, scale_y, scale_z, color_r, color_g, color_b, color_a,
lifetime_sec, frame_locked, points_count, colors_count

Outputs
-------
Per scenario:
- scenario_<id>_detection_counts_by_class.png
- scenario_<id>_closest_object_distance.png
- scenario_<id>_closest_object_per_class_distance.png
- scenario_<id>_bbox_size_distribution.png
- scenario_<id>_key_object_summary.csv
- scenario_<id>_timeline_object_presence.png
"""

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


def to_unix_ns_scalar(x) -> Optional[int]:
    try:
        v = float(x)
    except Exception:
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


def prepare_detection_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # choose timestamp column
    time_col = None
    for c in ["bag_time", "header_stamp", "ns"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found among bag_time / header_stamp / ns")

    df["t_unix_ns"] = df[time_col].apply(to_unix_ns_scalar)

    # class
    if "type_name" in df.columns:
        df["obj_class"] = df["type_name"].astype(str).str.strip().str.upper()
    elif "text" in df.columns:
        df["obj_class"] = df["text"].astype(str).str.extract(r"(CONE|PED|BIC|CAR|TRUCK_BUS|ULTRA_VEHICLE|UNKNOWN)", expand=False).fillna("UNKNOWN")
    else:
        df["obj_class"] = "UNKNOWN"

    df["obj_class"] = df["obj_class"].where(df["obj_class"].isin(VALID_CLASSES), "UNKNOWN")

    for c in ["pose_x", "pose_y", "pose_z", "scale_x", "scale_y", "scale_z", "id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["distance_m"] = np.sqrt(df["pose_x"] ** 2 + df["pose_y"] ** 2 + df["pose_z"] ** 2)
    return df.dropna(subset=["t_unix_ns"]).copy()


def load_scenarios(path: Path, wanted_ids=None) -> pd.DataFrame:
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
            colmap[c] = "start"
        elif cl == "end":
            colmap[c] = "end"
        elif cl == "note":
            colmap[c] = "note"
    df = df.rename(columns=colmap)
    df["start_ns"] = df["start"].apply(to_unix_ns_scalar)
    df["end_ns"] = df["end"].apply(to_unix_ns_scalar)
    df = df.dropna(subset=["start_ns", "end_ns"]).copy()
    df["scenario_id"] = df["scenario_id"].astype(str)
    if wanted_ids:
        wanted = set(str(x) for x in wanted_ids)
        df = df[df["scenario_id"].isin(wanted)].copy()
    return df


def make_presence_timeline(df: pd.DataFrame, scenario_start_ns: int, bin_s: float = 0.5) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    t_rel_s = (df["t_unix_ns"] - scenario_start_ns) / 1e9
    df = df.copy()
    df["t_bin"] = np.floor(t_rel_s / bin_s) * bin_s
    pres = (
        df.groupby(["t_bin", "obj_class"])
        .size()
        .reset_index(name="count")
        .pivot(index="t_bin", columns="obj_class", values="count")
        .fillna(0)
    )
    pres = (pres > 0).astype(int)
    return pres


def plot_presence_timeline(pres: pd.DataFrame, out_path: Path, title: str):
    if len(pres) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, max(3.5, 0.45 * len(pres.columns))))
    arr = pres.T.values
    ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(pres.columns)))
    ax.set_yticklabels(pres.columns)
    xt = np.linspace(0, max(0, len(pres.index) - 1), min(10, len(pres.index))).astype(int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{pres.index[i]:.1f}" for i in xt])
    ax.set_xlabel("Time since scenario start (s)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario-ids", nargs="*", default=["5", "7"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    det = prepare_detection_df(read_table(Path(args.detections)))
    scenarios = load_scenarios(Path(args.scenarios), wanted_ids=args.scenario_ids)

    det = det[det["obj_class"].isin(VALID_CLASSES)].copy()

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        sname = srow.get("scenario_type", "")
        start_ns = int(srow["start_ns"])
        end_ns = int(srow["end_ns"])

        ds = det[(det["t_unix_ns"] >= start_ns) & (det["t_unix_ns"] <= end_ns)].copy()
        if len(ds) == 0:
            print(f"[WARN] scenario {sid}: no detections in range")
            continue

        scenario_dir = outdir / f"scenario_{sid}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # counts by class
        counts = ds["obj_class"].value_counts().reindex(VALID_CLASSES, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Detection count")
        ax.set_title(f"Scenario {sid} ({sname}): detection counts by class")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(scenario_dir / f"scenario_{sid}_detection_counts_by_class.png", dpi=300)
        plt.close(fig)

        # closest object over time (global)
        ds_sorted = ds.sort_values("t_unix_ns")
        closest = ds_sorted.groupby("t_unix_ns")["distance_m"].min().reset_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot((closest["t_unix_ns"] - start_ns) / 1e9, closest["distance_m"])
        ax.set_xlabel("Time since scenario start (s)")
        ax.set_ylabel("Closest detected object distance (m)")
        ax.set_title(f"Scenario {sid} ({sname}): closest object distance")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_dir / f"scenario_{sid}_closest_object_distance.png", dpi=300)
        plt.close(fig)

        # closest per class
        fig, ax = plt.subplots(figsize=(10, 5))
        for cls in VALID_CLASSES:
            sub = ds[ds["obj_class"] == cls]
            if len(sub) == 0:
                continue
            tmp = sub.groupby("t_unix_ns")["distance_m"].min().reset_index()
            ax.plot((tmp["t_unix_ns"] - start_ns) / 1e9, tmp["distance_m"], label=cls)
        ax.set_xlabel("Time since scenario start (s)")
        ax.set_ylabel("Closest object distance per class (m)")
        ax.set_title(f"Scenario {sid} ({sname}): closest distance per class")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_dir / f"scenario_{sid}_closest_object_per_class_distance.png", dpi=300)
        plt.close(fig)

        # box size distribution
        if {"scale_x", "scale_y", "scale_z"}.issubset(ds.columns):
            ds["bbox_volume"] = ds["scale_x"] * ds["scale_y"] * ds["scale_z"]
            data = []
            labels = []
            for cls in VALID_CLASSES:
                vals = pd.to_numeric(ds.loc[ds["obj_class"] == cls, "bbox_volume"], errors="coerce").dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    labels.append(cls)
            if data:
                fig, ax = plt.subplots(figsize=(10, 4.5))
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.set_ylabel("Bounding-box volume (m³)")
                ax.set_title(f"Scenario {sid} ({sname}): bounding-box size distribution")
                ax.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                plt.savefig(scenario_dir / f"scenario_{sid}_bbox_size_distribution.png", dpi=300)
                plt.close(fig)

        # key-object summary
        key_rows = []
        for cls in VALID_CLASSES:
            sub = ds[ds["obj_class"] == cls].copy()
            if len(sub) == 0:
                continue
            key_rows.append({
                "scenario_id": sid,
                "scenario_type": sname,
                "obj_class": cls,
                "n_detections": len(sub),
                "n_unique_ids": sub["id"].nunique() if "id" in sub.columns else np.nan,
                "min_distance_m": float(sub["distance_m"].min()),
                "median_distance_m": float(sub["distance_m"].median()),
                "p95_distance_m": float(np.nanpercentile(sub["distance_m"], 95)),
            })
        if key_rows:
            pd.DataFrame(key_rows).to_csv(scenario_dir / f"scenario_{sid}_key_object_summary.csv", index=False)

        # timeline object presence
        pres = make_presence_timeline(ds, start_ns, bin_s=0.5)
        plot_presence_timeline(
            pres,
            scenario_dir / f"scenario_{sid}_timeline_object_presence.png",
            f"Scenario {sid} ({sname}): object presence by class"
        )

        print(f"[OK] scenario {sid} -> {scenario_dir}")


if __name__ == "__main__":
    main()
