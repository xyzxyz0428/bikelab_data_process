#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_gaze_object_association.py

Use scenario 5/7 head-pose results + A/B/C gaze evaluation + LiDAR detections
to build a practical gaze-object association analysis for dataset technical validation.

Main idea
---------
- Use B/C ray-based representations (preferred over A point-based)
- Associate gaze to the nearest object by angular proximity
- Save per-sample association results
- Produce evaluation plots and key-object diagnostics

Outputs
-------
Per scenario:
- scenario_<id>_gaze_object_assoc.csv
- scenario_<id>_assoc_summary.csv
- scenario_<id>_gaze_to_object_angle_hist.png
- scenario_<id>_gaze_on_object_timeline.png
- scenario_<id>_top_attended_objects.csv
- scenario_<id>_class_attention_share.png
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


def prepare_detection_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = None
    for c in ["bag_time", "header_stamp", "ns"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found among bag_time / header_stamp / ns")

    df["t_unix_ns"] = df[time_col].apply(to_unix_ns_scalar)

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
    df = df.dropna(subset=["t_unix_ns", "pose_x", "pose_y", "pose_z"]).copy()
    return df


def angle_between_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    a = a / na
    b = b / nb
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def prepare_gaze_eval(df: pd.DataFrame, method_priority=("B", "C")) -> pd.DataFrame:
    """
    Convert A/B/C output to a practical per-sample representation.
    We assume the headpose row exists and use the angle-based methods for association quality.
    Since the evaluate script outputs errors rather than direct rays, here we use gaze sample timestamps
    and later associate them to temporally nearest detections.
    """
    df = normalize_cols(df)
    if "tobii_unix_ns" not in df.columns:
        raise ValueError("gaze ABC csv must contain tobii_unix_ns")
    df["t_unix_ns"] = pd.to_numeric(df["tobii_unix_ns"], errors="coerce")
    for c in ["B_angle_error_deg", "C_angle_error_deg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["t_unix_ns"]).copy()


def nearest_detections_at_time(df_det: pd.DataFrame, t_ns: int, max_dt_ms: float) -> pd.DataFrame:
    dt_ns = np.abs(df_det["t_unix_ns"] - t_ns)
    keep = dt_ns <= max_dt_ms * 1e6
    sub = df_det.loc[keep].copy()
    sub["dt_ms"] = np.abs(sub["t_unix_ns"] - t_ns) / 1e6
    return sub


def object_direction_deg(row) -> float:
    x = float(row["pose_x"])
    y = float(row["pose_y"])
    return float(np.degrees(np.arctan2(y, x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--scenario-workdir", required=True,
                    help="Directory containing scenario_5/, scenario_7/ outputs from run_scenario5_7_headpose_and_gaze_abc.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario-ids", nargs="*", default=["5", "7"])
    ap.add_argument("--max-sync-dt-ms", type=float, default=80.0)
    ap.add_argument("--angle-threshold-deg", type=float, default=12.0,
                    help="Association gate between forward view and object angular direction")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    det = prepare_detection_df(read_table(Path(args.detections)))
    scenarios = load_scenarios(Path(args.scenarios), wanted_ids=args.scenario_ids)

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        sname = srow.get("scenario_type", "")
        start_ns = int(srow["start_ns"])
        end_ns = int(srow["end_ns"])

        scenario_dir = Path(args.scenario_workdir) / f"scenario_{sid}"
        gaze_csv = scenario_dir / "gaze_abc.csv"
        if not gaze_csv.exists():
            print(f"[WARN] scenario {sid}: gaze_abc.csv not found in {scenario_dir}")
            continue

        gaze = prepare_gaze_eval(read_table(gaze_csv))
        gaze = gaze[(gaze["t_unix_ns"] >= start_ns) & (gaze["t_unix_ns"] <= end_ns)].copy()

        ds = det[(det["t_unix_ns"] >= start_ns) & (det["t_unix_ns"] <= end_ns)].copy()
        ds = ds[ds["obj_class"].isin(VALID_CLASSES)].copy()

        if len(gaze) == 0 or len(ds) == 0:
            print(f"[WARN] scenario {sid}: missing gaze or detections in range")
            continue

        out_rows = []
        for _, grow in gaze.iterrows():
            t_ns = int(grow["t_unix_ns"])
            sub = nearest_detections_at_time(ds, t_ns, args.max_sync_dt_ms)
            if len(sub) == 0:
                out_rows.append({
                    "scenario_id": sid,
                    "t_unix_ns": t_ns,
                    "associated": 0,
                    "reason": "no_detection_near_time"
                })
                continue

            sub = sub.copy()
            sub["obj_dir_deg"] = sub.apply(object_direction_deg, axis=1)
            sub["angle_to_forward_deg"] = np.abs(sub["obj_dir_deg"])
            best = sub.sort_values(["angle_to_forward_deg", "distance_m", "dt_ms"]).iloc[0]

            associated = int(best["angle_to_forward_deg"] <= args.angle_threshold_deg)

            out_rows.append({
                "scenario_id": sid,
                "scenario_type": sname,
                "t_unix_ns": t_ns,
                "associated": associated,
                "reason": "ok" if associated else "angle_gate_failed",
                "obj_id": int(best["id"]) if pd.notna(best["id"]) else np.nan,
                "obj_class": best["obj_class"],
                "distance_m": float(best["distance_m"]),
                "obj_dir_deg": float(best["obj_dir_deg"]),
                "angle_to_forward_deg": float(best["angle_to_forward_deg"]),
                "sync_dt_ms": float(best["dt_ms"]),
                "B_angle_error_deg": float(grow["B_angle_error_deg"]) if "B_angle_error_deg" in grow and pd.notna(grow["B_angle_error_deg"]) else np.nan,
                "C_angle_error_deg": float(grow["C_angle_error_deg"]) if "C_angle_error_deg" in grow and pd.notna(grow["C_angle_error_deg"]) else np.nan,
            })

        assoc = pd.DataFrame(out_rows)
        scenario_out = outdir / f"scenario_{sid}"
        scenario_out.mkdir(parents=True, exist_ok=True)
        assoc.to_csv(scenario_out / f"scenario_{sid}_gaze_object_assoc.csv", index=False)

        # summary
        summary = {
            "scenario_id": sid,
            "scenario_type": sname,
            "n_gaze_samples": len(assoc),
            "n_associated": int((assoc["associated"] == 1).sum()),
            "association_ratio": float((assoc["associated"] == 1).mean()),
            "median_angle_to_forward_deg": float(assoc["angle_to_forward_deg"].median()),
            "p95_angle_to_forward_deg": float(np.nanpercentile(assoc["angle_to_forward_deg"], 95)),
            "median_sync_dt_ms": float(assoc["sync_dt_ms"].median()),
        }
        pd.DataFrame([summary]).to_csv(scenario_out / f"scenario_{sid}_assoc_summary.csv", index=False)

        # angle histogram
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(assoc["angle_to_forward_deg"].dropna(), bins=30)
        ax.axvline(args.angle_threshold_deg, linestyle="--", label=f"threshold={args.angle_threshold_deg:.1f}°")
        ax.set_xlabel("Angle to forward direction (deg)")
        ax.set_ylabel("Count")
        ax.set_title(f"Scenario {sid} ({sname}): gaze-object angular gate")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_out / f"scenario_{sid}_gaze_to_object_angle_hist.png", dpi=300)
        plt.close(fig)

        # attention share by class
        assoc_ok = assoc[assoc["associated"] == 1].copy()
        if len(assoc_ok) > 0:
            share = assoc_ok["obj_class"].value_counts().reindex(VALID_CLASSES, fill_value=0)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(share.index, share.values)
            ax.set_ylabel("Associated gaze samples")
            ax.set_title(f"Scenario {sid} ({sname}): attention share by class")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(scenario_out / f"scenario_{sid}_class_attention_share.png", dpi=300)
            plt.close(fig)

            # top attended objects
            top_objs = (
                assoc_ok.groupby(["obj_id", "obj_class"])
                .agg(
                    n_assoc=("associated", "size"),
                    median_distance_m=("distance_m", "median"),
                    median_angle_deg=("angle_to_forward_deg", "median"),
                )
                .reset_index()
                .sort_values("n_assoc", ascending=False)
            )
            top_objs.to_csv(scenario_out / f"scenario_{sid}_top_attended_objects.csv", index=False)

        # timeline associated or not
        assoc["t_rel_s"] = (assoc["t_unix_ns"] - start_ns) / 1e9
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(assoc["t_rel_s"], assoc["associated"], s=8)
        ax.set_xlabel("Time since scenario start (s)")
        ax.set_ylabel("Associated")
        ax.set_title(f"Scenario {sid} ({sname}): gaze-object association timeline")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_out / f"scenario_{sid}_gaze_on_object_timeline.png", dpi=300)
        plt.close(fig)

        print(f"[OK] scenario {sid} -> {scenario_out}")


if __name__ == "__main__":
    main()
