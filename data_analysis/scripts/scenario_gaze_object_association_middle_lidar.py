#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_gaze_object_association_middle_lidar.py

Association when gaze_repr.csv and LiDAR detections are both in middle_lidar frame.
No dynamic headpose is used here.
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


def normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v / n


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_vec(a)
    b = normalize_vec(b)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


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


def prepare_detection_df(df: pd.DataFrame) -> pd.DataFrame:
    time_col = None
    for c in ["bag_time", "header_stamp", "ns"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Need one of bag_time / header_stamp / ns")

    df = df.copy()
    df["t_unix_ns"] = df[time_col].apply(to_unix_ns_scalar)

    if "type_name" in df.columns:
        df["obj_class"] = df["type_name"].astype(str).str.strip().str.upper()
    elif "text" in df.columns:
        df["obj_class"] = df["text"].astype(str).str.extract(
            r"(CONE|PED|BIC|CAR|TRUCK_BUS|ULTRA_VEHICLE|UNKNOWN)", expand=False
        ).fillna("UNKNOWN").str.upper()
    else:
        df["obj_class"] = "UNKNOWN"

    df["obj_class"] = df["obj_class"].where(df["obj_class"].isin(VALID_CLASSES), "UNKNOWN")

    for c in ["pose_x", "pose_y", "pose_z", "id", "scale_x", "scale_y", "scale_z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["distance_m"] = np.sqrt(df["pose_x"]**2 + df["pose_y"]**2 + df["pose_z"]**2)

    if all(c in df.columns for c in ["scale_x", "scale_y", "scale_z"]):
        df["obj_radius_m"] = 0.5 * np.sqrt(df["scale_x"]**2 + df["scale_y"]**2 + df["scale_z"]**2)
    else:
        df["obj_radius_m"] = np.nan

    return df.dropna(subset=["t_unix_ns", "pose_x", "pose_y", "pose_z"]).copy()


def nearest_detections_at_time(df_det: pd.DataFrame, t_ns: int, max_dt_ms: float) -> pd.DataFrame:
    sub = df_det.loc[np.abs(df_det["t_unix_ns"] - t_ns) <= max_dt_ms * 1e6].copy()
    sub["sync_dt_ms"] = np.abs(sub["t_unix_ns"] - t_ns) / 1e6
    return sub


def associate_method_A(gaze: pd.DataFrame, det: pd.DataFrame, max_sync_dt_ms: float, point_margin_m: float, frame_suffix: str = "M") -> pd.DataFrame:
    out_rows = []
    gx = f"gaze_A_point_{frame_suffix}_x"
    gy = f"gaze_A_point_{frame_suffix}_y"
    gz = f"gaze_A_point_{frame_suffix}_z"
    for _, grow in gaze.iterrows():
        t_ns = int(grow["t_unix_ns"])
        det_sub = nearest_detections_at_time(det, t_ns, max_sync_dt_ms)
        if len(det_sub) == 0:
            out_rows.append({"t_unix_ns": t_ns, "associated": 0, "reason": "no_detection_near_time"})
            continue
        gp = np.array([grow.get(gx, np.nan), grow.get(gy, np.nan), grow.get(gz, np.nan)], dtype=float)
        if not np.all(np.isfinite(gp)):
            out_rows.append({"t_unix_ns": t_ns, "associated": 0, "reason": "invalid_gaze_point"})
            continue
        det_sub = det_sub.copy()
        det_sub["point_dist_m"] = np.sqrt((det_sub["pose_x"] - gp[0])**2 + (det_sub["pose_y"] - gp[1])**2 + (det_sub["pose_z"] - gp[2])**2)
        det_sub["point_gate_m"] = det_sub["obj_radius_m"].fillna(0.0) + point_margin_m
        det_sub = det_sub.sort_values(["point_dist_m", "distance_m", "sync_dt_ms"])
        best = det_sub.iloc[0]
        assoc = int(best["point_dist_m"] <= best["point_gate_m"])
        out_rows.append({
            "t_unix_ns": t_ns, "associated": assoc, "reason": "ok" if assoc else "point_gate_failed",
            "obj_id": best.get("id", np.nan), "obj_class": best["obj_class"], "distance_m": best["distance_m"],
            "point_dist_m": best["point_dist_m"], "point_gate_m": best["point_gate_m"], "sync_dt_ms": best["sync_dt_ms"],
        })
    return pd.DataFrame(out_rows)


def associate_method_ray(gaze: pd.DataFrame, det: pd.DataFrame, method_prefix: str, max_sync_dt_ms: float, angle_threshold_deg: float, frame_suffix: str = "M") -> pd.DataFrame:
    out_rows = []
    ox = f"gaze_{method_prefix}_ray_origin_{frame_suffix}_x"
    oy = f"gaze_{method_prefix}_ray_origin_{frame_suffix}_y"
    oz = f"gaze_{method_prefix}_ray_origin_{frame_suffix}_z"
    dx = f"gaze_{method_prefix}_ray_dir_{frame_suffix}_x"
    dy = f"gaze_{method_prefix}_ray_dir_{frame_suffix}_y"
    dz = f"gaze_{method_prefix}_ray_dir_{frame_suffix}_z"
    for _, grow in gaze.iterrows():
        t_ns = int(grow["t_unix_ns"])
        det_sub = nearest_detections_at_time(det, t_ns, max_sync_dt_ms)
        if len(det_sub) == 0:
            out_rows.append({"t_unix_ns": t_ns, "associated": 0, "reason": "no_detection_near_time"})
            continue
        ray_o = np.array([grow.get(ox, np.nan), grow.get(oy, np.nan), grow.get(oz, np.nan)], dtype=float)
        ray_d = np.array([grow.get(dx, np.nan), grow.get(dy, np.nan), grow.get(dz, np.nan)], dtype=float)
        if not np.all(np.isfinite(ray_d)):
            out_rows.append({"t_unix_ns": t_ns, "associated": 0, "reason": "invalid_gaze_ray"})
            continue
        candidates = []
        for _, drow in det_sub.iterrows():
            obj_vec = np.array([
                drow["pose_x"] - (ray_o[0] if np.isfinite(ray_o[0]) else 0.0),
                drow["pose_y"] - (ray_o[1] if np.isfinite(ray_o[1]) else 0.0),
                drow["pose_z"] - (ray_o[2] if np.isfinite(ray_o[2]) else 0.0),
            ], dtype=float)
            candidates.append({
                "obj_id": drow.get("id", np.nan), "obj_class": drow["obj_class"], "distance_m": drow["distance_m"],
                "angle_deg": angle_between_deg(ray_d, obj_vec), "sync_dt_ms": drow["sync_dt_ms"],
            })
        cdf = pd.DataFrame(candidates).sort_values(["angle_deg", "distance_m", "sync_dt_ms"])
        best = cdf.iloc[0]
        assoc = int(np.isfinite(best["angle_deg"]) and best["angle_deg"] <= angle_threshold_deg)
        out_rows.append({
            "t_unix_ns": t_ns, "associated": assoc, "reason": "ok" if assoc else "angle_gate_failed",
            "obj_id": best["obj_id"], "obj_class": best["obj_class"], "distance_m": best["distance_m"],
            "angle_deg": best["angle_deg"], "sync_dt_ms": best["sync_dt_ms"],
        })
    return pd.DataFrame(out_rows)


def save_assoc_products(df: pd.DataFrame, scenario_out: Path, sid: str, sname: str, method_label: str):
    df.to_csv(scenario_out / f"scenario_{sid}_gaze_object_assoc_{method_label}.csv", index=False)
    summary = {
        "scenario_id": sid, "scenario_type": sname, "method": method_label,
        "n_gaze_samples": len(df),
        "n_associated": int((df["associated"] == 1).sum()) if len(df) else 0,
        "association_ratio": float((df["associated"] == 1).mean()) if len(df) else np.nan,
    }
    if "angle_deg" in df.columns and df["angle_deg"].notna().any():
        summary["median_angle_deg"] = float(df["angle_deg"].median())
        summary["p95_angle_deg"] = float(np.nanpercentile(df["angle_deg"].dropna(), 95))
    if "point_dist_m" in df.columns and df["point_dist_m"].notna().any():
        summary["median_point_dist_m"] = float(df["point_dist_m"].median())
        summary["p95_point_dist_m"] = float(np.nanpercentile(df["point_dist_m"].dropna(), 95))
    if "sync_dt_ms" in df.columns and df["sync_dt_ms"].notna().any():
        summary["median_sync_dt_ms"] = float(df["sync_dt_ms"].median())
    pd.DataFrame([summary]).to_csv(scenario_out / f"scenario_{sid}_assoc_summary_{method_label}.csv", index=False)

    if "angle_deg" in df.columns and df["angle_deg"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df["angle_deg"].dropna(), bins=30)
        ax.set_xlabel("Gaze-to-object angle in middle_lidar frame (deg)")
        ax.set_ylabel("Count")
        ax.set_title(f"Scenario {sid} ({sname}) {method_label}: angle distribution")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_out / f"scenario_{sid}_{method_label}_angle_hist.png", dpi=300)
        plt.close(fig)

    if "point_dist_m" in df.columns and df["point_dist_m"].notna().any():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(df["point_dist_m"].dropna(), bins=30)
        ax.set_xlabel("Gaze point to object center distance in middle_lidar frame (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Scenario {sid} ({sname}) {method_label}: point distance distribution")
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_out / f"scenario_{sid}_{method_label}_pointdist_hist.png", dpi=300)
        plt.close(fig)

    assoc_ok = df[df["associated"] == 1].copy()
    if len(assoc_ok) > 0 and "obj_class" in assoc_ok.columns:
        share = assoc_ok["obj_class"].value_counts().reindex(VALID_CLASSES, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(share.index, share.values)
        ax.set_ylabel("Associated gaze samples")
        ax.set_title(f"Scenario {sid} ({sname}) {method_label}: attention share by class")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(scenario_out / f"scenario_{sid}_class_attention_share_{method_label}.png", dpi=300)
        plt.close(fig)
        top_objs = assoc_ok.groupby(["obj_id", "obj_class"]).agg(
            n_assoc=("associated", "size"), median_distance_m=("distance_m", "median")
        ).reset_index().sort_values("n_assoc", ascending=False)
        top_objs.to_csv(scenario_out / f"scenario_{sid}_top_attended_objects_{method_label}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", default=None)
    ap.add_argument("--detections-dir", default=None)
    ap.add_argument("--detections-glob", default="perception_info_rviz.xlsx_part*.csv")
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--scenario-workdir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--scenario-ids", nargs="*", default=["5", "7"])
    ap.add_argument("--max-sync-dt-ms", type=float, default=80.0)
    ap.add_argument("--angle-threshold-deg", type=float, default=12.0)
    ap.add_argument("--point-margin-m", type=float, default=0.8)
    ap.add_argument("--frame-suffix", default="M", help="Gaze frame suffix in gaze_repr.csv. Default: M for middle_lidar")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.detections_dir:
        raw_det = read_detection_parts(Path(args.detections_dir), args.detections_glob)
    elif args.detections:
        raw_det = read_table(Path(args.detections))
    else:
        raise ValueError("Please provide either --detections or --detections-dir")

    det = prepare_detection_df(raw_det)

    det = det[det["obj_class"].isin(VALID_CLASSES)].copy()
    scenarios = load_scenarios_relative(Path(args.scenarios), wanted_ids=args.scenario_ids)

    for _, srow in scenarios.iterrows():
        sid = str(srow["scenario_id"])
        sname = srow.get("scenario_type", "")
        start_ns, end_ns = int(srow["start_ns"]), int(srow["end_ns"])
        scenario_dir = Path(args.scenario_workdir) / f"scenario_{sid}"
        gaze_csv = scenario_dir / "gaze_repr.csv"
        if not gaze_csv.exists():
            print(f"[WARN] scenario {sid}: missing {gaze_csv}")
            continue
        gaze = read_table(gaze_csv)
        gaze["t_unix_ns"] = pd.to_numeric(gaze["t_unix_ns"], errors="coerce")
        gaze = gaze.dropna(subset=["t_unix_ns"]).copy()
        gaze = gaze[(gaze["t_unix_ns"] >= start_ns) & (gaze["t_unix_ns"] <= end_ns)].copy()
        ds = det[(det["t_unix_ns"] >= start_ns) & (det["t_unix_ns"] <= end_ns)].copy()
        if len(gaze) == 0 or len(ds) == 0:
            print(f"[WARN] scenario {sid}: no gaze or detections in range")
            continue
        scenario_out = outdir / f"scenario_{sid}"
        scenario_out.mkdir(parents=True, exist_ok=True)
        assoc_A = associate_method_A(gaze, ds, args.max_sync_dt_ms, args.point_margin_m, args.frame_suffix)
        assoc_B = associate_method_ray(gaze, ds, "B", args.max_sync_dt_ms, args.angle_threshold_deg, args.frame_suffix)
        assoc_C = associate_method_ray(gaze, ds, "C", args.max_sync_dt_ms, args.angle_threshold_deg, args.frame_suffix)
        save_assoc_products(assoc_A, scenario_out, sid, sname, "methodA")
        save_assoc_products(assoc_B, scenario_out, sid, sname, "methodB")
        save_assoc_products(assoc_C, scenario_out, sid, sname, "methodC")
        print(f"[OK] scenario {sid} -> {scenario_out}")


if __name__ == "__main__":
    main()
