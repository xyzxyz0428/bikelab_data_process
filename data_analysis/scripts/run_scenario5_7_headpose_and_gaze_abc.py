#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_scenario5_7_headpose_and_gaze_abc.py

Purpose
-------
For scenario 5 and 7 (or any selected scenario IDs), run:
1) head-pose estimation on the relevant frame range
2) A/B/C gaze evaluation using the generated head-pose CSV

This script is an orchestrator around:
- estimate_headpose_from_frames.py
- evaluate_gaze_abc_by_windows.py

It assumes:
- critical_scenarios.csv defines scenario start/end times
- frame timestamp CSV contains frame_idx and unix_ns
- image frames are in frame_dir
- the uploaded scripts are available locally

Outputs
-------
For each scenario:
- <outdir>/scenario_<id>/headpose.csv
- <outdir>/scenario_<id>/gaze_abc.csv
- <outdir>/scenario_<id>/gaze_abc_summary.csv
- <outdir>/scenario_<id>/run_meta.json
"""

import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_unix_ns_scalar(x):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios-csv", required=True)
    ap.add_argument("--scenario-ids", nargs="+", default=["5", "7"])
    ap.add_argument("--timestamps-csv", required=True, help="frame timestamps csv with frame_idx, unix_ns")
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--estimate-headpose-script", required=True)
    ap.add_argument("--evaluate-gaze-script", required=True)

    # head pose inputs
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--headpose-config-json", required=True)
    ap.add_argument("--rig-calib-json", required=True)
    ap.add_argument("--neutral-frame", default=None)

    # gaze ABC inputs
    ap.add_argument("--tag-windows-csv", required=True)
    ap.add_argument("--apriltag-baseline-json", required=True)
    ap.add_argument("--scene-camera-json", required=True)
    ap.add_argument("--transforms-json", required=True)
    ap.add_argument("--tobii-raw-xlsx", required=True)
    ap.add_argument("--recording-g3", required=True)

    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = normalize_cols(pd.read_csv(args.scenarios_csv))
    colmap = {}
    for c in scenarios.columns:
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
    scenarios = scenarios.rename(columns=colmap)

    scenarios["start_ns"] = scenarios["start"].apply(to_unix_ns_scalar)
    scenarios["end_ns"] = scenarios["end"].apply(to_unix_ns_scalar)
    scenarios = scenarios.dropna(subset=["start_ns", "end_ns"]).copy()
    scenarios["scenario_id"] = scenarios["scenario_id"].astype(str)

    frame_ts = normalize_cols(pd.read_csv(args.timestamps_csv))
    if not {"frame_idx", "unix_ns"}.issubset(frame_ts.columns):
        raise ValueError("timestamps csv must contain frame_idx and unix_ns")
    frame_ts["unix_ns"] = pd.to_numeric(frame_ts["unix_ns"], errors="coerce")
    frame_ts = frame_ts.dropna(subset=["unix_ns"]).copy()

    for sid in args.scenario_ids:
        sub = scenarios[scenarios["scenario_id"] == str(sid)]
        if len(sub) == 0:
            print(f"[WARN] scenario {sid} not found, skipped")
            continue
        row = sub.iloc[0]
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])

        frame_sub = frame_ts[(frame_ts["unix_ns"] >= start_ns) & (frame_ts["unix_ns"] <= end_ns)].copy()
        if len(frame_sub) == 0:
            print(f"[WARN] scenario {sid}: no frames in time range, skipped")
            continue

        scenario_dir = outdir / f"scenario_{sid}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        scenario_ts_csv = scenario_dir / "timestamps_scenario.csv"
        frame_sub.to_csv(scenario_ts_csv, index=False)

        headpose_csv = scenario_dir / "headpose.csv"
        gaze_abc_csv = scenario_dir / "gaze_abc.csv"

        cmd_head = [
            "python", args.estimate_headpose_script,
            "--camera", args.camera_json,
            "--config", args.headpose_config_json,
            "--rig-calib", args.rig_calib_json,
            "--frame-dir", args.frame_dir,
            "--timestamps-csv", str(scenario_ts_csv),
            "--output-csv", str(headpose_csv),
        ]
        if args.neutral_frame:
            cmd_head.extend(["--neutral-frame", args.neutral_frame])

        print("[INFO] running headpose:", " ".join(cmd_head))
        subprocess.run(cmd_head, check=True)

        cmd_gaze = [
            "python", args.evaluate_gaze_script,
            "--tag-windows-csv", args.tag_windows_csv,
            "--apriltag-baseline-json", args.apriltag_baseline_json,
            "--headpose-csv", str(headpose_csv),
            "--scene-camera-json", args.scene_camera_json,
            "--transforms-json", args.transforms_json,
            "--tobii-raw-xlsx", args.tobii_raw_xlsx,
            "--recording-g3", args.recording_g3,
            "--output-csv", str(gaze_abc_csv),
        ]

        print("[INFO] running gaze ABC:", " ".join(cmd_gaze))
        subprocess.run(cmd_gaze, check=True)

        meta = {
            "scenario_id": sid,
            "scenario_type": row.get("scenario_type", ""),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "headpose_csv": str(headpose_csv),
            "gaze_abc_csv": str(gaze_abc_csv),
        }
        with open(scenario_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] finished scenario {sid} -> {scenario_dir}")


if __name__ == "__main__":
    main()
