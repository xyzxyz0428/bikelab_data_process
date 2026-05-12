#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_bikelab_csvs_to_xlsx.py

Merge trimmed Bikelab sensor CSV/XLSX tables into a single XLSX workbook.
This version adds optional camera timestamps and LiDAR frame CSVs so that
all streams can be analysed within the same time window.

Typical usage
-------------
python merge_bikelab_csvs_to_xlsx.py \
  --input-dir ./session_folder \
  --output bike_interface_merged_with_lidar.xlsx \
  --start-unix-ns 1773159006000000000 \
  --end-unix-ns   1773159123000000000
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


CONFIG = [
    {
        "prefixes": ["ubx_nav"],
        "sheet": "gps",
        "time_mode": "ubx_nav",
        "keep_cols": [
            "t_unix_ns", "lon", "lat", "height", "h_acc", "v_acc",
            "g_speed", "head_mot", "s_acc", "head_acc"
        ],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": ["steering_angle"],
        "sheet": "potentiometer",
        "time_mode": "t_unix_ns",
        "keep_cols": ["t_unix_ns", "ok", "adc_raw", "angle_deg", "angle_deg_clamped", "error"],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": ["speed_decoded"],
        "sheet": "wheel_speed",
        "time_mode": "t_unix_ns",
        "keep_cols": ["t_unix_ns", "page", "speed_mps"],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": ["imu"],
        "sheet": "imu",
        "time_mode": "t_unix_ns",
        "keep_cols": [
            "t_unix_ns", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z",
            "mag_x_mG", "mag_y_mG", "mag_z_mG", "imu_temp_C", "pressure_Pa",
            "pressure_temp_C", "rollspeed", "pitchspeed", "headingspeed",
            "roll", "pitch", "heading", "qw", "qx", "qy", "qz"
        ],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": ["rally_payload_decoded"],
        "sheet": "powermeter",
        "time_mode": "t_unix_ns",
        "keep_cols": [
            "t_unix_ns", "page_hex", "page_name", "cadence_rpm",
            "p10_instantaneous_power_w", "p10_update_event_count",
            "p12_update_event_count", "p12_crank_ticks",
            "p12_crank_period_1_2048s", "p12_accumulated_torque_1_32nm"
        ],
        "rename_map": {},
        "required": True,
    },
    {
        "prefix": "tobii",
        "sheet": "eyetracker",
        "time_mode": "tobii_recording_time",
        "keep_cols": [
            "t_unix_ns",
            "Recording timestamp",
            "Sensor",
            "Gaze point X",
            "Gaze point Y",
            "Gaze point 3D X",
            "Gaze point 3D Y",
            "Gaze point 3D Z",
            "Gaze direction left X",
            "Gaze direction left Y",
            "Gaze direction left Z",
            "Gaze direction right X",
            "Gaze direction right Y",
            "Gaze direction right Z",
            "Pupil position left X",
            "Pupil position left Y",
            "Pupil position left Z",
            "Pupil position right X",
            "Pupil position right Y",
            "Pupil position right Z",
            "Pupil diameter left",
            "Pupil diameter right",
            "Pupil diameter filtered",
            "Validity left",
            "Validity right",
            "Recording media name",
            "Recording media width",
            "Recording media height",
            "Eye movement type",
            "Eye movement event duration",
            "Eye movement type index",
            "Fixation point X",
            "Fixation point Y",
            "Ungrouped",
            "Gyro X",
            "Gyro Y",
            "Gyro Z",
            "Accelerometer X",
            "Accelerometer Y",
            "Accelerometer Z",
            "Magnetometer X",
            "Magnetometer Y",
            "Magnetometer Z",
        ],
        "rename_map": {}
    },

    {
        "prefix": "fsr",
        "sheet": "brake_sensor_right",
        "time_mode": "t_unix_ns",
        "keep_cols": ["t_unix_ns", "ok", "adc_raw", "force_total_g", "force_total_n", "error"],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": "back_camera_timestamps",
        "sheet": "camera",
        "time_mode": "camera_unix_ns",
        "keep_cols": ["t_unix_ns","frame_idx"],
        "rename_map": {},
        "required": True,
    },
    {
        "prefixes": "scene_timestamps",
        "sheet": "camera",
        "time_mode": "camera_unix_ns",
        "keep_cols": ["t_unix_ns","frame_idx"],
        "rename_map": {},
        "required": True,
    },
    # Optional LiDAR frame CSVs exported from PointCloud2 topics
    # {
    #     "prefixes": "lidar_200_frames",
    #     "sheet": "lidar_f_200",
    #     "time_mode": "t_unix_ns",
    #     "keep_cols": [
    #         "t_unix_ns", "frame_id", "width", "height", "point_count",
    #         "row_step", "point_step", "is_dense"
    #     ],
    #     "rename_map": {},
    #     "required": False,
    # },
    # {
    #     "prefixes": "lidar_201_frames",
    #     "sheet": "lidar_f_201",
    #     "time_mode": "t_unix_ns",
    #     "keep_cols": [
    #         "t_unix_ns", "frame_id", "width", "height", "point_count",
    #         "row_step", "point_step", "is_dense"
    #     ],
    #     "rename_map": {},
    #     "required": False,
    # },
    # {
    #     "prefixes": "lidar_202_frames",
    #     "sheet": "lidar_f_202",
    #     "time_mode": "t_unix_ns",
    #     "keep_cols": [
    #         "t_unix_ns", "frame_id", "width", "height", "point_count",
    #         "row_step", "point_step", "is_dense"
    #     ],
    #     "rename_map": {},
    #     "required": False,
    # },
]


def read_table_robust(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path)

    try:
        return pd.read_excel(path)
    except Exception:
        pass

    attempts = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16-le", "utf-16-be"]
    last_err = None
    for enc in attempts:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read file {path}: {last_err}")


def drop_duplicate_columns(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        dup_cols = df.columns[dup_mask].tolist()
        print(f"Warning: {file_name} has duplicate columns; keeping first occurrence only: {dup_cols}")
        df = df.loc[:, ~dup_mask].copy()
    return df


def sanitize_time_column_ns(t_ns: pd.Series, file_name: str) -> pd.Series:
    """
    Remove obvious startup garbage such as 0 / tiny values mixed with valid Unix epoch ns.
    """
    s = pd.to_numeric(t_ns, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s.astype("Int64")

    valid_pos = valid[valid > 0]
    if valid_pos.empty:
        return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    med = valid_pos.median()
    mask = s > 0

    # If stream looks like Unix epoch ns, reject tiny outliers aggressively.
    if med > 1e17:
        lower = med * 0.1
        upper = med * 10.0
        mask &= (s >= lower) & (s <= upper)

    dropped = int((~mask & s.notna()).sum())
    if dropped > 0:
        print(f"Info: {file_name} dropped {dropped} invalid timestamp rows.")

    s = s.where(mask, pd.NA)
    return s.astype("Int64")


def find_matching_file(input_dir: Path, prefixes: list[str]):
    candidates = []
    for pfx in prefixes:
        matches = sorted(input_dir.rglob(f"{pfx}*"))
        matches = [
            m for m in matches
            if m.is_file() and m.suffix.lower() in [".csv", ".txt", ".xlsx", ".xlsm", ".xls"]
        ]
        candidates.extend(matches)

    # unique while preserving order
    seen = set()
    uniq = []
    for m in candidates:
        if m not in seen:
            uniq.append(m)
            seen.add(m)

    if not uniq:
        return None
    if len(uniq) > 1:
        print(f"Warning: multiple files found for prefixes {prefixes}, using: {uniq[0].name}")
    return uniq[0]


def add_unified_time_column(df: pd.DataFrame, time_mode: str, file_name: str) -> pd.DataFrame:
    df = drop_duplicate_columns(df, file_name)

    if time_mode == "ubx_nav":
        required = ["header.stamp.sec", "header.stamp.nanosec"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{file_name}: missing columns {missing}\nAvailable columns: {list(df.columns)}"
            )
        sec = pd.to_numeric(df["header.stamp.sec"], errors="coerce")
        nsec = pd.to_numeric(df["header.stamp.nanosec"], errors="coerce")
        df["t_unix_ns"] = (sec * 1_000_000_000 + nsec).astype("Int64")

    elif time_mode == "t_unix_ns":
        candidates = ["t_unix_ns", "unix_ns", "timestamp_ns", "ts_ns", "timestamp"]
        src = next((c for c in candidates if c in df.columns), None)
        if src is None:
            raise ValueError(
                f"{file_name}: missing a usable nanosecond timestamp column.\nAvailable columns: {list(df.columns)}"
            )
        df["t_unix_ns"] = pd.to_numeric(df[src], errors="coerce").astype("Int64")

    elif time_mode == "camera_unix_ns":
        candidates = ["unix_ns", "t_unix_ns", "timestamp_ns", "ts_ns"]
        src = next((c for c in candidates if c in df.columns), None)
        if src is None:
            raise ValueError(
                f"{file_name}: missing camera timestamp column (unix_ns / t_unix_ns / timestamp_ns).\n"
                f"Available columns: {list(df.columns)}"
            )
        df["t_unix_ns"] = pd.to_numeric(df[src], errors="coerce").astype("Int64")

    elif time_mode == "tobii_recording_time":
        # Priority 1: if t_unix_ns already exists, use it directly
        if "t_unix_ns" in df.columns:
            df["t_unix_ns"] = pd.to_numeric(df["t_unix_ns"], errors="coerce").astype("Int64")

        # Priority 2: reconstruct from UTC date + UTC start time + recording timestamp
        else:
            required_base = ["Recording date UTC", "Recording start time UTC"]
            missing_base = [c for c in required_base if c not in df.columns]
            if missing_base:
                raise ValueError(
                    f"{csv_name}: missing columns {missing_base}\n"
                    f"Available columns: {list(df.columns)}"
                )

            # support both 'Recording timestamp [μs]' and 'Recording timestamp'
            if "Recording timestamp [μs]" in df.columns:
                rec_ts = pd.to_numeric(df["Recording timestamp [μs]"], errors="coerce")
                rec_factor = 1000  # us -> ns
            elif "Recording timestamp" in df.columns:
                rec_ts = pd.to_numeric(df["Recording timestamp"], errors="coerce")
                # Pro Lab export is usually in microseconds
                rec_factor = 1000
            else:
                raise ValueError(
                    f"{csv_name}: missing 'Recording timestamp [μs]' or 'Recording timestamp'\n"
                    f"Available columns: {list(df.columns)}"
                )

            base_dt = pd.to_datetime(
                df["Recording date UTC"].astype(str).str.strip() + " " +
                df["Recording start time UTC"].astype(str).str.strip(),
                errors="coerce"
            )

            base_ns = pd.Series(base_dt.view("int64"), index=df.index)
            base_ns = base_ns.where(base_dt.notna(), pd.NA)

            t_ns = base_ns + rec_ts * rec_factor
            df["t_unix_ns"] = pd.array(t_ns, dtype="Int64")

    else:
        raise ValueError(f"{file_name}: unknown time_mode '{time_mode}'")

    df["t_unix_ns"] = sanitize_time_column_ns(df["t_unix_ns"], file_name)
    return df

def find_file_by_prefix(input_dir: Path, prefix: str) -> Path:
    matches = sorted(input_dir.rglob(f"{prefix}*"))
    matches = [
        m for m in matches
        if m.is_file() and m.suffix.lower() in [".csv", ".txt", ".xlsx", ".xlsm", ".xls"]
    ]

    if not matches:
        raise FileNotFoundError(f"No file found in {input_dir} with prefix '{prefix}'")
    if len(matches) > 1:
        print(f"Warning: multiple files found for prefix '{prefix}', using: {matches[0].name}")
    return matches[0]

def load_trim_and_filter(file_path: Path, cfg: dict, start_unix_ns: int, end_unix_ns: int) -> pd.DataFrame:
    df = read_table_robust(file_path)
    df = add_unified_time_column(df, cfg["time_mode"], file_path.name)

    missing = [c for c in cfg["keep_cols"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_path.name}: missing columns {missing}\nAvailable columns: {list(df.columns)}"
        )

    out_cols = ["t_unix_ns"] + cfg["keep_cols"]
    out_cols = list(dict.fromkeys(out_cols))
    df = df.loc[:, out_cols].copy()
    df = drop_duplicate_columns(df, file_path.name)

    t = df["t_unix_ns"]
    df = df[t.notna() & (t >= start_unix_ns) & (t <= end_unix_ns)].copy()

    if cfg.get("rename_map"):
        df = df.rename(columns=cfg["rename_map"])

    df = df.sort_values("t_unix_ns").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", "-i", required=True, help="Folder containing CSV/XLSX files")
    ap.add_argument("--output", "-o", required=True, help="Output XLSX path")
    ap.add_argument("--start-unix-ns", required=True, type=int, help="Start timestamp (Unix ns)")
    ap.add_argument("--end-unix-ns", required=True, type=int, help="End timestamp (Unix ns)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)

    if args.start_unix_ns > args.end_unix_ns:
        raise ValueError("start-unix-ns must be <= end-unix-ns")

    sheet_data = []

    for cfg in CONFIG:
        if "prefixes" in cfg:
            file_path = find_matching_file(input_dir, cfg["prefixes"])
        else:
            file_path = find_file_by_prefix(input_dir, cfg["prefix"])
        if file_path is None:
            if cfg.get("required", True):
                raise FileNotFoundError(
                    f"No file found in {input_dir} matching any of prefixes {cfg.get('prefixes', cfg.get('prefix'))}"
                )
            print(f"Info: optional stream not found for prefixes {cfg.get('prefixes', cfg.get('prefix'))}; skipped.")
            continue

        df = load_trim_and_filter(file_path, cfg, args.start_unix_ns, args.end_unix_ns)
        sheet_name = cfg["sheet"][:31]
        sheet_data.append((sheet_name, file_path.name, df))

        if len(df) > 0:
            tmin = int(df["t_unix_ns"].min())
            tmax = int(df["t_unix_ns"].max())
            print(f"Prepared sheet '{sheet_name}' from '{file_path.name}', rows={len(df)}, range=[{tmin}, {tmax}]")
        else:
            print(f"Prepared sheet '{sheet_name}' from '{file_path.name}', rows=0")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, src_name, df in sheet_data:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        meta_rows = []
        for sheet_name, src_name, df in sheet_data:
            meta_rows.append({
                "sheet": sheet_name,
                "source_file": src_name,
                "rows": len(df),
                "t_min": int(df["t_unix_ns"].min()) if len(df) else pd.NA,
                "t_max": int(df["t_unix_ns"].max()) if len(df) else pd.NA,
            })
        pd.DataFrame(meta_rows).to_excel(writer, sheet_name="_meta", index=False)

    print(f"\n✅ Wrote: {out_path}")
    print("Sheets:", [x[0] for x in sheet_data])


if __name__ == "__main__":
    main()
