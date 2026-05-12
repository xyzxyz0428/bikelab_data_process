#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

# --------- CONFIG: filename prefix -> (sheet name, keep columns, rename map) ----------

CONFIG = [
    {
        "prefix": "ubx_nav",
        "sheet": "gps",
        "time_mode": "ubx_nav",
        "keep_cols": ["t_unix_ns", "lon", "lat", "height", "h_acc", "v_acc", "g_speed",
      "head_mot", "s_acc", "head_acc"],
        "rename_map": {}
    },
    {
        "prefix": "steering_angle",
        "sheet": "potentiometer",
        "time_mode": "t_unix_ns",
        "keep_cols": ["t_unix_ns", "ok", "adc_raw", "angle_deg", "angle_deg_clamped", "error"],
        "rename_map": {}
    },
    {
        "prefix": "speed_decoded",
        "sheet": "wheel speed",
        "time_mode": "t_unix_ns",
        "keep_cols": ["t_unix_ns", "page", "speed_mps"],
        "rename_map": {}
    },
    {
        "prefix": "imu",
        "sheet": "imu",
        "time_mode": "t_unix_ns",
        "keep_cols":["t_unix_ns", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "mag_x_mG", "mag_y_mG", "mag_z_mG", "imu_temp_C", "pressure_Pa", "pressure_temp_C", "rollspeed", "pitchspeed", "headingspeed", "roll", "pitch", "heading", "qw", "qx", "qy", "qz"],

        "rename_map": {}
    },
    {
        "prefix": "rally_payload_decoded",
        "sheet": "powermeter",
        "time_mode": "t_unix_ns",
        "keep_cols":["t_unix_ns", "page_hex","page_name","cadence_rpm","p10_instantaneous_power_w","p10_update_event_count","p12_update_event_count","p12_crank_ticks","p12_crank_period_1_2048s","p12_accumulated_torque_1_32nm"],

        "rename_map": {}
    },
    {
        "prefix": "tobii",
        "sheet": "eyetracker",
        "time_mode": "tobii_recording_time",
        "keep_cols": [
            "Gaze point X [MCS px]",
            "Gaze point Y [MCS px]",
            "Gaze point 3D X [HUCS mm]",
            "Gaze point 3D Y [HUCS mm]",
            "Gaze point 3D Z [HUCS mm]",
            "Gaze direction left X [HUCS norm]",
            "Gaze direction left Y [HUCS norm]",
            "Gaze direction left Z [HUCS norm]",
            "Gaze direction right X [HUCS norm]",
            "Gaze direction right Y [HUCS norm]",
            "Gaze direction right Z [HUCS norm]",
            "Pupil position left X [HUCS mm]",
            "Pupil position left Y [HUCS mm]",
            "Pupil position left Z [HUCS mm]",
            "Pupil position right X [HUCS mm]",
            "Pupil position right Y [HUCS mm]",
            "Pupil position right Z [HUCS mm]",
            "Pupil diameter left [mm]",
            "Pupil diameter right [mm]",
            "Pupil diameter filtered [mm]",
            "Validity left",
            "Validity right",
            "Eye movement type",
            "Eye movement event duration [ms]",
            "Eye movement type index",
            "Fixation point X [MCS px]",
            "Fixation point Y [MCS px]",
            "Gyro X [°/s]",
            "Gyro Y [°/s]",
            "Gyro Z [°/s]",
            "Accelerometer X [m/s²]",
            "Accelerometer Y [m/s²]",
            "Accelerometer Z [m/s²]",
            "Magnetometer X [μT]",
            "Magnetometer Y [μT]",
            "Magnetometer Z [μT]",
        ],
        "rename_map": {}
    },
]

from pathlib import Path
import pandas as pd

def read_table_robust(path: Path) -> pd.DataFrame:
    # first try Excel if suffix suggests it
    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path)

    # try reading as Excel even if suffix is wrong
    try:
        return pd.read_excel(path)
    except Exception:
        pass

    # otherwise try CSV/text encodings
    attempts = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16-le", "utf-16-be"]
    last_err = None
    for enc in attempts:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not read file {path}: {last_err}")

def find_file_by_prefix(input_dir: Path, prefix: str) -> Path:
    matches = sorted(input_dir.glob(f"{prefix}*"))
    matches = [m for m in matches if m.is_file() and m.suffix.lower() in [".csv", ".txt", ".xlsx", ".xlsm", ".xls"]]

    if not matches:
        raise FileNotFoundError(f"No CSV file found in {input_dir} with prefix '{prefix}'")
    if len(matches) > 1:
        print(f"Warning: multiple files found for prefix '{prefix}', using: {matches[0].name}")
    return matches[0]


def drop_duplicate_columns(df: pd.DataFrame, csv_name: str) -> pd.DataFrame:
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        dup_cols = df.columns[dup_mask].tolist()
        print(f"Warning: {csv_name} has duplicate columns, keeping first occurrence only: {dup_cols}")
        df = df.loc[:, ~dup_mask].copy()
    return df


def add_unified_time_column(df: pd.DataFrame, time_mode: str, csv_name: str) -> pd.DataFrame:
    df = drop_duplicate_columns(df, csv_name)

    if time_mode == "ubx_nav":
        required = ["header.stamp.sec", "header.stamp.nanosec"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_name}: missing columns {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        sec = pd.to_numeric(df["header.stamp.sec"], errors="coerce")
        nsec = pd.to_numeric(df["header.stamp.nanosec"], errors="coerce")

        df["t_unix_ns"] = (sec * 1_000_000_000 + nsec).astype("Int64")

    elif time_mode == "t_unix_ns":
        if "t_unix_ns" not in df.columns:
            raise ValueError(
                f"{csv_name}: missing required column 't_unix_ns'\n"
                f"Available columns: {list(df.columns)}"
            )
        df["t_unix_ns"] = pd.to_numeric(df["t_unix_ns"], errors="coerce").astype("Int64")
    elif time_mode == "tobii_recording_time":
        required = [
            "Recording date UTC",
            "Recording start time UTC",
            "Recording timestamp [μs]",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_name}: missing columns {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        base_dt = pd.to_datetime(
            df["Recording date UTC"].astype(str).str.strip() + " " +
            df["Recording start time UTC"].astype(str).str.strip(),
            errors="coerce"
        )

        rec_us = pd.to_numeric(df["Recording timestamp [μs]"], errors="coerce")

        base_ns = pd.Series(base_dt.view("int64"), index=df.index)
        base_ns = base_ns.where(base_dt.notna(), pd.NA)

        t_ns = base_ns + rec_us * 1000
        df["t_unix_ns"] = pd.array(t_ns, dtype="Int64")

    else:
        raise ValueError(f"{csv_name}: unknown time_mode '{time_mode}'")

    df = drop_duplicate_columns(df, csv_name)
    return df


def load_trim_and_filter(csv_path: Path, cfg: dict, start_unix_ns: int, end_unix_ns: int) -> pd.DataFrame:
    df = read_table_robust(csv_path)
    df = add_unified_time_column(df, cfg["time_mode"], csv_path.name)

    missing = [c for c in cfg["keep_cols"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing columns {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    out_cols = ["t_unix_ns"]
    out_cols.extend(cfg["keep_cols"])
    out_cols = list(dict.fromkeys(out_cols))
   
    df = df.loc[:, out_cols].copy()
    df = drop_duplicate_columns(df, csv_path.name)

    t = df["t_unix_ns"]
    df = df[
        t.notna() &
        (t >= start_unix_ns) &
        (t <= end_unix_ns)
    ].copy()

    df = df.rename(columns=cfg["rename_map"])

    df = df.sort_values("t_unix_ns").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", "-i", required=True, help="Folder containing the CSV files")
    ap.add_argument("--output", "-o", required=True, help="Output XLSX path, e.g. merged.xlsx")
    ap.add_argument("--start-unix-ns", required=True, type=int, help="Start timestamp in unix nanoseconds")
    ap.add_argument("--end-unix-ns", required=True, type=int, help="End timestamp in unix nanoseconds")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)
    start_unix_ns = args.start_unix_ns
    end_unix_ns = args.end_unix_ns

    if start_unix_ns > end_unix_ns:
        raise ValueError("start-unix-ns must be <= end-unix-ns")

    sheet_data = []

    for cfg in CONFIG:
        csv_file = find_file_by_prefix(input_dir, cfg["prefix"])
        df = load_trim_and_filter(csv_file, cfg, start_unix_ns, end_unix_ns)

        safe_sheet = cfg["sheet"][:31]
        sheet_data.append((safe_sheet, csv_file.name, df))

        if len(df) > 0:
            tmin = int(df["t_unix_ns"].min())
            tmax = int(df["t_unix_ns"].max())
            print(f"Prepared sheet '{safe_sheet}' from '{csv_file.name}', rows={len(df)}, range=[{tmin}, {tmax}]")
        else:
            print(f"Prepared sheet '{safe_sheet}' from '{csv_file.name}', rows=0")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for safe_sheet, csv_name, df in sheet_data:
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    print(f"\n✅ Wrote: {out_path}")
    print(f"Sheets: {[x[0] for x in sheet_data]}")


if __name__ == "__main__":
    main()