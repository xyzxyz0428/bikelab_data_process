#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

# --------- CONFIG: input file -> (sheet name, keep columns, rename map) ----------
CONFIG = [
    ("ubx_nav_pvt.csv", "gps",
     ["timestamp", "lon", "lat", "height", "h_acc", "v_acc", "g_speed",
      "head_mot", "s_acc", "head_acc"],
     {}),

    ("adc_data.csv", "potentiometer",
     ["timestamp", "data"],
     {"data": "steering angle"}),

    ("bike_speed_data.csv", "wheel speed",
     ["timestamp", "speed"],
     {}),

    ("imu.csv", "imu",
     ["timestamp",
      "orientation.x", "orientation.y", "orientation.z", "orientation.w",
      "orientation_covariance",
      "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
      "angular_velocity_covariance",
      "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
      "linear_acceleration_covariance"],
     {}),

    ("power_meter_data.csv", "powermeter",
     ["timestamp", "cadence", "instantaneous_power", "torque"],
     {}),
]

def load_and_trim(csv_path: Path, keep_cols, rename_map):
    df = pd.read_csv(csv_path)

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name}: missing columns {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[keep_cols].rename(columns=rename_map)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", "-i", required=True, help="Folder containing the CSV files")
    ap.add_argument("--output", "-o", required=True, help="Output XLSX path, e.g. merged.xlsx")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for fname, sheet, keep_cols, rename_map in CONFIG:
            f = input_dir / fname
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")

            df = load_and_trim(f, keep_cols, rename_map)
            # Excel sheet names must be <= 31 chars
            safe_sheet = sheet[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    print(f"✅ Wrote: {out_path} (sheets: {[c[1] for c in CONFIG]})")

if __name__ == "__main__":
    main()