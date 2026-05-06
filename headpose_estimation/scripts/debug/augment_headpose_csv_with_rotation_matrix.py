#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def add_rotation_matrix_columns(df, prefix, euler_order):
    roll_col = f"{prefix}_roll_deg"
    pitch_col = f"{prefix}_pitch_deg"
    yaw_col = f"{prefix}_yaw_deg"

    if not all(c in df.columns for c in [roll_col, pitch_col, yaw_col]):
        print(f"[WARN] Missing Euler columns for {prefix}, skip.")
        return df

    r00 = []
    r01 = []
    r02 = []
    r10 = []
    r11 = []
    r12 = []
    r20 = []
    r21 = []
    r22 = []

    for _, row in df.iterrows():
        roll = row[roll_col]
        pitch = row[pitch_col]
        yaw = row[yaw_col]

        if pd.isna(roll) or pd.isna(pitch) or pd.isna(yaw):
            Rm = np.full((3, 3), np.nan)
        else:
            Rm = R.from_euler(
                euler_order,
                [float(roll), float(pitch), float(yaw)],
                degrees=True,
            ).as_matrix()

        r00.append(Rm[0, 0])
        r01.append(Rm[0, 1])
        r02.append(Rm[0, 2])

        r10.append(Rm[1, 0])
        r11.append(Rm[1, 1])
        r12.append(Rm[1, 2])

        r20.append(Rm[2, 0])
        r21.append(Rm[2, 1])
        r22.append(Rm[2, 2])

    df[f"{prefix}_R_00"] = r00
    df[f"{prefix}_R_01"] = r01
    df[f"{prefix}_R_02"] = r02
    df[f"{prefix}_R_10"] = r10
    df[f"{prefix}_R_11"] = r11
    df[f"{prefix}_R_12"] = r12
    df[f"{prefix}_R_20"] = r20
    df[f"{prefix}_R_21"] = r21
    df[f"{prefix}_R_22"] = r22

    return df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-csv", required=True)

    ap.add_argument(
        "--euler-order",
        default="xyz",
        help=(
            "Euler order used to reconstruct rotation matrix. "
            "Try xyz, xzy, yxz, yzx, zxy, zyx, or uppercase variants if needed."
        ),
    )

    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_csv)

    for prefix in ["cam_head", "cam_back", "back_head"]:
        df = add_rotation_matrix_columns(df, prefix, args.euler_order)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("[INFO] saved augmented CSV:")
    print(output_csv)
    print("[INFO] Euler order used:", args.euler_order)
    print("[INFO] Added rotation matrix columns for cam_head, cam_back, back_head if available.")


if __name__ == "__main__":
    main()