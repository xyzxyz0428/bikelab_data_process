#!/usr/bin/env python3
import argparse
import json
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def invert_T(T):
    T = np.array(T, dtype=np.float64)
    Rm = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4, dtype=np.float64)
    Tinv[:3, :3] = Rm.T
    Tinv[:3, 3] = -Rm.T @ t
    return Tinv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-W-C2-json", required=True)
    ap.add_argument("--T-H-C1-json", required=True)
    ap.add_argument("--T-C1-HUCS-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    data_w_c2 = load_json(args.T_W_C2_json)
    data_h_c1 = load_json(args.T_H_C1_json)
    data_c1_hucs = load_json(args.T_C1_HUCS_json)

    T_W_C2 = np.array(data_w_c2["T_W_C2"], dtype=np.float64)
    T_H_C1 = np.array(data_h_c1["T_H_C1"], dtype=np.float64)
    T_C1_HUCS = np.array(data_c1_hucs["T_C1_HUCS"], dtype=np.float64)

    T_H_HUCS = T_H_C1 @ T_C1_HUCS

    result = {
        "T_W_C2": T_W_C2.tolist(),
        "T_H_C1": T_H_C1.tolist(),
        "T_H_HUCS": T_H_HUCS.tolist()
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved to {args.output_json}")


if __name__ == "__main__":
    main()