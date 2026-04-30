#!/usr/bin/env python3
import argparse
import json
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # derive T_H_HUCS
    T_H_HUCS = T_H_C1 @ T_C1_HUCS

    result = {
        # main transforms
        "T_W_C2": T_W_C2.tolist(),
        "T_H_C1": T_H_C1.tolist(),
        "T_C1_HUCS": T_C1_HUCS.tolist(),
        "T_H_HUCS": T_H_HUCS.tolist(),

        # T_W_C2 metadata
        "T_W_C2_num_samples": data_w_c2.get("num_samples", None),
        "T_W_C2_ref_tag_id": data_w_c2.get("ref_tag_id", None),
        "T_W_C2_tag_size_m": data_w_c2.get("tag_size_m", None),
        "T_W_C2_translation_std_m": data_w_c2.get("translation_std_m", None),
        "T_W_C2_rotation_std_deg": data_w_c2.get("rotation_std_deg", None),
        "T_W_C2_low_confidence": data_w_c2.get("low_confidence", None),

        # T_H_C1 metadata
        "T_H_C1_num_samples": data_h_c1.get("num_samples", None),
        "T_H_C1_board_tag_id": data_h_c1.get("board_tag_id", None),
        "T_H_C1_board_tag_size_m": data_h_c1.get("board_tag_size_m", None),

        # T_C1_HUCS metadata
        "T_C1_HUCS_num_samples": data_c1_hucs.get("num_samples", None),
        "T_C1_HUCS_num_inliers": data_c1_hucs.get("num_inliers", None),
    }

    # optional pass-through metadata if present
    if "scene_window" in data_h_c1:
        result["T_H_C1_scene_window"] = data_h_c1["scene_window"]
    if "back_window" in data_h_c1:
        result["T_H_C1_back_window"] = data_h_c1["back_window"]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved to {args.output_json}")
    print("Included metadata:")
    print(f"  T_W_C2_num_samples = {result['T_W_C2_num_samples']}")
    print(f"  T_W_C2_translation_std_m = {result['T_W_C2_translation_std_m']}")
    print(f"  T_W_C2_rotation_std_deg = {result['T_W_C2_rotation_std_deg']}")
    print(f"  T_W_C2_low_confidence = {result['T_W_C2_low_confidence']}")
    print(f"  T_H_C1_num_samples = {result['T_H_C1_num_samples']}")
    print(f"  T_C1_HUCS_num_samples = {result['T_C1_HUCS_num_samples']}")
    print(f"  T_C1_HUCS_num_inliers = {result['T_C1_HUCS_num_inliers']}")


if __name__ == "__main__":
    main()