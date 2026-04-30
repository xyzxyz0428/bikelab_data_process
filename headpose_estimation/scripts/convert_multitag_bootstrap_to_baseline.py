#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    with open(args.bootstrap_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tag_map = data["tag_map"]

    tags = {}
    for sid, item in tag_map.items():
        if item.get("status") != "map_used":
            continue

        T = item["T_W_Tag"]
        tags[sid] = {
            "num_samples": item.get("num_samples", 0),
            "T_W_Tag": T,
            "center_W": item.get("center_W", [T[0][3], T[1][3], T[2][3]]),
            "normal_W": item.get("normal_W", [T[0][2], T[1][2], T[2][2]]),
            "translation_std_m": item.get("translation_std_m", None),
            "rotation_std_deg": item.get("rotation_std_deg", None),
            "low_confidence": item.get("low_confidence", False),
        }

    result = {
        "method": "converted_from_bootstrap_multitag",
        "ref_tag_id": data.get("ref_tag_id"),
        "default_size_m": data.get("tag_size_m"),
        "used_frames": data.get("num_samples"),
        "T_W_C2_source": {
            "num_samples": data.get("num_samples"),
            "translation_std_m": data.get("translation_std_m"),
            "rotation_std_deg": data.get("rotation_std_deg"),
            "low_confidence": data.get("low_confidence"),
            "multitag_rmse_px": data.get("multitag_rmse_px"),
        },
        "tags": tags,
        "note": (
            "Converted from bootstrap multi-tag tag_map. "
            "This baseline is internally consistent within the static baseline sequence, "
            "but it is not an independently measured physical tag layout."
        ),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved to {out}")
    print(f"num tags = {len(tags)}")


if __name__ == "__main__":
    main()