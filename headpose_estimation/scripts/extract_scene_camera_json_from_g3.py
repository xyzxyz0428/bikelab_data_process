#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording-g3", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    with open(args.recording_g3, "r", encoding="utf-8") as f:
        g3 = json.load(f)

    calib = g3["scenecamera"]["camera-calibration"]

    fx, fy = calib["focal-length"]
    cx, cy = calib["principal-point"]
    skew = calib.get("skew", 0.0)

    k1, k2, k3 = calib["radial-distortion"]
    p1, p2 = calib["tangential-distortion"]

    w, h = calib["resolution"]

    scene_cam = {
        "model": "pinhole",
        "K": [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        "dist": [k1, k2, p1, p2, k3],
        "image_width": w,
        "image_height": h
    }

    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scene_cam, f, indent=2)

    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()