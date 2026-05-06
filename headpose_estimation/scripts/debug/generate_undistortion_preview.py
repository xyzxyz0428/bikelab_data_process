#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_camera_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    K = np.array(data["K"], dtype=np.float64)
    dist = np.array(data["dist"], dtype=np.float64).reshape(-1, 1)

    return K, dist, data


def sample_files(files, max_frames):
    if max_frames <= 0 or max_frames >= len(files):
        return files

    idx = np.linspace(0, len(files) - 1, max_frames).astype(int)
    return [files[i] for i in idx]


def draw_grid(img, step=100):
    out = img.copy()
    h, w = out.shape[:2]

    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 1)

    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 0), 1)

    return out


def make_side_by_side(original, undistorted):
    h1, w1 = original.shape[:2]
    h2, w2 = undistorted.shape[:2]

    if h1 != h2:
        scale = h1 / h2
        undistorted = cv2.resize(
            undistorted,
            (int(w2 * scale), h1),
            interpolation=cv2.INTER_AREA,
        )

    return np.hstack([original, undistorted])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-json", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--grid-step", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="0 = crop strongly, 1 = keep all pixels")
    args = ap.parse_args()

    K, dist, cam_data = load_camera_json(args.camera_json)

    frame_dir = Path(args.frame_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(frame_dir.glob("frame_*.png"))
        + list(frame_dir.glob("frame_*.jpg"))
        + list(frame_dir.glob("frame_*.jpeg"))
    )

    files = sample_files(files, args.max_frames)

    if not files:
        raise RuntimeError(f"No frame files found in {frame_dir}")

    img0 = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
    if img0 is None:
        raise RuntimeError("Cannot read first image.")

    h, w = img0.shape[:2]
    image_size = (w, h)

    print("[INFO] frame size:", w, "x", h)
    print("[INFO] K:")
    print(K)
    print("[INFO] dist:", dist.reshape(-1).tolist())

    if "image_width" in cam_data and "image_height" in cam_data:
        print("[INFO] camera_json image size:",
              cam_data.get("image_width"), "x", cam_data.get("image_height"))

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K,
        dist,
        image_size,
        alpha=args.alpha,
        newImgSize=image_size,
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K,
        dist,
        None,
        new_K,
        image_size,
        cv2.CV_16SC2,
    )

    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue

        undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        img_grid = draw_grid(img, step=args.grid_step)
        undist_grid = draw_grid(undist, step=args.grid_step)

        side = make_side_by_side(img_grid, undist_grid)

        cv2.putText(
            side,
            "original",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            side,
            "undistorted",
            (w + 30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        cv2.imwrite(str(out_dir / p.name), side)

    print(f"[INFO] saved previews to {out_dir}")


if __name__ == "__main__":
    main()