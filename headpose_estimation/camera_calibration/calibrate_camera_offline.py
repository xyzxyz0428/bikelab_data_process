import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def collect_image_paths(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return paths


def build_object_points(cols: int, rows: int, square_size_m: float):
    """
    cols, rows = number of INNER corners
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size_m
    return objp


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err = 0.0
    total_points = 0
    per_view = []

    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        proj = proj.reshape(-1, 2)
        gt = imgpoints[i].reshape(-1, 2)
        err = np.linalg.norm(gt - proj, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))
        per_view.append(rmse)
        total_err += np.sum(err ** 2)
        total_points += len(err)

    overall_rmse = float(np.sqrt(total_err / total_points)) if total_points > 0 else np.nan
    return overall_rmse, per_view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="folder with calibration images")
    parser.add_argument("--cols", type=int, required=True, help="number of inner corners in x direction")
    parser.add_argument("--rows", type=int, required=True, help="number of inner corners in y direction")
    parser.add_argument("--square-size-m", type=float, required=True, help="square size in meters")
    parser.add_argument("--output-json", required=True, help="output camera json")
    parser.add_argument("--preview-dir", default=None, help="optional folder to save detected previews")
    parser.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_json = Path(args.output_json)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_size_m)

    objpoints = []
    imgpoints = []
    image_size = None
    used_images = []
    rejected_images = []

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            rejected_images.append((img_path.name, "read_failed"))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        vis = img.copy()

        if found:
            corners_subpix = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria,
            )

            objpoints.append(objp.copy())
            imgpoints.append(corners_subpix)
            used_images.append(img_path.name)

            cv2.drawChessboardCorners(vis, pattern_size, corners_subpix, found)

            if preview_dir:
                cv2.imwrite(str(preview_dir / img_path.name), vis)
        else:
            rejected_images.append((img_path.name, "corners_not_found"))
            if preview_dir:
                cv2.imwrite(str(preview_dir / img_path.name), vis)

    if len(objpoints) < 10:
        raise RuntimeError(
            f"Only {len(objpoints)} valid images found. "
            f"Please collect at least 10-15 good calibration images."
        )

    if args.model == "pinhole":
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            None,
            None,
        )
    else:
        # fisheye expects slightly different data shapes/types
        objpoints_fe = [op.reshape(1, -1, 3).astype(np.float64) for op in objpoints]
        imgpoints_fe = [ip.reshape(1, -1, 2).astype(np.float64) for ip in imgpoints]

        K = np.zeros((3, 3), dtype=np.float64)
        D = np.zeros((4, 1), dtype=np.float64)

        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW
        )

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_fe,
            imgpoints_fe,
            image_size,
            K,
            D,
            None,
            None,
            flags=flags,
            criteria=criteria,
        )
        ret = rms
        dist = D

    overall_rmse, per_view_rmse = compute_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, K, dist
    )

    result = {
        "model": args.model,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "calibration_rms": float(ret),
        "reprojection_rmse": overall_rmse,
        "board": {
            "inner_corners_cols": args.cols,
            "inner_corners_rows": args.rows,
            "square_size_m": args.square_size_m,
        },
        "num_input_images": len(image_paths),
        "num_valid_images": len(objpoints),
        "used_images": used_images,
        "rejected_images": rejected_images,
        "per_view_rmse": per_view_rmse,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Calibration done ===")
    print(f"Model: {args.model}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"Valid images: {len(objpoints)} / {len(image_paths)}")
    print(f"Calibration RMS: {ret:.6f}")
    print(f"Reprojection RMSE: {overall_rmse:.6f} px")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()