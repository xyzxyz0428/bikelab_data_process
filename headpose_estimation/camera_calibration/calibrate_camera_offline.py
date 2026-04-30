#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def collect_image_paths(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]


def build_object_points(cols: int, rows: int, square_size_m: float):
    """
    cols, rows = number of INNER chessboard corners
    """
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid
    objp *= square_size_m
    return objp


def find_corners(gray, pattern_size, criteria):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)

    if not found:
        # retry without FAST_CHECK, slower but sometimes more reliable
        flags2 = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags2)

    if not found:
        return False, None

    corners_subpix = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    return True, corners_subpix


def calibrate_pinhole(objpoints, imgpoints, image_size):
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    return float(ret), K, dist, rvecs, tvecs


def calibrate_fisheye(objpoints, imgpoints, image_size, criteria):
    objpoints_fe = [op.reshape(1, -1, 3).astype(np.float64) for op in objpoints]
    imgpoints_fe = [ip.reshape(1, -1, 2).astype(np.float64) for ip in imgpoints]

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
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

    return float(rms), K, D, rvecs, tvecs


def project_points(model, objp, rvec, tvec, K, dist):
    if model == "fisheye":
        objp_fe = objp.reshape(1, -1, 3).astype(np.float64)
        proj, _ = cv2.fisheye.projectPoints(objp_fe, rvec, tvec, K, dist)
        return proj.reshape(-1, 2)

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def compute_reprojection_error(model, objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err = 0.0
    total_points = 0
    per_view = []

    for i in range(len(objpoints)):
        proj = project_points(model, objpoints[i], rvecs[i], tvecs[i], K, dist)
        gt = imgpoints[i].reshape(-1, 2)

        err = np.linalg.norm(gt - proj, axis=1)
        rmse = float(np.sqrt(np.mean(err ** 2)))

        per_view.append(rmse)
        total_err += float(np.sum(err ** 2))
        total_points += len(err)

    overall_rmse = float(np.sqrt(total_err / total_points)) if total_points > 0 else np.nan
    return overall_rmse, per_view


def calibrate_model(model, objpoints, imgpoints, image_size, criteria):
    if model == "pinhole":
        return calibrate_pinhole(objpoints, imgpoints, image_size)
    if model == "fisheye":
        return calibrate_fisheye(objpoints, imgpoints, image_size, criteria)
    raise ValueError(f"Unsupported model: {model}")


def save_detection_preview(preview_dir, img_path, img, pattern_size, corners, found):
    vis = img.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    cv2.imwrite(str(preview_dir / img_path.name), vis)


def save_undistorted_preview(model, undistort_dir, image_paths, K, dist, image_size, max_images=20):
    undistort_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths[:max_images]:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if model == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                dist,
                image_size,
                np.eye(3),
                balance=0.0,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                dist,
                np.eye(3),
                new_K,
                image_size,
                cv2.CV_16SC2,
            )
            undist = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K,
                dist,
                image_size,
                alpha=0,
                newImgSize=image_size,
            )
            undist = cv2.undistort(img, K, dist, None, new_K)

        cv2.imwrite(str(undistort_dir / img_path.name), undist)


def filter_by_rmse(
    names,
    objpoints,
    imgpoints,
    per_view_rmse,
    max_per_view_rmse_px,
):
    kept_names = []
    kept_obj = []
    kept_img = []
    rejected = []

    for name, obj, img, rmse in zip(names, objpoints, imgpoints, per_view_rmse):
        if rmse <= max_per_view_rmse_px:
            kept_names.append(name)
            kept_obj.append(obj)
            kept_img.append(img)
        else:
            rejected.append({
                "image": name,
                "reason": "high_per_view_rmse",
                "per_view_rmse_px": float(rmse),
            })

    return kept_names, kept_obj, kept_img, rejected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--preview-dir", default=None)
    parser.add_argument("--undistort-preview-dir", default=None)

    parser.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")

    parser.add_argument("--auto-filter", action="store_true",
                        help="iteratively remove images with high per-view RMSE")
    parser.add_argument("--max-per-view-rmse-px", type=float, default=1.5)
    parser.add_argument("--filter-iterations", type=int, default=3)
    parser.add_argument("--min-valid-images", type=int, default=15)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_json = Path(args.output_json)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    undistort_preview_dir = Path(args.undistort_preview_dir) if args.undistort_preview_dir else None

    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    pattern_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_size_m)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )

    objpoints = []
    imgpoints = []
    used_images = []
    rejected_images = []
    image_size = None

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            rejected_images.append({
                "image": img_path.name,
                "reason": "read_failed",
            })
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        else:
            this_size = (gray.shape[1], gray.shape[0])
            if this_size != image_size:
                rejected_images.append({
                    "image": img_path.name,
                    "reason": f"image_size_mismatch_{this_size}",
                })
                continue

        found, corners_subpix = find_corners(gray, pattern_size, criteria)

        if found:
            objpoints.append(objp.copy())
            imgpoints.append(corners_subpix)
            used_images.append(img_path.name)
        else:
            rejected_images.append({
                "image": img_path.name,
                "reason": "corners_not_found",
            })

        if preview_dir:
            save_detection_preview(preview_dir, img_path, img, pattern_size, corners_subpix, found)

    if len(objpoints) < args.min_valid_images:
        raise RuntimeError(
            f"Only {len(objpoints)} valid images found. "
            f"Need at least {args.min_valid_images}."
        )

    # First calibration
    rms, K, dist, rvecs, tvecs = calibrate_model(
        args.model,
        objpoints,
        imgpoints,
        image_size,
        criteria,
    )

    overall_rmse, per_view_rmse = compute_reprojection_error(
        args.model,
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        K,
        dist,
    )

    filter_history = []
    bad_images_by_rmse = []

    # Iterative filtering
    if args.auto_filter:
        cur_names = used_images
        cur_obj = objpoints
        cur_img = imgpoints

        for it in range(args.filter_iterations):
            rms_it, K_it, dist_it, rvecs_it, tvecs_it = calibrate_model(
                args.model,
                cur_obj,
                cur_img,
                image_size,
                criteria,
            )

            overall_it, per_view_it = compute_reprojection_error(
                args.model,
                cur_obj,
                cur_img,
                rvecs_it,
                tvecs_it,
                K_it,
                dist_it,
            )

            max_rmse = max(per_view_it)
            num_bad = sum(1 for x in per_view_it if x > args.max_per_view_rmse_px)

            filter_history.append({
                "iteration": it + 1,
                "num_images": len(cur_obj),
                "calibration_rms": float(rms_it),
                "reprojection_rmse": float(overall_it),
                "max_per_view_rmse": float(max_rmse),
                "num_bad_images": int(num_bad),
            })

            if num_bad == 0:
                rms, K, dist, rvecs, tvecs = rms_it, K_it, dist_it, rvecs_it, tvecs_it
                overall_rmse, per_view_rmse = overall_it, per_view_it
                used_images = cur_names
                objpoints = cur_obj
                imgpoints = cur_img
                break

            new_names, new_obj, new_img, rejected_rmse = filter_by_rmse(
                cur_names,
                cur_obj,
                cur_img,
                per_view_it,
                args.max_per_view_rmse_px,
            )

            bad_images_by_rmse.extend(rejected_rmse)

            if len(new_obj) < args.min_valid_images:
                print("[WARN] Filtering would leave too few images. Stop filtering.")
                rms, K, dist, rvecs, tvecs = rms_it, K_it, dist_it, rvecs_it, tvecs_it
                overall_rmse, per_view_rmse = overall_it, per_view_it
                used_images = cur_names
                objpoints = cur_obj
                imgpoints = cur_img
                break

            cur_names = new_names
            cur_obj = new_obj
            cur_img = new_img

            # final assignment if this is last iteration
            rms, K, dist, rvecs, tvecs = rms_it, K_it, dist_it, rvecs_it, tvecs_it
            overall_rmse, per_view_rmse = overall_it, per_view_it
            used_images = cur_names
            objpoints = cur_obj
            imgpoints = cur_img

        # final recalibration after last removal
        rms, K, dist, rvecs, tvecs = calibrate_model(
            args.model,
            objpoints,
            imgpoints,
            image_size,
            criteria,
        )

        overall_rmse, per_view_rmse = compute_reprojection_error(
            args.model,
            objpoints,
            imgpoints,
            rvecs,
            tvecs,
            K,
            dist,
        )

    per_view_results = []
    for name, rmse in zip(used_images, per_view_rmse):
        per_view_results.append({
            "image": name,
            "per_view_rmse_px": float(rmse),
        })

    per_view_results_sorted = sorted(
        per_view_results,
        key=lambda x: x["per_view_rmse_px"],
        reverse=True,
    )

    if undistort_preview_dir:
        used_image_paths = [image_dir / name for name in used_images]
        save_undistorted_preview(
            args.model,
            undistort_preview_dir,
            used_image_paths,
            K,
            dist,
            image_size,
            max_images=20,
        )

    result = {
        "model": args.model,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "K": K.tolist(),
        "dist": dist.reshape(-1).tolist(),
        "calibration_rms": float(rms),
        "reprojection_rmse": float(overall_rmse),
        "board": {
            "inner_corners_cols": args.cols,
            "inner_corners_rows": args.rows,
            "square_size_m": args.square_size_m,
        },
        "num_input_images": len(image_paths),
        "num_valid_images": len(objpoints),
        "num_initial_detected_images": len(used_images) + len(bad_images_by_rmse),
        "used_images": used_images,
        "rejected_images": rejected_images,
        "bad_images_by_rmse": bad_images_by_rmse,
        "per_view_results": per_view_results_sorted,
        "filter": {
            "auto_filter": bool(args.auto_filter),
            "max_per_view_rmse_px": args.max_per_view_rmse_px,
            "filter_iterations": args.filter_iterations,
            "history": filter_history,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== Calibration done ===")
    print(f"Model: {args.model}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"Valid images: {len(objpoints)} / {len(image_paths)}")
    print(f"Calibration RMS: {rms:.6f}")
    print(f"Reprojection RMSE: {overall_rmse:.6f} px")
    print(f"Saved to: {output_json}")

    print("\nWorst per-view RMSE images:")
    for item in per_view_results_sorted[:10]:
        print(f"  {item['image']}: {item['per_view_rmse_px']:.3f} px")


if __name__ == "__main__":
    main()