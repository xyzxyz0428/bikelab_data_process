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


def compute_image_quality(gray, corners, image_size):
    """
    Compute simple image/chessboard quality indicators.

    image_size: (width, height)
    corners: detected chessboard corners, shape Nx1x2
    """
    w, h = image_size

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    pts = corners.reshape(-1, 2)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)

    board_w = float(x_max - x_min)
    board_h = float(y_max - y_min)

    board_area_ratio = float((board_w * board_h) / (w * h))

    board_cx = float((x_min + x_max) / 2.0)
    board_cy = float((y_min + y_max) / 2.0)

    board_center_x_norm = float(board_cx / w)
    board_center_y_norm = float(board_cy / h)

    board_center_distance = float(
        np.sqrt((board_center_x_norm - 0.5) ** 2 + (board_center_y_norm - 0.5) ** 2)
    )

    return {
        "sharpness": sharpness,
        "mean_brightness": mean_brightness,
        "board_area_ratio": board_area_ratio,
        "board_center_x_norm": board_center_x_norm,
        "board_center_y_norm": board_center_y_norm,
        "board_center_distance": board_center_distance,
    }


def pass_quality_filter(q, args):
    if q["sharpness"] < args.min_sharpness:
        return False, "low_sharpness"

    if q["mean_brightness"] < args.min_mean_brightness:
        return False, "too_dark"

    if q["mean_brightness"] > args.max_mean_brightness:
        return False, "too_bright"

    if q["board_area_ratio"] < args.min_board_area_ratio:
        return False, "board_too_small"

    if q["board_area_ratio"] > args.max_board_area_ratio:
        return False, "board_too_large"

    return True, "ok"


def select_diverse_images(candidates, max_selected):
    """
    Select high-quality but spatially diverse calibration images.

    candidates: dicts containing:
      image, image_path, objp, corners, quality
    """
    if max_selected <= 0 or len(candidates) <= max_selected:
        return candidates

    def score(c):
        q = c["quality"]

        sharp_score = min(q["sharpness"] / 300.0, 3.0)

        # Prefer board size around 20-35% of image area, but not only very large boards.
        area_score = 1.0 - abs(q["board_area_ratio"] - 0.25)

        # Prefer board also near image edges/corners for better distortion calibration.
        edge_score = q["board_center_distance"]

        return sharp_score + area_score + 0.8 * edge_score

    candidates_sorted = sorted(candidates, key=score, reverse=True)

    # Spatial bins based on chessboard center position
    bins = {}
    for c in candidates_sorted:
        q = c["quality"]
        bx = int(q["board_center_x_norm"] * 4)
        by = int(q["board_center_y_norm"] * 3)
        bx = max(0, min(3, bx))
        by = max(0, min(2, by))
        bins.setdefault((bx, by), []).append(c)

    selected = []

    # Round-robin selection from bins
    while len(selected) < max_selected:
        added = False
        for key in sorted(bins.keys()):
            if bins[key]:
                selected.append(bins[key].pop(0))
                added = True
                if len(selected) >= max_selected:
                    break

        if not added:
            break

    return selected


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

    # Pre-calibration quality filter
    parser.add_argument("--enable-quality-filter", action="store_true")
    parser.add_argument("--max-selected-images", type=int, default=150)
    parser.add_argument("--min-selected-images", type=int, default=80)

    parser.add_argument("--min-sharpness", type=float, default=80.0)
    parser.add_argument("--min-mean-brightness", type=float, default=40.0)
    parser.add_argument("--max-mean-brightness", type=float, default=220.0)

    parser.add_argument("--min-board-area-ratio", type=float, default=0.03)
    parser.add_argument("--max-board-area-ratio", type=float, default=0.75)

    # Post-calibration RMSE filter
    parser.add_argument(
        "--auto-filter",
        action="store_true",
        help="iteratively remove images with high per-view RMSE",
    )
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

    rejected_images = []
    image_size = None
    quality_candidates = []

    # ------------------------------------------------------------
    # Step 1: detect chessboard and optionally quality-filter images
    # ------------------------------------------------------------
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

        if preview_dir:
            save_detection_preview(
                preview_dir,
                img_path,
                img,
                pattern_size,
                corners_subpix,
                found,
            )

        if not found:
            rejected_images.append({
                "image": img_path.name,
                "reason": "corners_not_found",
            })
            continue

        q = compute_image_quality(gray, corners_subpix, image_size)

        if args.enable_quality_filter:
            ok_quality, reason = pass_quality_filter(q, args)
            if not ok_quality:
                item = {
                    "image": img_path.name,
                    "reason": reason,
                }
                item.update(q)
                rejected_images.append(item)
                continue

        quality_candidates.append({
            "image": img_path.name,
            "image_path": img_path,
            "objp": objp.copy(),
            "corners": corners_subpix,
            "quality": q,
        })

    if len(quality_candidates) < args.min_valid_images:
        raise RuntimeError(
            f"Only {len(quality_candidates)} detected / quality-passed images found. "
            f"Need at least {args.min_valid_images}."
        )

    # ------------------------------------------------------------
    # Step 2: select diverse high-quality images
    # ------------------------------------------------------------
    if args.enable_quality_filter:
        selected_candidates = select_diverse_images(
            quality_candidates,
            args.max_selected_images,
        )
    else:
        selected_candidates = quality_candidates

    if len(selected_candidates) < args.min_valid_images:
        raise RuntimeError(
            f"Only {len(selected_candidates)} selected valid images found. "
            f"Need at least {args.min_valid_images}."
        )

    if len(selected_candidates) < args.min_selected_images:
        print(
            f"[WARN] Only {len(selected_candidates)} images selected. "
            f"Recommended minimum is {args.min_selected_images}."
        )

    objpoints = [c["objp"] for c in selected_candidates]
    imgpoints = [c["corners"] for c in selected_candidates]
    used_images = [c["image"] for c in selected_candidates]
    quality_results = [{"image": c["image"], **c["quality"]} for c in selected_candidates]

    # ------------------------------------------------------------
    # Step 3: initial calibration
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Step 4: iterative RMSE filtering
    # ------------------------------------------------------------
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

            rms, K, dist, rvecs, tvecs = rms_it, K_it, dist_it, rvecs_it, tvecs_it
            overall_rmse, per_view_rmse = overall_it, per_view_it
            used_images = cur_names
            objpoints = cur_obj
            imgpoints = cur_img

        # Final recalibration after last removal
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

    # ------------------------------------------------------------
    # Step 5: save outputs
    # ------------------------------------------------------------
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
        "num_detected_images_before_quality_selection": len(quality_candidates),
        "num_valid_images": len(objpoints),
        "used_images": used_images,
        "rejected_images": rejected_images,
        "bad_images_by_rmse": bad_images_by_rmse,
        "per_view_results": per_view_results_sorted,
        "quality_filter": {
            "enabled": bool(args.enable_quality_filter),
            "max_selected_images": args.max_selected_images,
            "min_selected_images": args.min_selected_images,
            "min_sharpness": args.min_sharpness,
            "min_mean_brightness": args.min_mean_brightness,
            "max_mean_brightness": args.max_mean_brightness,
            "min_board_area_ratio": args.min_board_area_ratio,
            "max_board_area_ratio": args.max_board_area_ratio,
        },
        "quality_results": quality_results,
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
    print(f"Input images: {len(image_paths)}")
    print(f"Detected / quality-passed images before selection: {len(quality_candidates)}")
    print(f"Valid images used: {len(objpoints)}")
    print(f"Calibration RMS: {rms:.6f}")
    print(f"Reprojection RMSE: {overall_rmse:.6f} px")
    print(f"Saved to: {output_json}")

    print("\nWorst per-view RMSE images:")
    for item in per_view_results_sorted[:10]:
        print(f"  {item['image']}: {item['per_view_rmse_px']:.3f} px")

    if args.auto_filter:
        print("\nFilter history:")
        for h in filter_history:
            print(
                f"  iter {h['iteration']}: "
                f"n={h['num_images']}, "
                f"rmse={h['reprojection_rmse']:.3f}, "
                f"max={h['max_per_view_rmse']:.3f}, "
                f"bad={h['num_bad_images']}"
            )


if __name__ == "__main__":
    main()