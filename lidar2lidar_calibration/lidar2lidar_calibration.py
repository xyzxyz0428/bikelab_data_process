#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

#python lidar2lidar_calibration.py   --source_csv rear_helios.csv   --target_csv front_helios.csv   --skip_header   --voxel_size 0.05   --cols 0 1 2

print("Starting lidar2lidar_calibration.py ...")

try:
    import open3d as o3d
except ImportError as e:
    print("ERROR: Failed to import open3d:", e)
    print("Make sure you installed it in this environment, e.g.:")
    print("  pip install open3d")
    raise SystemExit(1)


# ----------------------------------------------------------------------
# Hard-coded initial transform: source (rear LiDAR) -> target (front LiDAR)
# Convention:
#   - Translation in target frame [m]
#   - RPY in degrees, roll about X, pitch about Y, yaw about Z
#   - R = Rz(yaw) * Ry(pitch) * Rx(roll)
# ----------------------------------------------------------------------
#bpearl to front
#INIT_TX = -0.2 
#INIT_TY = 0.0      
#INIT_TZ = -0.07     

#INIT_ROLL_DEG = 0.0
#INIT_PITCH_DEG = -90 
#INIT_YAW_DEG   = 180

#rear to front
INIT_TX = -1.30   # rear is 1.3 m behind front
INIT_TY = 0.0     # no lateral offset
INIT_TZ = 0.38    # rear is 0.42 m higher
INIT_ROLL_DEG  = 0.0   # deg
INIT_PITCH_DEG = 0.0   # degv
INIT_YAW_DEG   = 180.0 # deg (rear LiDAR facing backward)
def load_csv_as_points(path, cols=(0, 1, 2), skip_header=True):
    """
    Load a CSV file as Nx3 numpy array of points.
    Assumes columns `cols` contain x, y, z (in meters) in the LiDAR frame.
    """
    print(f"Loading CSV: {path}")
    data = np.loadtxt(path, delimiter=",", skiprows=1 if skip_header else 0)
    points = data[:, cols]
    print(f"  Loaded {points.shape[0]} points from {path}")
    return points


def points_to_o3d_pcd(points):
    """
    Convert Nx3 numpy array to Open3D PointCloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def preprocess_pcd(pcd, voxel_size=0.05):
    """
    Voxel downsample + remove isolated points + estimate normals.
    """
    n0 = np.asarray(pcd.points).shape[0]
    print(f"Preprocess: original points = {n0}")

    # Voxel downsampling
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    n1 = np.asarray(pcd.points).shape[0]
    print(f"  After voxel downsampling ({voxel_size} m): {n1} points")

    # Remove isolated points
    pcd, ind = pcd.remove_radius_outlier(nb_points=15, radius=0.3)
    n2 = np.asarray(pcd.points).shape[0]
    print(f"  After radius outlier removal: {n2} points")

    # Estimate normals (needed for point-to-plane ICP)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)

    return pcd


def refine_registration_icp(source_pcd, target_pcd, init_trans, voxel_size):
    """
    Refine registration using ICP (point-to-plane).
    Returns the final 4x4 transformation matrix.
    """
    max_correspondence_distance = voxel_size * 2.0
    print("Start ICP refinement...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100
        ),
    )
    print("ICP finished.")
    print("  ICP fitness:      ", result_icp.fitness)
    print("  ICP inlier_rmse:  ", result_icp.inlier_rmse)
    print("  ICP transformation:\n", result_icp.transformation)
    return result_icp.transformation


def make_rpy_transform(tx, ty, tz, roll_deg, pitch_deg, yaw_deg):
    """
    Build a 4x4 transform from translation (meters) and RPY (degrees).
    Convention: R = Rz(yaw) * Ry(pitch) * Rx(roll),
    where roll about x, pitch about y, yaw about z.
    """
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )

    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=float)
    return T


def rotation_matrix_to_rpy(R):
    """
    Decompose rotation matrix R into roll, pitch, yaw (radians)
    for the convention R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    # Protect against numerical issues
    R = np.asarray(R)
    # pitch = asin(-R[2,0])
    pitch = np.arcsin(-R[2, 0])
    # roll  = atan2(R[2,1], R[2,2])
    roll = np.arctan2(R[2, 1], R[2, 2])
    # yaw   = atan2(R[1,0], R[0,0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def transform_points(points, T):
    """
    Apply 4x4 transform T to Nx3 points.
    """
    n = points.shape[0]
    homog = np.hstack([points, np.ones((n, 1), dtype=points.dtype)])
    transformed = (T @ homog.T).T
    return transformed[:, :3]


def main():
    parser = argparse.ArgumentParser(
        description="ICP calibration between two LiDAR point clouds (CSV)."
    )
    parser.add_argument("--source_csv", required=True,
                        help="source CSV file (e.g. rear LiDAR)")
    parser.add_argument("--target_csv", required=True,
                        help="target CSV file (e.g. front LiDAR)")
    parser.add_argument(
        "--voxel_size", type=float, default=0.05,
        help="voxel size for downsampling [m]"
    )
    parser.add_argument(
        "--skip_header",
        action="store_true",
        help="if set, skip 1 header line in the CSV.",
    )
    parser.add_argument(
        "--cols", type=int, nargs=3, default=[0, 1, 2],
        help="indices of x,y,z columns in the CSV (default: 0 1 2)"
    )

    args = parser.parse_args()

    print("Arguments:", args)

    # Load raw points
    source_pts_raw = load_csv_as_points(
        args.source_csv,
        cols=tuple(args.cols),
        skip_header=args.skip_header,
    )
    target_pts_raw = load_csv_as_points(
        args.target_csv,
        cols=tuple(args.cols),
        skip_header=args.skip_header,
    )

    # Convert to Open3D, preprocess, then back to numpy
    source_pcd = points_to_o3d_pcd(source_pts_raw)
    target_pcd = points_to_o3d_pcd(target_pts_raw)

    print("Preprocessing source point cloud ...")
    source_pcd = preprocess_pcd(source_pcd, voxel_size=args.voxel_size)
    print("Preprocessing target point cloud ...")
    target_pcd = preprocess_pcd(target_pcd, voxel_size=args.voxel_size)

    source_pts = np.asarray(source_pcd.points)
    target_pts = np.asarray(target_pcd.points)

    # Hard-coded initial transform
    print("Using hard-coded initial transform:")
    print(f"  translation [m]: tx={INIT_TX}, ty={INIT_TY}, tz={INIT_TZ}")
    print(f"  RPY [deg]: roll={INIT_ROLL_DEG}, pitch={INIT_PITCH_DEG}, yaw={INIT_YAW_DEG}")

    T_init = make_rpy_transform(
        INIT_TX, INIT_TY, INIT_TZ,
        INIT_ROLL_DEG, INIT_PITCH_DEG, INIT_YAW_DEG
    )
    print("Initial transform matrix:\n", T_init)

    # ICP refinement
    T_icp = refine_registration_icp(source_pcd, target_pcd, T_init, args.voxel_size)

    # ---- NEW: print ICP tx,ty,tz and roll,pitch,yaw ----
    tx, ty, tz = T_icp[0, 3], T_icp[1, 3], T_icp[2, 3]
    roll_rad, pitch_rad, yaw_rad = rotation_matrix_to_rpy(T_icp[:3, :3])
    roll_deg = np.rad2deg(roll_rad)
    pitch_deg = np.rad2deg(pitch_rad)
    yaw_deg = np.rad2deg(yaw_rad)

    print("\n--- ICP result as translation + RPY ---")
    print(f"Translation (source -> target) [m]:")
    print(f"  tx = {tx:.6f},  ty = {ty:.6f},  tz = {tz:.6f}")
    print("Rotation (source -> target):")
    print(f"  roll  = {roll_rad:.6f} rad  ({roll_deg:.3f} deg)")
    print(f"  pitch = {pitch_rad:.6f} rad  ({pitch_deg:.3f} deg)")
    print(f"  yaw   = {yaw_rad:.6f} rad  ({yaw_deg:.3f} deg)")
    print("-------------------------------------------------\n")

    # Transform source points with both transforms
    rear_init_pts = transform_points(source_pts, T_init)
    rear_icp_pts = transform_points(source_pts, T_icp)

    # Build colored point clouds for combined views
    # INIT: front (green) + rear_init (red)
    front_init_pcd = points_to_o3d_pcd(target_pts)
    rear_init_pcd = points_to_o3d_pcd(rear_init_pts)
    front_init_pcd.paint_uniform_color([0, 1, 0])  # green
    rear_init_pcd.paint_uniform_color([1, 0, 0])   # red
    alignment_init_pcd = front_init_pcd + rear_init_pcd

    # ICP: front (green) + rear_icp (red)
    front_icp_pcd = points_to_o3d_pcd(target_pts)
    rear_icp_pcd = points_to_o3d_pcd(rear_icp_pts)
    front_icp_pcd.paint_uniform_color([0, 1, 0])   # green
    rear_icp_pcd.paint_uniform_color([1, 0, 0])    # red
    alignment_icp_pcd = front_icp_pcd + rear_icp_pcd

    # Save two PCD files
    print("Saving PCD point clouds for 3D inspection ...")
    o3d.io.write_point_cloud("alignment_init.pcd", alignment_init_pcd)
    o3d.io.write_point_cloud("alignment_icp.pcd", alignment_icp_pcd)
    print("  alignment_init.pcd  (front + rear, initial guess)")
    print("  alignment_icp.pcd   (front + rear, ICP result)")
    print("Colors in both PCDs: GREEN = front (target), RED = rear (source)")

    print("\n=== Final extrinsic (4x4, transform source -> target frame) ===")
    print(T_icp)


if __name__ == "__main__":
    main()

