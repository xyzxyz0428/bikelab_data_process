#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

print("Starting lidar2lidar_calibration_rosbag.py ...")

try:
    import open3d as o3d
except ImportError as e:
    print("ERROR: Failed to import open3d:", e)
    print("Make sure you installed it in this environment, e.g.:")
    print("  pip install open3d")
    raise SystemExit(1)

# ROS2 / rosbag2 imports
try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from sensor_msgs_py import point_cloud2 as pc2
except ImportError as e:
    print("ERROR: Failed to import ROS2 / rosbag2 dependencies:", e)
    print("Make sure you have sourced your ROS2 workspace, e.g.:")
    print("  source /opt/ros/humble/setup.bash")
    print("And installed rosbag2_py and sensor_msgs_py.")
    raise SystemExit(1)

# ----------------------------------------------------------------------
# Hard-coded initial transform: source (rear LiDAR) -> target (front LiDAR)
# Convention:
#   - Translation in target frame [m]
#   - RPY in degrees, roll about X, pitch about Y, yaw about Z
#   - R = Rz(yaw) * Ry(pitch) * Rx(roll)
# ----------------------------------------------------------------------
#INIT_TX = -0.2 
INIT_TX = -1.30   # rear is 1.3 m behind front
#INIT_TY = 0.0      
INIT_TY = 0.0     # no lateral offset
#INIT_TZ = -0.07     
INIT_TZ = 0.38    # rear is 0.42 m higher

#INIT_ROLL_DEG = 0.0 
INIT_ROLL_DEG  = 0.0   # deg
#INIT_PITCH_DEG = -90  
INIT_PITCH_DEG = 0.0   # deg
#INIT_YAW_DEG   = 180
INIT_YAW_DEG   = 180.0 # deg (rear LiDAR facing backward)

TIMESTAMP_TOL_NS = 5_000_000  # 5 ms
# ----------------------------------------------------------------------
# Helper: find matching timestamp within tolerance
# ----------------------------------------------------------------------
def find_matching_stamp(stamp_ns, other_cache, tol_ns):
    """
    Find a timestamp in other_cache whose difference to stamp_ns is
    <= tol_ns (nanoseconds). Returns the matching timestamp or None.
    Uses a simple linear search (fine for moderate bag sizes).
    """
    if not other_cache:
        return None

    best_stamp = None
    best_diff = tol_ns + 1

    for s in other_cache.keys():
        diff = abs(s - stamp_ns)
        if diff <= tol_ns and diff < best_diff:
            best_diff = diff
            best_stamp = s

    return best_stamp
# ----------------------------------------------------------------------
# Helper: convert PointCloud2 -> Nx3 numpy (x,y,z)
# ----------------------------------------------------------------------
def pc2_to_xyz(msg):
    """
    Convert sensor_msgs/msg/PointCloud2 to an Nx3 numpy array (x,y,z).
    Uses sensor_msgs_py.point_cloud2.read_points.
    """
    points = []
    for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


# ----------------------------------------------------------------------
# Helper: read first synchronized pair of PointCloud2 from bag
# ----------------------------------------------------------------------
def load_synchronized_clouds_from_rosbag(
    bag_path,
    storage_id,
    source_topic,
    target_topic,
    time_tolerance_ns=0,
    max_duration_ns=None,
):
    """
    Open a rosbag2 and return a pair (source_points, target_points) as Nx3 arrays
    for the FIRST pair of messages whose timestamps differ by <= time_tolerance_ns,
    scanning only up to max_duration_ns from the first LiDAR timestamp (if given).

    max_duration_ns:
        - None  -> no limit
        - value -> stop scanning once (current_stamp - first_lidar_stamp) > value
    """
    print(f"Opening rosbag2 at: {bag_path}")
    print(f"  storage_id       = {storage_id}")
    print(f"  source_topic     = {source_topic}")
    print(f"  target_topic     = {target_topic}")
    print(f"  time_tolerance   = {time_tolerance_ns} ns "
          f"({time_tolerance_ns * 1e-6:.3f} ms)")
    if max_duration_ns is not None:
        print(f"  max_duration     = {max_duration_ns * 1e-9:.3f} s")
    else:
        print("  max_duration     = unlimited")

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for t in topic_types:
            if t.name == topic_name:
                return t.type
        raise ValueError(f"Topic {topic_name} not found in bag.")

    topic_names = [t.name for t in topic_types]
    if source_topic not in topic_names:
        raise RuntimeError(
            f"Source topic '{source_topic}' not found in bag.\n"
            f"Available topics: {topic_names}"
        )
    if target_topic not in topic_names:
        raise RuntimeError(
            f"Target topic '{target_topic}' not found in bag.\n"
            f"Available topics: {topic_names}"
        )

    source_type_str = typename(source_topic)
    target_type_str = typename(target_topic)
    print(f"  source_topic type: {source_type_str}")
    print(f"  target_topic type: {target_type_str}")

    source_msg_type = get_message(source_type_str)
    target_msg_type = get_message(target_type_str)

    # Caches indexed by timestamp (ns) until we find a match
    source_cache = {}
    target_cache = {}

    # First LiDAR timestamp we see (header.stamp) across either topic
    first_stamp_ns = None

    print("Scanning bag for synchronized PointCloud2 messages ...")

    while reader.has_next():
        topic, data, _storage_ts = reader.read_next()

        # Only care about the two LiDAR topics
        if topic not in (source_topic, target_topic):
            continue

        if topic == source_topic:
            msg = deserialize_message(data, source_msg_type)
        else:  # topic == target_topic
            msg = deserialize_message(data, target_msg_type)

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

        # Initialize reference time
        if first_stamp_ns is None:
            first_stamp_ns = stamp_ns
            print(f"First LiDAR header.stamp: {first_stamp_ns} ns")

        # Check max duration
        if max_duration_ns is not None and (stamp_ns - first_stamp_ns) > max_duration_ns:
            print("Reached max_duration_ns limit "
                  f"({(stamp_ns - first_stamp_ns) * 1e-9:.3f} s > "
                  f"{max_duration_ns * 1e-9:.3f} s).")
            break  # stop scanning further messages

        # Fill caches and try matching
        if topic == source_topic:
            source_cache[stamp_ns] = pc2_to_xyz(msg)
            match_stamp = find_matching_stamp(stamp_ns, target_cache, time_tolerance_ns)
            if match_stamp is not None:
                print("Found synchronized pair:")
                print(f"  source stamp = {stamp_ns}")
                print(f"  target stamp = {match_stamp}")
                print(f"  Δt = {(stamp_ns - match_stamp) * 1e-6:.3f} ms")
                src_pts = source_cache[stamp_ns]
                tgt_pts = target_cache[match_stamp]
                print(f"  source points: {src_pts.shape[0]}")
                print(f"  target points: {tgt_pts.shape[0]}")
                return src_pts, tgt_pts

        else:  # topic == target_topic
            target_cache[stamp_ns] = pc2_to_xyz(msg)
            match_stamp = find_matching_stamp(stamp_ns, source_cache, time_tolerance_ns)
            if match_stamp is not None:
                print("Found synchronized pair:")
                print(f"  target stamp = {stamp_ns}")
                print(f"  source stamp = {match_stamp}")
                print(f"  Δt = {(stamp_ns - match_stamp) * 1e-6:.3f} ms")
                src_pts = source_cache[match_stamp]
                tgt_pts = target_cache[stamp_ns]
                print(f"  source points: {src_pts.shape[0]}")
                print(f"  target points: {tgt_pts.shape[0]}")
                return src_pts, tgt_pts

    # If we exit the loop without returning:
    if max_duration_ns is not None:
        raise RuntimeError(
            "No synchronized PointCloud2 messages within tolerance "
            f"found in the first {max_duration_ns * 1e-9:.3f} seconds "
            f"between topics '{source_topic}' and '{target_topic}'."
        )
    else:
        raise RuntimeError(
            "No synchronized PointCloud2 messages within tolerance "
            f"found between topics '{source_topic}' and '{target_topic}'."
        )

# ----------------------------------------------------------------------
# Open3D helpers (unchanged from your CSV version)
# ----------------------------------------------------------------------
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
    R = np.asarray(R)
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
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


# ----------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ICP calibration between two LiDAR point clouds stored in a ROS2 bag."
    )
    
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to rosbag2 directory (contains metadata.yaml)",
    )
    parser.add_argument(
        "--storage_id",
        default="sqlite3",
        help="rosbag2 storage backend (default: sqlite3, use 'mcap' for MCAP bags)",
    )
    parser.add_argument(
        "--source_topic",
        default="/rslidar_points_200",
        help="source LiDAR topic (rear), default: /rslidar_points_200",
    )
    parser.add_argument(
        "--target_topic",
        default="/rslidar_points_201",
        help="target LiDAR topic (front), default: /rslidar_points_201",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="voxel size for downsampling [m]",
    )
    parser.add_argument(
    "--time_tolerance_ms",
    type=float,
    default=0.0,
    help="max allowed timestamp difference between clouds [ms] "
            "(0.0 = exact match only)",
    )
    parser.add_argument(
    "--max_duration_s",
    type=float,
    default=None,
    help="Only scan the first X seconds of LiDAR data (based on header.stamp). "
            "Default: no limit.",
    )

    args = parser.parse_args()
    print("Arguments:", args)
    time_tolerance_ns = int(args.time_tolerance_ms * 1e6)

    if args.max_duration_s is not None:
        max_duration_ns = int(args.max_duration_s * 1e9)
    else:
        max_duration_ns = None

    source_pts_raw, target_pts_raw = load_synchronized_clouds_from_rosbag(
        bag_path=args.bag,
        storage_id=args.storage_id,
        source_topic=args.source_topic,
        target_topic=args.target_topic,
        time_tolerance_ns=time_tolerance_ns,
        max_duration_ns=max_duration_ns,
    )

    # Convert to Open3D, preprocess
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
        INIT_TX,
        INIT_TY,
        INIT_TZ,
        INIT_ROLL_DEG,
        INIT_PITCH_DEG,
        INIT_YAW_DEG,
    )
    print("Initial transform matrix:\n", T_init)

    # ICP refinement
    T_icp = refine_registration_icp(source_pcd, target_pcd, T_init, args.voxel_size)

    # ---- print ICP tx,ty,tz and roll,pitch,yaw ----
    tx, ty, tz = T_icp[0, 3], T_icp[1, 3], T_icp[2, 3]
    roll_rad, pitch_rad, yaw_rad = rotation_matrix_to_rpy(T_icp[:3, :3])
    roll_deg = np.rad2deg(roll_rad)
    pitch_deg = np.rad2deg(pitch_rad)
    yaw_deg = np.rad2deg(yaw_rad)

    print("\n--- ICP result as translation + RPY ---")
    print("Translation (source -> target) [m]:")
    print(f"  tx = {tx:.6f},  ty = {ty:.6f},  tz = {tz:.6f}")
    print("Rotation (source -> target):")
    print(f"  roll  = {roll_rad:.6f} rad  ({roll_deg:.3f} deg)")
    print(f"  pitch = {pitch_rad:.6f} rad  ({pitch_deg:.3f} deg)")
    print(f"  yaw   = {yaw_rad:.6f} rad  ({yaw_deg:.3f} deg)")
    print("-------------------------------------------------\n")

    # Transform source points with both transforms (for visualization)
    rear_init_pts = transform_points(source_pts, T_init)
    rear_icp_pts = transform_points(source_pts, T_icp)

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