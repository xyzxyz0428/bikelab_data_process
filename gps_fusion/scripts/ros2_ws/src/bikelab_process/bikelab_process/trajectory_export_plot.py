#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


EARTH_RADIUS_M = 6378137.0


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_from_quaternion(x, y, z, w):
    # ROS ENU yaw from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def sanitize_topic(topic_name):
    return topic_name.strip("/").replace("/", "_") or "root"


def local_enu_from_llh(lat, lon, lat0, lon0):
    """Use an equirectangular approximation for a short ENU trajectory."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    x = (lon_rad - lon0_rad) * math.cos(lat0_rad) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x, y


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def open_reader(bag_uri, storage_id):
    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_uri),
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)
    return reader


def main():
    parser = argparse.ArgumentParser(
        description="Export and plot /fix and odometry from a ROS 2 bag."
    )

    parser.add_argument(
        "--bag",
        required=True,
        help="Path to a ROS2 bag directory, e.g. results/local_runs/fused_result",
    )

    parser.add_argument(
        "--out",
        default="trajectory_export",
        help="Output directory",
    )

    parser.add_argument(
        "--storage_id",
        default="sqlite3",
        help="rosbag2 storage id: sqlite3 or mcap",
    )

    parser.add_argument(
        "--fix_topic",
        default="/fix",
    )

    parser.add_argument(
        "--odom_topics",
        nargs="+",
        default=[
            "/odometry/gps",
            "/odometry/filtered",
            "/odometry/filtered_global",
        ],
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create trajectory_xy.png",
    )

    args = parser.parse_args()

    bag_uri = Path(args.bag)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_topics = [args.fix_topic] + args.odom_topics

    reader = open_reader(bag_uri, args.storage_id)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    print("\n[INFO] Topics in bag:")
    for name, typ in sorted(type_map.items()):
        if name in selected_topics:
            print(f"  SELECTED {name}: {typ}")

    msg_type_cache = {}
    for topic in selected_topics:
        if topic in type_map:
            msg_type_cache[topic] = get_message(type_map[topic])
        else:
            print(f"[WARN] Topic not found in bag: {topic}")

    fix_rows = []
    odom_rows_by_topic = defaultdict(list)
    counts = defaultdict(int)

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()

        if topic not in msg_type_cache:
            continue

        msg_type = msg_type_cache[topic]
        msg = deserialize_message(data, msg_type)
        counts[topic] += 1

        bag_time_sec = float(bag_time_ns) * 1e-9

        if type_map[topic] == "sensor_msgs/msg/NavSatFix":
            t = stamp_to_sec(msg.header.stamp)

            cov = list(msg.position_covariance)

            fix_rows.append({
                "t": t,
                "bag_time": bag_time_sec,
                "frame_id": msg.header.frame_id,
                "status": int(msg.status.status),
                "service": int(msg.status.service),
                "latitude": float(msg.latitude),
                "longitude": float(msg.longitude),
                "altitude": float(msg.altitude),
                "cov_xx": cov[0],
                "cov_xy": cov[1],
                "cov_xz": cov[2],
                "cov_yx": cov[3],
                "cov_yy": cov[4],
                "cov_yz": cov[5],
                "cov_zx": cov[6],
                "cov_zy": cov[7],
                "cov_zz": cov[8],
                "covariance_type": int(msg.position_covariance_type),
            })

        elif type_map[topic] == "nav_msgs/msg/Odometry":
            t = stamp_to_sec(msg.header.stamp)

            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            v = msg.twist.twist.linear
            w = msg.twist.twist.angular

            yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

            pose_cov = list(msg.pose.covariance)
            twist_cov = list(msg.twist.covariance)

            odom_rows_by_topic[topic].append({
                "t": t,
                "bag_time": bag_time_sec,
                "frame_id": msg.header.frame_id,
                "child_frame_id": msg.child_frame_id,
                "x": float(p.x),
                "y": float(p.y),
                "z": float(p.z),
                "qx": float(q.x),
                "qy": float(q.y),
                "qz": float(q.z),
                "qw": float(q.w),
                "yaw": yaw,
                "vx": float(v.x),
                "vy": float(v.y),
                "vz": float(v.z),
                "wx": float(w.x),
                "wy": float(w.y),
                "wz": float(w.z),
                "pose_cov_xx": pose_cov[0],
                "pose_cov_xy": pose_cov[1],
                "pose_cov_yy": pose_cov[7],
                "pose_cov_yaw": pose_cov[35],
                "twist_cov_vx": twist_cov[0],
                "twist_cov_vy": twist_cov[7],
                "twist_cov_wz": twist_cov[35],
            })

    print("\n[INFO] Counts:")
    for topic in selected_topics:
        print(f"  {topic}: {counts[topic]}")

    if fix_rows:
        fix_csv = out_dir / "fix.csv"
        write_csv(
            fix_csv,
            fix_rows,
            [
                "t", "bag_time", "frame_id",
                "status", "service",
                "latitude", "longitude", "altitude",
                "cov_xx", "cov_xy", "cov_xz",
                "cov_yx", "cov_yy", "cov_yz",
                "cov_zx", "cov_zy", "cov_zz",
                "covariance_type",
            ],
        )
        print(f"[OK] Wrote {fix_csv}")

        lat0 = fix_rows[0]["latitude"]
        lon0 = fix_rows[0]["longitude"]
        alt0 = fix_rows[0]["altitude"]

        fix_enu_rows = []
        for row in fix_rows:
            x, y = local_enu_from_llh(
                row["latitude"],
                row["longitude"],
                lat0,
                lon0,
            )

            fix_enu_rows.append({
                "t": row["t"],
                "bag_time": row["bag_time"],
                "frame_id": row["frame_id"],
                "x_east_m": x,
                "y_north_m": y,
                "z_up_m": row["altitude"] - alt0,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "altitude": row["altitude"],
                "status": row["status"],
                "cov_xx": row["cov_xx"],
                "cov_yy": row["cov_yy"],
                "cov_zz": row["cov_zz"],
            })

        fix_enu_csv = out_dir / "fix_enu.csv"
        write_csv(
            fix_enu_csv,
            fix_enu_rows,
            [
                "t", "bag_time", "frame_id",
                "x_east_m", "y_north_m", "z_up_m",
                "latitude", "longitude", "altitude",
                "status",
                "cov_xx", "cov_yy", "cov_zz",
            ],
        )
        print(f"[OK] Wrote {fix_enu_csv}")

    for topic, rows in odom_rows_by_topic.items():
        if not rows:
            continue

        filename = sanitize_topic(topic) + ".csv"
        path = out_dir / filename

        write_csv(
            path,
            rows,
            [
                "t", "bag_time",
                "frame_id", "child_frame_id",
                "x", "y", "z",
                "qx", "qy", "qz", "qw", "yaw",
                "vx", "vy", "vz",
                "wx", "wy", "wz",
                "pose_cov_xx", "pose_cov_xy", "pose_cov_yy", "pose_cov_yaw",
                "twist_cov_vx", "twist_cov_vy", "twist_cov_wz",
            ],
        )
        print(f"[OK] Wrote {path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib not installed. Install with: pip install matplotlib")
            return

        plt.figure()

        if fix_rows:
            xs = []
            ys = []
            lat0 = fix_rows[0]["latitude"]
            lon0 = fix_rows[0]["longitude"]
            for row in fix_rows:
                x, y = local_enu_from_llh(
                    row["latitude"],
                    row["longitude"],
                    lat0,
                    lon0,
                )
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, label="/fix local ENU")

        for topic, rows in odom_rows_by_topic.items():
            if not rows:
                continue
            xs = [r["x"] for r in rows]
            ys = [r["y"] for r in rows]
            plt.plot(xs, ys, label=topic)

        plt.xlabel("x / East [m]")
        plt.ylabel("y / North [m]")
        plt.title("GNSS / EKF Trajectory Comparison")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

        png_path = out_dir / "trajectory_xy.png"
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"[OK] Wrote {png_path}")


if __name__ == "__main__":
    main()
