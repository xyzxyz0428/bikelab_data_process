#!/usr/bin/env python3
"""Create a short read-only-derived GNSS bag and reconstructed IMU CSV.

The source is a completed four-way fusion bag. GNSS input messages are copied
without deserialization. The IMU CSV is reconstructed from the previously
published raw-rate and AHRS-rate messages so the same file player can be used
for parameter screening when the original data volume is offline.
"""

import argparse
import csv
import math
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


GNSS_TOPICS = {
    "/fix", "/ubx_nav_vel_ned", "/ubx_nav_pvt", "/ubx_nav_hp_pos_llh",
}
RAW_IMU = "/imu/raw_gyro_rate"
AHRS_IMU = "/imu/ahrs_heading_rate"


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--bag-out", required=True)
    ap.add_argument("--imu-out", required=True)
    ap.add_argument("--start-ns", type=int, required=True)
    ap.add_argument("--end-ns", type=int, required=True)
    args = ap.parse_args()

    source = Path(args.source).resolve()
    bag_out = Path(args.bag_out).resolve()
    imu_out = Path(args.imu_out).resolve()
    if bag_out.exists() or imu_out.exists():
        raise SystemExit("Refusing to overwrite an existing output")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(source), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    metadata = {x.name: x for x in reader.get_all_topics_and_types()}

    bag_out.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_out), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for name in sorted(GNSS_TOPICS):
        item = metadata[name]
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=name, type=item.type, serialization_format="cdr",
        ))

    imu_type = get_message(metadata[RAW_IMU].type)
    raw_rows = []
    ahrs_rows = []
    copied = {name: 0 for name in GNSS_TOPICS}
    while reader.has_next():
        topic, data, record_ns = reader.read_next()
        if topic in GNSS_TOPICS and args.start_ns <= record_ns <= args.end_ns:
            writer.write(topic, data, record_ns)
            copied[topic] += 1
        if topic not in {RAW_IMU, AHRS_IMU}:
            continue
        msg = deserialize_message(data, imu_type)
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        # Keep the original stationary start for covariance/bias estimation,
        # plus the short replay interval used in the sweep.
        if not (stamp_ns <= args.start_ns or args.start_ns <= stamp_ns <= args.end_ns):
            continue
        yaw = yaw_from_quaternion(msg.orientation)
        # Inverse of yaw = pi/2 - heading - pi.
        heading = math.atan2(math.sin(-math.pi / 2.0 - yaw), math.cos(-math.pi / 2.0 - yaw))
        common = {
            "t_unix_ns": stamp_ns,
            "heading": heading,
            "roll": 0.0,
            "pitch": 0.0,
            "gyro_x": msg.angular_velocity.x,
            "gyro_y": -msg.angular_velocity.y,
            "gyro_z": -msg.angular_velocity.z,
            "acc_x": msg.linear_acceleration.x,
            "acc_y": -msg.linear_acceleration.y,
            "acc_z": -msg.linear_acceleration.z,
            "crc8_ok": 1,
            "crc16_ok": 1,
            "end_ok": 1,
        }
        if topic == RAW_IMU:
            raw_rows.append({**common, "dtype": 64, "headingspeed": ""})
        else:
            ahrs_rows.append({
                **common, "dtype": 65,
                "headingspeed": -msg.angular_velocity.z,
            })

    # Only the first 12 s are needed before the replay interval to estimate
    # stationary covariance/bias. Keeping the entire earlier IMU stream would
    # only increase loading time.
    earliest = min(r["t_unix_ns"] for r in raw_rows + ahrs_rows)
    static_end = earliest + 12_000_000_000
    rows = [
        r for r in raw_rows + ahrs_rows
        if r["t_unix_ns"] <= static_end
        or args.start_ns <= r["t_unix_ns"] <= args.end_ns
    ]
    rows.sort(key=lambda r: (r["t_unix_ns"], r["dtype"]))
    imu_out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t_unix_ns", "dtype", "heading", "roll", "pitch",
        "gyro_x", "gyro_y", "gyro_z", "headingspeed",
        "acc_x", "acc_y", "acc_z", "crc8_ok", "crc16_ok", "end_ok",
    ]
    with imu_out.open("w", newline="") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)
    print({"copied_gnss": copied, "imu_rows": len(rows)})


if __name__ == "__main__":
    main()
