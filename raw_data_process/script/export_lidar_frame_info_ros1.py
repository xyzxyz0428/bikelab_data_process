#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export ROS1 PointCloud2 frame metadata directly from an offline rosbag.

Example:
  python3 export_lidar_frame_info_from_rosbag_ros1.py \
      --bag your_data.bag \
      --topic /rslidar_points_200 \
      --out lidar_200_frames.csv
"""

import os
import csv
import argparse

import rosbag
from sensor_msgs.msg import PointCloud2


def main():
    parser = argparse.ArgumentParser(
        description="Export PointCloud2 frame metadata from a ROS1 bag"
    )
    parser.add_argument("--bag", required=True, help="Path to ROS1 bag file")
    parser.add_argument("--topic", required=True, help="PointCloud2 topic name")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    count = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t_unix_ns",
            "frame_id",
            "width",
            "height",
            "point_count",
            "row_step",
            "point_step",
            "is_dense",
        ])

        with rosbag.Bag(args.bag, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=[args.topic]):
                if not isinstance(msg, PointCloud2):
                    continue

                stamp_sec = int(msg.header.stamp.secs)
                stamp_nsec = int(msg.header.stamp.nsecs)
                t_unix_ns = stamp_sec * 10**9 + stamp_nsec

                width = int(msg.width)
                height = int(msg.height)
                point_count = width * height

                writer.writerow([
                    t_unix_ns,
                    msg.header.frame_id,
                    width,
                    height,
                    point_count,
                    int(msg.row_step),
                    int(msg.point_step),
                    int(msg.is_dense),
                ])
                count += 1

                if count % 500 == 0:
                    print(f"[{args.topic}] exported {count} frames")

    print(f"Done. Exported {count} frames to {args.out}")


if __name__ == "__main__":
    main()