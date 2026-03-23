#!/usr/bin/env python3
"""
Subscribe to a point cloud topic and export frame timestamps + point counts.

Example:
  python3 export_lidar_frame_info_ros2.py \
      --topic /rslidar_points_200 \
      --out lidar_200_frames.csv

Run one instance per LiDAR topic.
"""

import csv
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class FrameExporter(Node):
    def __init__(self, topic: str, out_csv: str):
        super().__init__('lidar_frame_exporter_' + topic.strip('/').replace('/', '_'))
        self.out_csv = out_csv
        self.csv_f = open(out_csv, 'w', newline='')
        self.writer = csv.writer(self.csv_f)
        self.writer.writerow([
            'topic', 'frame_id', 'stamp_sec', 'stamp_nanosec', 't_unix_ns',
            'width', 'height', 'point_count'
        ])
        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 10)
        self.get_logger().info(f'Listening on {topic}, writing to {out_csv}')

    def cb(self, msg: PointCloud2):
        t_unix_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        # Fast point count: width * height is usually enough for PointCloud2
        point_count = int(msg.width) * int(msg.height)
        self.writer.writerow([
            msg._type if hasattr(msg, '_type') else 'sensor_msgs/PointCloud2',
            msg.header.frame_id,
            msg.header.stamp.sec,
            msg.header.stamp.nanosec,
            t_unix_ns,
            msg.width,
            msg.height,
            point_count,
        ])
        self.csv_f.flush()

    def destroy_node(self):
        try:
            self.csv_f.close()
        finally:
            super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    rclpy.init()
    node = FrameExporter(args.topic, args.out)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
