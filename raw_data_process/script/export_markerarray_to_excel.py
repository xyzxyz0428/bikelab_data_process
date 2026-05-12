#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys

import rosbag


def safe_to_sec(stamp):
    """Convert ROS time/duration to float seconds safely."""
    try:
        return stamp.to_sec()
    except Exception:
        try:
            return float(stamp.secs) + float(stamp.nsecs) * 1e-9
        except Exception:
            return None


def marker_type_to_name(marker_type):
    mapping = {
        0: "ARROW",
        1: "CUBE",
        2: "SPHERE",
        3: "CYLINDER",
        4: "LINE_STRIP",
        5: "LINE_LIST",
        6: "CUBE_LIST",
        7: "SPHERE_LIST",
        8: "POINTS",
        9: "TEXT_VIEW_FACING",
        10: "MESH_RESOURCE",
        11: "TRIANGLE_LIST",
    }
    return mapping.get(marker_type, f"UNKNOWN_{marker_type}")


def marker_action_to_name(action):
    mapping = {
        0: "ADD/MODIFY",
        1: "DEPRECATED",
        2: "DELETE",
        3: "DELETEALL",
    }
    return mapping.get(action, f"UNKNOWN_{action}")


FIELDNAMES = [
    # order keys / traceability
    "bag_time",
    "header_stamp",
    "ns",
    "id",
    "marker_index_in_array",
    "bag_msg_index",

    # basic info
    "topic",
    "frame_id",
    "type",
    "type_name",
    "action",
    "action_name",
    "text",
    "mesh_resource",

    # pose
    "pose_x",
    "pose_y",
    "pose_z",
    "ori_x",
    "ori_y",
    "ori_z",
    "ori_w",

    # scale
    "scale_x",
    "scale_y",
    "scale_z",

    # color
    "color_r",
    "color_g",
    "color_b",
    "color_a",

    # other
    "lifetime_sec",
    "frame_locked",
    "points_count",
    "colors_count",
]


class CsvChunkWriter:
    def __init__(self, output_prefix, max_rows):
        self.output_prefix = output_prefix
        self.max_rows = max_rows
        self.part_idx = 0
        self.rows_in_current_file = 0
        self.total_rows = 0
        self.current_fp = None
        self.current_writer = None

    def _open_new_file(self):
        self.close()

        self.part_idx += 1
        out_path = f"{self.output_prefix}_part{self.part_idx:03d}.csv"
        self.current_fp = open(out_path, "w", newline="", encoding="utf-8-sig")
        self.current_writer = csv.DictWriter(self.current_fp, fieldnames=FIELDNAMES)
        self.current_writer.writeheader()
        self.rows_in_current_file = 0
        print(f"[INFO] Opened {out_path}")

    def write_row(self, row):
        if self.current_fp is None or self.rows_in_current_file >= self.max_rows:
            self._open_new_file()

        self.current_writer.writerow(row)
        self.rows_in_current_file += 1
        self.total_rows += 1

    def close(self):
        if self.current_fp is not None:
            self.current_fp.close()
            self.current_fp = None
            self.current_writer = None


def build_row(topic, bag_msg_index, bag_time_sec, marker_index, m):
    header_stamp_sec = safe_to_sec(m.header.stamp)
    lifetime_sec = safe_to_sec(m.lifetime)

    return {
        "bag_time": bag_time_sec,
        "header_stamp": header_stamp_sec,
        "ns": m.ns,
        "id": m.id,
        "marker_index_in_array": marker_index,
        "bag_msg_index": bag_msg_index,

        "topic": topic,
        "frame_id": m.header.frame_id,
        "type": m.type,
        "type_name": marker_type_to_name(m.type),
        "action": m.action,
        "action_name": marker_action_to_name(m.action),
        "text": m.text,
        "mesh_resource": m.mesh_resource,

        "pose_x": m.pose.position.x,
        "pose_y": m.pose.position.y,
        "pose_z": m.pose.position.z,
        "ori_x": m.pose.orientation.x,
        "ori_y": m.pose.orientation.y,
        "ori_z": m.pose.orientation.z,
        "ori_w": m.pose.orientation.w,

        "scale_x": m.scale.x,
        "scale_y": m.scale.y,
        "scale_z": m.scale.z,

        "color_r": m.color.r,
        "color_g": m.color.g,
        "color_b": m.color.b,
        "color_a": m.color.a,

        "lifetime_sec": lifetime_sec,
        "frame_locked": bool(m.frame_locked),
        "points_count": len(m.points),
        "colors_count": len(m.colors),
    }


def write_ns_summary(output_prefix, ns_counter):
    out_path = f"{output_prefix}_ns_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(["ns", "count"])
        for ns, count in sorted(ns_counter.items(), key=lambda x: (-x[1], str(x[0]))):
            writer.writerow([ns, count])
    print(f"[INFO] Saved namespace summary -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream-export ROS1 MarkerArray topic from rosbag to CSV chunks."
    )
    parser.add_argument("bag", help="Path to input .bag file")
    parser.add_argument(
        "--topic",
        default="/perception_info_rviz",
        help="MarkerArray topic name (default: /perception_info_rviz)",
    )
    parser.add_argument(
        "--output-prefix",
        default="perception_info_rviz",
        help="Prefix of output CSV files",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000000,
        help="Maximum data rows per CSV file, excluding header",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Print progress every N bag messages",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        raise FileNotFoundError(f"Bag file not found: {args.bag}")
    if args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    writer = CsvChunkWriter(args.output_prefix, args.max_rows)
    ns_counter = {}

    processed_msgs = 0
    matched_msgs = 0
    matched_markers = 0

    try:
        with rosbag.Bag(args.bag, "r") as bag:
            for bag_msg_index, (topic, msg, bag_time) in enumerate(
                bag.read_messages(topics=[args.topic])
            ):
                processed_msgs += 1
                matched_msgs += 1
                bag_time_sec = safe_to_sec(bag_time)

                markers = getattr(msg, "markers", [])
                if not markers:
                    continue

                # 只对同一条 MarkerArray 内部排序，避免同一 timestamp 下顺序混乱
                indexed_markers = list(enumerate(markers))
                indexed_markers.sort(
                    key=lambda pair: (
                        safe_to_sec(pair[1].header.stamp)
                        if safe_to_sec(pair[1].header.stamp) is not None else float("inf"),
                        str(pair[1].ns),
                        pair[1].id,
                        pair[0],
                    )
                )

                for marker_index, m in indexed_markers:
                    row = build_row(topic, bag_msg_index, bag_time_sec, marker_index, m)
                    writer.write_row(row)
                    matched_markers += 1
                    ns_counter[m.ns] = ns_counter.get(m.ns, 0) + 1

                if processed_msgs % args.progress_every == 0:
                    print(
                        f"[INFO] messages={processed_msgs}, markers={matched_markers}, "
                        f"current_part={writer.part_idx}, rows_in_current_file={writer.rows_in_current_file}"
                    )
                    sys.stdout.flush()

    finally:
        writer.close()

    if matched_msgs == 0:
        print(f"[WARN] No messages found on topic {args.topic} in {args.bag}")
        return

    write_ns_summary(args.output_prefix, ns_counter)

    print("[DONE]")
    print(f"  topic            : {args.topic}")
    print(f"  matched messages : {matched_msgs}")
    print(f"  exported markers : {matched_markers}")
    print(f"  output files     : {writer.part_idx} CSV part file(s)")


if __name__ == "__main__":
    main()