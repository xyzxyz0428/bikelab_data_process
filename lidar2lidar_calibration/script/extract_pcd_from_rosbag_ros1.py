#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
from pathlib import Path

import rosbag
import sensor_msgs.point_cloud2 as pc2


def write_pcd_ascii(path, points, fields=("x", "y", "z", "intensity")):
    """
    Write ASCII PCD file.

    points: list of tuples, e.g. (x, y, z, intensity)
    """
    n = len(points)
    field_names = list(fields)

    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS " + " ".join(field_names) + "\n")
        f.write("SIZE " + " ".join(["4"] * len(field_names)) + "\n")
        f.write("TYPE " + " ".join(["F"] * len(field_names)) + "\n")
        f.write("COUNT " + " ".join(["1"] * len(field_names)) + "\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")

        for p in points:
            f.write(" ".join(f"{float(v):.8f}" for v in p) + "\n")


def get_available_fields(msg):
    return [f.name for f in msg.fields]


def extract_points(msg, prefer_intensity=True):
    available = get_available_fields(msg)

    if prefer_intensity and "intensity" in available:
        wanted_fields = ["x", "y", "z", "intensity"]
    else:
        wanted_fields = ["x", "y", "z"]

    points = []
    for p in pc2.read_points(
        msg,
        field_names=wanted_fields,
        skip_nans=True,
    ):
        vals = tuple(float(v) for v in p)

        # extra finite check
        if all(math.isfinite(v) for v in vals[:3]):
            points.append(vals)

    return points, wanted_fields


def main():
    ap = argparse.ArgumentParser(
        description="Extract PointCloud2 frames from ROS1 bag to PCD files."
    )
    ap.add_argument("--bag", required=True, help="Input ROS1 bag file")
    ap.add_argument("--topic", required=True, help="PointCloud2 topic, e.g. /middle_lidar/points")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--start-unix-ns", type=int, default=None, help="Optional start time")
    ap.add_argument("--end-unix-ns", type=int, default=None, help="Optional end time")
    ap.add_argument("--max-frames", type=int, default=None, help="Optional limit")
    ap.add_argument("--every-n", type=int, default=1, help="Save every Nth frame")
    ap.add_argument("--no-intensity", action="store_true", help="Only save x y z")
    args = ap.parse_args()

    bag_path = Path(args.bag)
    outdir = Path(args.outdir)
    pcd_dir = outdir / "pcd"
    pcd_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "lidar_frames.csv"

    saved = 0
    seen = 0

    with rosbag.Bag(str(bag_path), "r") as bag, open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "frame_idx",
                "t_unix_ns",
                "t_unix_s",
                "pcd_path",
                "n_points",
                "frame_id",
                "topic",
            ],
        )
        writer.writeheader()

        for topic, msg, bag_time in bag.read_messages(topics=[args.topic]):
            # Prefer message header timestamp, because it should represent sensor/frame time.
            if hasattr(msg, "header") and msg.header.stamp is not None:
                t_unix_ns = int(msg.header.stamp.secs) * 1_000_000_000 + int(msg.header.stamp.nsecs)
                frame_id = msg.header.frame_id
            else:
                t_unix_ns = int(bag_time.secs) * 1_000_000_000 + int(bag_time.nsecs)
                frame_id = ""

            if args.start_unix_ns is not None and t_unix_ns < args.start_unix_ns:
                continue
            if args.end_unix_ns is not None and t_unix_ns > args.end_unix_ns:
                continue

            seen += 1
            if (seen - 1) % args.every_n != 0:
                continue

            points, fields = extract_points(
                msg,
                prefer_intensity=not args.no_intensity,
            )

            if len(points) == 0:
                print(f"[WARN] empty cloud at t={t_unix_ns}")
                continue

            pcd_name = f"lidar_frame_{saved:06d}.pcd"
            pcd_path = pcd_dir / pcd_name

            write_pcd_ascii(pcd_path, points, fields=fields)

            writer.writerow({
                "frame_idx": saved,
                "t_unix_ns": t_unix_ns,
                "t_unix_s": t_unix_ns / 1e9,
                "pcd_path": pcd_name,
                "n_points": len(points),
                "frame_id": frame_id,
                "topic": topic,
            })

            saved += 1

            if saved % 20 == 0:
                print(f"[INFO] saved {saved} frames, latest t={t_unix_ns}, points={len(points)}")

            if args.max_frames is not None and saved >= args.max_frames:
                break

    print("\n[OK] finished")
    print(f"saved frames: {saved}")
    print(f"pcd folder: {pcd_dir}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()