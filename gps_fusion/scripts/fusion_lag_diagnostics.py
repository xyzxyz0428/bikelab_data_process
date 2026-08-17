#!/usr/bin/env python3
"""Measure input arrival lag and trajectory jumps in a fusion result bag."""

import argparse
import csv
import json
import math
from bisect import bisect_left
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


NS_PER_SECOND = 1_000_000_000
INPUT_TOPICS = (
    "/compare_input/gps",
    "/compare_input/course",
    "/compare_input/raw_gyro_rate",
    "/compare_input/ahrs_heading_rate",
)
SOURCE_TOPICS = (
    "/fix",
    "/ubx_nav_vel_ned",
)
OUTPUT_TOPICS = (
    "/compare/g02_gps_course",
    "/compare/g03_gps_course_raw_gyro",
    "/compare/g04_gps_course_ahrs_rate",
)
COURSE_TOPIC = "/gnss/course_imu"


def stamp_ns(message):
    """Return the ROS header stamp as integer nanoseconds."""
    return (
        int(message.header.stamp.sec) * NS_PER_SECOND
        + int(message.header.stamp.nanosec)
    )


def quaternion_yaw(orientation):
    """Return ENU yaw from a geometry_msgs quaternion."""
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def percentile(values, probability):
    """Return a linearly interpolated percentile without NumPy."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = probability * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values):
    """Return compact distribution statistics."""
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "median": percentile(finite, 0.5),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
    }


def angular_difference_deg(first, second):
    """Return the absolute wrapped angular difference in degrees."""
    difference = math.atan2(
        math.sin(first - second),
        math.cos(first - second),
    )
    return abs(math.degrees(difference))


def read_bag(bag_path):
    """Read only the topics needed for lag and trajectory diagnostics."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    wanted = (
        set(SOURCE_TOPICS)
        | set(INPUT_TOPICS)
        | set(OUTPUT_TOPICS)
        | {COURSE_TOPIC}
    )
    missing = sorted(wanted - set(topic_types))
    if missing:
        raise RuntimeError("Missing topics: " + ", ".join(missing))
    message_types = {
        topic: get_message(topic_types[topic]) for topic in wanted
    }

    input_events = []
    outputs = {topic: [] for topic in OUTPUT_TOPICS}
    courses = []
    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        if topic not in wanted:
            continue
        message = deserialize_message(serialized, message_types[topic])
        header_ns = stamp_ns(message)
        if topic in SOURCE_TOPICS or topic in INPUT_TOPICS:
            input_events.append({
                "topic": topic,
                "record_ns": int(record_ns),
                "header_ns": header_ns,
                "arrival_minus_header_s": (
                    int(record_ns) - header_ns
                ) / NS_PER_SECOND,
            })
        elif topic == COURSE_TOPIC:
            courses.append((
                header_ns,
                quaternion_yaw(message.orientation),
            ))
        else:
            position = message.pose.pose.position
            outputs[topic].append({
                "record_ns": int(record_ns),
                "header_ns": header_ns,
                "x": float(position.x),
                "y": float(position.y),
                "yaw": quaternion_yaw(message.pose.pose.orientation),
            })
    return input_events, outputs, courses


def source_metrics(events):
    """Summarize record/header timing for the replayed receiver messages."""
    result = {}
    for topic in SOURCE_TOPICS:
        selected = [event for event in events if event["topic"] == topic]
        result[topic] = {
            "message_count": len(selected),
            "arrival_minus_header_s": summarize([
                event["arrival_minus_header_s"] for event in selected
            ]),
        }
    return result


def input_metrics(events):
    """Summarize record/header lag and cross-topic out-of-sequence arrival."""
    events = sorted(
        (event for event in events if event["topic"] in INPUT_TOPICS),
        key=lambda item: item["record_ns"],
    )
    latest_header_ns = None
    for event in events:
        if latest_header_ns is None:
            event["out_of_sequence_s"] = 0.0
            latest_header_ns = event["header_ns"]
            continue
        event["out_of_sequence_s"] = max(
            0.0,
            (latest_header_ns - event["header_ns"]) / NS_PER_SECOND,
        )
        latest_header_ns = max(latest_header_ns, event["header_ns"])

    result = {}
    for topic in INPUT_TOPICS:
        selected = [event for event in events if event["topic"] == topic]
        oos = [
            event["out_of_sequence_s"]
            for event in selected
            if event["out_of_sequence_s"] > 0.0
        ]
        result[topic] = {
            "message_count": len(selected),
            "arrival_minus_header_s": summarize([
                event["arrival_minus_header_s"] for event in selected
            ]),
            "out_of_sequence": {
                "count": len(oos),
                "fraction": len(oos) / len(selected) if selected else None,
                "delay_s": summarize(oos),
            },
        }
    return result, events


def trajectory_metrics(rows, turn_range=None):
    """Summarize output rate, apparent speed, and backward corrections."""
    if turn_range is not None:
        start_ns, end_ns = turn_range
        rows = [
            row for row in rows
            if start_ns <= row["header_ns"] <= end_ns
        ]
    rows = sorted(rows, key=lambda item: item["record_ns"])
    speeds = []
    step_distances = []
    backward_distances = []
    nonpositive_dt = 0
    for previous, current in zip(rows, rows[1:]):
        dt = (current["header_ns"] - previous["header_ns"]) / NS_PER_SECOND
        if dt <= 0.0:
            nonpositive_dt += 1
            continue
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        distance = math.hypot(dx, dy)
        speed = distance / dt
        step_distances.append(distance)
        speeds.append(speed)
        forward = dx * math.cos(previous["yaw"]) + dy * math.sin(
            previous["yaw"]
        )
        if forward < -0.05:
            backward_distances.append(-forward)

    if len(rows) >= 2:
        header_span_s = (
            max(row["header_ns"] for row in rows)
            - min(row["header_ns"] for row in rows)
        ) / NS_PER_SECOND
    else:
        header_span_s = 0.0
    return {
        "message_count": len(rows),
        "header_span_s": header_span_s,
        "effective_rate_hz": (
            (len(rows) - 1) / header_span_s if header_span_s > 0 else None
        ),
        "nonpositive_header_step_count": nonpositive_dt,
        "step_distance_m": summarize(step_distances),
        "apparent_speed_mps": summarize(speeds),
        "apparent_speed_over_15_mps_count": sum(
            speed > 15.0 for speed in speeds
        ),
        "backward_along_heading_over_0_05_m_count": len(
            backward_distances
        ),
        "backward_along_heading_m": summarize(backward_distances),
    }


def yaw_course_metrics(rows, courses, maximum_time_difference_s=0.25):
    """Compare output yaw with the nearest direct GNSS course message."""
    course_rows = sorted(courses)
    course_times = [item[0] for item in course_rows]
    differences = []
    for row in rows:
        index = bisect_left(course_times, row["header_ns"])
        candidates = []
        if index < len(course_rows):
            candidates.append(course_rows[index])
        if index > 0:
            candidates.append(course_rows[index - 1])
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda item: abs(item[0] - row["header_ns"]),
        )
        time_difference_s = abs(nearest[0] - row["header_ns"]) / NS_PER_SECOND
        if time_difference_s <= maximum_time_difference_s:
            differences.append(angular_difference_deg(row["yaw"], nearest[1]))
    return summarize(differences)


def write_input_events(path, events):
    """Write per-message input timing values for audit and plotting."""
    fields = (
        "topic",
        "record_ns",
        "header_ns",
        "arrival_minus_header_s",
        "out_of_sequence_s",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)


def write_summary_csv(path, report):
    """Write a compact table of the main comparison metrics."""
    fields = (
        "scope",
        "topic",
        "count",
        "median",
        "p95",
        "p99",
        "max",
    )
    rows = []
    for topic, metrics in report["source_timing"].items():
        rows.append({
            "scope": "source_arrival_minus_header_s",
            "topic": topic,
            **metrics["arrival_minus_header_s"],
        })
    for topic, metrics in report["input_timing"].items():
        for name in ("arrival_minus_header_s",):
            values = metrics[name]
            rows.append({
                "scope": name,
                "topic": topic,
                **values,
            })
        values = metrics["out_of_sequence"]["delay_s"]
        rows.append({
            "scope": "out_of_sequence_delay_s",
            "topic": topic,
            **values,
        })
    for interval_name in ("full", "selected_turn"):
        for topic, metrics in report["trajectory"][interval_name].items():
            for name in (
                "step_distance_m",
                "apparent_speed_mps",
                "backward_along_heading_m",
            ):
                values = metrics[name]
                rows.append({
                    "scope": f"{interval_name}_{name}",
                    "topic": topic,
                    **values,
                })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turn-manifest", type=Path)
    args = parser.parse_args()

    if not args.bag.is_dir():
        parser.error(f"Bag directory not found: {args.bag}")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite: {args.output_dir}")

    turn_range = None
    turn_manifest = None
    if args.turn_manifest:
        turn_manifest = json.loads(
            args.turn_manifest.read_text(encoding="utf-8")
        )
        turn_range = (
            int(turn_manifest["start_ns"]),
            int(turn_manifest["end_ns"]),
        )

    timing_events, outputs, courses = read_bag(args.bag)
    sources = source_metrics(timing_events)
    timing, timed_events = input_metrics(timing_events)
    report = {
        "schema_version": 1,
        "label": args.label,
        "bag": str(args.bag.resolve()),
        "turn_manifest": (
            str(args.turn_manifest.resolve()) if args.turn_manifest else None
        ),
        "selected_turn": turn_manifest,
        "definitions": {
            "arrival_minus_header_s": (
                "bag record timestamp minus message header timestamp"
            ),
            "out_of_sequence_delay_s": (
                "latest header timestamp already seen on any comparison "
                "input minus the current input header timestamp"
            ),
            "apparent_speed_mps": (
                "distance between consecutive output positions divided by "
                "their positive header-time difference"
            ),
            "backward_along_heading": (
                "consecutive displacement projected opposite to the prior "
                "fused yaw, counted when the backward component exceeds 0.05 m"
            ),
            "yaw_course_difference_deg": (
                "absolute wrapped heading-angle difference between fused yaw "
                "and the nearest direct GNSS-derived course over ground"
            ),
        },
        "source_timing": sources,
        "input_timing": timing,
        "trajectory": {
            "full": {
                topic: trajectory_metrics(rows)
                for topic, rows in outputs.items()
            },
            "selected_turn": {
                topic: trajectory_metrics(rows, turn_range)
                for topic, rows in outputs.items()
            },
        },
        "yaw_course_consistency_deg": {
            topic: yaw_course_metrics(rows, courses)
            for topic, rows in outputs.items()
        },
    }

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "lag_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    write_input_events(args.output_dir / "input_timing.csv", timed_events)
    write_summary_csv(args.output_dir / "diagnostic_summary.csv", report)
    print(args.output_dir / "lag_diagnostics.json")


if __name__ == "__main__":
    main()
