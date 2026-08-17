#!/usr/bin/env python3
"""Compare raw-z and AHRS-assisted tilt-compensated Group 3 runs."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COURSE_TOPIC = "/gnss/course_imu"
GROUP2_TOPIC = "/compare/g02_gps_course"
GROUP3_TOPIC = "/compare/g03_gps_course_raw_gyro"
RATE_TOPIC = "/imu/raw_gyro_rate"
TOPICS = {COURSE_TOPIC, GROUP2_TOPIC, GROUP3_TOPIC, RATE_TOPIC}


def message_stamp_ns(message):
    """Return a message header stamp as integer nanoseconds."""
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def quaternion_yaw(quaternion):
    """Return planar yaw from a ROS quaternion."""
    return math.atan2(
        2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        ),
        1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        ),
    )


def percentile(values, probability):
    """Return one quantile or NaN for an empty sequence."""
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def read_run(path, start_ns, end_ns):
    """Read the fixed evaluation interval from one result bag."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    message_types = {
        item.name: get_message(item.type)
        for item in reader.get_all_topics_and_types()
        if item.name in TOPICS
    }
    missing = TOPICS.difference(message_types)
    if missing:
        raise RuntimeError(f"Missing topics in {path}: {sorted(missing)}")

    data = {topic: [] for topic in TOPICS}
    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        if topic not in TOPICS:
            continue
        message = deserialize_message(serialized, message_types[topic])
        header_ns = message_stamp_ns(message)
        if not start_ns <= header_ns <= end_ns:
            continue
        if topic == COURSE_TOPIC:
            data[topic].append({
                "record_ns": int(record_ns),
                "header_ns": header_ns,
                "yaw": quaternion_yaw(message.orientation),
            })
        elif topic == RATE_TOPIC:
            data[topic].append({
                "record_ns": int(record_ns),
                "header_ns": header_ns,
                "rate": float(message.angular_velocity.z),
                "variance": float(message.angular_velocity_covariance[8]),
            })
        else:
            data[topic].append({
                "record_ns": int(record_ns),
                "header_ns": header_ns,
                "yaw": quaternion_yaw(message.pose.pose.orientation),
                "x": float(message.pose.pose.position.x),
                "y": float(message.pose.pose.position.y),
            })
    return data


def course_reference(course_rows):
    """Return unique, increasing COG samples suitable for interpolation."""
    ordered = sorted(course_rows, key=lambda row: row["header_ns"])
    by_stamp = {row["header_ns"]: row["yaw"] for row in ordered}
    times = np.asarray(sorted(by_stamp), dtype=np.int64)
    yaw = np.unwrap(np.asarray([by_stamp[t] for t in times], dtype=float))
    return times, yaw


def yaw_course_metrics(output_rows, course_rows):
    """Measure internal yaw consistency with the interpolated COG input."""
    course_time, course_yaw = course_reference(course_rows)
    if len(course_time) < 2:
        raise RuntimeError("At least two COG samples are required")
    rows = [
        row for row in output_rows
        if course_time[0] <= row["header_ns"] <= course_time[-1]
    ]
    output_time = np.asarray(
        [row["header_ns"] for row in rows], dtype=np.int64
    )
    output_yaw = np.asarray([row["yaw"] for row in rows], dtype=float)
    reference = np.interp(
        (output_time - course_time[0]).astype(float),
        (course_time - course_time[0]).astype(float),
        course_yaw,
    )
    difference = np.degrees(np.arctan2(
        np.sin(output_yaw - reference),
        np.cos(output_yaw - reference),
    ))
    span_s = (output_time.max() - output_time.min()) / 1e9
    return {
        "count": int(len(difference)),
        "effective_rate_hz": (
            float((len(difference) - 1) / span_s) if span_s > 0 else None
        ),
        "median_abs_deg": float(np.median(np.abs(difference))),
        "p95_abs_deg": float(np.quantile(np.abs(difference), 0.95)),
        "rmse_deg": float(np.sqrt(np.mean(difference ** 2))),
        "mean_signed_deg": float(np.mean(difference)),
    }


def trajectory_metrics(rows):
    """Calculate position-step diagnostics in bag-record order."""
    rows = sorted(rows, key=lambda row: row["record_ns"])
    step_distance = []
    apparent_speed = []
    backward_distance = []
    nonpositive_steps = 0
    for previous, current in zip(rows, rows[1:]):
        dt = (current["header_ns"] - previous["header_ns"]) / 1e9
        if dt <= 0.0:
            nonpositive_steps += 1
            continue
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        distance = math.hypot(dx, dy)
        step_distance.append(distance)
        apparent_speed.append(distance / dt)
        forward = (
            dx * math.cos(previous["yaw"])
            + dy * math.sin(previous["yaw"])
        )
        if forward < -0.05:
            backward_distance.append(-forward)
    return {
        "count": len(rows),
        "nonpositive_header_step_count": nonpositive_steps,
        "step_distance_p95_m": percentile(step_distance, 0.95),
        "apparent_speed_p95_mps": percentile(apparent_speed, 0.95),
        "backward_step_count": len(backward_distance),
        "backward_step_max_m": (
            max(backward_distance) if backward_distance else 0.0
        ),
    }


def paired_position_metrics(group2_rows, group3_rows):
    """Pair Group 2 and 3 positions by nearest header time."""
    group3 = sorted(group3_rows, key=lambda row: row["header_ns"])
    times = np.asarray([row["header_ns"] for row in group3], dtype=np.int64)
    distances = []
    for row in group2_rows:
        if not len(times):
            break
        index = int(np.searchsorted(times, row["header_ns"]))
        candidates = [candidate for candidate in (index - 1, index)
                      if 0 <= candidate < len(times)]
        nearest = min(
            candidates,
            key=lambda candidate: abs(
                int(times[candidate]) - row["header_ns"]
            ),
        )
        if abs(int(times[nearest]) - row["header_ns"]) > 30_000_000:
            continue
        other = group3[nearest]
        distances.append(math.hypot(
            row["x"] - other["x"],
            row["y"] - other["y"],
        ))
    return {
        "count": len(distances),
        "median_m": float(np.median(distances)),
        "p95_m": percentile(distances, 0.95),
    }


def rate_metrics(rows):
    """Summarize the actual Group 3 yaw-rate input in the result bag."""
    by_stamp = {row["header_ns"]: row for row in rows}
    ordered = [by_stamp[stamp] for stamp in sorted(by_stamp)]
    times = np.asarray([row["header_ns"] for row in ordered], dtype=np.int64)
    rates = np.asarray([row["rate"] for row in ordered], dtype=float)
    variances = np.asarray([row["variance"] for row in ordered], dtype=float)
    span_s = (times[-1] - times[0]) / 1e9 if len(times) > 1 else 0.0
    relative_time = (
        (times - times[0]).astype(float) / 1e9 if len(times) else []
    )
    integral_deg = (
        math.degrees(float(np.trapz(rates, relative_time)))
        if len(times) > 1 else float("nan")
    )
    return {
        "count": len(times),
        "effective_rate_hz": (
            float((len(times) - 1) / span_s) if span_s > 0 else None
        ),
        "integral_deg": integral_deg,
        "variance_median_rad2_s2": (
            float(np.median(variances)) if len(variances) else float("nan")
        ),
    }


def flatten_run_metrics(variant, repeat, bag, data):
    """Return one flat CSV row and a structured report item."""
    yaw2 = yaw_course_metrics(data[GROUP2_TOPIC], data[COURSE_TOPIC])
    yaw3 = yaw_course_metrics(data[GROUP3_TOPIC], data[COURSE_TOPIC])
    trajectory2 = trajectory_metrics(data[GROUP2_TOPIC])
    trajectory3 = trajectory_metrics(data[GROUP3_TOPIC])
    paired = paired_position_metrics(
        data[GROUP2_TOPIC], data[GROUP3_TOPIC]
    )
    rate = rate_metrics(data[RATE_TOPIC])
    row = {
        "variant": variant,
        "repeat": repeat,
        "bag": str(bag.resolve()),
        "course_count": len(data[COURSE_TOPIC]),
    }
    for prefix, values in (
        ("group2_yaw", yaw2),
        ("group3_yaw", yaw3),
        ("group2_trajectory", trajectory2),
        ("group3_trajectory", trajectory3),
        ("group2_group3_position", paired),
        ("group3_rate", rate),
    ):
        for key, value in values.items():
            row[f"{prefix}_{key}"] = value
    for metric in ("median_abs_deg", "p95_abs_deg", "rmse_deg"):
        row[f"group3_minus_group2_{metric}"] = (
            yaw3[metric] - yaw2[metric]
        )
    report = {
        "variant": variant,
        "repeat": repeat,
        "bag": str(bag.resolve()),
        "yaw_course": {"group2": yaw2, "group3": yaw3},
        "trajectory": {"group2": trajectory2, "group3": trajectory3},
        "group2_group3_position": paired,
        "group3_rate": rate,
    }
    return row, report


def summarize_repeats(rows):
    """Aggregate selected metrics for each input variant."""
    selected = [
        "group2_yaw_median_abs_deg",
        "group2_yaw_p95_abs_deg",
        "group2_yaw_rmse_deg",
        "group3_yaw_median_abs_deg",
        "group3_yaw_p95_abs_deg",
        "group3_yaw_rmse_deg",
        "group3_minus_group2_median_abs_deg",
        "group3_minus_group2_p95_abs_deg",
        "group3_minus_group2_rmse_deg",
        "group3_trajectory_step_distance_p95_m",
        "group3_trajectory_apparent_speed_p95_mps",
        "group3_trajectory_backward_step_count",
        "group2_group3_position_p95_m",
        "group3_rate_integral_deg",
        "group3_rate_variance_median_rad2_s2",
    ]
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        item = {"variant": variant, "repeat_count": len(subset)}
        for metric in selected:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            item[f"{metric}_min"] = float(np.min(values))
            item[f"{metric}_max"] = float(np.max(values))
        summary.append(item)
    return summary


def write_csv(path, rows):
    """Write dictionaries to CSV."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_summary(path, summary):
    """Plot Group 2/3 yaw consistency and Group 3 step diagnostics."""
    labels = [
        "Raw gyro z" if row["variant"] == "raw_gyro_z"
        else "Tilt-compensated raw gyro"
        for row in summary
    ]
    x = np.arange(len(summary))
    colors = {"Group 2": "#4C78A8", "Group 3": "#F58518"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1))
    width = 0.34
    for axis, metric, title in zip(
        axes,
        ("median_abs_deg", "p95_abs_deg", "rmse_deg"),
        ("Median absolute difference", "P95 absolute difference", "RMSE"),
    ):
        for offset, group in ((-width / 2, "Group 2"), (width / 2, "Group 3")):
            prefix = group.lower().replace(" ", "")
            mean = [
                row[f"{prefix}_yaw_{metric}_mean"] for row in summary
            ]
            std = [
                row[f"{prefix}_yaw_{metric}_std"] for row in summary
            ]
            axis.bar(
                x + offset,
                mean,
                width,
                yerr=std,
                capsize=3,
                label=group,
                color=colors[group],
            )
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=12, ha="right")
        axis.set_ylabel("Yaw--COG difference (deg)")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_rate_comparison(path, reports, start_ns):
    """Plot one actual rate trace from each variant on the fixed turn."""
    chosen = {}
    for report in reports:
        chosen.setdefault(report["variant"], report)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), sharex=True)
    traces = {}
    for variant, report in chosen.items():
        data = read_run(Path(report["bag"]), start_ns, report["end_ns"])
        rows = sorted(data[RATE_TOPIC], key=lambda row: row["header_ns"])
        time = np.asarray([
            (row["header_ns"] - start_ns) / 1e9 for row in rows
        ])
        rate = np.asarray([row["rate"] for row in rows])
        traces[variant] = (time, rate)
        label = (
            "Raw gyro z" if variant == "raw_gyro_z"
            else "Tilt-compensated raw gyro"
        )
        axes[0].plot(time, np.degrees(rate), linewidth=1.3, label=label)
    raw_time, raw_rate = traces["raw_gyro_z"]
    tilt_time, tilt_rate = traces["tilt_compensated_raw_gyro"]
    interpolated_tilt = np.interp(raw_time, tilt_time, tilt_rate)
    axes[1].plot(
        raw_time,
        np.degrees(interpolated_tilt - raw_rate),
        color="#7A5195",
        linewidth=1.3,
    )
    axes[0].set_ylabel("Yaw rate (deg/s)")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Tilt minus raw (deg/s)")
    axes[1].set_xlabel("Time in selected turn (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def main():
    """Run the repeated fixed-window evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="VARIANT,REPEAT,BAG",
    )
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--end-ns", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    rows = []
    reports = []
    for run in args.run:
        variant, repeat, bag_text = run.split(",", 2)
        bag = Path(bag_text).resolve()
        if not bag.is_dir():
            parser.error(f"Bag directory not found: {bag}")
        data = read_run(bag, args.start_ns, args.end_ns)
        row, report = flatten_run_metrics(
            variant, int(repeat), bag, data
        )
        report["start_ns"] = args.start_ns
        report["end_ns"] = args.end_ns
        rows.append(row)
        reports.append(report)
    rows.sort(key=lambda row: (row["variant"], row["repeat"]))
    summary = summarize_repeats(rows)
    write_csv(output / "per_run_metrics.csv", rows)
    write_csv(output / "repeatability_summary.csv", summary)
    (output / "evaluation.json").write_text(
        json.dumps({
            "schema_version": 1,
            "start_ns": args.start_ns,
            "end_ns": args.end_ns,
            "definitions": {
                "yaw_course_difference": (
                    "wrapped fused-yaw difference from linearly interpolated "
                    "GNSS-derived COG; COG is an EKF input, not ground truth"
                ),
                "backward_step": (
                    "consecutive displacement with more than 0.05 m "
                    "projected opposite to the previous fused yaw"
                ),
                "trajectory_order": "bag record order",
            },
            "runs": reports,
            "repeatability": summary,
        }, indent=2),
        encoding="utf-8",
    )
    plot_metric_summary(output / "tilt_compensation_metrics.png", summary)
    plot_rate_comparison(
        output / "yaw_rate_input_comparison.png",
        reports,
        args.start_ns,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
