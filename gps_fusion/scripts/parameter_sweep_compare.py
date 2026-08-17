#!/usr/bin/env python3
"""Compare short-turn fusion parameter runs and make a tuning figure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR / "ros2_ws" / "src" / "bikelab_process"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
DATA_SCRIPT_DIR = SCRIPT_DIR.parent.parent / "data_analysis" / "scripts"
if str(DATA_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCRIPT_DIR))

from bikelab_process.four_way_compare_evaluate import (  # noqa: E402
    GROUP_TOPICS,
    build_result,
    extract_odometry,
    quaternion_yaw,
    read_selected_topics,
)
from paper_style import apply_paper_style  # noqa: E402


COURSE_TOPIC = "/gnss/course_imu"
GROUP3_TOPIC = "/compare/g03_gps_course_raw_gyro"
GROUP2_TOPIC = "/compare/g02_gps_course"
CONFIG_DESCRIPTION = {
    "baseline": "history 1 s; gyro covariance x1; no static-bias subtraction; IMU offset 0 s",
    "bias_cov4": "history 1 s; gyro covariance x4; static-bias subtraction; IMU offset 0 s",
    "bias_cov16": "history 1 s; gyro covariance x16; static-bias subtraction; IMU offset 0 s",
    "time_cov4": "history 1 s; gyro covariance x4; static-bias subtraction; IMU offset +1.34 s",
    "time_cov16": "history 1 s; gyro covariance x16; static-bias subtraction; IMU offset +1.34 s",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    return label, Path(path).resolve()


def course_points(rows):
    points = []
    for header_time, _, message in rows:
        if header_time is None:
            continue
        points.append((header_time, quaternion_yaw(message.orientation)))
    return sorted(points)


def metric(result, topic: str, name: str):
    return (result["odometry_yaw_vs_gnss_course"].get(topic) or {}).get(name)


def collect_run(label: str, bag: Path) -> dict:
    args = argparse.Namespace(
        bag=str(bag), storage_id="sqlite3", fix_topic="/fix/fusion",
        imu_topic="/imu/ahrs_heading_rate", course_topic=COURSE_TOPIC,
        odom_topics=list(GROUP_TOPICS.values()), time_tolerance=0.25,
        minimum_course_speed=2.0, course_half_window=5, pair_tolerance=0.03,
        rate_tolerance=0.01,
    )
    result = build_result(args)
    pair = result["four_way"]["pairwise"].get(
        "gps_course__vs__gps_course_raw_gyro", {}
    )
    group3_stats = result["four_way"]["group_statistics"]["gps_course_raw_gyro"]
    g3_p95 = float(metric(result, GROUP3_TOPIC, "p95_abs_error_deg"))
    g3_median = float(metric(result, GROUP3_TOPIC, "median_abs_error_deg"))
    pos_p95 = float((pair.get("position") or {}).get("p95_m", np.nan))
    # A yaw improvement that requires a large Group-2/Group-3 position jump is
    # not a useful fusion setting. The score is a transparent screening score,
    # not an accuracy estimate.
    score = g3_p95 + g3_median + 20.0 * pos_p95
    _, rows = read_selected_topics(
        str(bag), "sqlite3", [COURSE_TOPIC, GROUP2_TOPIC, GROUP3_TOPIC]
    )
    course = course_points(rows.get(COURSE_TOPIC, []))
    group2 = extract_odometry(rows.get(GROUP2_TOPIC, []))
    group3 = extract_odometry(rows.get(GROUP3_TOPIC, []))
    all_times = [p[0] for p in course] + [p["t"] for p in group2] + [p["t"] for p in group3]
    t0 = min(all_times) if all_times else 0.0
    return {
        "label": label,
        "configuration": CONFIG_DESCRIPTION.get(label, "Configuration supplied by the caller"),
        "bag": str(bag),
        "result": result,
        "effective_rate_hz": group3_stats.get("effective_rate_hz"),
        "group2_median_deg": metric(result, GROUP2_TOPIC, "median_abs_error_deg"),
        "group2_p95_deg": metric(result, GROUP2_TOPIC, "p95_abs_error_deg"),
        "group3_median_deg": g3_median,
        "group3_p95_deg": g3_p95,
        "group3_vs_group2_position_p95_m": pos_p95,
        "screening_score": score,
        "course": [(t - t0, np.degrees(y)) for t, y in course],
        "group2": [(p["t"] - t0, np.degrees(p["yaw"])) for p in group2],
        "group3": [(p["t"] - t0, np.degrees(p["yaw"])) for p in group3],
    }


def plot_runs(runs: list[dict], output: Path):
    labels = [r["label"] for r in runs]
    x = np.arange(len(runs))
    fig = plt.figure(figsize=(10.5, 8.0))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 2.2], hspace=0.48)
    ax0 = fig.add_subplot(grid[0])
    width = 0.19
    ax0.bar(x - 1.5 * width, [r["group2_median_deg"] for r in runs], width, label="Group 2 median")
    ax0.bar(x - 0.5 * width, [r["group2_p95_deg"] for r in runs], width, label="Group 2 P95")
    ax0.bar(x + 0.5 * width, [r["group3_median_deg"] for r in runs], width, label="Group 3 median")
    ax0.bar(x + 1.5 * width, [r["group3_p95_deg"] for r in runs], width, label="Group 3 P95")
    ax0.set_ylabel("Yaw–COG difference (deg)")
    ax0.set_title("Yaw consistency on the selected turn")
    ax0.set_xticks(x, labels, rotation=20, ha="right")
    ax0.legend(ncol=2, fontsize=8)
    ax0.grid(True, axis="y")

    ax1 = fig.add_subplot(grid[1])
    ax1.bar(x, [r["group3_vs_group2_position_p95_m"] for r in runs], color="#009E73")
    ax1.set_ylabel("Group 2–3 position P95 (m)")
    ax1.set_title("Position change caused by the raw-gyro branch")
    ax1.set_xticks(x, labels, rotation=20, ha="right")
    ax1.grid(True, axis="y")

    ax2 = fig.add_subplot(grid[2])
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for index, run in enumerate(runs):
        color = colors[index % len(colors)]
        course = np.asarray(run["course"])
        g3 = np.asarray(run["group3"])
        if len(course) and len(g3):
            course_t = course[:, 0]
            course_yaw = np.radians(course[:, 1])
            errors = []
            times = []
            for t, yaw in g3:
                nearest = int(np.argmin(np.abs(course_t - t)))
                difference = np.arctan2(
                    np.sin(np.radians(yaw) - course_yaw[nearest]),
                    np.cos(np.radians(yaw) - course_yaw[nearest]),
                )
                times.append(t)
                errors.append(np.degrees(difference))
            ax2.plot(times, errors, color=color, linewidth=0.9,
                     label=run["label"])
    ax2.set_xlabel("Time within the 15 s turn window (s)")
    ax2.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
    ax2.set_ylabel("Group 3 yaw − COG (deg)")
    ax2.set_title("Group 3 course-consistency error for each parameter setting")
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True)
    fig.suptitle("Raw-gyro parameter screening; same P9 turn and GNSS inputs", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    runs = [collect_run(label, path) for label, path in args.run]
    best = min(runs, key=lambda r: r["screening_score"])
    rows = []
    for run in runs:
        row = {key: value for key, value in run.items() if key not in {"result", "course", "group2", "group3"}}
        rows.append(row)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output / "parameter_sweep_metrics.csv", index=False)
    (output / "parameter_sweep_metrics.json").write_text(
        json.dumps({"best_screening_setting": best["label"], "runs": rows}, indent=2),
        encoding="utf-8",
    )
    plot_runs(runs, output / "parameter_sweep_comparison.png")
    (output / "README.txt").write_text(
        "Five settings were replayed on the same 15 s segment containing the selected P9 turn. "
        "Group 3 is evaluated against the course derived from the same GNSS receiver, so the values "
        "are internal consistency measures rather than independent accuracy. The screening score is "
        "Group-3 P95 + median yaw difference + 20 times the Group-2/Group-3 position P95. The position "
        "term rejects settings that improve yaw only by introducing a large trajectory displacement. "
        f"The lowest screening score was {best['label']}; this is a parameter-screening result, not a "
        "final accuracy claim.\n",
        encoding="utf-8",
    )
    print(f"Best screening setting: {best['label']}")


if __name__ == "__main__":
    apply_paper_style()
    main()
