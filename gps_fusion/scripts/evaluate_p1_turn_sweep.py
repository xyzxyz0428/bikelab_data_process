#!/usr/bin/env python3
"""Evaluate P1 gyro-covariance sweep on one fixed absolute time window."""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


COURSE = "/gnss/course_imu"
G2 = "/compare/g02_gps_course"
G3 = "/compare/g03_gps_course_raw_gyro"


def yaw(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))


def read_bag(path, start_ns, end_ns):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {x.name: get_message(x.type) for x in reader.get_all_topics_and_types()
             if x.name in {COURSE, G2, G3}}
    out = {COURSE: [], G2: [], G3: []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in out:
            continue
        msg = deserialize_message(data, types[topic])
        t = int(msg.header.stamp.sec)*1_000_000_000 + int(msg.header.stamp.nanosec)
        if not start_ns <= t <= end_ns:
            continue
        if topic == COURSE:
            out[topic].append((t, yaw(msg.orientation)))
        else:
            out[topic].append((t, yaw(msg.pose.pose.orientation),
                               msg.pose.pose.position.x, msg.pose.pose.position.y))
    return {k: sorted(v) for k, v in out.items()}


def percentile(v, p):
    return float(np.quantile(np.asarray(v, float), p)) if len(v) else float("nan")


def evaluate(label, scale, bag, start_ns, end_ns):
    d = read_bag(bag, start_ns, end_ns)
    ct = np.asarray([x[0] for x in d[COURSE]], np.int64)
    cy = np.unwrap(np.asarray([x[1] for x in d[COURSE]], float))
    row = {"label": label, "gyro_covariance_scale": scale, "bag": str(bag),
           "course_count": len(ct)}
    for name, topic in [("group2", G2), ("group3", G3)]:
        a = d[topic]
        t = np.asarray([x[0] for x in a], np.int64)
        yy = np.unwrap(np.asarray([x[1] for x in a], float))
        valid = (t >= ct[0]) & (t <= ct[-1]) if len(ct) else np.zeros(len(t), bool)
        t, yy = t[valid], yy[valid]
        ref = np.interp((t-ct[0]).astype(float), (ct-ct[0]).astype(float), cy)
        e = np.degrees(np.arctan2(np.sin(yy-ref), np.cos(yy-ref)))
        row.update({
            f"{name}_count": len(e),
            f"{name}_median_abs_deg": float(np.median(np.abs(e))),
            f"{name}_p95_abs_deg": percentile(np.abs(e), .95),
            f"{name}_rmse_deg": float(np.sqrt(np.mean(e*e))),
            f"{name}_mean_signed_deg": float(np.mean(e)),
            f"{name}_effective_rate_hz": ((len(t)-1)/((t[-1]-t[0])/1e9)
                                           if len(t)>1 else float("nan")),
        })
    a, b = d[G2], d[G3]
    bt = np.asarray([x[0] for x in b], np.int64)
    dist = []
    for x in a:
        if not len(bt): break
        j = int(np.argmin(np.abs(bt-x[0])))
        if abs(int(bt[j])-x[0]) <= 30_000_000:
            dist.append(math.hypot(x[2]-b[j][2], x[3]-b[j][3]))
    row["group2_group3_position_median_m"] = float(np.median(dist))
    row["group2_group3_position_p95_m"] = percentile(dist, .95)
    # Balanced consistency score. This remains an internal screening metric.
    row["screening_score"] = (row["group3_median_abs_deg"] +
                              row["group3_p95_abs_deg"] +
                              row["group3_rmse_deg"])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL,SCALE,BAG")
    ap.add_argument("--start-ns", type=int, required=True)
    ap.add_argument("--end-ns", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if out.exists(): raise SystemExit("Refusing to overwrite output")
    out.mkdir(parents=True)
    rows = []
    for value in args.run:
        label, scale, bag = value.split(",", 2)
        rows.append(evaluate(label, float(scale), Path(bag), args.start_ns, args.end_ns))
    rows.sort(key=lambda x: x["gyro_covariance_scale"])
    with (out/"turn_parameter_metrics.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    x=np.arange(len(rows)); labels=[r["label"] for r in rows]
    fig, ax=plt.subplots(figsize=(8.2,4.8)); width=.25
    ax.bar(x-width,[r["group3_median_abs_deg"] for r in rows],width,label="Median absolute difference")
    ax.bar(x,[r["group3_p95_abs_deg"] for r in rows],width,label="P95 absolute difference")
    ax.bar(x+width,[r["group3_rmse_deg"] for r in rows],width,label="RMSE")
    ax.set_xticks(x,labels); ax.set_xlabel("Parameter setting / repeat")
    ax.set_ylabel("Group 3 yaw--COG difference (deg)"); ax.grid(axis="y",alpha=.3)
    ax.legend(); fig.tight_layout(); fig.savefig(out/"turn_parameter_comparison.png",dpi=220)
    fig.savefig(out/"turn_parameter_comparison.svg"); plt.close(fig)
    # When labels end in r1, r2, ..., also report repeatability by setting.
    grouped = {}
    for row in rows:
        setting = re.sub(r"\s+r\d+$", "", row["label"])
        grouped.setdefault(setting, []).append(row)
    repeated = {key: value for key, value in grouped.items() if len(value) > 1}
    if repeated:
        metrics = [
            "group2_median_abs_deg", "group2_p95_abs_deg", "group2_rmse_deg",
            "group3_median_abs_deg", "group3_p95_abs_deg", "group3_rmse_deg",
        ]
        summary = []
        for setting, values in repeated.items():
            item = {"setting": setting, "repeat_count": len(values)}
            for metric in metrics:
                data = np.asarray([value[metric] for value in values], float)
                item[f"{metric}_mean"] = float(np.mean(data))
                item[f"{metric}_std"] = float(np.std(data, ddof=1))
            summary.append(item)
        with (out/"repeatability_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0]))
            w.writeheader(); w.writerows(summary)

        fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.0), sharex=True)
        metric_names = ["median_abs_deg", "p95_abs_deg", "rmse_deg"]
        titles = ["Median absolute difference", "P95 absolute difference", "RMSE"]
        settings = [item["setting"] for item in summary]
        sx = np.arange(len(settings)); width = .34
        for ax, metric, title in zip(axes, metric_names, titles):
            for offset, group, name, color in [
                (-width/2, "group2", "Group 2", "#4c78a8"),
                ( width/2, "group3", "Group 3", "#f58518"),
            ]:
                means = [item[f"{group}_{metric}_mean"] for item in summary]
                stds = [item[f"{group}_{metric}_std"] for item in summary]
                ax.bar(sx+offset, means, width, yerr=stds, capsize=3,
                       label=name, color=color)
            ax.set_title(title); ax.set_xticks(sx, settings)
            ax.grid(axis="y", alpha=.3); ax.set_ylabel("Yaw--COG difference (deg)")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out/"repeatability_summary.png", dpi=220)
        fig.savefig(out/"repeatability_summary.svg")
        plt.close(fig)

    best=min(rows,key=lambda r:r["screening_score"])
    (out/"README.txt").write_text(
        f"Fixed-window parameter screening. Lowest median+P95+RMSE score: {best['label']}. "
        "COG is a fusion input, so this is internal consistency rather than independent accuracy.\n")
    print("best",best["label"],best["screening_score"])


if __name__ == "__main__": main()
