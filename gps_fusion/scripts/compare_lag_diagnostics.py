#!/usr/bin/env python3
"""Build compact CSV and Markdown tables from fusion lag diagnostics."""

import argparse
import csv
import json
from pathlib import Path


GROUP_NAMES = {
    "/compare/g02_gps_course": "Group 2: GNSS position + COG",
    "/compare/g03_gps_course_raw_gyro": (
        "Group 3: GNSS position + COG + raw gyro z"
    ),
    "/compare/g04_gps_course_ahrs_rate": (
        "Group 4: GNSS position + COG + AHRS heading rate"
    ),
}


def format_value(value, digits=3):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=JSON",
        help="Run label and lag_diagnostics.json path; repeat for each run.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite: {args.output_dir}")

    reports = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"Invalid --run value: {item}")
        label, path_text = item.split("=", 1)
        path = Path(path_text)
        if not path.is_file():
            parser.error(f"Diagnostic JSON not found: {path}")
        reports.append((label, path, json.loads(path.read_text())))

    timing_rows = []
    trajectory_rows = []
    for label, path, report in reports:
        gps = report["input_timing"]["/compare_input/gps"]
        arrival = gps["arrival_minus_header_s"]
        oos = gps["out_of_sequence"]
        timing_rows.append({
            "run": label,
            "diagnostic_json": str(path.resolve()),
            "gps_input_count": gps["message_count"],
            "gps_arrival_lag_median_s": arrival["median"],
            "gps_arrival_lag_p95_s": arrival["p95"],
            "gps_arrival_lag_max_s": arrival["max"],
            "gps_out_of_sequence_fraction": oos["fraction"],
            "gps_out_of_sequence_median_s": oos["delay_s"]["median"],
            "gps_out_of_sequence_p95_s": oos["delay_s"]["p95"],
            "gps_out_of_sequence_max_s": oos["delay_s"]["max"],
        })
        for topic, group in GROUP_NAMES.items():
            full = report["trajectory"]["full"][topic]
            turn = report["trajectory"]["selected_turn"][topic]
            yaw = report["yaw_course_consistency_deg"][topic]
            trajectory_rows.append({
                "run": label,
                "group": group,
                "topic": topic,
                "effective_rate_hz": full["effective_rate_hz"],
                "full_backward_step_fraction": (
                    full["backward_along_heading_over_0_05_m_count"]
                    / max(1, full["message_count"] - 1)
                ),
                "full_backward_step_max_m": full[
                    "backward_along_heading_m"
                ]["max"],
                "full_apparent_speed_p95_mps": full[
                    "apparent_speed_mps"
                ]["p95"],
                "full_apparent_speed_p99_mps": full[
                    "apparent_speed_mps"
                ]["p99"],
                "full_apparent_speed_max_mps": full[
                    "apparent_speed_mps"
                ]["max"],
                "turn_backward_step_count": turn[
                    "backward_along_heading_over_0_05_m_count"
                ],
                "turn_backward_step_fraction": (
                    turn["backward_along_heading_over_0_05_m_count"]
                    / max(1, turn["message_count"] - 1)
                ),
                "turn_backward_step_max_m": turn[
                    "backward_along_heading_m"
                ]["max"],
                "turn_step_distance_p95_m": turn["step_distance_m"]["p95"],
                "turn_apparent_speed_p95_mps": turn[
                    "apparent_speed_mps"
                ]["p95"],
                "yaw_cog_median_abs_difference_deg": yaw["median"],
                "yaw_cog_p95_abs_difference_deg": yaw["p95"],
            })

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "timing_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=timing_rows[0].keys())
        writer.writeheader()
        writer.writerows(timing_rows)
    with (args.output_dir / "trajectory_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=trajectory_rows[0].keys())
        writer.writeheader()
        writer.writerows(trajectory_rows)

    lines = [
        "# Lag-aware fusion comparison",
        "",
        "Arrival lag is the bag record time minus the message header time. "
        "Out-of-sequence delay is measured against the latest comparison-input "
        "header timestamp already received.",
        "",
        "## GPS input timing",
        "",
        "| Run | GPS inputs | Median arrival lag (s) | P95 (s) | "
        "Out-of-sequence (%) | Median out-of-sequence delay (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing_rows:
        lines.append(
            f"| {row['run']} | {row['gps_input_count']} | "
            f"{format_value(row['gps_arrival_lag_median_s'])} | "
            f"{format_value(row['gps_arrival_lag_p95_s'])} | "
            f"{format_value(100.0 * row['gps_out_of_sequence_fraction'], 1)} | "
            f"{format_value(row['gps_out_of_sequence_median_s'])} |"
        )
    lines.extend([
        "",
        "## Full-trajectory jump diagnostics",
        "",
        "| Run | Group | Backward steps (%) | Largest backward step (m) | "
        "Apparent speed P95 (m/s) | P99 (m/s) | Maximum (m/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in trajectory_rows:
        lines.append(
            f"| {row['run']} | {row['group']} | "
            f"{format_value(100.0 * row['full_backward_step_fraction'], 1)} | "
            f"{format_value(row['full_backward_step_max_m'])} | "
            f"{format_value(row['full_apparent_speed_p95_mps'])} | "
            f"{format_value(row['full_apparent_speed_p99_mps'])} | "
            f"{format_value(row['full_apparent_speed_max_mps'])} |"
        )
    lines.extend([
        "",
        "## Selected-turn trajectory diagnostics",
        "",
        "A backward step is a consecutive displacement with more than 0.05 m "
        "projected opposite to the previous fused heading.",
        "",
        "| Run | Group | Rate (Hz) | Backward steps (%) | Largest backward "
        "step (m) | Turn speed P95 (m/s) | Yaw--COG median (deg) | "
        "Yaw--COG P95 (deg) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in trajectory_rows:
        lines.append(
            f"| {row['run']} | {row['group']} | "
            f"{format_value(row['effective_rate_hz'])} | "
            f"{format_value(100.0 * row['turn_backward_step_fraction'], 1)} | "
            f"{format_value(row['turn_backward_step_max_m'])} | "
            f"{format_value(row['turn_apparent_speed_p95_mps'])} | "
            f"{format_value(row['yaw_cog_median_abs_difference_deg'])} | "
            f"{format_value(row['yaw_cog_p95_abs_difference_deg'])} |"
        )
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
