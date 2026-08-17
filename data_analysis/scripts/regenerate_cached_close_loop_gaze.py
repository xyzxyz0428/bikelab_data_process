#!/usr/bin/env python3
"""Regenerate a cached closed-loop figure with auditable gaze validity.

This helper uses derived common-interval files and does not require the raw
session mount. A separate Tobii fixation export can be supplied to add the
fixation intervals without changing the raw Method A/B/C signals.
"""

import argparse
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
P9_PATH = SCRIPT_DIR / "p9_speed_timing_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("p9_workflow_functions", P9_PATH)
P9 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(P9)

from paper_style import apply_paper_style  # noqa: E402


NS_PER_SECOND = 1_000_000_000


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} under {directory}, found {len(matches)}")
    return matches[0]


def merged_duration_s(intervals: pd.DataFrame, start_ns: int, end_ns: int) -> float:
    clipped = []
    for _, row in intervals.iterrows():
        start = max(start_ns, int(row["start_ns"]))
        end = min(end_ns, int(row["end_ns"]))
        if end > start:
            clipped.append((start, end))
    if not clipped:
        return 0.0
    clipped.sort()
    merged = [list(clipped[0])]
    for start, end in clipped[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / NS_PER_SECOND


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-dir", required=True)
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--riding-input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--extrinsics", required=True)
    parser.add_argument("--fixation-table", default="")
    args = parser.parse_args()

    common = Path(args.common_dir).resolve()
    workflow = Path(args.workflow_dir).resolve()
    riding = Path(args.riding_input_dir).resolve()
    output = Path(args.out).resolve()
    extrinsics = Path(args.extrinsics).resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    manifest = json.loads((workflow / "run_manifest.json").read_text(encoding="utf-8"))
    interval = json.loads(
        (workflow / "tables" / "video_interval_global_time.json").read_text(
            encoding="utf-8"
        )
    )
    start_ns, end_ns = int(interval["start_ns"]), int(interval["end_ns"])
    created_text = manifest["tobii_recording"]["created"]
    created = datetime.fromisoformat(created_text.replace("Z", "+00:00"))
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))

    gaze_path = find_one(common, "gazedata.gz")
    gaze = P9.read_tobii_gaze(gaze_path, created_ns)
    steering = P9.read_csv(find_one(common, "steering_angle_*.csv"))
    brake = P9.read_csv(find_one(common, "brake_sensors_force_*.csv"))
    imu = P9.read_csv(find_one(common, "imu_*.csv"))
    power = P9.read_csv(find_one(common, "rally_payload_decoded_*.csv"))
    wheel = P9.read_csv(find_one(common, "speed_decoded_*.csv"))
    velocity = P9.read_csv(find_one(common, "ubx_nav_vel_ned.csv"))
    wheel["wheel_speed_mps"] = pd.to_numeric(
        wheel["speed_mps"], errors="coerce"
    ) / 3.6
    raw_imu = imu[pd.to_numeric(imu["dtype"], errors="coerce").eq(64)].copy()
    power_p10 = power[power["page_name"].eq("standard_power")].copy()

    steering_neutral = json.loads(
        (riding / "tables" / "steering_neutral_reference.json").read_text(
            encoding="utf-8"
        )
    )
    band_table = pd.read_csv(riding / "tables" / "brake_zero_input_band.csv")
    brake_bands = {
        str(row["force_column"]): (
            float(row["zero_band_lower_n"]), float(row["zero_band_upper_n"])
        )
        for _, row in band_table.iterrows()
    }

    fixation_intervals = pd.DataFrame()
    fixation_summary = None
    fixation_path = Path(args.fixation_table).resolve() if args.fixation_table else None
    if fixation_path is not None:
        fixation_intervals, fixation_summary = P9.read_tobii_fixation_intervals(
            fixation_path, created_text
        )

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)
    P9.plot_closed_loop(
        gaze, steering, brake, power_p10, raw_imu,
        P9.gnss_course_rate(velocity), velocity, wheel,
        start_ns, end_ns, figures / "P9_representative_closed_loop",
        extrinsics,
        steering_neutral=steering_neutral,
        brake_bands=brake_bands,
        fixation_intervals=fixation_intervals,
    )

    gaze_window = P9.add_gaze_angle_methods(
        P9.smooth_gaze(P9.crop(gaze, start_ns, end_ns)), extrinsics
    )
    total = int(len(gaze_window))
    validity = []
    definitions = {
        "Method A": (
            "finite Tobii 3-D gaze-point x/y/z and positive forward component "
            "after HUCS-to-scene-camera rotation"
        ),
        "Method B": (
            "finite left- and right-eye 3-D gaze-direction vectors and positive "
            "forward component of their mean after rotation"
        ),
        "Method C": (
            "finite normalized gaze2d x/y inside [0,1] x [0,1] and positive "
            "forward component of the back-projected camera ray"
        ),
    }
    for key, method in (("a", "Method A"), ("b", "Method B"), ("c", "Method C")):
        fields = gaze_window[f"method_{key}_fields_valid"].fillna(False).astype(bool)
        forward = gaze_window[f"method_{key}_forward_valid"].fillna(False).astype(bool)
        valid = gaze_window[f"method_{key}_valid"].fillna(False).astype(bool)
        validity.append({
            "method": method,
            "definition": definitions[method],
            "raw_gaze_records": total,
            "required_fields_valid": int(fields.sum()),
            "image_domain_valid": (
                int(gaze_window["gaze_2d_image_valid"].sum()) if key == "c" else None
            ),
            "forward_geometry_valid": int(forward.sum()),
            "output_valid": int(valid.sum()),
            "output_valid_percent": 100.0 * float(valid.mean()) if total else None,
        })
    validity_frame = pd.DataFrame(validity)
    validity_frame.to_csv(tables / "gaze_method_validity_summary.csv", index=False)

    series_columns = [
        "t_unix_ns", "gaze_x_norm", "gaze_y_norm",
        "gaze_x_relative_norm", "gaze_2d_relative_angle_deg",
        "gaze_2d_fields_valid", "gaze_2d_image_valid",
        "method_a_ego_angle_deg", "method_b_ego_angle_deg",
        "method_c_ego_angle_deg", "method_a_valid", "method_b_valid",
        "method_c_valid",
    ]
    gaze_window[series_columns].to_csv(
        tables / "gaze_window_series.csv", index=False
    )

    raw_times = gaze_window["t_unix_ns"].sort_values().to_numpy(dtype=np.int64)
    raw_steps_ms = np.diff(raw_times).astype(float) / 1.0e6
    timing = {
        "record_count": total,
        "median_interval_ms": float(np.median(raw_steps_ms)),
        "maximum_interval_ms": float(np.max(raw_steps_ms)),
    }
    fixation_text = (
        "No fixation export was available for this run; no fixation intervals "
        "are marked in the provisional figure."
    )
    if fixation_summary is not None:
        selected_fixations = fixation_intervals[
            (fixation_intervals["end_ns"] >= start_ns)
            & (fixation_intervals["start_ns"] <= end_ns)
        ].copy()
        selected_fixations.to_csv(
            tables / "tobii_fixation_intervals.csv", index=False
        )
        fixation_coverage = merged_duration_s(
            selected_fixations, start_ns, end_ns
        )
        fixation_summary.update({
            "selected_window_event_count": int(len(selected_fixations)),
            "selected_window_covered_duration_s": fixation_coverage,
            "selected_window_covered_fraction": (
                fixation_coverage / ((end_ns - start_ns) / NS_PER_SECOND)
            ),
        })
        P9.write_json(tables / "tobii_fixation_summary.json", fixation_summary)
        fixation_text = (
            f"The shaded intervals show {len(selected_fixations)} fixation events "
            "selected from the separate Tobii fixation export. Their merged "
            f"duration is {fixation_coverage:.2f} s "
            f"({100.0 * fixation_summary['selected_window_covered_fraction']:.2f}% "
            "of the selected window). Rows were required "
            "to be labelled Fixation with finite fixation-point coordinates and "
            "were grouped by eye-movement event index."
        )

    row_by_method = {row["method"]: row for row in validity}
    text = (
        "Panel (a) was calculated from the raw Tobii gazedata stream, rather "
        "than from the fixation export. A sample was considered valid for "
        "Method A when all three coordinates of the 3-D gaze point were finite "
        "and the transformed point had a positive forward component. Method B "
        "required finite left- and right-eye gaze-direction vectors and a "
        "positive forward component of their mean direction. Method C required "
        "a finite normalized 2-D gaze point inside the recorded image and a "
        "forward back-projected camera ray. Validity therefore describes "
        "numerical and geometric availability; it does not indicate a Tobii-"
        "classified fixation or independently verified gaze accuracy. Of "
        f"{total:,} raw gaze records, Method A produced "
        f"{row_by_method['Method A']['output_valid']:,} valid estimates "
        f"({row_by_method['Method A']['output_valid_percent']:.2f}%), Method B "
        f"produced {row_by_method['Method B']['output_valid']:,} "
        f"({row_by_method['Method B']['output_valid_percent']:.2f}%), and Method C "
        f"produced {row_by_method['Method C']['output_valid']:,} "
        f"({row_by_method['Method C']['output_valid_percent']:.2f}%). The raw "
        f"median and maximum sampling intervals were {timing['median_interval_ms']:.3f} "
        f"and {timing['maximum_interval_ms']:.3f} ms. The plotting routine breaks "
        "the line at invalid samples and does not join across them. The raw "
        "normalized horizontal 2-D position is not plotted; it remains in the "
        "derived audit table together with its camera-calibrated angle. "
        "This closed-loop analysis does not call the AprilTag head/back-pose "
        "estimation pipeline. Camera-frame availability, tag detections, and pose "
        "quality are therefore not included in these Method A--C validity counts. "
        + fixation_text + " The Method A--C curves remain calculated from the raw "
        "gaze stream in a static camera frame."
    )
    (output / "technical_validation_gaze_text.txt").write_text(
        text + "\n", encoding="utf-8"
    )
    caption = (
        "Rider input and bicycle response during a representative dynamic "
        "interval. Panel (a) shows three static camera-frame gaze-angle methods "
        "calculated from the raw Tobii gaze stream. Line breaks identify "
        "samples that fail the method-specific numerical or geometric validity "
        "checks, and shaded intervals show Tobii-classified fixations from a "
        "separate fixation export. No dynamic head/back-pose compensation is "
        "applied in this closed-loop panel."
    )
    (output / "figure_caption_gaze_revision.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    P9.write_json(output / "run_manifest.json", {
        "source_workflow": str(workflow),
        "common_derived_input": str(common),
        "raw_gaze_source": str(gaze_path),
        "fixation_source": str(fixation_path) if fixation_path else None,
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "timing": timing,
        "validity": validity,
        "fixation": fixation_summary,
        "note": "Raw and previous generated data were not modified.",
    })
    print(output)


if __name__ == "__main__":
    apply_paper_style()
    main()
