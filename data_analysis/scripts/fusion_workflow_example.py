#!/usr/bin/env python3
"""Create one GNSS/IMU workflow figure and summary table."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_style import COLORS, apply_paper_style, panel_label, save_figure  # noqa: E402


NS_PER_SECOND = 1_000_000_000
GROUPS = {
    "Group 1": {
        "topic": "/odometry/gps_common",
        "label": "Group 1: GNSS position only",
        "inputs": "GNSS position only",
        "color": COLORS["black"],
        "linestyle": "--",
        "has_yaw": False,
    },
    "Group 2": {
        "topic": "/compare/g02_gps_course",
        "label": "Group 2: EKF with GNSS position + COG",
        "inputs": "GNSS position + GNSS-derived course over ground",
        "color": COLORS["blue"],
        "linestyle": "-",
        "has_yaw": True,
    },
    "Group 3": {
        "topic": "/compare/g03_gps_course_raw_gyro",
        "label": "Group 3: Group 2 + raw gyro z",
        "inputs": (
            "GNSS position + GNSS-derived course over ground + "
            "raw gyro z yaw rate"
        ),
        "color": COLORS["vermillion"],
        "linestyle": "--",
        "has_yaw": True,
    },
    "Group 4": {
        "topic": "/compare/g04_gps_course_ahrs_rate",
        "label": "Group 4: Group 2 + AHRS heading rate",
        "inputs": (
            "GNSS position + GNSS-derived course over ground + "
            "AHRS headingspeed"
        ),
        "color": COLORS["green"],
        "linestyle": "-.",
        "has_yaw": True,
    },
}
COURSE_TOPIC = "/gnss/course_imu"
RAW_GYRO_TOPIC = "/compare_input/raw_gyro_rate"


def stamp_ns(message):
    return (
        int(message.header.stamp.sec) * NS_PER_SECOND
        + int(message.header.stamp.nanosec)
    )


def quaternion_yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def sha256_path(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read_result(result_bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(result_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    wanted = (
        {item["topic"] for item in GROUPS.values()}
        | {COURSE_TOPIC, RAW_GYRO_TOPIC}
    )
    missing = sorted(wanted - set(topics))
    if missing:
        raise RuntimeError(f"Missing result topics: {', '.join(missing)}")
    message_types = {topic: get_message(topics[topic]) for topic in wanted}
    rows = {name: [] for name in GROUPS}
    course_rows = []
    raw_gyro_rows = []
    topic_to_group = {item["topic"]: name for name, item in GROUPS.items()}

    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        t_ns = stamp_ns(message)
        if topic == COURSE_TOPIC:
            course_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "yaw_rad": quaternion_yaw(message.orientation),
            })
        elif topic == RAW_GYRO_TOPIC:
            raw_gyro_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "yaw_rate_rad_s": float(message.angular_velocity.z),
            })
        else:
            pose = message.pose.pose
            rows[topic_to_group[topic]].append({
                "t_ns": t_ns,
                "x_m": float(pose.position.x),
                "y_m": float(pose.position.y),
                "yaw_rad": quaternion_yaw(pose.orientation),
            })

    frames = {
        name: pd.DataFrame(values).sort_values("t_ns").reset_index(drop=True)
        for name, values in rows.items()
    }
    course = pd.DataFrame(course_rows).sort_values("t_ns").reset_index(drop=True)
    raw_gyro = (
        pd.DataFrame(raw_gyro_rows).sort_values("t_ns").reset_index(drop=True)
    )
    for name, frame in frames.items():
        if len(frame) < 2:
            raise RuntimeError(f"Too few {name} samples in the result bag")
    if len(course) < 2:
        raise RuntimeError(
            "Too few GNSS-derived course-over-ground samples in the result bag"
        )
    if len(raw_gyro) < 2:
        raise RuntimeError("Too few raw gyroscope samples in the result bag")
    return frames, course, raw_gyro


def _finite_distribution(values):
    """Return compact finite-value statistics for a diagnostic table."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _nearest_distance_ns(query_ns, reference_ns):
    """Return the distance to the nearest reference timestamp."""
    query = np.asarray(query_ns, dtype=np.int64)
    reference = np.sort(np.asarray(reference_ns, dtype=np.int64))
    if not len(reference):
        return np.full(len(query), np.iinfo(np.int64).max, dtype=np.int64)
    indices = np.searchsorted(reference, query)
    left = reference[np.clip(indices - 1, 0, len(reference) - 1)]
    right = reference[np.clip(indices, 0, len(reference) - 1)]
    return np.minimum(np.abs(query - left), np.abs(query - right))


def yaw_update_diagnostic(
    frames,
    course,
    raw_gyro,
    course_proximity_s=0.06,
    negative_step_threshold_deg=1.0,
):
    """Quantify Group 3 corrections near discrete COG observations."""
    group = frames["Group 3"].sort_values("t_ns").reset_index(drop=True)
    if len(group) < 3 or len(course) < 2 or len(raw_gyro) < 2:
        raise RuntimeError("Too few samples for the Group 3 yaw diagnostic")

    output_ns = group["t_ns"].to_numpy(dtype=np.int64)
    output_s = output_ns.astype(float) / NS_PER_SECOND
    yaw = np.unwrap(group["yaw_rad"].to_numpy(dtype=float))
    dt_s = np.diff(output_s)
    observed_step_deg = np.rad2deg(np.diff(yaw))
    midpoint_s = 0.5 * (output_s[:-1] + output_s[1:])

    gyro_s = (
        raw_gyro["t_ns"].to_numpy(dtype=np.int64).astype(float)
        / NS_PER_SECOND
    )
    gyro_rate = raw_gyro["yaw_rate_rad_s"].to_numpy(dtype=float)
    interpolated_rate = np.interp(midpoint_s, gyro_s, gyro_rate)
    propagation_step_deg = np.rad2deg(interpolated_rate * dt_s)
    propagation_residual_abs_deg = np.abs(
        observed_step_deg - propagation_step_deg
    )

    nearest_course_ns = _nearest_distance_ns(
        output_ns[1:], course["t_ns"].to_numpy(dtype=np.int64)
    )
    proximity_ns = int(round(course_proximity_s * NS_PER_SECOND))
    near_course = nearest_course_ns <= proximity_ns
    negative_step = observed_step_deg < -abs(negative_step_threshold_deg)

    raw_lag_ms = (
        raw_gyro["record_ns"].to_numpy(dtype=np.int64)
        - raw_gyro["t_ns"].to_numpy(dtype=np.int64)
    ) / 1e6
    course_lag_ms = (
        course["record_ns"].to_numpy(dtype=np.int64)
        - course["t_ns"].to_numpy(dtype=np.int64)
    ) / 1e6
    raw_rate_deg_s = np.rad2deg(
        raw_gyro["yaw_rate_rad_s"].to_numpy(dtype=float)
    )

    negative_count = int(np.count_nonzero(negative_step))
    negative_near_count = int(np.count_nonzero(negative_step & near_course))
    return {
        "scope": "Displayed 10 s yaw panel",
        "interpretation": (
            "A large residual close to a COG timestamp indicates a discrete EKF "
            "correction of gyro-propagated yaw; it is not an independent yaw error."
        ),
        "course_proximity_threshold_ms": float(course_proximity_s * 1000.0),
        "negative_yaw_step_threshold_deg": float(
            abs(negative_step_threshold_deg)
        ),
        "group3_output_step_count": int(len(observed_step_deg)),
        "group3_negative_yaw_step_count": negative_count,
        "negative_steps_near_course_count": negative_near_count,
        "negative_steps_near_course_fraction": (
            float(negative_near_count / negative_count)
            if negative_count else None
        ),
        "group3_observed_yaw_step_deg": _finite_distribution(
            observed_step_deg
        ),
        "gyro_propagation_residual_abs_deg_all": _finite_distribution(
            propagation_residual_abs_deg
        ),
        "gyro_propagation_residual_abs_deg_near_course": _finite_distribution(
            propagation_residual_abs_deg[near_course]
        ),
        "gyro_propagation_residual_abs_deg_far_from_course": _finite_distribution(
            propagation_residual_abs_deg[~near_course]
        ),
        "raw_gyro_z_deg_s": _finite_distribution(raw_rate_deg_s),
        "raw_gyro_record_minus_header_ms": _finite_distribution(raw_lag_ms),
        "course_record_minus_header_ms": _finite_distribution(course_lag_ms),
    }


def pairwise_position_separation(frames):
    """Compare pairwise positions at timestamps from the lower-rate series."""
    rows = []
    names = list(frames)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            first = frames[first_name].sort_values("t_ns")
            second = frames[second_name].sort_values("t_ns")
            reference = first if len(first) <= len(second) else second
            overlap_start = max(int(first["t_ns"].min()), int(second["t_ns"].min()))
            overlap_end = min(int(first["t_ns"].max()), int(second["t_ns"].max()))
            query = reference[
                reference["t_ns"].between(
                    overlap_start, overlap_end, inclusive="both"
                )
            ]["t_ns"].to_numpy(dtype=np.int64)
            if len(query) < 2:
                continue
            first_x = np.interp(query, first["t_ns"], first["x_m"])
            first_y = np.interp(query, first["t_ns"], first["y_m"])
            second_x = np.interp(query, second["t_ns"], second["x_m"])
            second_y = np.interp(query, second["t_ns"], second["y_m"])
            distance = np.hypot(first_x - second_x, first_y - second_y)
            rows.append({
                "first_group": first_name,
                "second_group": second_name,
                "sample_count": int(len(distance)),
                "median_separation_m": float(np.median(distance)),
                "p95_separation_m": float(np.percentile(distance, 95)),
                "maximum_separation_m": float(np.max(distance)),
            })
    return pd.DataFrame(rows)


def unwrap_near_reference(frame, course):
    time = frame["t_ns"].to_numpy(dtype=np.int64).astype(float) / NS_PER_SECOND
    yaw = np.unwrap(frame["yaw_rad"].to_numpy(dtype=float))
    course_time = course["t_ns"].to_numpy(dtype=np.int64).astype(float) / NS_PER_SECOND
    course_yaw = np.unwrap(course["yaw_rad"].to_numpy(dtype=float))
    reference = np.interp(time, course_time, course_yaw)
    shift = round(float(np.median(reference - yaw)) / (2.0 * np.pi)) * 2.0 * np.pi
    return yaw + shift


def frame_statistics(frame):
    """Return count and effective header-time rate for one displayed frame."""
    count = int(len(frame))
    if count < 2:
        return count, None
    duration_s = (
        int(frame["t_ns"].iloc[-1]) - int(frame["t_ns"].iloc[0])
    ) / NS_PER_SECOND
    rate = (count - 1) / duration_s if duration_s > 0 else None
    return count, rate


def build_summary(evaluation, full_frames):
    consistency = evaluation["odometry_yaw_vs_gnss_course"]
    rows = []
    for name, item in GROUPS.items():
        topic = item["topic"]
        yaw = consistency.get(topic)
        output_count, effective_rate = frame_statistics(full_frames[name])
        rows.append({
            "group": name,
            "inputs": item["inputs"],
            "output_count": output_count,
            "effective_rate_hz": effective_rate,
            "yaw_trajectory_course_samples": (
                int(yaw["samples"]) if yaw is not None else None
            ),
            "median_abs_yaw_trajectory_course_difference_deg": (
                float(yaw["median_abs_error_deg"])
                if yaw is not None else None
            ),
            "p95_abs_yaw_trajectory_course_difference_deg": (
                float(yaw["p95_abs_error_deg"])
                if yaw is not None else None
            ),
            "circular_yaw_minus_trajectory_course_bias_deg": (
                float(yaw["circular_bias_deg"])
                if yaw is not None else None
            ),
            "output_metric_scope": "Displayed common workflow interval",
            "yaw_trajectory_course_metric_scope": (
                "Full result bag; trajectory-derived course; may extend outside "
                "the displayed common interval"
            ),
        })
    return pd.DataFrame(rows)


def direct_turn_yaw_cog_metrics(frames, course):
    """Compare fused yaw with the receiver COG shown in panel (c)."""
    course_time = (
        course["t_ns"].to_numpy(dtype=np.int64).astype(float)
        / NS_PER_SECOND
    )
    course_yaw = np.unwrap(course["yaw_rad"].to_numpy(dtype=float))
    rows = []
    for name, frame in frames.items():
        if not GROUPS[name]["has_yaw"]:
            continue
        frame_time = (
            frame["t_ns"].to_numpy(dtype=np.int64).astype(float)
            / NS_PER_SECOND
        )
        inside = (
            (frame_time >= course_time[0])
            & (frame_time <= course_time[-1])
        )
        frame_time = frame_time[inside]
        if not len(frame_time):
            raise RuntimeError(f"No {name} yaw samples overlap the turn COG")
        frame_yaw = unwrap_near_reference(frame, course)[inside]
        reference = np.interp(frame_time, course_time, course_yaw)
        difference = (
            (frame_yaw - reference + np.pi) % (2.0 * np.pi) - np.pi
        )
        absolute_deg = np.abs(np.rad2deg(difference))
        rows.append({
            "group": name,
            "reference": "GNSS-derived COG from UBX-NAV-VELNED",
            "sample_count": int(len(absolute_deg)),
            "median_abs_yaw_cog_difference_deg": float(
                np.median(absolute_deg)
            ),
            "p95_abs_yaw_cog_difference_deg": float(
                np.percentile(absolute_deg, 95)
            ),
            "rmse_yaw_cog_difference_deg": float(
                np.sqrt(np.mean(np.square(absolute_deg)))
            ),
            "metric_scope": "Displayed turn interval",
        })
    return pd.DataFrame(rows)


def read_repeatability_summary(path):
    """Read the raw-gyro repeated-turn row used in the paper text."""
    if not path:
        return None, None
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise RuntimeError("Repeatability evaluation must be JSON")
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = pd.DataFrame(payload.get("repeatability", []))
    interval = {
        "start_ns": int(payload["start_ns"]),
        "end_ns": int(payload["end_ns"]),
    }
    selected = table[table["variant"] == "raw_gyro_z"]
    if len(selected) != 1:
        raise RuntimeError(
            "Repeatability table must contain one raw_gyro_z row"
        )
    return selected.iloc[0].to_dict(), interval


def read_launch_parameters(path):
    """Extract launch key/value arguments from a run provenance file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    parameters = {}
    for item in payload.get("command", []):
        if ":=" not in item:
            continue
        key, value = item.split(":=", 1)
        parameters[key] = value
    return parameters, payload


def bool_parameter(parameters, key):
    """Parse one required true/false launch argument."""
    value = parameters[key].strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"Invalid Boolean launch parameter {key}={value}")
    return value == "true"


def effective_config(provenance, path):
    """Return the captured effective-config entry matching an absolute path."""
    expected = Path(path).resolve()
    matches = [
        item for item in provenance.get("effective_configs", [])
        if Path(item.get("path", "")).resolve() == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one captured effective config for {expected}, found "
            f"{len(matches)}"
        )
    selected = matches[0]
    content = selected.get("content")
    digest = selected.get("sha256", "")
    if (
        selected.get("exists") is not True
        or selected.get("type") != "file"
        or not isinstance(content, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise RuntimeError(f"Invalid effective-config record for {expected}")
    content_bytes = content.encode("utf-8")
    if hashlib.sha256(content_bytes).hexdigest() != digest:
        raise RuntimeError(f"Effective-config content hash mismatch for {expected}")
    if int(selected.get("size_bytes", -1)) != len(content_bytes):
        raise RuntimeError(f"Effective-config size mismatch for {expected}")
    return selected


def validate_selected_yaml(configuration, parameters, provenance):
    """Validate the selected Group 2/3 YAML settings against provenance."""
    import yaml

    repo_root = Path(provenance["repo_root"]).resolve()
    expected_config = (
        repo_root / configuration["expected_config_path"]
    ).resolve()
    configured_path = Path(parameters["config_file"]).resolve()
    if configured_path != expected_config:
        raise RuntimeError(
            f"Unexpected EKF config: {configured_path} != {expected_config}"
        )
    captured = effective_config(provenance, configured_path)
    payload = yaml.safe_load(captured["content"])
    expected_sources = {
        "compare_gps_course": {
            "odom0": "/compare_input/gps",
            "imu0": "/compare_input/course",
            "imu1": None,
        },
        "compare_gps_course_raw_gyro": {
            "odom0": "/compare_input/gps",
            "imu0": "/compare_input/course",
            "imu1": "/compare_input/raw_gyro_rate",
        },
    }
    for node, sources in expected_sources.items():
        values = payload[node]["ros__parameters"]
        matrix = values["process_noise_covariance"]
        if len(matrix) != 225:
            raise RuntimeError(f"{node} process-noise matrix is not 15x15")
        if not math.isclose(
            float(matrix[5 * 15 + 5]), float(configuration["q_yaw"]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise RuntimeError(f"{node} Q_yaw does not match screening choice")
        if not math.isclose(
            float(matrix[11 * 15 + 11]),
            float(configuration["q_yaw_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{node} Q_yaw_rate does not match screening choice"
            )
        for source_key, expected_topic in sources.items():
            actual_topic = values.get(source_key)
            if actual_topic != expected_topic:
                raise RuntimeError(
                    f"{node} {source_key}={actual_topic!r}, expected "
                    f"{expected_topic!r}"
                )
        expected_configs = {
            "odom0_config": [
                True, True, False,
                False, False, False,
                False, False, False,
                False, False, False,
                False, False, False,
            ],
            "imu0_config": [
                False, False, False,
                False, False, True,
                False, False, False,
                False, False, False,
                False, False, False,
            ],
        }
        if node == "compare_gps_course_raw_gyro":
            expected_configs["imu1_config"] = [
                False, False, False,
                False, False, False,
                False, False, False,
                False, False, True,
                False, False, False,
            ]
        for config_key, expected_config in expected_configs.items():
            if values.get(config_key) != expected_config:
                raise RuntimeError(f"{node} {config_key} is not the selected mapping")
        if any(
            key.startswith("twist") and key[5:].split("_")[0].isdigit()
            for key in values
        ):
            raise RuntimeError(f"{node} unexpectedly uses a velocity input")
        rejection_keys = ["imu0_pose_rejection_threshold"]
        if node == "compare_gps_course_raw_gyro":
            rejection_keys.append("imu1_twist_rejection_threshold")
        rejection_keys = [key for key in rejection_keys if key in values]
        if rejection_keys:
            raise RuntimeError(
                f"{node} unexpectedly enables innovation rejection: "
                + ", ".join(rejection_keys)
            )
    expected_launch = (
        repo_root / configuration["expected_launch_path"]
    ).resolve()
    effective_config(provenance, expected_launch)


def select_turn_core(course, start_ns, end_ns, rate_threshold=0.08, padding_s=0.5):
    selected = course[
        (course["t_ns"] >= start_ns) & (course["t_ns"] <= end_ns)
    ].copy()
    if len(selected) < 7:
        raise RuntimeError(
            "Too few GNSS-derived course-over-ground samples in the selected turn"
        )
    time_s = selected["t_ns"].to_numpy(dtype=np.int64) / NS_PER_SECOND
    yaw = np.unwrap(selected["yaw_rad"].to_numpy(dtype=float))
    yaw_smooth = pd.Series(yaw).rolling(5, center=True, min_periods=1).mean().to_numpy()
    rate = np.gradient(yaw_smooth, time_s)
    active = np.abs(rate) >= rate_threshold
    if not np.any(active):
        return start_ns, end_ns, rate_threshold
    active_times = selected.loc[active, "t_ns"].to_numpy(dtype=np.int64)
    padding_ns = int(round(padding_s * NS_PER_SECOND))
    core_start = max(start_ns, int(active_times[0]) - padding_ns)
    core_end = min(end_ns, int(active_times[-1]) + padding_ns)
    if core_end - core_start < 4 * NS_PER_SECOND:
        return start_ns, end_ns, rate_threshold
    return core_start, core_end, rate_threshold


def make_figure(
    full_frames,
    full_course,
    gnss_route,
    start_ns,
    end_ns,
    output_base,
    detail_start_offset_s=None,
    detail_end_offset_s=None,
    position_zoom_start_offset_s=None,
    position_zoom_end_offset_s=None,
    position_zoom_margin_m=0.5,
    candidate_label=None,
):
    if detail_start_offset_s is not None and detail_end_offset_s is not None:
        core_start_ns = start_ns + int(round(
            detail_start_offset_s * NS_PER_SECOND
        ))
        core_end_ns = start_ns + int(round(
            detail_end_offset_s * NS_PER_SECOND
        ))
        if not start_ns <= core_start_ns < core_end_ns <= end_ns:
            raise RuntimeError("Turn-detail offsets fall outside the selected turn")
        rate_threshold = None
        selection = "Explicit offsets within the GNSS-selected turn"
    else:
        core_start_ns, core_end_ns, rate_threshold = select_turn_core(
            full_course, start_ns, end_ns
        )
        selection = "Automatic course-rate threshold with 0.5 s padding"
    frames = {
        name: frame[
            (frame["t_ns"] >= core_start_ns) & (frame["t_ns"] <= core_end_ns)
        ].copy()
        for name, frame in full_frames.items()
    }
    course = full_course[
        (full_course["t_ns"] >= core_start_ns)
        & (full_course["t_ns"] <= core_end_ns)
    ].copy()
    for name, frame in frames.items():
        if len(frame) < 2:
            raise RuntimeError(f"Too few {name} samples in the turn zoom")
        frame["time_s"] = (frame["t_ns"] - core_start_ns) / NS_PER_SECOND
        if GROUPS[name]["has_yaw"]:
            frame["yaw_unwrapped_rad"] = unwrap_near_reference(frame, course)
    course["time_s"] = (course["t_ns"] - core_start_ns) / NS_PER_SECOND
    course["yaw_unwrapped_rad"] = np.unwrap(course["yaw_rad"].to_numpy(dtype=float))

    origin_x = float(full_frames["Group 2"]["x_m"].iloc[0])
    origin_y = float(full_frames["Group 2"]["y_m"].iloc[0])
    for frame in full_frames.values():
        frame["relative_x_m"] = frame["x_m"] - origin_x
        frame["relative_y_m"] = frame["y_m"] - origin_y
    for frame in frames.values():
        frame["relative_x_m"] = frame["x_m"] - origin_x
        frame["relative_y_m"] = frame["y_m"] - origin_y

    if (position_zoom_start_offset_s is None) != (
        position_zoom_end_offset_s is None
    ):
        raise RuntimeError(
            "Both position-zoom offsets must be provided together"
        )
    if position_zoom_start_offset_s is not None:
        position_zoom_start_ns = start_ns + int(round(
            position_zoom_start_offset_s * NS_PER_SECOND
        ))
        position_zoom_end_ns = start_ns + int(round(
            position_zoom_end_offset_s * NS_PER_SECOND
        ))
        if not (
            core_start_ns
            <= position_zoom_start_ns
            < position_zoom_end_ns
            <= core_end_ns
        ):
            raise RuntimeError(
                "Position-zoom offsets fall outside the turn-detail window"
            )
        position_zoom_selection = (
            "Explicit offsets within the selected turn"
        )
    else:
        position_zoom_start_ns = core_start_ns
        position_zoom_end_ns = core_end_ns
        position_zoom_selection = "Full turn-detail window"

    position_frames = {
        name: frame[
            (frame["t_ns"] >= position_zoom_start_ns)
            & (frame["t_ns"] <= position_zoom_end_ns)
        ].copy()
        for name, frame in frames.items()
    }
    for name, frame in position_frames.items():
        if len(frame) < 2:
            raise RuntimeError(
                f"Too few {name} samples in the position zoom"
            )

    if position_zoom_margin_m < 0:
        raise RuntimeError("Position-zoom margin must be non-negative")
    zoom_points = pd.concat(
        [
            frame[["relative_x_m", "relative_y_m"]]
            for frame in position_frames.values()
        ],
        ignore_index=True,
    )
    x_min = float(zoom_points["relative_x_m"].min())
    x_max = float(zoom_points["relative_x_m"].max())
    y_min = float(zoom_points["relative_y_m"].min())
    y_max = float(zoom_points["relative_y_m"].max())
    circle_center_x = 0.5 * (x_min + x_max)
    circle_center_y = 0.5 * (y_min + y_max)
    circle_radius = (
        0.5 * math.hypot(x_max - x_min, y_max - y_min)
        + position_zoom_margin_m
    )
    x_limits = (
        x_min - position_zoom_margin_m,
        x_max + position_zoom_margin_m,
    )
    y_limits = (
        y_min - position_zoom_margin_m,
        y_max + position_zoom_margin_m,
    )

    apply_paper_style()
    fig = plt.figure(figsize=(7.2, 6.7))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8])
    full_ax = fig.add_subplot(grid[0, 0])
    zoom_ax = fig.add_subplot(grid[0, 1])
    yaw_ax = fig.add_subplot(grid[1, :])

    for name, item in GROUPS.items():
        full = full_frames[name]
        zoom = position_frames[name]
        turn = frames[name]
        full_ax.plot(
            full["relative_x_m"].to_numpy(dtype=float),
            full["relative_y_m"].to_numpy(dtype=float),
            color=item["color"], linestyle=item["linestyle"],
            label=item["label"],
            linewidth=1.0,
        )
        zoom_ax.plot(
            zoom["relative_x_m"].to_numpy(dtype=float),
            zoom["relative_y_m"].to_numpy(dtype=float),
            color=item["color"], linestyle=item["linestyle"],
            label=item["label"],
            linewidth=1.2,
        )
        if item["has_yaw"]:
            yaw_ax.plot(
                turn["time_s"].to_numpy(dtype=float),
                np.rad2deg(turn["yaw_unwrapped_rad"].to_numpy(dtype=float)),
                color=item["color"], linestyle=item["linestyle"],
                label=item["label"],
                linewidth=1.4, zorder=3,
            )
    if gnss_route is not None and len(gnss_route):
        full_ax.plot(
            gnss_route["relative_x_m"].to_numpy(dtype=float),
            gnss_route["relative_y_m"].to_numpy(dtype=float),
            color=COLORS["grey"], linestyle=":", linewidth=0.9,
            label="Valid /ubx_nav_pvt route (context)", zorder=1,
        )
    full_ax.add_patch(
        Circle(
            (circle_center_x, circle_center_y),
            max(circle_radius, 15.0),
            fill=False,
            edgecolor=COLORS["black"],
            linestyle="-",
            linewidth=0.8,
            zorder=6,
        )
    )
    yaw_ax.plot(
        course["time_s"].to_numpy(dtype=float),
        np.rad2deg(course["yaw_unwrapped_rad"].to_numpy(dtype=float)),
        color=COLORS["purple"], linewidth=1.0, alpha=0.9,
        marker="o", markersize=1.8, markerfacecolor="white",
        markeredgecolor=COLORS["purple"], markeredgewidth=0.6,
        markevery=2, label="GNSS-derived COG measurement", zorder=2,
    )

    full_ax.set_xlabel("Relative map x (m)")
    full_ax.set_ylabel("Relative map y (m)")
    candidate_suffix = (
        f" (Candidate {candidate_label})" if candidate_label else ""
    )
    full_ax.set_title(f"Full trajectory{candidate_suffix}")
    full_ax.set_aspect("equal", adjustable="datalim")

    zoom_ax.set_xlim(x_limits)
    zoom_ax.set_ylim(y_limits)
    zoom_ax.set_xlabel("Relative map x (m)")
    zoom_ax.set_ylabel("Relative map y (m)")
    zoom_ax.set_title(f"Turn detail{candidate_suffix}")
    zoom_ax.set_aspect("equal", adjustable="box")

    yaw_ax.set_xlabel("Time within turn detail (s)")
    yaw_ax.set_ylabel("Yaw (deg)")
    yaw_ax.set_title("Fused yaw and GNSS-derived course over ground")

    legend_handles = [
        Line2D(
            [0], [0],
            color=item["color"],
            linestyle=item["linestyle"],
            linewidth=1.6,
            label=(
                item["label"] + " (no yaw)"
                if not item["has_yaw"] else item["label"]
            ),
        )
        for item in GROUPS.values()
    ]
    optional_route_handle = []
    if gnss_route is not None and len(gnss_route):
        optional_route_handle.append(
            Line2D(
                [0], [0], color=COLORS["grey"], linestyle=":", linewidth=1.0,
                label="Valid /ubx_nav_pvt route (context)",
            )
        )
    legend_handles.extend(optional_route_handle + [
        Line2D(
            [0], [0],
            color=COLORS["purple"],
            linewidth=1.0,
            marker="o",
            markersize=3.0,
            markerfacecolor="white",
            label="GNSS-derived COG measurement",
        ),
        Line2D(
            [0], [0],
            color=COLORS["black"],
            linestyle="None",
            marker="o",
            markersize=6.0,
            markerfacecolor="none",
            markeredgewidth=0.8,
            label="Enlarged turn segment",
        ),
    ])
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=6.5,
        frameon=True,
    )

    panel_label(full_ax, "(a)")
    panel_label(zoom_ax, "(b)")
    panel_label(yaw_ax, "(c)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    save_figure(fig, output_base)
    return (
        frames,
        course,
        position_frames,
        core_start_ns,
        core_end_ns,
        position_zoom_start_ns,
        position_zoom_end_ns,
        position_zoom_selection,
        rate_threshold,
        selection,
    )


def write_checksums(root):
    root = Path(root)
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256"):
        lines.append(f"{sha256_path(path)}  {path.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="Four-way result bag")
    parser.add_argument("--evaluation-json", required=True)
    parser.add_argument("--turn-json", required=True)
    parser.add_argument("--turn-detail-start-offset-s", type=float)
    parser.add_argument("--turn-detail-end-offset-s", type=float)
    parser.add_argument(
        "--position-zoom-start-offset-s",
        type=float,
        help="Panel (b) start offset from the selected-turn start",
    )
    parser.add_argument(
        "--position-zoom-end-offset-s",
        type=float,
        help="Panel (b) end offset from the selected-turn start",
    )
    parser.add_argument(
        "--position-zoom-margin-m",
        type=float,
        default=0.5,
        help="Per-axis margin around panel (b) trajectories",
    )
    parser.add_argument(
        "--hide-candidate-label",
        action="store_true",
        help="Omit the candidate identifier from figure titles and prose",
    )
    parser.add_argument(
        "--exclude-group-4",
        action="store_true",
        help="Omit the AHRS heading-rate group from the paper figure and text",
    )
    parser.add_argument("--common-start-ns", type=int)
    parser.add_argument("--common-end-ns", type=int)
    parser.add_argument(
        "--gnss-route-csv", default="",
        help="Valid PVT route table used as full-coverage context in panel (a)",
    )
    parser.add_argument(
        "--session-label",
        default="P9 session",
        help="Session name used in the generated caption and technical text",
    )
    parser.add_argument(
        "--run-provenance-json",
        required=True,
        help="Run provenance used to report the effective workflow settings",
    )
    parser.add_argument(
        "--repeatability-evaluation-json",
        default="",
        help="Repeated-turn evaluation with required interval metadata",
    )
    parser.add_argument(
        "--screening-json",
        default="",
        help="Structured parameter-screening decisions and next steps",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result_bag = Path(args.bag).resolve()
    evaluation_path = Path(args.evaluation_json).resolve()
    turn_path = Path(args.turn_json).resolve()
    provenance_path = Path(args.run_provenance_json).resolve()
    repeatability_path = (
        Path(args.repeatability_evaluation_json).resolve()
        if args.repeatability_evaluation_json else None
    )
    screening_path = (
        Path(args.screening_json).resolve()
        if args.screening_json else None
    )
    output = Path(args.out).resolve()
    optional_inputs = [
        path for path in [repeatability_path, screening_path]
        if path is not None
    ]
    for path in [
        result_bag, evaluation_path, turn_path, provenance_path,
        *optional_inputs,
    ]:
        if not path.exists():
            raise SystemExit(f"Input does not exist: {path}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    if args.exclude_group_4:
        GROUPS.pop("Group 4", None)
    if (args.common_start_ns is None) != (args.common_end_ns is None):
        raise SystemExit("Both common interval boundaries must be supplied together")

    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    turn = json.loads(turn_path.read_text(encoding="utf-8"))
    launch_parameters, run_provenance = read_launch_parameters(provenance_path)
    if run_provenance.get("status") != "completed":
        raise SystemExit(
            "Run provenance is not completed; refuse to read a bag that may still "
            "be changing"
        )
    if run_provenance.get("exit_code") != 0:
        raise SystemExit("Run provenance records a non-zero exit code")
    provenance_output = Path(
        run_provenance.get("output", {}).get("path", "")
    ).resolve()
    if provenance_output != result_bag:
        raise SystemExit(
            "Run provenance output does not match --bag: "
            f"{provenance_output} != {result_bag}"
        )
    repeatability, repeatability_interval = read_repeatability_summary(
        repeatability_path
    )
    if repeatability is not None and "Group 4" in GROUPS:
        raise SystemExit(
            "The supplied repeatability summary reports Groups 2 and 3 only; "
            "use --exclude-group-4"
        )
    screening = (
        json.loads(screening_path.read_text(encoding="utf-8"))
        if screening_path else None
    )
    required_launch_parameters = {
        "playback_rate",
        "smooth_lagged_data",
        "history_length",
        "predict_to_current_time",
        "imu_time_offset_ns",
        "gyro_covariance_scale",
        "subtract_static_gyro_bias",
        "minimum_course_speed_mps",
        "maximum_course_accuracy_deg",
        "group3_yaw_rate_source",
        "static_start_s",
        "static_duration_s",
    }
    missing_launch_parameters = sorted(
        required_launch_parameters - set(launch_parameters)
    )
    if missing_launch_parameters:
        raise SystemExit(
            "Run provenance lacks required launch parameters: "
            + ", ".join(missing_launch_parameters)
        )
    if launch_parameters["group3_yaw_rate_source"] != "raw_gyro_z":
        raise SystemExit(
            "This paper figure labels Group 3 as raw gyro z; the run provenance "
            "uses a different source"
        )
    ros_packages = run_provenance.get("environment", {}).get("packages", {})
    robot_localization_package = ros_packages.get(
        "ros-humble-robot-localization"
    )
    if not robot_localization_package:
        raise SystemExit(
            "Run provenance does not record the robot_localization package version"
        )
    robot_localization_version = robot_localization_package.split("-1", 1)[0]
    if screening is not None and screening.get("selected_configuration"):
        selected_configuration = screening["selected_configuration"]
        numeric_checks = {
            "playback_rate": float(launch_parameters["playback_rate"]),
            "history_length_s": float(launch_parameters["history_length"]),
            "minimum_course_speed_mps": float(
                launch_parameters["minimum_course_speed_mps"]
            ),
            "maximum_course_accuracy_estimate_deg": float(
                launch_parameters["maximum_course_accuracy_deg"]
            ),
            "imu_time_offset_ms": (
                float(launch_parameters["imu_time_offset_ns"]) / 1e6
            ),
            "gyro_covariance_scale": float(
                launch_parameters["gyro_covariance_scale"]
            ),
            "static_window_start_s_from_first_imu_timestamp": float(
                launch_parameters["static_start_s"]
            ),
            "static_window_duration_s": float(
                launch_parameters["static_duration_s"]
            ),
        }
        for key, actual in numeric_checks.items():
            expected = float(selected_configuration[key])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
                raise SystemExit(
                    f"Screening configuration mismatch for {key}: "
                    f"run={actual}, screening={expected}"
                )
        boolean_checks = {
            "smooth_lagged_data": bool_parameter(
                launch_parameters, "smooth_lagged_data"
            ),
            "predict_to_current_time": bool_parameter(
                launch_parameters, "predict_to_current_time"
            ),
            "subtract_static_gyro_bias": bool_parameter(
                launch_parameters, "subtract_static_gyro_bias"
            ),
        }
        for key, actual in boolean_checks.items():
            if actual is not bool(selected_configuration[key]):
                raise SystemExit(
                    f"Screening configuration mismatch for {key}"
                )
        if (
            launch_parameters["group3_yaw_rate_source"]
            != selected_configuration["group3_yaw_rate_source"]
        ):
            raise SystemExit(
                "Screening configuration mismatch for Group 3 source"
            )
        validate_selected_yaml(
            selected_configuration,
            launch_parameters,
            run_provenance,
        )
    start_ns = int(turn["start_ns"])
    end_ns = int(turn["end_ns"])
    source_candidate_label = turn.get("candidate_label")
    display_candidate_label = (
        None if args.hide_candidate_label else source_candidate_label
    )
    full_frames, full_course, full_raw_gyro = read_result(result_bag)
    if args.common_start_ns is not None:
        full_frames = {
            name: frame[
                (frame["t_ns"] >= args.common_start_ns)
                & (frame["t_ns"] <= args.common_end_ns)
            ].copy().reset_index(drop=True)
            for name, frame in full_frames.items()
        }
        full_course = full_course[
            (full_course["t_ns"] >= args.common_start_ns)
            & (full_course["t_ns"] <= args.common_end_ns)
        ].copy().reset_index(drop=True)
        full_raw_gyro = full_raw_gyro[
            (full_raw_gyro["t_ns"] >= args.common_start_ns)
            & (full_raw_gyro["t_ns"] <= args.common_end_ns)
        ].copy().reset_index(drop=True)
    gnss_route = None
    if args.gnss_route_csv:
        route_path = Path(args.gnss_route_csv).resolve()
        if not route_path.is_file():
            raise SystemExit(f"GNSS route table does not exist: {route_path}")
        gnss_route = pd.read_csv(route_path)
        gnss_route = gnss_route[
            gnss_route["t_ns"].between(
                args.common_start_ns, args.common_end_ns, inclusive="both"
            )
        ].copy() if args.common_start_ns is not None else gnss_route.copy()
        route_origin_time = int(full_frames["Group 2"]["t_ns"].iloc[0])
        route_reference = gnss_route.iloc[
            np.abs(gnss_route["t_ns"].to_numpy(dtype=np.int64) - route_origin_time).argmin()
        ]
        gnss_route["relative_x_m"] = gnss_route["east_m"] - float(route_reference["east_m"])
        gnss_route["relative_y_m"] = gnss_route["north_m"] - float(route_reference["north_m"])
    evaluation_bag = Path(evaluation.get("bag", "")).resolve()
    if evaluation_bag != result_bag:
        raise SystemExit(
            "Evaluation JSON does not match --bag: "
            f"{evaluation_bag} != {result_bag}"
        )
    summary = build_summary(evaluation, full_frames)
    output_coverage = {
        name: {
            "sample_count": int(len(frame)),
            "first_ns": int(frame["t_ns"].min()),
            "last_ns": int(frame["t_ns"].max()),
            "first_offset_from_common_start_s": (
                float((int(frame["t_ns"].min()) - args.common_start_ns) / NS_PER_SECOND)
                if args.common_start_ns is not None else None
            ),
            "last_offset_from_common_start_s": (
                float((int(frame["t_ns"].max()) - args.common_start_ns) / NS_PER_SECOND)
                if args.common_start_ns is not None else None
            ),
        }
        for name, frame in full_frames.items()
    }
    route_coverage = {
        "sample_count": int(len(gnss_route)) if gnss_route is not None else 0,
        "first_ns": int(gnss_route["t_ns"].min()) if gnss_route is not None and len(gnss_route) else None,
        "last_ns": int(gnss_route["t_ns"].max()) if gnss_route is not None and len(gnss_route) else None,
        "first_offset_from_common_start_s": (
            float((int(gnss_route["t_ns"].min()) - args.common_start_ns) / NS_PER_SECOND)
            if gnss_route is not None and len(gnss_route) and args.common_start_ns is not None else None
        ),
        "last_offset_from_common_start_s": (
            float((int(gnss_route["t_ns"].max()) - args.common_start_ns) / NS_PER_SECOND)
            if gnss_route is not None and len(gnss_route) and args.common_start_ns is not None else None
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        figures = temporary / "figures"
        tables = temporary / "tables"
        figures.mkdir()
        tables.mkdir()
        (
            frames,
            course,
            position_frames,
            core_start_ns,
            core_end_ns,
            position_zoom_start_ns,
            position_zoom_end_ns,
            position_zoom_selection,
            rate_threshold,
            detail_selection,
        ) = make_figure(
            full_frames,
            full_course,
            gnss_route,
            start_ns,
            end_ns,
            figures / "full_trajectory_and_turn_detail",
            detail_start_offset_s=args.turn_detail_start_offset_s,
            detail_end_offset_s=args.turn_detail_end_offset_s,
            position_zoom_start_offset_s=(
                args.position_zoom_start_offset_s
            ),
            position_zoom_end_offset_s=args.position_zoom_end_offset_s,
            position_zoom_margin_m=args.position_zoom_margin_m,
            candidate_label=display_candidate_label,
        )
        position_course = course[
            (course["t_ns"] >= position_zoom_start_ns)
            & (course["t_ns"] <= position_zoom_end_ns)
        ]
        if len(position_course) >= 2:
            position_course_change_deg = abs(math.degrees(
                float(position_course["yaw_unwrapped_rad"].iloc[-1])
                - float(position_course["yaw_unwrapped_rad"].iloc[0])
            ))
        else:
            position_course_change_deg = None
        position_zoom_duration_s = (
            position_zoom_end_ns - position_zoom_start_ns
        ) / NS_PER_SECOND
        summary.to_csv(tables / "fusion_workflow_summary.csv", index=False)

        full_rows = []
        for name, frame in full_frames.items():
            selected = frame[[
                "t_ns", "relative_x_m", "relative_y_m",
            ]].copy()
            selected.insert(0, "group", name)
            full_rows.append(selected)
        pd.concat(full_rows, ignore_index=True).to_csv(
            tables / "full_trajectory_series.csv", index=False
        )

        long_rows = []
        for name, frame in frames.items():
            columns = [
                "t_ns", "time_s", "relative_x_m", "relative_y_m",
            ]
            if GROUPS[name]["has_yaw"]:
                columns.append("yaw_unwrapped_rad")
            selected = frame[columns].copy()
            if "yaw_unwrapped_rad" not in selected:
                selected["yaw_unwrapped_rad"] = np.nan
            selected.insert(0, "group", name)
            long_rows.append(selected)
        pd.concat(long_rows, ignore_index=True).to_csv(
            tables / "turn_detail_fused_series.csv", index=False
        )
        position_rows = []
        for name, frame in position_frames.items():
            selected = frame[[
                "t_ns", "time_s", "relative_x_m", "relative_y_m",
            ]].copy()
            selected.insert(0, "group", name)
            position_rows.append(selected)
        pd.concat(position_rows, ignore_index=True).to_csv(
            tables / "position_zoom_fused_series.csv", index=False
        )
        position_separation = pairwise_position_separation(position_frames)
        position_separation.to_csv(
            tables / "position_zoom_pairwise_separation.csv", index=False
        )
        course[["t_ns", "time_s", "yaw_unwrapped_rad"]].to_csv(
            tables / "turn_detail_course.csv", index=False
        )
        if repeatability_interval is not None and (
            repeatability_interval["start_ns"] != core_start_ns
            or repeatability_interval["end_ns"] != core_end_ns
        ):
            raise RuntimeError(
                "Repeatability interval does not match the displayed turn: "
                f"{repeatability_interval} versus "
                f"{{'start_ns': {core_start_ns}, 'end_ns': {core_end_ns}}}"
            )
        direct_metrics = direct_turn_yaw_cog_metrics(frames, course)
        direct_metrics.to_csv(
            tables / "turn_yaw_cog_consistency.csv", index=False
        )
        turn_raw_gyro = full_raw_gyro[
            (full_raw_gyro["t_ns"] >= core_start_ns)
            & (full_raw_gyro["t_ns"] <= core_end_ns)
        ].copy().reset_index(drop=True)
        yaw_diagnostic = yaw_update_diagnostic(
            frames,
            course,
            turn_raw_gyro,
        )
        write_json(
            tables / "group3_yaw_update_diagnostic.json",
            yaw_diagnostic,
        )
        write_json(
            tables / "effective_workflow_parameters.json",
            launch_parameters,
        )
        if repeatability_path is not None:
            shutil.copy2(
                repeatability_path,
                tables / f"source_repeatability{repeatability_path.suffix}",
            )
        if screening is not None:
            screening_rows = pd.DataFrame(screening.get("tested_processes", []))
            screening_rows.to_csv(
                tables / "fusion_improvement_summary.csv", index=False
            )
            write_json(
                tables / "selected_fusion_configuration.json",
                screening.get("selected_configuration", {}),
            )
            next_steps = screening.get("next_steps", [])
            (temporary / "recommended_next_tests.txt").write_text(
                "\n".join(
                    f"{index}. {item}"
                    for index, item in enumerate(next_steps, start=1)
                ) + ("\n" if next_steps else ""),
                encoding="utf-8",
            )
            assessment_lines = [
                "Fusion parameter-screening assessment",
                "",
                "Selected paper workflow",
                "-----------------------",
                json.dumps(
                    screening.get("selected_configuration", {}),
                    indent=2,
                    ensure_ascii=False,
                ),
                "",
                "Tested processes",
                "----------------",
            ]
            for item in screening.get("tested_processes", []):
                assessment_lines.extend([
                    item["process"],
                    f"Tested: {item['tested_settings']}",
                    f"Observation: {item['observation']}",
                    f"Decision: {item['decision']}",
                    f"Evidence: {item['evidence']}",
                    "",
                ])
            assessment_lines.extend([
                "Recommended next tests",
                "----------------------",
                *[
                    f"{index}. {item}"
                    for index, item in enumerate(next_steps, start=1)
                ],
                "",
            ])
            (temporary / "fusion_improvement_assessment.txt").write_text(
                "\n".join(assessment_lines),
                encoding="utf-8",
            )

        route_note = (
            "The dotted PVT route is shown as context in panel (a).\n"
            if gnss_route is not None and len(gnss_route)
            else "Panel (a) shows only the fusion outputs; no separate PVT context route is overlaid.\n"
        )
        first_output_offset = min(
            item["first_offset_from_common_start_s"]
            for item in output_coverage.values()
            if item["first_offset_from_common_start_s"] is not None
        )
        output_gate_note = (
            f"The first fusion output is {first_output_offset:.3f} s after the "
            "analysis start. This offset reflects filter initialization and input "
            "availability and is not interpreted as sensor delay.\n"
        )
        note = (
            "Scope: workflow/application example only.\n"
            "Group 1 contains GNSS position only and has no yaw estimate.\n"
            "The yaw-estimating groups use GNSS position and GNSS-derived course over ground "
            "from the same receiver.\n"
            "The full-trajectory summary uses course derived from the position trajectory.\n"
            "The turn-specific table uses the receiver COG displayed in panel (c).\n"
            "Both are internal consistency measures, not independent accuracy.\n"
            "The selected turn is defined by the accompanying technical-validation output.\n"
            "Panel (b) shows a spatially enlarged segment within the turn.\n"
            + route_note
            + output_gate_note
            + "Panel (c) uses the full selected-turn interval.\n"
        )
        (temporary / "README.txt").write_text(note, encoding="utf-8")
        candidate_sentence = (
            f"This comparison version shows turn candidate {display_candidate_label}. "
            if display_candidate_label else ""
        )
        group_count_word = "three" if len(GROUPS) == 3 else "four"
        optional_group_4_caption = (
            " Group 4 additionally uses the AHRS heading rate."
            if "Group 4" in GROUPS else ""
        )
        yaw_group_range = "Groups 2-4" if "Group 4" in GROUPS else "Groups 2 and 3"
        route_caption = (
            "The dotted line is the valid PVT position context; "
            if gnss_route is not None and len(gnss_route) else ""
        )
        figure_caption = (
            f"Figure X. Exploratory GNSS-IMU fusion workflow for the {args.session_label}. "
            f"{candidate_sentence}"
            f"(a) Full trajectory; {route_caption}the black circle marks the area enlarged in "
            "panel (b). (b) Enlarged view of the main turning section, "
            f"showing the position estimates from the {group_count_word} groups. Group 1 uses "
            "GNSS position only. Group 2 combines GNSS position with "
            "GNSS-derived course over ground (COG). Group 3 additionally uses "
            "bias-corrected raw gyroscope z-axis angular velocity."
            f"{optional_group_4_caption} (c) Fused yaw from {yaw_group_range} compared "
            "with the GNSS-derived COG measurement. Group 1 is excluded from "
            "panel (c) because it does not estimate yaw. The yaw--COG differences "
            "show internal consistency rather than independent heading error.\n"
        )
        (temporary / "figure_caption.txt").write_text(
            figure_caption,
            encoding="utf-8",
        )

        summary_by_group = summary.set_index("group")
        rates_text = ", ".join(
            f"{summary_by_group.loc[name, 'effective_rate_hz']:.2f} Hz for {name}"
            for name in GROUPS
        )
        optional_group_4_method = (
            " Group 4 uses AHRS headingspeed as the yaw-rate input."
            if "Group 4" in GROUPS else ""
        )
        overlap_text = (
            " Groups 3 and 4 nearly overlap, indicating similar responses to the two yaw-rate inputs."
            if "Group 4" in GROUPS else ""
        )
        route_text = (
            f"The dotted route in panel (a) shows {route_coverage['sample_count']} valid /ubx_nav_pvt positions in the common interval. "
            if gnss_route is not None and len(gnss_route) else
            "Panel (a) contains only the fusion outputs; no independent PVT context route is overlaid. "
        )
        direct_by_group = direct_metrics.set_index("group")
        group23_separation = position_separation[
            (position_separation["first_group"] == "Group 2")
            & (position_separation["second_group"] == "Group 3")
        ].iloc[0]
        maximum_group_separation_m = float(
            position_separation["maximum_separation_m"].max()
        )
        if repeatability is not None:
            yaw_result_text = (
                "A separate repeatability check used the same fixed 10 s turn "
                "and selected raw-gyro configuration. Across "
                f"{int(repeatability['repeat_count'])} repeated replays, the "
                "median absolute yaw--COG difference was "
                f"$({repeatability['group2_yaw_median_abs_deg_mean']:.3f} "
                f"\\pm {repeatability['group2_yaw_median_abs_deg_std']:.3f})^"
                "\\circ$ "
                "for Group 2 and "
                f"$({repeatability['group3_yaw_median_abs_deg_mean']:.3f} "
                f"\\pm {repeatability['group3_yaw_median_abs_deg_std']:.3f})^"
                "\\circ$ "
                "for Group 3. The corresponding P95 values were "
                f"$({repeatability['group2_yaw_p95_abs_deg_mean']:.3f} "
                f"\\pm {repeatability['group2_yaw_p95_abs_deg_std']:.3f})^"
                "\\circ$ and "
                f"$({repeatability['group3_yaw_p95_abs_deg_mean']:.3f} "
                f"\\pm {repeatability['group3_yaw_p95_abs_deg_std']:.3f})^"
                "\\circ$."
            )
        else:
            yaw_result_text = (
                "In the displayed run, the median absolute yaw--COG difference "
                f"was {direct_by_group.loc['Group 2', 'median_abs_yaw_cog_difference_deg']:.3f} "
                "degrees for Group 2 and "
                f"{direct_by_group.loc['Group 3', 'median_abs_yaw_cog_difference_deg']:.3f} "
                "degrees for Group 3. The corresponding P95 values were "
                f"{direct_by_group.loc['Group 2', 'p95_abs_yaw_cog_difference_deg']:.3f} "
                "degrees and "
                f"{direct_by_group.loc['Group 3', 'p95_abs_yaw_cog_difference_deg']:.3f} "
                "degrees."
            )

        playback_rate = launch_parameters.get("playback_rate", "1.0")
        lag_history = launch_parameters["history_length"]
        gyro_scale = launch_parameters["gyro_covariance_scale"]
        static_duration = launch_parameters["static_duration_s"]
        imu_offset_ms = (
            float(launch_parameters["imu_time_offset_ns"]) / 1e6
        )
        smooth_lagged = bool_parameter(
            launch_parameters, "smooth_lagged_data"
        )
        predict_current = bool_parameter(
            launch_parameters, "predict_to_current_time"
        )
        subtract_bias = bool_parameter(
            launch_parameters, "subtract_static_gyro_bias"
        )
        processing_sentences = [
            f"The result was generated at {playback_rate}x playback."
        ]
        if smooth_lagged:
            processing_sentences.append(
                f"Delayed-measurement smoothing used a {lag_history} s history."
            )
        else:
            processing_sentences.append(
                "Delayed-measurement smoothing was disabled."
            )
        processing_sentences.append(
            "Prediction to the current filter time was "
            + ("enabled." if predict_current else "disabled.")
        )
        processing_sentences.append(
            f"The raw gyroscope covariance was scaled by {gyro_scale}."
        )
        if subtract_bias:
            processing_sentences.append(
                "Its bias was estimated from a video-confirmed "
                f"{float(static_duration):.1f} s stationary interval and removed "
                "before fusion."
            )
        else:
            processing_sentences.append(
                "No static gyroscope-bias correction was applied."
            )
        if imu_offset_ms == 0:
            processing_sentences.append("No fixed IMU time shift was applied (0 ms).")
        else:
            processing_sentences.append(
                f"A fixed IMU time shift of {imu_offset_ms:.1f} ms was applied."
            )
        processing_text = " ".join(processing_sentences)
        method_text = (
            "In two-dimensional mode, the state components directly observed in "
            "this comparison include $p_x$, $p_y$, $\\psi$, and $\\dot{\\psi}$. "
            "Prediction "
            "and covariance propagation follow "
            "$\\mathbf{x}_k^-=f(\\mathbf{x}_{k-1}^+,\\Delta t)$ and "
            "$\\mathbf{P}_k^-=\\mathbf{F}_k\\mathbf{P}_{k-1}^+"
            "\\mathbf{F}_k^T+\\mathbf{Q}_k$. For Group 3, the gyroscope supplies "
            "the yaw-rate observation used between GNSS updates. GNSS position "
            "updates $p_x$ and $p_y$, COG updates $\\psi$, and raw gyroscope z "
            "updates $\\dot{\\psi}$. The effective parameters and launch setup "
            "are provided in the repository files "
            "\\texttt{offline\\_four\\_way\\_compare.yaml} and "
            "\\texttt{offline\\_four\\_way\\_compare.launch.py}."
        )
        screening_text = (
            "\n\n" + screening.get("paper_summary", "").strip()
            if screening and screening.get("paper_summary") else ""
        )
        negative_count = yaw_diagnostic["group3_negative_yaw_step_count"]
        negative_near_count = yaw_diagnostic[
            "negative_steps_near_course_count"
        ]
        near_p95 = yaw_diagnostic[
            "gyro_propagation_residual_abs_deg_near_course"
        ]["p95"]
        far_p95 = yaw_diagnostic[
            "gyro_propagation_residual_abs_deg_far_from_course"
        ]["p95"]
        yaw_update_text = (
            "The small saw-tooth changes in Group 3 are discrete filter "
            "corrections rather than angle wrapping. Between COG observations, "
            "the 20 Hz output is propagated with the gyroscope. A new COG "
            "observation then corrects the accumulated yaw state. In the displayed "
            f"interval, {negative_near_count} of {negative_count} negative "
            "one-step yaw changes larger than 1 degree occurred within 60 ms of "
            "a COG timestamp. The P95 absolute difference between the observed "
            "Group 3 step and gyro-only propagation was "
            f"{near_p95:.3f} degrees near COG updates and {far_p95:.3f} degrees "
            "away from them. The raw gyroscope was continuous over this section; "
            "the pattern therefore reflects the disagreement between body-frame "
            "gyro propagation and the lower-rate velocity-course observation. "
            "Connected 5 Hz COG points also appear visually smoother than the "
            "20 Hz EKF output."
        )
        technical_validation_text = (
            "As an application example, a two-dimensional extended Kalman filter "
            "was implemented with ROS 2 robot_localization (version "
            f"{robot_localization_version}). GNSS "
            "position from /fix was transformed to a local ENU frame. Course over "
            "ground (COG) was taken from UBX-NAV-VELNED and converted to ENU yaw. "
            "COG measurements were accepted at ground speeds of at least "
            f"{float(launch_parameters['minimum_course_speed_mps']):g} m/s and "
            "with a receiver-reported course-accuracy estimate not greater than "
            f"{float(launch_parameters['maximum_course_accuracy_deg']):g} degrees. "
            "Group 1 used GNSS position only. Group 2 added GNSS-derived COG, and "
            "Group 3 further added the bias-corrected raw gyroscope z-axis rate."
            f"{optional_group_4_method}\n\n"
            + method_text
            + "\n\n"
            + processing_text
            + "\n\n"
            + route_text
            + f"Panel (b) enlarges a {position_zoom_duration_s:.2f} s interval"
            + (
                f", over which COG changes by approximately "
                f"{position_course_change_deg:.1f} degrees. "
                if position_course_change_deg is not None else ". "
            )
            + f"The {group_count_word} position estimates remain close because all "
            "groups use the same GNSS position observations. Their local separation "
            "reflects filter update timing and state propagation and is not an "
            "independent position error. In this enlarged section, the maximum "
            f"temporally aligned separation was {maximum_group_separation_m:.3f} m "
            "across all displayed pairs and "
            f"{float(group23_separation['maximum_separation_m']):.3f} m between "
            f"Groups 2 and 3.{overlap_text} The effective output rates "
            f"were {rates_text}.\n\n"
            + yaw_result_text
            + " The gyroscope provides higher-rate rotational updates between GNSS "
            "observations, but Group 3 did not improve yaw--COG consistency under "
            "the selected configuration. "
            + yaw_update_text
            + " These differences are internal consistency "
            "measures because COG is an EKF input, not independent heading ground "
            "truth. The example demonstrates timestamp handling, sensor compatibility, "
            "and different update rates; it is not an absolute positioning or heading "
            "accuracy validation."
            + screening_text
            + "\n"
        )
        (temporary / "technical_validation_text.txt").write_text(
            technical_validation_text,
            encoding="utf-8",
        )
        write_json(temporary / "run_manifest.json", {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "command": [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ],
            "software_versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "source_files": {
                "figure_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": sha256_path(Path(__file__).resolve()),
                },
                "paper_style": {
                    "path": str(SCRIPT_DIR / "paper_style.py"),
                    "sha256": sha256_path(SCRIPT_DIR / "paper_style.py"),
                },
            },
            "scope": "Workflow/application example; not independent accuracy validation",
            "included_groups": list(GROUPS),
            "group_4_excluded": args.exclude_group_4,
            "common_interval_start_ns": args.common_start_ns,
            "common_interval_end_ns": args.common_end_ns,
            "session_label": args.session_label,
            "result_bag": str(result_bag),
            "result_bag_sha256": sha256_path(result_bag),
            "evaluation_json": str(evaluation_path),
            "evaluation_json_sha256": sha256_path(evaluation_path),
            "turn_json": str(turn_path),
            "turn_json_sha256": sha256_path(turn_path),
            "run_provenance_json": (
                str(provenance_path) if provenance_path else None
            ),
            "run_provenance_json_sha256": (
                sha256_path(provenance_path) if provenance_path else None
            ),
            "repeatability_evaluation_json": (
                str(repeatability_path) if repeatability_path else None
            ),
            "repeatability_evaluation_json_sha256": (
                sha256_path(repeatability_path) if repeatability_path else None
            ),
            "screening_json": str(screening_path) if screening_path else None,
            "screening_json_sha256": (
                sha256_path(screening_path) if screening_path else None
            ),
            "effective_launch_parameters": launch_parameters,
            "turn_selection_label": source_candidate_label,
            "turn_selection_label_displayed": bool(display_candidate_label),
            "turn_start_ns": start_ns,
            "turn_end_ns": end_ns,
            "turn_duration_s": (end_ns - start_ns) / NS_PER_SECOND,
            "turn_detail_start_ns": core_start_ns,
            "turn_detail_end_ns": core_end_ns,
            "turn_detail_duration_s": (
                core_end_ns - core_start_ns
            ) / NS_PER_SECOND,
            "turn_detail_selection": detail_selection,
            "turn_detail_start_offset_s": args.turn_detail_start_offset_s,
            "turn_detail_end_offset_s": args.turn_detail_end_offset_s,
            "turn_rate_threshold_rad_s": rate_threshold,
            "position_zoom_start_ns": position_zoom_start_ns,
            "position_zoom_end_ns": position_zoom_end_ns,
            "position_zoom_duration_s": position_zoom_duration_s,
            "position_zoom_selection": position_zoom_selection,
            "position_zoom_start_offset_s": (
                args.position_zoom_start_offset_s
            ),
            "position_zoom_end_offset_s": args.position_zoom_end_offset_s,
            "position_zoom_margin_m": args.position_zoom_margin_m,
            "position_zoom_course_change_deg": position_course_change_deg,
            "full_trajectory_samples": {
                **{name: int(len(frame)) for name, frame in full_frames.items()},
                "GNSS-derived course over ground": int(len(full_course)),
                "bias-corrected raw gyroscope": int(len(full_raw_gyro)),
            },
            "gnss_route_context": {
                "path": str(Path(args.gnss_route_csv).resolve()) if args.gnss_route_csv else None,
                "sha256": (
                    sha256_path(Path(args.gnss_route_csv).resolve())
                    if args.gnss_route_csv else None
                ),
                "sample_count": int(len(gnss_route)) if gnss_route is not None else 0,
                "source": "/ubx_nav_pvt valid positions",
            },
            "output_coverage": output_coverage,
            "route_coverage": route_coverage,
            "turn_detail_samples": {
                **{name: int(len(frame)) for name, frame in frames.items()},
                "GNSS-derived course over ground": int(len(course)),
            },
            "position_zoom_samples": {
                **{
                    name: int(len(frame))
                    for name, frame in position_frames.items()
                },
                "GNSS-derived course over ground": int(len(position_course)),
            },
            "displayed_turn_yaw_cog_metrics": direct_metrics.to_dict(
                orient="records"
            ),
            "position_zoom_pairwise_separation": position_separation.to_dict(
                orient="records"
            ),
            "group3_yaw_update_diagnostic": yaw_diagnostic,
        })
        write_checksums(temporary)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Fusion workflow example: {output}")


if __name__ == "__main__":
    main()
