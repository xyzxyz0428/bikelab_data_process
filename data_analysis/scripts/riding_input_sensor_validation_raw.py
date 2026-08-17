#!/usr/bin/env python3
"""Validate riding-input sensors during a GNSS-speed-selected interval.

The selected interval runs from the first to the last /ubx_nav_vel_ned sample
whose ground speed is greater than the configured threshold. An optional
rule-based cleaned steering table can be supplied; limit values are not
removed merely because they equal +/-45 degrees. The plotted steering signal
is centred on a data-derived neutral position. The brake zero-input band is
estimated from a separately specified static reference interval when available.
Raw files are read-only; use --overwrite only for a derived result directory.
"""

import argparse
import importlib.util
import json
import math
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_style import COLORS, apply_paper_style, panel_label, save_figure  # noqa: E402
import video_time_mapping as VIDEO_TIME  # noqa: E402


BASE_PATH = SCRIPT_DIR / "riding_input_sensor_validation.py"
SPEC = importlib.util.spec_from_file_location("riding_input_sensor_validation", BASE_PATH)
BASE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE)

NS_PER_SECOND = BASE.NS_PER_SECOND
STEERING_LIMIT_DEG = 45.0


def stamp_ns(message):
    return (
        int(message.header.stamp.sec) * NS_PER_SECOND
        + int(message.header.stamp.nanosec)
    )


def read_gnss_speed(bag_dir: Path) -> pd.DataFrame:
    """Read GNSS ground speed from UBX-NAV-VEL-NED."""
    if bag_dir.is_file():
        gnss_path = SCRIPT_DIR / "gnss_imu_technical_validation.py"
        gnss_spec = importlib.util.spec_from_file_location("gnss_technical_validation", gnss_path)
        gnss = importlib.util.module_from_spec(gnss_spec)
        assert gnss_spec.loader is not None
        gnss_spec.loader.exec_module(gnss)
        frame = gnss.read_bag_topics(bag_dir, "sqlite3")["vel"].rename(
            columns={"t_ns": "t_unix_ns"}
        )
        return frame.sort_values("t_unix_ns").drop_duplicates("t_unix_ns").reset_index(drop=True)
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    type_map = {item.name: item.type for item in reader.get_all_topics_and_types()}
    topic = "/ubx_nav_vel_ned"
    if topic not in type_map:
        raise RuntimeError(f"Missing GNSS speed topic: {topic}")
    message_type = get_message(type_map[topic])
    rows = []
    while reader.has_next():
        current_topic, serialized, record_ns = reader.read_next()
        if current_topic != topic:
            continue
        message = deserialize_message(serialized, message_type)
        rows.append({
            # Use rosbag receive time for alignment with camera timestamps.
            # Keep the ROS header stamp for diagnostics.
            "t_unix_ns": int(record_ns),
            "header_ns": stamp_ns(message),
            "ground_speed_mps": float(message.g_speed) * 0.01,
            "course_deg": float(message.heading) * 1.0e-5,
            "course_accuracy_deg": float(message.c_acc) * 1.0e-5,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No GNSS speed samples were found")
    return frame.sort_values("t_unix_ns").drop_duplicates("t_unix_ns").reset_index(drop=True)


def select_gnss_positive_interval(speed: pd.DataFrame, threshold_mps: float):
    positive = np.isfinite(speed["ground_speed_mps"]) & (
        speed["ground_speed_mps"] > threshold_mps
    )
    if not positive.any():
        raise RuntimeError(
            f"No GNSS speed samples greater than {threshold_mps} m/s were found"
        )
    selected = speed.loc[positive].copy()
    return (
        int(selected["t_unix_ns"].min()),
        int(selected["t_unix_ns"].max()),
        selected,
    )


def crop(frame: pd.DataFrame, start_ns: int, end_ns: int) -> pd.DataFrame:
    time = pd.to_numeric(frame["t_unix_ns"], errors="coerce")
    return frame[(time >= start_ns) & (time <= end_ns)].copy().reset_index(drop=True)


def modulo_difference(series: pd.Series, modulus: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(values.size, np.nan)
    if values.size > 1:
        result[1:] = np.mod(values[1:] - values[:-1], modulus)
    return result


def unwrap_modulo_counter(series: pd.Series, modulus: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = np.full(values.size, np.nan)
    if not values.size or not np.isfinite(values[0]):
        return result
    result[0] = values[0]
    for index in range(1, values.size):
        if not np.isfinite(values[index]) or not np.isfinite(values[index - 1]):
            continue
        result[index] = result[index - 1] + np.mod(
            values[index] - values[index - 1], modulus
        )
    return result


def derive_standard_torque_page(power: pd.DataFrame) -> pd.DataFrame:
    """Derive event-level crank torque, cadence and power from page 0x12."""
    torque = power.loc[power["page_name"].eq("standard_torque")].copy()
    torque = torque.sort_values("t_unix_ns").reset_index(drop=True)
    event_delta = modulo_difference(torque["p12_update_event_count"], 256)
    tick_delta = modulo_difference(torque["p12_crank_ticks"], 256)
    period_delta = modulo_difference(torque["p12_crank_period_1_2048s"], 65536)
    torque_delta = modulo_difference(torque["p12_accumulated_torque_1_32nm"], 65536)

    valid = (
        np.isfinite(event_delta)
        & np.isfinite(tick_delta)
        & np.isfinite(period_delta)
        & np.isfinite(torque_delta)
        & (event_delta > 0)
        & (tick_delta > 0)
        & (period_delta > 0)
    )
    derived_cadence = np.full(len(torque), np.nan)
    derived_torque = np.full(len(torque), np.nan)
    derived_cadence[valid] = 60.0 * tick_delta[valid] / (period_delta[valid] / 2048.0)
    derived_torque[valid] = (torque_delta[valid] / 32.0) / event_delta[valid]

    torque["event_count_delta"] = event_delta
    torque["unwrapped_update_event_count"] = unwrap_modulo_counter(
        torque["p12_update_event_count"], 256
    )
    torque["crank_tick_delta"] = tick_delta
    torque["crank_period_delta_s"] = period_delta / 2048.0
    torque["accumulated_torque_delta_nm"] = torque_delta / 32.0
    torque["derived_cadence_rpm"] = derived_cadence
    torque["derived_crank_torque_nm"] = derived_torque
    torque["derived_power_w"] = (
        derived_torque * (2.0 * np.pi * derived_cadence / 60.0)
    )
    torque["derived_valid"] = valid
    return torque


def associate_brake_and_steering(
    brake: pd.DataFrame,
    steering: pd.DataFrame,
    tolerance_s: float,
) -> pd.DataFrame:
    brake_values = brake[["t_unix_ns", "left_force_n", "right_force_n"]].copy()
    steering_values = steering[["t_unix_ns", "angle_deg"]].copy()
    brake_values["t_unix_ns"] = pd.to_numeric(
        brake_values["t_unix_ns"], errors="coerce"
    ).astype("int64")
    steering_values["t_unix_ns"] = pd.to_numeric(
        steering_values["t_unix_ns"], errors="coerce"
    ).astype("int64")
    steering_values = steering_values.rename(columns={"t_unix_ns": "steering_t_unix_ns"})
    associated = pd.merge_asof(
        brake_values.sort_values("t_unix_ns"),
        steering_values.sort_values("steering_t_unix_ns"),
        left_on="t_unix_ns",
        right_on="steering_t_unix_ns",
        direction="nearest",
        tolerance=int(tolerance_s * NS_PER_SECOND),
    )
    associated["association_abs_dt_ms"] = (
        associated["t_unix_ns"] - associated["steering_t_unix_ns"]
    ).abs() / 1.0e6
    return associated


def estimate_zero_brake_bands(
    associated: pd.DataFrame,
    steering_max_abs_deg: float,
    minimum_reference_samples: int,
    speed: pd.DataFrame | None = None,
    speed_threshold_mps: float = 0.1,
    reference_brake: pd.DataFrame | None = None,
    reference_description: str | None = None,
):
    reference_mask = pd.to_numeric(associated["angle_deg"], errors="coerce").abs() <= steering_max_abs_deg
    stop_associated = None
    if speed is not None and len(speed):
        speed_values = speed[["t_unix_ns", "ground_speed_mps"]].copy()
        speed_values["t_unix_ns"] = pd.to_numeric(speed_values["t_unix_ns"], errors="coerce")
        speed_values["ground_speed_mps"] = pd.to_numeric(speed_values["ground_speed_mps"], errors="coerce")
        speed_values = speed_values.dropna().sort_values("t_unix_ns")
        stop_associated = pd.merge_asof(
            associated.sort_values("t_unix_ns"), speed_values, on="t_unix_ns",
            direction="nearest", tolerance=int(0.25 * NS_PER_SECOND),
        )
        stop_associated = stop_associated[stop_associated["ground_speed_mps"] < speed_threshold_mps]

    static_values = {}
    if reference_brake is not None and len(reference_brake):
        for column in ["left_force_n", "right_force_n"]:
            if column in reference_brake:
                values = pd.to_numeric(reference_brake[column], errors="coerce")
                values = values[np.isfinite(values)].to_numpy(dtype=float)
                if values.size:
                    static_values[column] = values

    bands, rows = {}, []
    for channel, column in [("Left brake", "left_force_n"), ("Right brake", "right_force_n")]:
        steering_values = pd.to_numeric(associated.loc[reference_mask, column], errors="coerce").dropna().to_numpy(float)
        stop_values = (
            pd.to_numeric(stop_associated[column], errors="coerce").dropna().to_numpy(float)
            if stop_associated is not None else np.array([], dtype=float)
        )
        if static_values.get(column, np.array([], dtype=float)).size >= minimum_reference_samples:
            # Use the pre-ride static interval rather than inferring a baseline
            # from the riding signal, which may contain real brake events.
            values = static_values[column]
            lower = float(max(0.0, np.min(values)))
            upper = float(np.max(values))
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            method = "Full observed force range in static reference interval"
            condition = reference_description or "Static reference interval"
            sufficient, fallback = True, False
            baseline_q25 = float(np.quantile(values, 0.25))
            baseline_count = int(values.size)
            static_reference_count = int(values.size)
        elif stop_values.size >= minimum_reference_samples:
            # A stopped bicycle can still have a non-zero sensor output caused
            # by preload or electronics noise.  Use the complete observed
            # stopped-speed range as a conservative diagnostic band.  A force
            # above its maximum is then marked as a candidate brake pulse.
            values = stop_values
            lower = float(max(0.0, np.min(values)))
            upper = float(np.max(values))
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            method = "Full observed stopped-speed range (0 to maximum)"
            condition = f"GNSS ground speed < {speed_threshold_mps:g} m/s; full stopped-speed force range"
            sufficient, fallback = True, False
            baseline_q25 = float(np.quantile(values, 0.25))
            baseline_count = int(values.size)
            static_reference_count = 0
        elif steering_values.size:
            values = steering_values
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            baseline_q25 = float(np.quantile(values, 0.25))
            lower, upper = float(np.min(values)), float(np.max(values))
            method = "Empirical minimum and maximum of steering-reference samples"
            condition = f"|steering angle| <= {steering_max_abs_deg:g} deg"
            sufficient, fallback = values.size >= minimum_reference_samples, values.size < minimum_reference_samples
            baseline_count = int(values.size)
            static_reference_count = 0
            if fallback:
                all_values = pd.to_numeric(associated[column], errors="coerce").dropna().to_numpy(float)
                seed = float(np.quantile(all_values, 0.25))
                baseline = all_values[all_values <= seed]
                median = float(np.median(baseline))
                mad = float(np.median(np.abs(baseline - median)))
                lower = float(np.min(baseline))
                upper = float(min(max(median + 3.0 * 1.4826 * mad, seed), np.quantile(all_values, 0.5)))
                method = "Robust lower-force baseline (lower quartile, median + 3 MAD)"
                condition = f"steering reference |delta| <= {steering_max_abs_deg:g} deg; insufficient, fallback from lower-force quartile"
                baseline_q25, baseline_count = seed, int(baseline.size)
        else:
            raise RuntimeError(f"No {channel.lower()} samples were associated with a valid reference")
        bands[column] = (lower, upper)
        rows.append({
            "channel": channel, "force_column": column, "reference_condition": condition,
            "reference_sample_count": int(values.size),
            "reference_fraction_of_brake_samples": float(values.size / len(associated)),
            "band_method": method, "baseline_seed_sample_count": baseline_count,
            "baseline_median_n": median, "baseline_mad_n": mad, "baseline_q25_n": baseline_q25,
            "zero_band_lower_n": lower, "zero_band_median_n": median, "zero_band_upper_n": upper,
            "minimum_reference_samples": minimum_reference_samples,
            "reference_sample_count_sufficient": bool(sufficient), "fallback_used": bool(fallback),
            "speed_reference_sample_count": int(stop_values.size),
            "speed_reference_threshold_mps": speed_threshold_mps,
            "static_reference_sample_count": static_reference_count,
            "static_reference_description": reference_description or "",
        })
    return bands, pd.DataFrame(rows), reference_mask


def estimate_steering_neutral(
    steering: pd.DataFrame,
    bin_width_deg: float = 0.5,
    cluster_half_width_deg: float = 2.5,
):
    """Estimate the steering neutral position from the dominant signal cluster.

    The absolute calibration value is not treated as the bicycle's neutral
    angle.  Instead, the densest valid angle cluster is selected, refined by
    its median, and converted to a relative angle.  The neutral band is the
    symmetric 1--99 percent range of that cluster, so isolated turns do not
    determine the displayed zero reference.
    """
    values = pd.to_numeric(steering.get("angle_deg"), errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (np.abs(values) < STEERING_LIMIT_DEG)]
    if values.size < 20:
        raise RuntimeError("Too few valid steering samples for neutral estimation")
    minimum = float(np.floor(values.min() / bin_width_deg) * bin_width_deg)
    maximum = float(np.ceil(values.max() / bin_width_deg) * bin_width_deg + bin_width_deg)
    edges = np.arange(minimum, maximum + 0.5 * bin_width_deg, bin_width_deg)
    counts, edges = np.histogram(values, bins=edges)
    peak_count = int(counts.max())
    peak_indices = np.flatnonzero(counts == peak_count)
    median_all = float(np.median(values))
    centres = (edges[peak_indices] + edges[peak_indices + 1]) / 2.0
    mode_centre = float(centres[np.argmin(np.abs(centres - median_all))])
    cluster = values[np.abs(values - mode_centre) <= cluster_half_width_deg]
    if cluster.size < 20:
        cluster = values[np.abs(values - mode_centre) <= 2.0 * cluster_half_width_deg]
    neutral_deg = float(np.median(cluster))
    for _ in range(2):
        refined = values[np.abs(values - neutral_deg) <= cluster_half_width_deg]
        if refined.size < 20:
            break
        neutral_deg = float(np.median(refined))
        cluster = refined
    residual = cluster - neutral_deg
    half_width = float(max(
        0.5,
        abs(np.quantile(residual, 0.01)),
        abs(np.quantile(residual, 0.99)),
    ))
    return {
        "method": "Dominant steering-angle cluster; median-refined",
        "neutral_angle_deg": neutral_deg,
        "neutral_band_lower_relative_deg": -half_width,
        "neutral_band_upper_relative_deg": half_width,
        "histogram_bin_width_deg": bin_width_deg,
        "cluster_half_width_deg": cluster_half_width_deg,
        "reference_sample_count": int(cluster.size),
        "valid_sample_count": int(values.size),
        "reference_p01_relative_deg": float(np.quantile(residual, 0.01)),
        "reference_p99_relative_deg": float(np.quantile(residual, 0.99)),
    }


def brake_pulse_statistics(
    brake: pd.DataFrame,
    column: str,
    band_upper_n: float,
    interval_duration_s: float,
    minimum_pulse_duration_s: float,
):
    time_ns = pd.to_numeric(brake["t_unix_ns"], errors="coerce").to_numpy(dtype=float)
    force = pd.to_numeric(brake[column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_ns) & np.isfinite(force)
    time_ns = time_ns[valid]
    force = force[valid]
    positive_dt_s = np.diff(time_ns) / NS_PER_SECOND
    positive_dt_s = positive_dt_s[np.isfinite(positive_dt_s) & (positive_dt_s > 0)]
    sample_period_s = float(np.median(positive_dt_s)) if positive_dt_s.size else 0.0
    gap_limit_s = 1.5 * sample_period_s if sample_period_s else math.inf
    minimum_pulse_samples = max(
        1,
        int(math.floor(minimum_pulse_duration_s / sample_period_s + 0.5))
        if sample_period_s else 1,
    )
    above = force > band_upper_n

    events = []
    start_index = None
    for index, is_above in enumerate(above):
        separated = (
            index > 0
            and (time_ns[index] - time_ns[index - 1]) / NS_PER_SECOND > gap_limit_s
        )
        if is_above and (start_index is None or separated):
            if start_index is not None:
                events.append((start_index, index - 1))
            start_index = index
        elif not is_above and start_index is not None:
            events.append((start_index, index - 1))
            start_index = None
    if start_index is not None:
        events.append((start_index, len(above) - 1))

    event_rows = []
    for event_index, (first, last) in enumerate(events, start=1):
        duration_s = float((last - first + 1) * sample_period_s)
        excess = np.maximum(force[first:last + 1] - band_upper_n, 0.0)
        event_rows.append({
            "event_index": event_index,
            "force_column": column,
            "start_ns": int(time_ns[first]),
            "end_ns": int(time_ns[last]),
            "sample_count": int(last - first + 1),
            "duration_s": duration_s,
            "peak_force_n": float(np.max(force[first:last + 1])),
            "excess_force_impulse_n_s": float(np.sum(excess) * sample_period_s),
            "meets_minimum_duration": bool(
                last - first + 1 >= minimum_pulse_samples
            ),
        })
    event_table = pd.DataFrame(event_rows)
    qualified = (
        event_table[event_table["meets_minimum_duration"]]
        if len(event_table)
        else event_table
    )
    active_duration_s = float(np.count_nonzero(above) * sample_period_s)
    qualified_duration_s = (
        float(qualified["duration_s"].sum()) if len(qualified) else 0.0
    )
    summary = {
        "force_column": column,
        "zero_band_upper_n": band_upper_n,
        "sample_period_s": sample_period_s,
        "samples_above_band": int(np.count_nonzero(above)),
        "fraction_samples_above_band": float(np.mean(above)) if above.size else None,
        "above_band_duration_s": active_duration_s,
        "above_band_fraction_of_selected_interval": (
            active_duration_s / interval_duration_s if interval_duration_s else None
        ),
        "all_excursion_count": int(len(event_table)),
        "minimum_pulse_duration_s": minimum_pulse_duration_s,
        "minimum_pulse_samples": minimum_pulse_samples,
        "qualified_pulse_count": int(len(qualified)),
        "qualified_pulse_duration_s": qualified_duration_s,
        "qualified_pulse_fraction_of_selected_interval": (
            qualified_duration_s / interval_duration_s if interval_duration_s else None
        ),
        "median_qualified_pulse_duration_s": (
            float(qualified["duration_s"].median()) if len(qualified) else None
        ),
        "p95_qualified_pulse_duration_s": (
            float(qualified["duration_s"].quantile(0.95)) if len(qualified) else None
        ),
        "maximum_qualified_pulse_duration_s": (
            float(qualified["duration_s"].max()) if len(qualified) else None
        ),
        "maximum_force_n": float(np.max(force)) if force.size else None,
    }
    return summary, event_table


def add_gap_aware_line(axis, time_s, values, *, gap_factor=3.0, **kwargs):
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(values)
    time_s = time_s[valid]
    values = values[valid]
    if not time_s.size:
        return None
    order = np.argsort(time_s)
    time_s = time_s[order]
    values = values[order]
    positive_dt = np.diff(time_s)
    positive_dt = positive_dt[positive_dt > 0]
    if positive_dt.size:
        gap_limit = gap_factor * float(np.median(positive_dt))
        break_indices = np.flatnonzero(np.diff(time_s) > gap_limit) + 1
        time_s = np.insert(time_s, break_indices, np.nan)
        values = np.insert(values, break_indices, np.nan)
    return axis.plot(time_s, values, **kwargs)[0]


def power_page_validation(power: pd.DataFrame, derived_torque: pd.DataFrame):
    p10 = power.loc[power["page_name"].eq("standard_power")].copy()
    p10 = p10.sort_values("t_unix_ns")
    p10["unwrapped_update_event_count"] = unwrap_modulo_counter(
        p10["p10_update_event_count"], 256
    )
    valid_torque = derived_torque.loc[derived_torque["derived_valid"]].copy()
    comparison = pd.merge(
        valid_torque[[
            "t_unix_ns",
            "unwrapped_update_event_count",
            "derived_cadence_rpm",
            "derived_crank_torque_nm",
            "derived_power_w",
        ]],
        p10[[
            "t_unix_ns",
            "unwrapped_update_event_count",
            "cadence_rpm",
            "p10_instantaneous_power_w",
        ]]
        .rename(columns={
            "t_unix_ns": "p10_t_unix_ns",
            "cadence_rpm": "p10_cadence_rpm_for_comparison",
            "p10_instantaneous_power_w": "p10_power_w_for_comparison",
        }),
        on="unwrapped_update_event_count",
        how="left",
        validate="one_to_one",
    )
    comparison["cross_page_abs_time_difference_s"] = (
        comparison["t_unix_ns"] - comparison["p10_t_unix_ns"]
    ).abs() / NS_PER_SECOND
    matched = (
        comparison["p10_t_unix_ns"].notna()
        & (comparison["cross_page_abs_time_difference_s"] <= 2.0)
    )
    cadence_difference = (
        comparison.loc[matched, "derived_cadence_rpm"]
        - comparison.loc[matched, "p10_cadence_rpm_for_comparison"]
    ).abs()
    power_difference = (
        comparison.loc[matched, "derived_power_w"]
        - comparison.loc[matched, "p10_power_w_for_comparison"]
    ).abs()
    time_difference_s = comparison.loc[matched, "cross_page_abs_time_difference_s"]

    p10_timing = BASE.timing_summary(
        p10["t_unix_ns"].to_numpy(),
        int(power["t_unix_ns"].min()),
        int(power["t_unix_ns"].max()),
    )
    p12_timing = BASE.timing_summary(
        derived_torque["t_unix_ns"].to_numpy(),
        int(power["t_unix_ns"].min()),
        int(power["t_unix_ns"].max()),
    )
    event_delta = pd.to_numeric(
        derived_torque["event_count_delta"], errors="coerce"
    ).dropna()
    power_p95 = (
        float(power_difference.quantile(0.95)) if len(power_difference) else None
    )
    upper_tail = (
        matched
        & (
            (
                comparison["derived_power_w"]
                - comparison["p10_power_w_for_comparison"]
            ).abs()
            >= power_p95
        )
        if power_p95 is not None else pd.Series(False, index=comparison.index)
    )
    upper_tail_zero_mismatch = (
        upper_tail
        & comparison["p10_power_w_for_comparison"].eq(0)
        & comparison["derived_power_w"].gt(0)
    )
    summary = {
        "standard_power_rows": int(len(p10)),
        "standard_torque_rows": int(len(derived_torque)),
        "standard_power_rate_hz": p10_timing["inferred_rate_hz"],
        "standard_torque_rate_hz": p12_timing["inferred_rate_hz"],
        "standard_power_max_gap_s": p10_timing["max_gap_s"],
        "standard_torque_max_gap_s": p12_timing["max_gap_s"],
        "standard_power_field_valid_fraction": float(
            pd.to_numeric(p10["p10_instantaneous_power_w"], errors="coerce").notna().mean()
        ),
        "standard_torque_derived_valid_rows": int(
            derived_torque["derived_valid"].sum()
        ),
        "standard_torque_derived_valid_fraction": float(
            derived_torque["derived_valid"].mean()
        ),
        "missing_update_events_inferred": int(
            np.maximum(event_delta.to_numpy(dtype=float) - 1.0, 0.0).sum()
        ),
        "cross_page_matching_rule": "Equal unwrapped update-event count and <=2 s time difference",
        "cross_page_matched_event_rows_within_2_s": int(matched.sum()),
        "cross_page_median_abs_time_difference_s": (
            float(time_difference_s.median()) if len(time_difference_s) else None
        ),
        "cross_page_p95_abs_time_difference_s": (
            float(time_difference_s.quantile(0.95)) if len(time_difference_s) else None
        ),
        "cross_page_median_abs_cadence_difference_rpm": (
            float(cadence_difference.median()) if len(cadence_difference) else None
        ),
        "cross_page_p95_abs_cadence_difference_rpm": (
            float(cadence_difference.quantile(0.95)) if len(cadence_difference) else None
        ),
        "cross_page_median_abs_power_difference_w": (
            float(power_difference.median()) if len(power_difference) else None
        ),
        "cross_page_p95_abs_power_difference_w": (
            power_p95
        ),
        "cross_page_power_upper_5_percent_rows": int(upper_tail.sum()),
        "cross_page_power_upper_5_percent_p10_zero_p12_positive_rows": int(
            upper_tail_zero_mismatch.sum()
        ),
    }
    return summary, comparison


def plot_filtered_inputs(
    frames: dict[str, pd.DataFrame],
    derived_torque: pd.DataFrame,
    brake_bands: dict[str, tuple[float, float]],
    steering_neutral: dict,
    pulse_events: pd.DataFrame,
    start_ns: int,
    output_base: Path,
):
    brake = frames["brake"]
    power = frames["power"]
    steering = frames["steering"]
    t0_s = start_ns / NS_PER_SECOND
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.2), sharex=True)

    steering_time = pd.to_numeric(steering["t_unix_ns"], errors="coerce").to_numpy(dtype=float)
    steering_angle = pd.to_numeric(
        steering.get("angle_relative_deg", steering["angle_deg"]), errors="coerce"
    ).to_numpy(dtype=float)
    add_gap_aware_line(
        axes[0],
        steering_time / NS_PER_SECOND - t0_s,
        steering_angle,
        color=COLORS["blue"],
        linewidth=0.9,
        label="Steering angle",
    )
    neutral_lower = steering_neutral["neutral_band_lower_relative_deg"]
    neutral_upper = steering_neutral["neutral_band_upper_relative_deg"]
    axes[0].axhspan(
        neutral_lower, neutral_upper, color=COLORS["blue"], alpha=0.14,
        zorder=0, label="Estimated neutral band",
    )
    axes[0].axhline(0.0, color=COLORS["grey"], linestyle="--", linewidth=0.7, zorder=1)
    axes[0].set_ylabel("Relative angle (deg)")
    axes[0].set_title("Steering angle relative to estimated neutral")
    axes[0].legend(loc="upper right")

    brake_time = pd.to_numeric(brake["t_unix_ns"], errors="coerce").to_numpy(dtype=float)
    left = pd.to_numeric(brake["left_force_n"], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(brake["right_force_n"], errors="coerce").to_numpy(dtype=float)
    left_band = brake_bands["left_force_n"]
    right_band = brake_bands["right_force_n"]
    axes[1].plot(
        brake_time / NS_PER_SECOND - t0_s,
        left,
        color=COLORS["blue"],
        linewidth=1.0,
        linestyle="-",
        label="Left brake",
        zorder=5,
    )
    axes[1].plot(
        brake_time / NS_PER_SECOND - t0_s,
        right,
        color=COLORS["orange"],
        linewidth=0.75,
        linestyle="--",
        label="Right brake",
        zorder=5,
    )
    # Draw the estimated zero-input bands after the force traces so that they
    # remain visible in front of both channels.  Pulse events are retained in
    # the tables, but are intentionally not overlaid on this paper figure.
    axes[1].axhspan(
        left_band[0], left_band[1], color=COLORS["blue"], alpha=0.18,
        label="Left zero-input band", zorder=10,
    )
    axes[1].axhspan(
        right_band[0], right_band[1], color=COLORS["orange"], alpha=0.18,
        label="Right zero-input band", zorder=10,
    )
    axes[1].axhline(left_band[1], color=COLORS["blue"], linewidth=0.8, linestyle=":", zorder=11)
    axes[1].axhline(right_band[1], color=COLORS["orange"], linewidth=0.8, linestyle=":", zorder=11)
    axes[1].set_ylabel("Force (N)")
    axes[1].set_title("Brake-force channels and estimated zero-input bands")
    axes[1].legend(loc="upper right", ncol=1)

    p10 = power.loc[power["page_name"].eq("standard_power")]
    p10_time = pd.to_numeric(p10["t_unix_ns"], errors="coerce").to_numpy(dtype=float)
    p10_power = pd.to_numeric(p10["p10_instantaneous_power_w"], errors="coerce").to_numpy(dtype=float)
    p10_cadence = pd.to_numeric(p10["cadence_rpm"], errors="coerce").to_numpy(dtype=float)
    add_gap_aware_line(
        axes[2], p10_time / NS_PER_SECOND - t0_s, p10_power,
        color=COLORS["blue"], linewidth=1.0, drawstyle="steps-post",
        label="Instantaneous power",
    )
    axes[2].set_ylabel("Power (W)")
    axes[2].set_title("Rally standard power page (0x10)")
    p10_cadence_axis = axes[2].twinx()
    add_gap_aware_line(
        p10_cadence_axis, p10_time / NS_PER_SECOND - t0_s, p10_cadence,
        color=COLORS["orange"], linewidth=0.9, linestyle="--", label="Cadence",
    )
    p10_cadence_axis.set_ylabel("Cadence (rpm)", color=COLORS["sky"])
    p10_cadence_axis.tick_params(axis="y", colors=COLORS["sky"])
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = p10_cadence_axis.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    for axis, label in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(axis, label)
        axis.grid(True)
    axes[-1].set_xlabel("Elapsed time from common-interval start (s)")
    fig.tight_layout()
    save_figure(fig, output_base)


def technical_validation_text(
    interval: dict,
    summary: pd.DataFrame,
    zero_bands: pd.DataFrame,
    pulse_summary: pd.DataFrame,
    power_summary: dict,
    steering_neutral: dict,
) -> str:
    brake_timing = summary.loc[summary["sensor"].eq("Brake force")].iloc[0]
    steering_timing = summary.loc[summary["sensor"].eq("Steering angle")].iloc[0]
    power_timing = summary.loc[summary["sensor"].eq("Power meter")].iloc[0]
    if str(interval.get("selection", "")).lower().startswith("explicit"):
        interval_description = "the explicitly supplied timestamp interval"
    else:
        interval_description = (
            f"the interval from the first to the last GNSS sample with ground speed above "
            f"{interval['threshold_mps']:.1f} m s$^{{-1}}$"
        )
    left_band = zero_bands.loc[zero_bands["force_column"].eq("left_force_n")].iloc[0]
    right_band = zero_bands.loc[zero_bands["force_column"].eq("right_force_n")].iloc[0]
    left_pulse = pulse_summary.loc[pulse_summary["force_column"].eq("left_force_n")].iloc[0]
    right_pulse = pulse_summary.loc[pulse_summary["force_column"].eq("right_force_n")].iloc[0]
    reference_count = int(left_band["reference_sample_count"])
    reference_sufficient = bool(left_band["reference_sample_count_sufficient"])
    if "static reference" in str(left_band["band_method"]).lower():
        band_basis_text = (
            f"{reference_count} samples were available in the pre-ride static reference interval; "
            "their complete observed force range was used as the no-input band."
        )
    elif str(left_band["band_method"]).startswith("Full observed"):
        band_basis_text = (
            f"{reference_count} stopped-speed samples were available; their full observed force "
            "range was used as a conservative no-input band."
        )
    elif reference_sufficient:
        band_basis_text = (
            f"{reference_count} associated samples met the minimum of "
            f"{int(left_band['minimum_reference_samples'])}; the empirical reference range was used."
        )
    else:
        band_basis_text = (
            f"Only {reference_count} associated samples met the steering condition, below the "
            f"minimum of {int(left_band['minimum_reference_samples'])}; the robust lower-force "
            "fallback was used."
        )
    return (
        f"The riding-input sensors were examined over {interval['duration_s']:.2f} s in "
        f"{interval_description}. The brake and cleaned steering "
        f"streams operated at approximately {brake_timing['inferred_rate_hz']:.2f} Hz "
        f"and {steering_timing['inferred_rate_hz']:.2f} Hz, with maximum gaps of "
        f"{brake_timing['max_gap_s']:.3f} s and {steering_timing['max_gap_s']:.3f} s. "
        "Steering samples at the configured limits were not removed automatically. The "
        "logger clamps out-of-range ADC readings to $\\pm45^\\circ$, so each limit run was "
        "checked using its ADC distance from the fitted calibration boundary and its entry "
        "and exit rates. Of "
        f"{interval['steering_limit_samples_in_source_interval']} limit-valued samples, "
        f"{interval['steering_plausible_boundary_samples_kept']} were retained as continuous "
        f"near-boundary measurements and {interval['steering_abnormal_clamped_samples_removed']} "
        f"were rejected as implausible clamped readings. The cleaned trace contains "
        f"{interval['steering_samples_after_filter']} samples. No steering value was "
        "interpolated; retained limit values remain boundary measurements.\n\n"
        f"The plotted steering signal is relative to a data-derived neutral angle of "
        f"{steering_neutral['neutral_angle_deg']:.2f} degrees. This reference is the median of "
        f"the dominant {steering_neutral['reference_sample_count']}-sample steering cluster; "
        f"the displayed neutral band is {steering_neutral['neutral_band_lower_relative_deg']:.2f} "
        f"to {steering_neutral['neutral_band_upper_relative_deg']:.2f} degrees. "
        "Candidate zero-input brake ranges were estimated from a pre-ride static reference interval "
        f"({interval.get('zero_band_reference_video_start_s', 0):g}--{interval.get('zero_band_reference_video_end_s', 0):g} video seconds) "
        "when available. The complete observed force range in that interval is used as a conservative "
        "no-input band; "
        "force above the upper boundary is marked as a candidate pulse. "
        f"{band_basis_text} The reported band uses the "
        f"{left_band['band_method'].lower()}. "
        f"The ranges were {left_band['zero_band_lower_n']:.3f}--{left_band['zero_band_upper_n']:.3f} N "
        f"for the left channel and {right_band['zero_band_lower_n']:.3f}--{right_band['zero_band_upper_n']:.3f} N "
        f"for the right channel. The band and pulse counts are preliminary rather "
        "than a calibrated brake classifier. Using the band upper limits and a minimum pulse "
        f"duration of {left_pulse['minimum_pulse_duration_s']:.1f} s, "
        f"{int(left_pulse['qualified_pulse_count'])} left-brake pulses "
        f"({left_pulse['qualified_pulse_duration_s']:.1f} s) and "
        f"{int(right_pulse['qualified_pulse_count'])} right-brake pulses "
        f"({right_pulse['qualified_pulse_duration_s']:.1f} s) were identified. The peak "
        f"recorded forces were {left_pulse['maximum_force_n']:.2f} N and "
        f"{right_pulse['maximum_force_n']:.2f} N.\n\n"
        f"The Rally power meter provided {power_summary['standard_power_rows']} standard "
        f"power messages and {power_summary['standard_torque_rows']} standard crank-torque "
        f"messages at {power_summary['standard_power_rate_hz']:.2f} Hz and "
        f"{power_summary['standard_torque_rate_hz']:.2f} Hz. "
        f"{power_summary['standard_torque_derived_valid_rows']} torque-page rows supported a "
        "valid difference calculation. For equal update-event counts paired within 2 s, the "
        "median absolute differences between the ANT+ pages were "
        f"{power_summary['cross_page_median_abs_cadence_difference_rpm']:.2f} rpm for cadence "
        f"and {power_summary['cross_page_median_abs_power_difference_w']:.2f} W for power; "
        "the corresponding 95th percentiles were "
        f"{power_summary['cross_page_p95_abs_cadence_difference_rpm']:.2f} rpm and "
        f"{power_summary['cross_page_p95_abs_power_difference_w']:.2f} W. These results "
        "describe timing, coverage, and internal agreement, not absolute accuracy, because "
        "no independent steering-angle, torque, or brake-force reference was available."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--gnss-bag", default="")
    parser.add_argument(
        "--steering-file",
        default="",
        help="Optional cleaned steering CSV with the original six columns",
    )
    parser.add_argument(
        "--steering-quality-audit",
        default="",
        help="Optional audit CSV created by p9_clean_and_crop.py",
    )
    parser.add_argument("--brake-file", default="")
    parser.add_argument("--power-file", default="")
    parser.add_argument("--imu-file", default="")
    parser.add_argument("--wheel-file", default="")
    parser.add_argument("--speed-threshold-mps", type=float, default=0.1)
    parser.add_argument("--stop-reference-speed-threshold-mps", type=float, default=0.1)
    parser.add_argument("--start-ns", type=int, default=None)
    parser.add_argument("--end-ns", type=int, default=None)
    parser.add_argument("--zero-band-steering-max-abs-deg", type=float, default=2.0)
    parser.add_argument("--association-tolerance-s", type=float, default=0.05)
    parser.add_argument("--minimum-zero-band-samples", type=int, default=30)
    parser.add_argument(
        "--zero-reference-video-start-s", type=float, default=None,
        help="Video start (s) of the static brake reference interval",
    )
    parser.add_argument(
        "--zero-reference-video-end-s", type=float, default=None,
        help="Video end (s) of the static brake reference interval",
    )
    parser.add_argument("--minimum-brake-pulse-duration-s", type=float, default=0.3)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow updating an existing derived output directory",
    )
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory does not exist: {session_dir}")
    if (output.exists() or output.is_symlink()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    bag_dir = Path(args.gnss_bag).resolve() if args.gnss_bag else None
    if bag_dir is None:
        matches = sorted(session_dir.glob("rosbag2_*/metadata.yaml"))
        if len(matches) == 1:
            bag_dir = matches[0].parent
        else:
            db_matches = sorted(session_dir.glob("rosbag2_*/*.db3"))
            if len(db_matches) != 1:
                raise SystemExit("Expected one rosbag2 directory or db3 file; pass --gnss-bag explicitly")
            bag_dir = db_matches[0]
    if not bag_dir.exists():
        raise SystemExit(f"GNSS bag does not exist: {bag_dir}")

    paths = {name: session_dir / filename for name, filename in BASE.INPUT_FILES.items()}
    overrides = {
        "steering": args.steering_file,
        "brake": args.brake_file,
        "power": args.power_file,
        "imu": args.imu_file,
        "wheel": args.wheel_file,
    }
    for name, value in overrides.items():
        if value:
            paths[name] = Path(value).resolve()
    # P8 and later sessions use timestamped files with a different capture time.
    # Resolve absent defaults by sensor prefix, without touching the source files.
    prefixes = {
        "brake": "brake_sensors_force_*.csv",
        "power": "rally_payload_decoded_*.csv",
        "steering": "steering_angle_*.csv",
        "imu": "imu_*.csv",
        "wheel": "speed_decoded_*.csv",
    }
    for name, pattern in prefixes.items():
        if name not in paths or not paths[name].is_file():
            matches = sorted(session_dir.glob(pattern))
            if matches:
                paths[name] = matches[0]
    required = ["brake", "power", "steering"]
    missing = [name for name in required if not paths.get(name, Path()).is_file()]
    if missing:
        raise SystemExit("Missing input CSV files: " + ", ".join(missing))
    raw_frames = {name: BASE.read_input(path) for name, path in paths.items()}
    if (args.zero_reference_video_start_s is None) != (args.zero_reference_video_end_s is None):
        raise SystemExit("Both zero-reference video boundaries are required together")
    static_reference = None
    static_reference_description = None
    static_reference_start_ns = None
    static_reference_end_ns = None
    static_reference_time_mapping = None
    if args.zero_reference_video_start_s is not None:
        timestamp_matches = sorted(session_dir.glob("camera_*/timestamps.csv"))
        video_matches = sorted(session_dir.glob("camera_*/video_mjpg.avi"))
        if len(timestamp_matches) != 1 or len(video_matches) != 1:
            raise SystemExit("Expected one camera video and timestamps.csv for the static reference")
        (
            static_reference_start_ns,
            static_reference_end_ns,
            static_reference_time_mapping,
        ) = VIDEO_TIME.map_playback_interval(
            video_matches[0], timestamp_matches[0],
            args.zero_reference_video_start_s,
            args.zero_reference_video_end_s,
        )
        static_reference = crop(
            raw_frames["brake"], static_reference_start_ns, static_reference_end_ns
        )
        static_reference_description = (
            f"Video static reference, {args.zero_reference_video_start_s:g}--"
            f"{args.zero_reference_video_end_s:g} s"
        )
    speed = read_gnss_speed(bag_dir)
    if (args.start_ns is None) != (args.end_ns is None):
        raise SystemExit("Both --start-ns and --end-ns are required together")
    if args.start_ns is not None:
        start_ns, end_ns = int(args.start_ns), int(args.end_ns)
        positive_speed = speed[
            speed["t_unix_ns"].between(start_ns, end_ns, inclusive="both")
            & (speed["ground_speed_mps"] > args.speed_threshold_mps)
        ].copy()
        interval_selection = "Explicit timestamp interval"
    else:
        start_ns, end_ns, positive_speed = select_gnss_positive_interval(
            speed, args.speed_threshold_mps
        )
        interval_selection = "First to last GNSS speed sample above threshold"
    frames = {name: crop(frame, start_ns, end_ns) for name, frame in raw_frames.items()}

    steering_before = frames["steering"]
    steering_angle = pd.to_numeric(steering_before["angle_deg"], errors="coerce")
    steering_filtered = steering_before.loc[
        pd.to_numeric(steering_before["ok"], errors="coerce").eq(1)
        & steering_angle.notna()
    ].copy().reset_index(drop=True)
    frames["steering"] = steering_filtered
    frames_for_plot = dict(frames)
    frames_for_plot["steering"] = steering_filtered
    steering_neutral = estimate_steering_neutral(steering_filtered)
    steering_filtered["angle_relative_deg"] = (
        pd.to_numeric(steering_filtered["angle_deg"], errors="coerce")
        - steering_neutral["neutral_angle_deg"]
    )
    frames["steering"] = steering_filtered
    frames_for_plot["steering"] = steering_filtered

    quality_audit_path = (
        Path(args.steering_quality_audit).resolve()
        if args.steering_quality_audit else None
    )
    if quality_audit_path is not None:
        quality_audit = crop(
            pd.read_csv(quality_audit_path, low_memory=False), start_ns, end_ns
        )
        limit_samples_in_source = int(
            quality_audit["is_limit_value"].astype(bool).sum()
        )
        plausible_boundary_kept = int(
            quality_audit["quality_classification"]
            .eq("plausible_boundary_measurement").sum()
        )
        abnormal_clamped_removed = int(
            quality_audit["quality_classification"]
            .eq("abnormal_clamped_limit").sum()
        )
    else:
        quality_audit = None
        limit_samples_in_source = int(
            steering_angle.abs().eq(STEERING_LIMIT_DEG).sum()
        )
        plausible_boundary_kept = limit_samples_in_source
        abnormal_clamped_removed = 0

    associated = associate_brake_and_steering(
        frames["brake"], steering_filtered, args.association_tolerance_s
    )
    bands, zero_band_table, reference_mask = estimate_zero_brake_bands(
        associated,
        args.zero_band_steering_max_abs_deg,
        args.minimum_zero_band_samples,
        speed=speed,
        speed_threshold_mps=args.stop_reference_speed_threshold_mps,
        reference_brake=static_reference,
        reference_description=static_reference_description,
    )
    interval_duration_s = (end_ns - start_ns) / NS_PER_SECOND
    pulse_summaries = []
    pulse_event_tables = []
    for column in ["left_force_n", "right_force_n"]:
        summary_row, event_table = brake_pulse_statistics(
            frames["brake"],
            column,
            bands[column][1],
            interval_duration_s,
            args.minimum_brake_pulse_duration_s,
        )
        pulse_summaries.append(summary_row)
        pulse_event_tables.append(event_table)
    pulse_summary = pd.DataFrame(pulse_summaries)
    pulse_events = pd.concat(pulse_event_tables, ignore_index=True)

    derived_torque = derive_standard_torque_page(frames["power"])
    power_summary, power_comparison = power_page_validation(
        frames["power"], derived_torque
    )

    summary = BASE.build_summary(frames, start_ns, end_ns)
    # BASE.build_summary keeps the original P9 filenames for historical
    # compatibility.  Replace them with the actual resolved files so reports
    # for later recordings identify their real inputs.
    source_names = {
        "Brake force": paths["brake"].name,
        "Power meter": paths["power"].name,
        "Steering angle": paths["steering"].name,
    }
    summary["source_file"] = summary["sensor"].map(source_names).fillna(summary["source_file"])
    summary.insert(0, "session_id", args.session_id or session_dir.name)
    summary.insert(1, "gnss_positive_start_ns", start_ns)
    summary.insert(2, "gnss_positive_end_ns", end_ns)

    interval = {
        "selection": interval_selection,
        "topic": "/ubx_nav_vel_ned",
        "gnss_timestamp_source": "rosbag record time",
        "threshold_mps": args.speed_threshold_mps,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_s": interval_duration_s,
        "positive_speed_samples": len(positive_speed),
        "minimum_positive_speed_mps": float(positive_speed["ground_speed_mps"].min()),
        "median_positive_speed_mps": float(positive_speed["ground_speed_mps"].median()),
        "maximum_positive_speed_mps": float(positive_speed["ground_speed_mps"].max()),
        "steering_limit_deg": STEERING_LIMIT_DEG,
        "steering_filter_method": (
            "ADC boundary distance and entry/exit angular-rate continuity"
            if quality_audit is not None else "No automatic limit-value removal"
        ),
        "steering_samples_before_filter": int(
            len(quality_audit) if quality_audit is not None else len(steering_before)
        ),
        "steering_limit_samples_in_source_interval": limit_samples_in_source,
        "steering_plausible_boundary_samples_kept": plausible_boundary_kept,
        "steering_abnormal_clamped_samples_removed": abnormal_clamped_removed,
        "steering_samples_after_filter": int(len(steering_filtered)),
        "brake_steering_association_tolerance_s": args.association_tolerance_s,
        "brake_samples_with_matched_steering": int(associated["angle_deg"].notna().sum()),
        "zero_band_steering_max_abs_deg": args.zero_band_steering_max_abs_deg,
        "zero_band_reference_speed_threshold_mps": args.stop_reference_speed_threshold_mps,
        "zero_band_reference_method": (
            "Full observed force range from the specified video static interval; "
            "stopped-speed fallback or steering-near-zero fallback if unavailable"
            if static_reference is not None else
            "Full observed force range from stopped GNSS speed samples; steering-near-zero fallback if insufficient"
        ),
        "zero_band_reference_video_start_s": args.zero_reference_video_start_s,
        "zero_band_reference_video_end_s": args.zero_reference_video_end_s,
        "zero_band_reference_start_ns": static_reference_start_ns,
        "zero_band_reference_end_ns": static_reference_end_ns,
        "zero_band_reference_video_time_mapping": static_reference_time_mapping,
        "zero_band_reference_static_rows": int(len(static_reference)) if static_reference is not None else 0,
        "zero_band_reference_samples": int(
            zero_band_table["speed_reference_sample_count"].max()
            if zero_band_table["speed_reference_sample_count"].notna().any()
            and zero_band_table["speed_reference_sample_count"].max() > 0
            else reference_mask.sum()
        ),
        "zero_band_reference_minimum_required": args.minimum_zero_band_samples,
        "zero_band_reference_sample_count_sufficient": bool(
            max(
                int(zero_band_table["speed_reference_sample_count"].max())
                if zero_band_table["speed_reference_sample_count"].notna().any()
                else 0,
                int(reference_mask.sum()),
            ) >= args.minimum_zero_band_samples
        ),
        "minimum_brake_pulse_duration_s": args.minimum_brake_pulse_duration_s,
        "steering_neutral_method": steering_neutral["method"],
        "steering_neutral_angle_deg": steering_neutral["neutral_angle_deg"],
        "steering_neutral_band_lower_relative_deg": steering_neutral["neutral_band_lower_relative_deg"],
        "steering_neutral_band_upper_relative_deg": steering_neutral["neutral_band_upper_relative_deg"],
        "steering_neutral_reference_sample_count": steering_neutral["reference_sample_count"],
    }

    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    summary.to_csv(tables / "riding_input_sensor_summary.csv", index=False)
    signal_stats = pd.concat([
        BASE.signal_statistics(steering_filtered, ["angle_relative_deg"]),
        BASE.signal_statistics(frames["brake"], ["left_force_n", "right_force_n"]),
        BASE.signal_statistics(
            frames["power"], ["cadence_rpm", "p10_instantaneous_power_w"]
        ),
        BASE.signal_statistics(
            derived_torque,
            ["derived_cadence_rpm", "derived_crank_torque_nm", "derived_power_w"],
        ),
    ], ignore_index=True)
    signal_stats.to_csv(tables / "riding_input_signal_statistics.csv", index=False)
    positive_speed.to_csv(tables / "gnss_positive_speed_samples.csv", index=False)
    associated.to_csv(tables / "brake_steering_association.csv", index=False)
    zero_band_table.to_csv(tables / "brake_zero_input_band.csv", index=False)
    pulse_summary.to_csv(tables / "brake_pulse_summary.csv", index=False)
    pulse_events.to_csv(tables / "brake_pulse_events.csv", index=False)
    derived_torque.to_csv(tables / "powermeter_standard_torque_derived.csv", index=False)
    power_comparison.to_csv(tables / "powermeter_cross_page_comparison.csv", index=False)
    pd.DataFrame([power_summary]).to_csv(
        tables / "powermeter_page_validation.csv", index=False
    )
    page_counts = (
        frames["power"].groupby(["page", "page_hex", "page_name"], dropna=False)
        .size().reset_index(name="n_rows")
    )
    page_counts.to_csv(tables / "powermeter_page_counts.csv", index=False)
    BASE.write_json(tables / "selection_and_filtering.json", interval)
    BASE.write_json(tables / "steering_neutral_reference.json", steering_neutral)

    plot_filtered_inputs(
        frames_for_plot,
        derived_torque,
        bands,
        steering_neutral,
        pulse_events,
        start_ns,
        figures / "riding_input_sensor_validation_gnss_positive",
    )

    interval_label = (
        "the explicitly supplied timestamp interval"
        if str(interval.get("selection", "")).lower().startswith("explicit")
        else "the common GNSS-speed-selected interval"
    )
    caption = (
        f"Riding-input sensor validation during {interval_label}. "
        "(a) Steering angle after rule-based ADC and continuity filtering, shown relative to "
        f"the estimated neutral position ({steering_neutral['neutral_angle_deg']:.2f} degrees). "
        f"The shaded neutral band is {steering_neutral['neutral_band_lower_relative_deg']:.2f} to "
        f"{steering_neutral['neutral_band_upper_relative_deg']:.2f} degrees. (b) Left and right "
        "brake-force channels. The shaded horizontal bands are the full observed force ranges "
        "during the pre-ride static reference interval and are drawn in front of the curves. Pulse counts are "
        "reported in the tables but are not overlaid in this figure. (c) Instantaneous "
        "power held until the next message and cadence from the Rally standard power page. "
        "Crank torque is retained in the tables but is not plotted."
    )
    (output / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    (output / "technical_validation_text.txt").write_text(
        technical_validation_text(
            interval, summary, zero_band_table, pulse_summary, power_summary,
            steering_neutral,
        ) + "\n",
        encoding="utf-8",
    )
    if "static reference" in str(zero_band_table.iloc[0]["band_method"]).lower():
        reference_note = (
            f"{int(zero_band_table.iloc[0]['static_reference_sample_count'])} samples from the "
            f"video static interval {args.zero_reference_video_start_s:g}--{args.zero_reference_video_end_s:g} s "
            "were used to estimate the full observed zero-input force range."
        )
    elif str(zero_band_table.iloc[0]["band_method"]).startswith("Full observed"):
        reference_note = (
            f"{int(zero_band_table.iloc[0]['speed_reference_sample_count'])} stopped-speed samples "
            "were used to estimate the full observed zero-input force range."
        )
    elif bool(zero_band_table.iloc[0]["reference_sample_count_sufficient"]):
        reference_note = f"{int(reference_mask.sum())} steering-near-zero samples were sufficient for the empirical range."
    else:
        reference_note = f"Only {int(reference_mask.sum())} steering-near-zero samples were available; the robust fallback was used."
    readme = (
        "This report uses the raw brake, steering, Rally power-meter, and GNSS files from the "
        f"{args.session_id or session_dir.name} session. It does not modify the source data. The riding interval is "
        + ("the explicitly supplied timestamp interval. " if str(interval.get("selection", "")).lower().startswith("explicit") else
           "selected from the first to the last /ubx_nav_vel_ned ground-speed sample above the configured threshold. ")
        + "Steering limit values are retained only when the separate quality audit "
        "classifies them as continuous near-boundary measurements; no value is interpolated. "
        f"The plotted steering signal is relative to the estimated neutral position ({steering_neutral['neutral_angle_deg']:.2f} degrees). "
        f"Brake zero-input bands use the pre-ride static reference when supplied. {reference_note} "
        "The band and pulse results remain preliminary and are not a calibrated brake/no-brake classifier. Power is "
        "plotted as a held-value step line; crank-torque checks remain in the tables.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    bag_files = (
        [bag_dir]
        if bag_dir.is_file()
        else sorted(item for item in bag_dir.iterdir() if item.is_file())
    )
    manifest = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id or session_dir.name,
        "session_dir": str(session_dir),
        "gnss_bag": str(bag_dir),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            **{
                name: {"path": str(path), "sha256": BASE.sha256_file(path)}
                for name, path in paths.items()
            },
            "gnss_bag_files": [
                {"path": str(path), "sha256": BASE.sha256_file(path)}
                for path in bag_files
            ],
            "steering_quality_audit": (
                {
                    "path": str(quality_audit_path),
                    "sha256": BASE.sha256_file(quality_audit_path),
                }
                if quality_audit_path is not None else None
            ),
        },
        "scripts": {
            "main": {"path": str(Path(__file__).resolve()), "sha256": BASE.sha256_file(Path(__file__).resolve())},
            "base": {"path": str(BASE_PATH.resolve()), "sha256": BASE.sha256_file(BASE_PATH.resolve())},
        },
        "selection_and_filtering": interval,
        "power_page_validation": power_summary,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    BASE.write_json(output / "run_manifest.json", manifest)
    checksum_lines = []
    for path in sorted(
        item for item in output.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    ):
        checksum_lines.append(f"{BASE.sha256_file(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote GNSS-positive riding-input validation to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
