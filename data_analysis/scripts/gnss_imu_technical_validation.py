#!/usr/bin/env python3
"""Generate GNSS/IMU technical-validation tables and figures for one session."""

import argparse
import calendar
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_style import (  # noqa: E402
    COLORS,
    RTK_COLORS,
    RTK_LABELS,
    apply_paper_style,
    panel_label,
    save_figure,
)


NS_PER_SECOND = 1_000_000_000
CONVENTIONAL_GRAVITY = 9.80665
STATE_ORDER = [0, 1, 2]


def wgs84_normal_gravity(latitude_deg, ellipsoidal_height_m):
    """Approximate normal gravity at latitude and ellipsoidal height."""
    latitude_rad = math.radians(float(latitude_deg))
    sin_squared = math.sin(latitude_rad) ** 2
    semi_major_axis_m = 6378137.0
    flattening = 1.0 / 298.257223563
    geodetic_constant = 0.00344978650684
    surface_gravity = (
        9.7803253359
        * (1.0 + 0.00193185265241 * sin_squared)
        / math.sqrt(1.0 - 0.00669437999013 * sin_squared)
    )
    height_m = float(ellipsoidal_height_m)
    height_factor = (
        1.0
        - (2.0 / semi_major_axis_m)
        * (
            1.0
            + flattening
            + geodetic_constant
            - 2.0 * flattening * sin_squared
        )
        * height_m
        + 3.0 * height_m ** 2 / semi_major_axis_m ** 2
    )
    return surface_gravity * height_factor


def stamp_ns(message):
    """Return a ROS header timestamp as integer nanoseconds."""
    return (
        int(message.header.stamp.sec) * NS_PER_SECOND
        + int(message.header.stamp.nanosec)
    )


def pvt_utc_ns(message):
    """Return receiver UTC from UBX-NAV-PVT, or None when it is invalid."""
    if not (message.valid_date and message.valid_time and message.fully_resolved):
        return None
    try:
        whole_seconds = calendar.timegm((
            int(message.year), int(message.month), int(message.day),
            int(message.hour), int(message.min), int(message.sec),
        ))
    except (OverflowError, ValueError):
        return None
    return whole_seconds * NS_PER_SECOND + int(message.nano)


def sha256_path(path):
    """Hash one file or a directory tree without modifying it."""
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


def json_safe(value):
    """Convert NumPy values and non-finite floats to strict JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def find_single(directory, pattern, label):
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {label} matching {pattern!r}; found {len(matches)}"
        )
    return matches[0]


def load_bag_metadata(bag_dir):
    path = Path(bag_dir) / "metadata.yaml"
    info = yaml.safe_load(path.read_text(encoding="utf-8"))[
        "rosbag2_bagfile_information"
    ]
    start_ns = int(info["starting_time"]["nanoseconds_since_epoch"])
    duration_ns = int(info["duration"]["nanoseconds"])
    topic_counts = {
        item["topic_metadata"]["name"]: int(item["message_count"])
        for item in info["topics_with_message_count"]
    }
    return {
        "path": path,
        "start_ns": start_ns,
        "end_ns": start_ns + duration_ns,
        "duration_s": duration_ns / NS_PER_SECOND,
        "message_count": int(info["message_count"]),
        "topic_counts": topic_counts,
        "storage_identifier": info["storage_identifier"],
    }


def read_bag_topics(bag_dir, storage_id):
    """Read only the GNSS topics needed by this report."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    wanted = {
        "/fix",
        "/ubx_nav_pvt",
        "/ubx_nav_hp_pos_llh",
        "/ubx_nav_vel_ned",
        "/ubx_rxm_rawx",
        "/ubx_nav_sat",
    }
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    type_map = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    message_types = {
        topic: get_message(type_map[topic])
        for topic in wanted
        if topic in type_map
    }

    pvt_rows = []
    hp_rows = []
    vel_rows = []
    fix_rows = []
    rawx_rows = []
    rawx_cno = []
    rawx_cno_t_ns = []
    rawx_cno_record_ns = []
    sat_rows = []

    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        t_ns = stamp_ns(message)

        if topic == "/ubx_nav_pvt":
            pvt_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "gnss_utc_ns": pvt_utc_ns(message),
                "itow_ms": int(message.itow),
                "rtk_state": int(message.carr_soln.status),
                "gnss_fix_ok": bool(message.gnss_fix_ok),
                "invalid_llh": bool(message.invalid_llh),
                "latitude_deg": float(message.lat) * 1.0e-7,
                "longitude_deg": float(message.lon) * 1.0e-7,
                "height_m": float(message.height) * 1.0e-3,
                "hacc_m": float(message.h_acc) * 1.0e-3,
                "vacc_m": float(message.v_acc) * 1.0e-3,
                "ground_speed_mps": float(message.g_speed) * 1.0e-3,
                "course_deg": float(message.head_mot) * 1.0e-5,
                "course_accuracy_deg": float(message.head_acc) * 1.0e-5,
                "num_sv": int(message.num_sv),
                "pdop": float(message.p_dop) * 0.01,
            })
        elif topic == "/ubx_nav_hp_pos_llh":
            valid = not any([
                message.invalid_lon,
                message.invalid_lat,
                message.invalid_height,
                message.invalid_lon_hp,
                message.invalid_lat_hp,
                message.invalid_height_hp,
            ])
            hp_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "itow_ms": int(message.itow),
                "hp_valid": bool(valid),
                "latitude_deg": (
                    float(message.lat) * 1.0e-7
                    + float(message.lat_hp) * 1.0e-9
                ),
                "longitude_deg": (
                    float(message.lon) * 1.0e-7
                    + float(message.lon_hp) * 1.0e-9
                ),
                "height_m": (
                    float(message.height) + 0.1 * float(message.height_hp)
                ) * 1.0e-3,
                "hacc_m": float(message.h_acc) * 1.0e-4,
                "vacc_m": float(message.v_acc) * 1.0e-4,
            })
        elif topic == "/ubx_nav_vel_ned":
            vel_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "ground_speed_mps": float(message.g_speed) * 0.01,
                "course_deg": float(message.heading) * 1.0e-5,
                "course_accuracy_deg": float(message.c_acc) * 1.0e-5,
            })
        elif topic == "/fix":
            covariance = list(message.position_covariance)
            fix_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "status": int(message.status.status),
                "latitude_deg": float(message.latitude),
                "longitude_deg": float(message.longitude),
                "altitude_m": float(message.altitude),
                "covariance_finite": bool(
                    len(covariance) == 9
                    and all(math.isfinite(float(item)) for item in covariance)
                ),
            })
        elif topic == "/ubx_rxm_rawx":
            observations = list(message.rawx_data)
            pr_valid = sum(bool(item.trk_stat.pr_valid) for item in observations)
            cp_valid = sum(bool(item.trk_stat.cp_valid) for item in observations)
            cno_values = [float(item.c_no) for item in observations]
            rawx_cno.extend(cno_values)
            rawx_cno_t_ns.extend([t_ns] * len(cno_values))
            rawx_cno_record_ns.extend([int(record_ns)] * len(cno_values))
            rawx_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "num_measurements": int(message.num_meas),
                "decoded_measurements": len(observations),
                "pseudorange_valid": int(pr_valid),
                "carrier_phase_valid": int(cp_valid),
                "cno_median_dbhz": (
                    float(np.median(cno_values)) if cno_values else np.nan
                ),
            })
        elif topic == "/ubx_nav_sat":
            sat_rows.append({
                "t_ns": t_ns,
                "record_ns": int(record_ns),
                "num_svs_visible": int(message.num_svs),
            })

    def frame(rows):
        result = pd.DataFrame(rows)
        if len(result):
            result = result.sort_values("t_ns").reset_index(drop=True)
            result["t_ns"] = result["t_ns"].astype("int64")
        return result

    return {
        "topic_types": type_map,
        "pvt": frame(pvt_rows),
        "hp": frame(hp_rows),
        "vel": frame(vel_rows),
        "fix": frame(fix_rows),
        "rawx": frame(rawx_rows),
        "rawx_cno": np.asarray(rawx_cno, dtype=float),
        "rawx_cno_t_ns": np.asarray(rawx_cno_t_ns, dtype=np.int64),
        "rawx_cno_record_ns": np.asarray(rawx_cno_record_ns, dtype=np.int64),
        "sat": frame(sat_rows),
    }


def apply_rosbag_time_source(data, source="header"):
    """Select ROS header time or rosbag receive time for analysis.

    When record time is selected, the original header timestamp is preserved
    in ``header_ns`` and ``t_ns`` becomes the rosbag receive timestamp.
    """
    if source not in {"header", "record"}:
        raise ValueError(f"Unsupported rosbag time source: {source}")
    if source == "header":
        return data
    for name, frame in data.items():
        if not isinstance(frame, pd.DataFrame) or "record_ns" not in frame.columns:
            continue
        frame["header_ns"] = frame["t_ns"].astype("int64")
        frame["t_ns"] = frame["record_ns"].astype("int64")
        data[name] = frame.sort_values("t_ns").reset_index(drop=True)
    if "rawx_cno_record_ns" in data:
        data["rawx_cno_header_ns"] = data.get(
            "rawx_cno_t_ns", np.array([], dtype=np.int64)
        )
        data["rawx_cno_t_ns"] = data["rawx_cno_record_ns"]
    return data


def load_imu(path, start_ns, end_ns):
    imu = pd.read_csv(path, low_memory=False)
    required = {
        "t_unix_ns",
        "dtype",
        "crc8_ok",
        "crc16_ok",
        "end_ok",
    }
    missing = sorted(required.difference(imu.columns))
    if missing:
        raise RuntimeError(f"IMU CSV missing columns: {missing}")
    imu["t_unix_ns"] = pd.to_numeric(
        imu["t_unix_ns"], errors="coerce"
    ).astype("Int64")
    imu["dtype"] = pd.to_numeric(imu["dtype"], errors="coerce").astype("Int64")
    imu = imu.dropna(subset=["t_unix_ns", "dtype"]).copy()
    imu["t_unix_ns"] = imu["t_unix_ns"].astype("int64")
    imu["dtype"] = imu["dtype"].astype("int64")

    clock_audit = {"timestamp_field": "t_unix_ns", "full_file": {}}
    for dtype in [64, 65]:
        sequence = imu.loc[
            imu["dtype"] == dtype, "t_unix_ns"
        ].to_numpy(dtype=np.int64)
        differences = np.diff(sequence)
        rollback_indices = np.flatnonzero(differences < 0)
        events = []
        for index in rollback_indices:
            events.append({
                "previous_ns": int(sequence[index]),
                "current_ns": int(sequence[index + 1]),
                "change_s": float(differences[index] / NS_PER_SECOND),
                "before_bag_start": bool(sequence[index + 1] < int(start_ns)),
            })
        clock_audit["full_file"][str(dtype)] = {
            "n_rows": int(len(sequence)),
            "start_ns": int(sequence.min()) if len(sequence) else None,
            "end_ns": int(sequence.max()) if len(sequence) else None,
            "rollback_count": int(len(events)),
            "rollback_events": events,
        }

    imu = imu[
        (imu["t_unix_ns"] >= int(start_ns))
        & (imu["t_unix_ns"] <= int(end_ns))
    ].copy()

    checks = ["crc8_ok", "crc16_ok", "end_ok"]
    for column in checks:
        imu[column] = pd.to_numeric(imu[column], errors="coerce")
    imu["framing_valid"] = imu[checks].eq(1).all(axis=1)

    streams = {}
    requirements = {
        64: ["gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"],
        65: ["rollspeed", "pitchspeed", "headingspeed", "roll", "pitch", "heading"],
    }
    for dtype, columns in requirements.items():
        part = imu[imu["dtype"] == dtype].copy()
        for column in columns:
            part[column] = pd.to_numeric(part[column], errors="coerce")
        part["numeric_valid"] = part[columns].notna().all(axis=1)
        part["valid"] = part["framing_valid"] & part["numeric_valid"]
        part = part.sort_values("t_unix_ns").reset_index(drop=True)
        streams[dtype] = part
    clock_audit["bag_overlap"] = {
        str(dtype): {
            "n_rows": int(len(frame)),
            "rollback_count": int(np.sum(
                np.diff(frame["t_unix_ns"].to_numpy(dtype=np.int64)) < 0
            )),
        }
        for dtype, frame in streams.items()
    }
    return streams, clock_audit


def build_timestamp_audit(pvt, imu_clock_audit):
    """Summarize clock relationships without treating them as calibrated UTC."""
    header_column = "header_ns" if "header_ns" in pvt.columns else "t_ns"
    audit = {
        "gnss_header_timestamp_field": "header.stamp",
        "gnss_record_timestamp_field": "rosbag record_ns",
        "imu_timestamp_field": "t_unix_ns",
        "imu": imu_clock_audit,
        "interpretation": [
            "GNSS header stamps and IMU t_unix_ns are host-side timestamps.",
            "Receiver UTC is taken from fully resolved UBX-NAV-PVT date/time fields.",
            "Cross-correlation lag is an effective signal alignment, not a direct clock calibration.",
        ],
    }
    if not len(pvt):
        audit["gnss"] = {"available": False}
        return audit

    valid_utc = pvt.dropna(subset=["gnss_utc_ns"]).copy()
    header_minus_utc = (
        valid_utc[header_column].to_numpy(dtype=np.int64)
        - valid_utc["gnss_utc_ns"].to_numpy(dtype=np.int64)
    ) / NS_PER_SECOND
    record_minus_header = (
        pvt["record_ns"].to_numpy(dtype=np.int64)
        - pvt[header_column].to_numpy(dtype=np.int64)
    ) / NS_PER_SECOND
    audit["gnss"] = {
        "available": bool(len(valid_utc)),
        "fully_resolved_utc_samples": int(len(valid_utc)),
        "header_minus_receiver_utc_median_s": (
            float(np.median(header_minus_utc)) if len(header_minus_utc) else None
        ),
        "header_minus_receiver_utc_p05_s": (
            float(np.quantile(header_minus_utc, 0.05)) if len(header_minus_utc) else None
        ),
        "header_minus_receiver_utc_p95_s": (
            float(np.quantile(header_minus_utc, 0.95)) if len(header_minus_utc) else None
        ),
        "bag_record_minus_header_median_ms": float(
            np.median(record_minus_header) * 1000.0
        ),
        "bag_record_minus_header_p95_ms": float(
            np.quantile(record_minus_header, 0.95) * 1000.0
        ),
    }
    return audit


def stream_health(name, source, timestamp_field, timestamps, valid):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    valid = valid[order]
    unique = np.insert(np.diff(timestamps) > 0, 0, True) if len(timestamps) else []
    timestamps = timestamps[unique]
    valid = valid[unique]
    dt = np.diff(timestamps).astype(float) / NS_PER_SECOND
    duration = (
        (timestamps[-1] - timestamps[0]) / NS_PER_SECOND
        if len(timestamps) >= 2 else 0.0
    )
    median_dt = float(np.median(dt)) if len(dt) else np.nan
    gap_threshold = 2.5 * median_dt if math.isfinite(median_dt) else np.nan
    return {
        "stream": name,
        "source": source,
        "timestamp_field": timestamp_field,
        "n_frames": int(len(timestamps)),
        "n_valid_frames": int(valid.sum()),
        "valid_frame_fraction": float(valid.mean()) if len(valid) else np.nan,
        "start_ns": int(timestamps[0]) if len(timestamps) else None,
        "end_ns": int(timestamps[-1]) if len(timestamps) else None,
        "duration_s": float(duration),
        "effective_rate_hz": (
            float((len(timestamps) - 1) / duration)
            if len(timestamps) >= 2 and duration > 0 else np.nan
        ),
        "median_interval_ms": median_dt * 1000.0,
        "p99_interval_ms": (
            float(np.quantile(dt, 0.99) * 1000.0) if len(dt) else np.nan
        ),
        "longest_gap_ms": float(dt.max() * 1000.0) if len(dt) else np.nan,
        "gap_threshold_ms": gap_threshold * 1000.0,
        "gap_count": int(np.sum(dt > gap_threshold)) if len(dt) else 0,
    }


def build_stream_table(data, imu_streams, gnss_timestamp_field="header.stamp"):
    rows = []
    definitions = [
        (
            "NavSatFix",
            "/fix",
            data["fix"],
            (
                (data["fix"]["status"] >= 0)
                & np.isfinite(data["fix"]["latitude_deg"])
                & np.isfinite(data["fix"]["longitude_deg"])
            ) if len(data["fix"]) else [],
        ),
        (
            "GNSS PVT",
            "/ubx_nav_pvt",
            data["pvt"],
            (
                data["pvt"]["gnss_fix_ok"]
                & ~data["pvt"]["invalid_llh"]
            ) if len(data["pvt"]) else [],
        ),
        (
            "GNSS velocity/course",
            "/ubx_nav_vel_ned",
            data["vel"],
            np.isfinite(data["vel"]["ground_speed_mps"])
            if len(data["vel"]) else [],
        ),
        (
            "Raw GNSS observations",
            "/ubx_rxm_rawx",
            data["rawx"],
            (
                data["rawx"]["num_measurements"]
                == data["rawx"]["decoded_measurements"]
            ) if len(data["rawx"]) else [],
        ),
    ]
    for name, source, frame, valid in definitions:
        timestamps = frame["t_ns"].to_numpy(dtype=np.int64) if len(frame) else []
        rows.append(stream_health(name, source, gnss_timestamp_field, timestamps, valid))

    for dtype, name in [(64, "IMU raw"), (65, "IMU AHRS")]:
        frame = imu_streams[dtype]
        rows.append(stream_health(
            name,
            f"IMU CSV dtype {dtype}",
            "t_unix_ns",
            frame["t_unix_ns"].to_numpy(dtype=np.int64),
            frame["valid"].to_numpy(dtype=bool),
        ))
    return pd.DataFrame(rows)


def nearest_merge(left, right, tolerance_ns, suffix):
    if not len(left) or not len(right):
        return left.copy()
    return pd.merge_asof(
        left.sort_values("t_ns"),
        right.sort_values("t_ns"),
        on="t_ns",
        direction="nearest",
        tolerance=int(tolerance_ns),
        suffixes=("", suffix),
    )


def build_gnss_quality(data):
    pvt = data["pvt"].copy()
    if not len(pvt):
        raise RuntimeError("No /ubx_nav_pvt messages were found")
    pvt = pvt[pvt["gnss_fix_ok"] & ~pvt["invalid_llh"]].copy()

    if len(data["hp"]):
        quality = data["hp"][data["hp"]["hp_valid"]].copy()
        quality = nearest_merge(
            quality,
            pvt[[
                "t_ns", "rtk_state", "num_sv", "pdop",
                "ground_speed_mps", "course_deg", "course_accuracy_deg",
            ]],
            tolerance_ns=150_000_000,
            suffix="_pvt",
        )
        quality["hacc_source"] = "UBX-NAV-HPPOSLLH (0.1 mm scale)"
    else:
        quality = pvt.copy()
        quality["hacc_source"] = "UBX-NAV-PVT (mm scale)"

    quality = quality.dropna(subset=[
        "rtk_state", "latitude_deg", "longitude_deg", "hacc_m",
    ]).copy()
    quality["rtk_state"] = quality["rtk_state"].astype(int)
    quality["rtk_label"] = quality["rtk_state"].map(RTK_LABELS)
    return quality.sort_values("t_ns").reset_index(drop=True)


def time_weighted_state_fraction(frame):
    if len(frame) < 2:
        return {state: np.nan for state in STATE_ORDER}
    t = frame["t_ns"].to_numpy(dtype=np.int64)
    dt = np.diff(t).astype(float) / NS_PER_SECOND
    median_dt = float(np.median(dt[dt > 0]))
    weights = np.r_[np.minimum(dt, 3.0 * median_dt), median_dt]
    total = weights.sum()
    states = frame["rtk_state"].to_numpy(dtype=int)
    return {
        state: float(weights[states == state].sum() / total)
        for state in STATE_ORDER
    }


def summarize_gnss(quality):
    weighted = time_weighted_state_fraction(quality)
    rows = []
    for state in [None, *STATE_ORDER]:
        subset = quality if state is None else quality[quality["rtk_state"] == state]
        if state is not None and not len(subset):
            continue
        rows.append({
            "subset": "All epochs" if state is None else RTK_LABELS[state],
            "rtk_state": "all" if state is None else state,
            "n_epochs": int(len(subset)),
            "sample_fraction": (
                1.0 if state is None else float(len(subset) / len(quality))
            ),
            "time_weighted_fraction": (
                1.0 if state is None else weighted[state]
            ),
            "receiver_reported_hacc_median_m": float(subset["hacc_m"].median()),
            "receiver_reported_hacc_p95_m": float(subset["hacc_m"].quantile(0.95)),
            "num_sv_median": float(subset["num_sv"].median()),
            "num_sv_p05": float(subset["num_sv"].quantile(0.05)),
            "pdop_median": float(subset["pdop"].median()),
            "pdop_p95": float(subset["pdop"].quantile(0.95)),
        })
    return pd.DataFrame(rows)


def geodetic_to_enu(latitude, longitude, altitude):
    """Project a short route into a local WGS84 tangent approximation."""
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    alt = np.asarray(altitude, dtype=float)
    lat0 = float(lat[0])
    lon0 = float(lon[0])
    alt0 = float(alt[0])
    radius = 6378137.0
    east = radius * math.cos(lat0) * (lon - lon0)
    north = radius * (lat - lat0)
    up = alt - alt0
    return east, north, up


def longest_true_run(t_ns, mask, max_gap_s):
    t_ns = np.asarray(t_ns, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    best = None
    start = None
    for index in range(len(t_ns)):
        continues = (
            index == 0
            or (t_ns[index] - t_ns[index - 1]) / NS_PER_SECOND <= max_gap_s
        )
        if mask[index] and (start is None or continues):
            if start is None:
                start = index
        elif mask[index]:
            start = index
        elif start is not None:
            candidate = (start, index - 1)
            if best is None or t_ns[candidate[1]] - t_ns[candidate[0]] > t_ns[best[1]] - t_ns[best[0]]:
                best = candidate
            start = None
    if start is not None:
        candidate = (start, len(t_ns) - 1)
        if best is None or t_ns[candidate[1]] - t_ns[candidate[0]] > t_ns[best[1]] - t_ns[best[0]]:
            best = candidate
    return best


def stationary_quality(
    vel,
    raw_imu,
    speed_threshold,
    min_duration_s,
    reference_gravity_mps2,
    gravity_reference,
    analysis_start_offset_s=0.0,
):
    result = {
        "available": False,
        "selection": (
            "First analysis window within the longest contiguous "
            "GNSS speed-gated interval"
        ),
        "speed_threshold_mps": speed_threshold,
        "minimum_duration_s": min_duration_s,
        "allan_deviation_computed": False,
        "gravity_reference": gravity_reference,
    }
    if not len(vel) or not len(raw_imu):
        result["reason"] = "Missing GNSS velocity or raw IMU samples"
        return result, pd.DataFrame(), pd.DataFrame()

    times = vel["t_ns"].to_numpy(dtype=np.int64)
    speed = vel["ground_speed_mps"].to_numpy(dtype=float)
    median_dt = float(np.median(np.diff(times)) / NS_PER_SECOND)
    # GNSS topic scheduling is bursty in this recording.  A one-second
    # continuity allowance preserves the physical low-speed interval without
    # interpolating speed values across a long outage.
    run = longest_true_run(
        times,
        speed <= speed_threshold,
        max(1.0, 5.0 * median_dt),
    )
    if run is None:
        result["reason"] = "No contiguous low-speed interval"
        return result, pd.DataFrame(), pd.DataFrame()
    start_ns = int(times[run[0]])
    end_ns = int(times[run[1]])
    duration_s = (end_ns - start_ns) / NS_PER_SECOND
    if duration_s < min_duration_s:
        result["reason"] = f"Longest low-speed interval is only {duration_s:.2f} s"
        return result, pd.DataFrame(), pd.DataFrame()

    candidate = raw_imu[
        (raw_imu["t_unix_ns"] >= start_ns)
        & (raw_imu["t_unix_ns"] <= end_ns)
        & raw_imu["valid"]
    ].copy()
    if len(candidate) < 20:
        result["reason"] = "Too few valid raw IMU samples in the selected interval"
        return result, pd.DataFrame(), pd.DataFrame()

    # Use a predefined window at the beginning of the low-speed interval.
    # This avoids reporting statistics from the later handling/vibration seen
    # near the end of the 91 s GNSS low-speed candidate and avoids selecting a
    # retrospectively optimal (lowest-noise) window.
    first_imu_ns = int(candidate["t_unix_ns"].iloc[0])
    analysis_start_ns = first_imu_ns + int(round(analysis_start_offset_s * NS_PER_SECOND))
    analysis_end_ns = analysis_start_ns + int(round(min_duration_s * NS_PER_SECOND))
    sample = candidate[
        (candidate["t_unix_ns"] >= analysis_start_ns)
        & (candidate["t_unix_ns"] <= analysis_end_ns)
    ].copy()
    if len(sample) < 20:
        result["reason"] = "Too few valid raw IMU samples in the analysis window"
        return result, pd.DataFrame(), pd.DataFrame()
    selected_start_ns = int(sample["t_unix_ns"].iloc[0])
    selected_end_ns = int(sample["t_unix_ns"].iloc[-1])
    selected_duration_s = (
        selected_end_ns - selected_start_ns
    ) / NS_PER_SECOND

    gyro = sample[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    acceleration = sample[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    confirmation = bool(
        np.quantile(gyro_norm, 0.95) <= 0.05
        and np.std(acceleration_norm, ddof=1) <= 0.15
    )

    result.update({
        "available": True,
        "selection": (
            f"{analysis_start_offset_s:.1f} s after the start of the longest "
            f"low-speed interval, then {min_duration_s:.1f} s; "
            "GNSS speed-gated interval; checked against IMU stability gates"
        ),
        "analysis_start_offset_s": float(analysis_start_offset_s),
        "candidate_start_ns": start_ns,
        "candidate_end_ns": end_ns,
        "candidate_duration_s": duration_s,
        "start_ns": selected_start_ns,
        "end_ns": selected_end_ns,
        "duration_s": selected_duration_s,
        "n_raw_imu_samples": int(len(sample)),
        "gnss_speed_max_mps": float(
            vel.loc[
                (vel["t_ns"] >= selected_start_ns)
                & (vel["t_ns"] <= selected_end_ns),
                "ground_speed_mps",
            ].max()
        ),
        "stationary_checks_passed": confirmation,
        "stationary_check_definition": (
            "GNSS speed gate plus gyro-norm P95 <= 0.05 rad/s and "
            "acceleration-norm standard deviation <= 0.15 m/s^2"
        ),
        "acceleration_norm_mean_mps2": float(acceleration_norm.mean()),
        "acceleration_norm_std_mps2": float(acceleration_norm.std(ddof=1)),
        "reference_normal_gravity_mps2": float(reference_gravity_mps2),
        "acceleration_norm_minus_reference_normal_gravity_mps2": float(
            acceleration_norm.mean() - reference_gravity_mps2
        ),
        "gyro_norm_p95_rad_s": float(np.quantile(gyro_norm, 0.95)),
        "allan_deviation_reason": (
            "Not computed: the selected stationary interval is shorter "
            "than the 600 s minimum used for this report."
            if selected_duration_s < 600.0 else
            "Not computed automatically; stationary status requires manual review."
        ),
    })

    axes = []
    for index, axis in enumerate(["x", "y", "z"]):
        axes.append({
            "axis": axis,
            "gyro_bias_rad_s": float(gyro[:, index].mean()),
            "gyro_std_rad_s": float(gyro[:, index].std(ddof=1)),
            "acceleration_mean_mps2": float(acceleration[:, index].mean()),
            "acceleration_std_mps2": float(acceleration[:, index].std(ddof=1)),
        })
    sample["acceleration_norm_mps2"] = acceleration_norm
    return result, pd.DataFrame(axes), sample


def smooth_series(values, window=5):
    values = np.asarray(values, dtype=float)
    if len(values) < window or window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def build_course_rate(vel, min_speed, max_accuracy):
    if len(vel) < 10:
        return pd.DataFrame()
    vel = vel.sort_values("t_ns").reset_index(drop=True)
    t = vel["t_ns"].to_numpy(dtype=np.int64)
    valid = (
        np.isfinite(vel["course_deg"])
        & np.isfinite(vel["ground_speed_mps"])
        & np.isfinite(vel["course_accuracy_deg"])
        & (vel["ground_speed_mps"] >= min_speed)
        & (vel["course_accuracy_deg"] <= max_accuracy)
    ).to_numpy(dtype=bool)
    median_dt = float(np.median(np.diff(t)) / NS_PER_SECOND)
    rows = []
    start = None
    for index in range(len(t) + 1):
        is_valid = index < len(t) and valid[index]
        continues = (
            index == 0
            or index >= len(t)
            or (t[index] - t[index - 1]) / NS_PER_SECOND <= 2.5 * median_dt
        )
        if is_valid and start is None:
            start = index
        elif is_valid and not continues:
            stop = index
            if stop - start >= 7:
                rows.extend(course_segment_rows(vel.iloc[start:stop]))
            start = index
        elif not is_valid and start is not None:
            stop = index
            if stop - start >= 7:
                rows.extend(course_segment_rows(vel.iloc[start:stop]))
            start = None
    return pd.DataFrame(rows).sort_values("t_ns").reset_index(drop=True) if rows else pd.DataFrame()


def course_segment_rows(segment):
    t_ns = segment["t_ns"].to_numpy(dtype=np.int64)
    t_s = t_ns.astype(float) / NS_PER_SECOND
    heading_rad = np.deg2rad(segment["course_deg"].to_numpy(dtype=float))
    yaw_rad = np.unwrap(np.pi / 2.0 - heading_rad)
    yaw_smooth = smooth_series(yaw_rad, 5)
    rate = np.gradient(yaw_smooth, t_s)
    rows = []
    for index in range(2, len(segment) - 2):
        rows.append({
            "t_ns": int(t_ns[index]),
            "course_yaw_rad": float(yaw_rad[index]),
            "course_yaw_smooth_rad": float(yaw_smooth[index]),
            "course_rate_rad_s": float(rate[index]),
            "ground_speed_mps": float(segment.iloc[index]["ground_speed_mps"]),
            "course_accuracy_deg": float(segment.iloc[index]["course_accuracy_deg"]),
        })
    return rows


def lag_correlation(course, raw_imu, max_lag_s, lag_step_s, turn_threshold):
    raw = raw_imu[raw_imu["valid"]].copy()
    t_imu = raw["t_unix_ns"].to_numpy(dtype=np.int64).astype(float) / NS_PER_SECOND
    yaw_rate = -raw["gyro_z"].to_numpy(dtype=float)  # native FRD -> ROS FLU
    yaw_rate = smooth_series(yaw_rate, 9)
    t_course = course["t_ns"].to_numpy(dtype=np.int64).astype(float) / NS_PER_SECOND
    course_rate = course["course_rate_rad_s"].to_numpy(dtype=float)
    turning = np.abs(course_rate) >= turn_threshold

    rows = []
    shifts = np.arange(-max_lag_s, max_lag_s + 0.5 * lag_step_s, lag_step_s)
    for shift in shifts:
        query = t_course + shift
        valid = turning & (query >= t_imu[0]) & (query <= t_imu[-1])
        if valid.sum() < 20:
            correlation = np.nan
        else:
            imu_values = np.interp(query[valid], t_imu, yaw_rate)
            left = course_rate[valid]
            correlation = (
                float(np.corrcoef(left, imu_values)[0, 1])
                if np.std(left) > 0 and np.std(imu_values) > 0 else np.nan
            )
        rows.append({
            "lag_s": float(shift),
            "normalized_correlation": correlation,
            "n_turn_samples": int(valid.sum()),
        })
    curve = pd.DataFrame(rows)
    valid_curve = curve.dropna(subset=["normalized_correlation"])
    if not len(valid_curve):
        raise RuntimeError("No valid GNSS/IMU lag correlations could be computed")
    best = valid_curve.loc[valid_curve["normalized_correlation"].idxmax()]
    return curve, float(best["lag_s"]), float(best["normalized_correlation"])


def select_turn(course, duration_s):
    t = course["t_ns"].to_numpy(dtype=np.int64).astype(float) / NS_PER_SECOND
    yaw = course["course_yaw_smooth_rad"].to_numpy(dtype=float)
    best = None
    for start in range(len(course)):
        end = int(np.searchsorted(t, t[start] + duration_s, side="right"))
        if end - start < max(12, int(0.6 * duration_s * 5.0)):
            continue
        local_t = t[start:end]
        if local_t[-1] - local_t[0] < 0.85 * duration_s:
            continue
        if np.max(np.diff(local_t)) > 0.6:
            continue
        local_yaw = yaw[start:end]
        score = float(np.sum(np.abs(np.diff(local_yaw))))
        candidate = (score, start, end)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No contiguous representative turn window was found")
    _, start, end = best
    selected = course.iloc[start:end].copy()
    return selected, {
        "start_ns": int(selected["t_ns"].iloc[0]),
        "end_ns": int(selected["t_ns"].iloc[-1]),
        "duration_s": float(
            (selected["t_ns"].iloc[-1] - selected["t_ns"].iloc[0])
            / NS_PER_SECOND
        ),
        "course_net_change_deg": float(np.rad2deg(
            selected["course_yaw_smooth_rad"].iloc[-1]
            - selected["course_yaw_smooth_rad"].iloc[0]
        )),
        "course_total_variation_deg": float(np.rad2deg(np.sum(np.abs(
            np.diff(selected["course_yaw_smooth_rad"].to_numpy(dtype=float))
        )))),
        "selection_rule": (
            f"Highest integrated absolute GNSS course change in a contiguous "
            f"{duration_s:.1f} s window after speed and course-accuracy gates"
        ),
    }


def build_turn_signals(selected, raw_imu, effective_lag_s):
    start_ns = int(selected["t_ns"].iloc[0])
    end_ns = int(selected["t_ns"].iloc[-1])
    start_s = start_ns / NS_PER_SECOND
    end_s = end_ns / NS_PER_SECOND

    raw = raw_imu[raw_imu["valid"]].copy()
    raw["aligned_time_s"] = (
        raw["t_unix_ns"].astype(float) / NS_PER_SECOND - effective_lag_s
    )
    raw = raw[
        (raw["aligned_time_s"] >= start_s)
        & (raw["aligned_time_s"] <= end_s)
    ].copy()
    raw["time_from_turn_start_s"] = raw["aligned_time_s"] - start_s
    raw["imu_yaw_rate_flu_rad_s"] = -raw["gyro_z"].astype(float)
    t = raw["aligned_time_s"].to_numpy(dtype=float)
    rate = raw["imu_yaw_rate_flu_rad_s"].to_numpy(dtype=float)
    increments = 0.5 * (rate[1:] + rate[:-1]) * np.diff(t)
    raw["integrated_imu_yaw_change_deg"] = np.rad2deg(
        np.r_[0.0, np.cumsum(increments)]
    )

    course = selected.copy()
    course["time_from_turn_start_s"] = (
        course["t_ns"].astype(float) / NS_PER_SECOND - start_s
    )
    course["gnss_yaw_change_deg"] = np.rad2deg(
        course["course_yaw_smooth_rad"]
        - course["course_yaw_smooth_rad"].iloc[0]
    )
    return course, raw


def plot_stream_timing(streams, bag_start_ns, bag_end_ns, output_base):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    labels = streams["stream"].tolist()
    y = np.arange(len(labels))
    start = (streams["start_ns"].astype(float) - bag_start_ns) / NS_PER_SECOND
    duration = streams["duration_s"].to_numpy(dtype=float)
    axes[0, 0].barh(y, duration, left=start, color=COLORS["blue"], alpha=0.8)
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Time from bag start (s)")
    axes[0, 0].set_title("Stream coverage")
    axes[0, 0].axvline((bag_end_ns - bag_start_ns) / NS_PER_SECOND, color=COLORS["black"], linestyle="--")

    axes[0, 1].barh(y, streams["effective_rate_hz"], color=COLORS["blue"])
    axes[0, 1].set_yticks(y, labels)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Effective rate (Hz)")
    axes[0, 1].set_title("Effective sampling rate")

    height = 0.36
    axes[1, 0].barh(y - height / 2, streams["p99_interval_ms"], height, label="P99 interval", color=COLORS["orange"])
    axes[1, 0].barh(y + height / 2, streams["longest_gap_ms"], height, label="Longest gap", color=COLORS["vermillion"])
    axes[1, 0].set_yticks(y, labels)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Interval (ms)")
    axes[1, 0].set_title("Sampling intervals")
    axes[1, 0].set_xscale("log")
    axes[1, 0].legend()

    axes[1, 1].barh(y, 100.0 * streams["valid_frame_fraction"], color=COLORS["green"])
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 102)
    axes[1, 1].set_xlabel("Valid frames (%)")
    axes[1, 1].set_title("Frame validity")
    for index, value in enumerate(100.0 * streams["valid_frame_fraction"]):
        axes[1, 1].text(
            min(value - 1.0, 99.0),
            index,
            f"{value:.1f}",
            va="center",
            ha="right",
            color="white" if value >= 20.0 else COLORS["black"],
            fontsize=7.5,
        )

    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(ax, label)
    fig.suptitle("GNSS and IMU stream completeness and timing")
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_rtk_hacc(quality, output_base):
    t_min = quality["t_ns"].iloc[0]
    time_min = (quality["t_ns"].to_numpy(dtype=float) - t_min) / (60.0 * NS_PER_SECOND)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for state in STATE_ORDER:
        mask = quality["rtk_state"].to_numpy(dtype=int) == state
        if mask.any():
            ax.scatter(time_min[mask], np.full(mask.sum(), state), s=8, color=RTK_COLORS[state], label=RTK_LABELS[state], zorder=3)
    ax.set_yticks(STATE_ORDER, ["None", "Float", "Fixed"])
    ax.set_ylim(-0.2, 2.35)
    ax.set_xlabel("Elapsed time (min)")
    ax.set_ylabel("Carrier solution state")

    right = ax.twinx()
    right.plot(time_min, quality["hacc_m"].to_numpy(dtype=float), color=COLORS["purple"], alpha=0.75, label="Receiver-reported hAcc")
    right.axhline(0.10, color=COLORS["purple"], linestyle="--", linewidth=1.0, label="hAcc = 0.10 m")
    right.set_ylabel("Receiver-reported hAcc (m)", color=COLORS["purple"])
    right.tick_params(axis="y", colors=COLORS["purple"])

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.98),
        framealpha=0.92,
    )
    ax.set_title("RTK state and receiver-reported hAcc over the full interval")
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_gnss_route(quality, output_base, route=None):
    route_frame = route if route is not None and len(route) else quality
    if {"east_m", "north_m"}.issubset(route_frame.columns):
        east = route_frame["east_m"].to_numpy(dtype=float)
        north = route_frame["north_m"].to_numpy(dtype=float)
    else:
        east, north, _ = geodetic_to_enu(
            route_frame["latitude_deg"], route_frame["longitude_deg"], route_frame["height_m"]
        )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(east, north, color=COLORS["light_grey"], linewidth=0.8, zorder=1)
    states = route_frame["rtk_state"].to_numpy(dtype=int)
    for state in STATE_ORDER:
        mask = states == state
        if mask.any():
            ax.scatter(east[mask], north[mask], s=7, color=RTK_COLORS[state], label=RTK_LABELS[state], zorder=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("GNSS trajectory by RTK carrier solution over the full interval")
    ax.legend(loc="best")
    fig.tight_layout()
    save_figure(fig, output_base)
    return east, north


def plot_gnss_distributions(
    quality,
    output_base,
    route=None,
    interval_start_ns=None,
    interval_end_ns=None,
    route_title="Valid PVT route in the analysis interval",
    quality_title="RTK state and hAcc over the full riding interval",
):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    # The hAcc ECDF and the hAcc violin plot show the same distribution.  Use
    # Use every valid PVT position for the route panel.  The hAcc and
    # time-series panels remain based on the quality pairing.
    route_frame = route if route is not None and len(route) else quality
    if {"east_m", "north_m"}.issubset(route_frame.columns):
        east = route_frame["east_m"].to_numpy(dtype=float)
        north = route_frame["north_m"].to_numpy(dtype=float)
    else:
        east, north, _ = geodetic_to_enu(
            route_frame["latitude_deg"],
            route_frame["longitude_deg"],
            route_frame["height_m"],
        )
    axes[0, 0].plot(
        east,
        north,
        color=COLORS["light_grey"],
        linewidth=0.45,
        zorder=1,
    )
    route_states = route_frame["rtk_state"].to_numpy(dtype=int)
    for state in STATE_ORDER:
        if np.any(route_states == state):
            # Empty handles keep the legend while the trajectory itself is
            # drawn as thin state-coloured line segments below.
            axes[0, 0].plot(
                [],
                [],
                color=RTK_COLORS[state],
                linewidth=0.9,
                label=RTK_LABELS[state],
            )
    for index in range(len(east) - 1):
        if route_states[index] == route_states[index + 1]:
            axes[0, 0].plot(
                east[index:index + 2],
                north[index:index + 2],
                color=RTK_COLORS[route_states[index]],
                linewidth=0.9,
                solid_capstyle="round",
                zorder=2,
            )
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].set_xlabel("East (m)")
    axes[0, 0].set_ylabel("North (m)")
    axes[0, 0].set_title(route_title)
    axes[0, 0].legend(loc="best")

    # Use the time-series quality view for panel (b).  This preserves both
    # the RTK state changes and the receiver-reported hAcc threshold in one
    # panel; the distribution-only violin view is retained in older results.
    time_origin_ns = (
        int(interval_start_ns)
        if interval_start_ns is not None
        else int(quality["t_ns"].iloc[0])
    )
    elapsed_s = (
        quality["t_ns"].to_numpy(dtype=float) - float(time_origin_ns)
    ) / NS_PER_SECOND
    interval_duration_s = (
        (int(interval_end_ns) - int(interval_start_ns)) / NS_PER_SECOND
        if interval_start_ns is not None and interval_end_ns is not None
        else None
    )
    quality_ax = axes[0, 1]
    for state in STATE_ORDER:
        mask = quality["rtk_state"].to_numpy(dtype=int) == state
        if mask.any():
            quality_ax.scatter(
                elapsed_s[mask],
                np.full(mask.sum(), state),
                s=5,
                color=RTK_COLORS[state],
                label=RTK_LABELS[state],
                zorder=3,
            )
    quality_ax.set_yticks(STATE_ORDER, ["None", "Float", "Fixed"])
    quality_ax.set_ylim(-0.2, 2.35)
    quality_ax.set_xlabel("Elapsed time (s)")
    quality_ax.set_ylabel("Carrier solution state")
    if interval_duration_s is not None:
        quality_ax.set_xlim(0.0, interval_duration_s)

    hacc_ax = quality_ax.twinx()
    hacc_ax.plot(
        elapsed_s,
        quality["hacc_m"].to_numpy(dtype=float),
        color=COLORS["purple"],
        alpha=0.75,
        label="Receiver-reported hAcc",
    )
    hacc_ax.axhline(
        0.10,
        color=COLORS["purple"],
        linestyle="--",
        linewidth=1.0,
        label="hAcc = 0.10 m",
    )
    hacc_ax.set_ylabel("Receiver-reported hAcc (m)", color=COLORS["purple"])
    hacc_ax.tick_params(axis="y", colors=COLORS["purple"])
    handles1, labels1 = quality_ax.get_legend_handles_labels()
    handles2, labels2 = hacc_ax.get_legend_handles_labels()
    quality_ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=7,
        framealpha=0.92,
    )
    quality_ax.set_title(quality_title)

    elapsed = (quality["t_ns"].to_numpy(dtype=float) - float(time_origin_ns)) / NS_PER_SECOND
    axes[1, 0].plot(elapsed, quality["num_sv"].to_numpy(dtype=float), color=COLORS["blue"])
    axes[1, 0].set_xlabel("Elapsed time (s)")
    axes[1, 0].set_ylabel("Satellites used")
    axes[1, 0].set_title("Satellites used in navigation solution")
    if interval_duration_s is not None:
        axes[1, 0].set_xlim(0.0, interval_duration_s)

    axes[1, 1].plot(elapsed, quality["pdop"].to_numpy(dtype=float), color=COLORS["vermillion"])
    axes[1, 1].set_xlabel("Elapsed time (s)")
    axes[1, 1].set_ylabel("PDOP")
    axes[1, 1].set_title("Position dilution of precision")
    if interval_duration_s is not None:
        axes[1, 1].set_xlim(0.0, interval_duration_s)

    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(ax, label)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_raw_gnss(rawx, cno, output_base):
    if not len(rawx):
        return False
    elapsed = (
        rawx["t_ns"].to_numpy(dtype=float)
        - float(rawx["t_ns"].iloc[0])
    ) / NS_PER_SECOND
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(elapsed, rawx["num_measurements"].to_numpy(dtype=float), color=COLORS["blue"], label="Observations")
    axes[0].plot(elapsed, rawx["carrier_phase_valid"].to_numpy(dtype=float), color=COLORS["green"], label="Valid carrier phase")
    axes[0].set_xlabel("Elapsed time (s)")
    axes[0].set_ylabel("Observations per epoch")
    axes[0].set_title("Raw GNSS observation availability")
    axes[0].legend()

    axes[1].hist(cno[np.isfinite(cno)], bins=np.arange(0, 61, 2), color=COLORS["blue"], alpha=0.85)
    axes[1].set_xlabel("C/N0 (dB-Hz)")
    axes[1].set_ylabel("Observations")
    axes[1].set_title("Carrier-to-noise density")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def plot_imu_stationary(
    sample,
    axes_summary,
    reference_gravity_mps2,
    output_base,
):
    if not len(sample):
        return False
    t0 = sample["t_unix_ns"].iloc[0]
    elapsed = (sample["t_unix_ns"].to_numpy(dtype=float) - t0) / NS_PER_SECOND
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    axis_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for axis, color in zip(["x", "y", "z"], axis_colors):
        axes[0, 0].plot(
            elapsed, sample[f"gyro_{axis}"].to_numpy(dtype=float),
            color=color, label=axis,
        )
    axes[0, 0].set_xlabel("Time within selected stationary interval (s)")
    axes[0, 0].set_ylabel("Angular rate (rad/s)")
    axes[0, 0].set_title("Raw gyroscope, native FRD frame")
    axes[0, 0].legend(title="Axis", ncol=3)

    values = [sample[f"gyro_{axis}"].to_numpy(dtype=float) for axis in ["x", "y", "z"]]
    boxes = axes[0, 1].boxplot(values, labels=["x", "y", "z"], showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], axis_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0, 1].set_xlabel("Axis")
    axes[0, 1].set_ylabel("Angular rate (rad/s)")
    axes[0, 1].set_title("Gyroscope bias and spread")

    axes[1, 0].plot(
        elapsed, sample["acceleration_norm_mps2"].to_numpy(dtype=float),
        color=COLORS["vermillion"],
    )
    axes[1, 0].axhline(
        reference_gravity_mps2,
        color=COLORS["black"],
        linestyle="--",
        label="Estimated local gravity",
    )
    axes[1, 0].set_xlabel("Time within selected stationary interval (s)")
    axes[1, 0].set_ylabel("Acceleration norm (m/s²)")
    axes[1, 0].set_title("Acceleration norm")
    axes[1, 0].legend()

    axes[1, 1].hist(
        sample["acceleration_norm_mps2"].to_numpy(dtype=float),
        bins=35, color=COLORS["vermillion"], alpha=0.8,
    )
    axes[1, 1].axvline(
        reference_gravity_mps2,
        color=COLORS["black"],
        linestyle="--",
    )
    axes[1, 1].set_xlabel("Acceleration norm (m/s²)")
    axes[1, 1].set_ylabel("Samples")
    axes[1, 1].set_title("Acceleration-norm distribution")

    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(ax, label)
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def plot_temporal_alignment(curve, course_turn, imu_turn, best_lag, best_corr, output_base):
    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(2, 2)
    lag_ax = fig.add_subplot(grid[0, 0])
    rate_ax = fig.add_subplot(grid[0, 1])
    heading_ax = fig.add_subplot(grid[1, :])

    lag_ax.plot(
        curve["lag_s"].to_numpy(dtype=float),
        curve["normalized_correlation"].to_numpy(dtype=float),
        color=COLORS["blue"],
    )
    lag_ax.axvline(best_lag, color=COLORS["vermillion"], linestyle="--", label=f"Peak: {best_lag:+.2f} s")
    lag_ax.set_xlabel("IMU timestamp minus GNSS timestamp (s)")
    lag_ax.set_ylabel("Normalized correlation")
    lag_ax.set_title("Course-rate / gyro cross-correlation")
    lag_ax.legend(title=f"r = {best_corr:.3f}")

    rate_ax.plot(
        imu_turn["time_from_turn_start_s"].to_numpy(dtype=float),
        imu_turn["imu_yaw_rate_flu_rad_s"].to_numpy(dtype=float),
        color=COLORS["vermillion"],
        label="Raw gyro yaw rate (20 Hz)",
    )
    rate_ax.plot(
        course_turn["time_from_turn_start_s"].to_numpy(dtype=float),
        course_turn["course_rate_rad_s"].to_numpy(dtype=float),
        marker="o",
        markersize=2.5,
        color=COLORS["blue"],
        label="GNSS course rate",
    )
    rate_ax.set_xlabel("Time within selected turn (s)")
    rate_ax.set_ylabel("Yaw rate (rad/s)")
    rate_ax.set_title("Yaw-rate complementarity")
    rate_ax.legend()

    heading_ax.plot(
        imu_turn["time_from_turn_start_s"].to_numpy(dtype=float),
        imu_turn["integrated_imu_yaw_change_deg"].to_numpy(dtype=float),
        color=COLORS["vermillion"],
        label="Integrated raw gyro",
    )
    heading_ax.plot(
        course_turn["time_from_turn_start_s"].to_numpy(dtype=float),
        course_turn["gnss_yaw_change_deg"].to_numpy(dtype=float),
        marker="o",
        markersize=3,
        color=COLORS["blue"],
        label="GNSS course samples",
    )
    heading_ax.set_xlabel("Time within selected turn (s)")
    heading_ax.set_ylabel("Relative heading change (deg)")
    heading_ax.set_title("Low-rate absolute course and high-rate angular motion")
    heading_ax.legend()

    panel_label(lag_ax, "(a)")
    panel_label(rate_ax, "(b)")
    panel_label(heading_ax, "(c)")
    fig.tight_layout()
    save_figure(fig, output_base)


def write_checksums(root):
    root = Path(root)
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256"):
        lines.append(f"{sha256_path(path)}  {path.relative_to(root)}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_info(repo_root):
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--", "data_analysis"], text=True
        ).splitlines()
        return {"commit": commit, "data_analysis_dirty_entries": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "data_analysis_dirty_entries": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--stationary-speed-threshold", type=float, default=0.15)
    parser.add_argument("--stationary-min-duration", type=float, default=10.0)
    parser.add_argument("--course-min-speed", type=float, default=2.0)
    parser.add_argument("--course-max-accuracy", type=float, default=30.0)
    parser.add_argument("--max-lag", type=float, default=2.0)
    parser.add_argument("--lag-step", type=float, default=0.02)
    parser.add_argument("--turn-rate-threshold", type=float, default=0.015)
    parser.add_argument("--turn-duration", type=float, default=15.0)
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory does not exist: {session_dir}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    bag_dir = find_single(session_dir, "rosbag2_*", "ROS 2 bag directory")
    imu_csv = find_single(session_dir, "imu_*.csv", "IMU CSV")
    metadata = load_bag_metadata(bag_dir)
    data = read_bag_topics(bag_dir, metadata["storage_identifier"])
    imu_streams, imu_clock_audit = load_imu(
        imu_csv, metadata["start_ns"], metadata["end_ns"]
    )
    timestamp_audit = build_timestamp_audit(data["pvt"], imu_clock_audit)

    session_id = args.session_id or session_dir.name
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    tables = temporary / "tables"
    figures = temporary / "figures"
    tables.mkdir()
    figures.mkdir()

    streams = build_stream_table(data, imu_streams)
    streams.insert(0, "session_id", session_id)
    streams.to_csv(tables / "stream_timing.csv", index=False)
    write_json(tables / "timestamp_audit.json", json_safe(timestamp_audit))

    pvt_start = int(data["pvt"]["t_ns"].min())
    pvt_end = int(data["pvt"]["t_ns"].max())
    raw_start = int(imu_streams[64]["t_unix_ns"].min())
    raw_end = int(imu_streams[64]["t_unix_ns"].max())
    ahrs_start = int(imu_streams[65]["t_unix_ns"].min())
    ahrs_end = int(imu_streams[65]["t_unix_ns"].max())
    overlap = {
        "pvt_raw_imu_overlap_s": max(0.0, (min(pvt_end, raw_end) - max(pvt_start, raw_start)) / NS_PER_SECOND),
        "pvt_ahrs_overlap_s": max(0.0, (min(pvt_end, ahrs_end) - max(pvt_start, ahrs_start)) / NS_PER_SECOND),
    }
    write_json(tables / "gnss_imu_overlap.json", json_safe(overlap))
    plot_stream_timing(streams, metadata["start_ns"], metadata["end_ns"], figures / "stream_completeness_and_timing")

    quality = build_gnss_quality(data)
    east, north, up = geodetic_to_enu(quality["latitude_deg"], quality["longitude_deg"], quality["height_m"])
    quality["east_m"] = east
    quality["north_m"] = north
    quality["up_m"] = up
    gravity_window_end_ns = (
        int(quality["t_ns"].min())
        + int(round(args.stationary_min_duration * NS_PER_SECOND))
    )
    gravity_window = quality[quality["t_ns"] <= gravity_window_end_ns]
    gravity_latitude_deg = float(gravity_window["latitude_deg"].median())
    gravity_longitude_deg = float(gravity_window["longitude_deg"].median())
    gravity_height_m = float(gravity_window["height_m"].median())
    reference_gravity_mps2 = wgs84_normal_gravity(
        gravity_latitude_deg,
        gravity_height_m,
    )
    gravity_reference = {
        "model": (
            "WGS84 Somigliana normal gravity with second-order ellipsoidal "
            "height correction"
        ),
        "location": "Dresden P9 initial stationary-analysis interval",
        "latitude_deg": gravity_latitude_deg,
        "longitude_deg": gravity_longitude_deg,
        "ellipsoidal_height_m": gravity_height_m,
        "normal_gravity_mps2": reference_gravity_mps2,
        "conventional_gravity_mps2": CONVENTIONAL_GRAVITY,
        "interpretation": (
            "Model-derived normal gravity, not a local gravimeter measurement"
        ),
    }
    write_json(tables / "gravity_reference.json", json_safe(gravity_reference))
    quality.to_csv(tables / "gnss_quality_epochs.csv", index=False)
    gnss_summary = summarize_gnss(quality)
    gnss_summary.insert(0, "session_id", session_id)
    gnss_summary.to_csv(tables / "gnss_solution_quality.csv", index=False)
    plot_rtk_hacc(quality, figures / "rtk_status_and_hacc")
    plot_gnss_route(quality, figures / "gnss_route_by_rtk_status")
    plot_gnss_distributions(quality, figures / "gnss_quality_distributions")

    rawx = data["rawx"]
    rawx.to_csv(tables / "raw_gnss_epoch_quality.csv", index=False)
    total_observations = int(rawx["decoded_measurements"].sum()) if len(rawx) else 0
    raw_summary = {
        "session_id": session_id,
        "available": bool(len(rawx)),
        "topic": "/ubx_rxm_rawx",
        "epochs": int(len(rawx)),
        "observations": total_observations,
        "observations_per_epoch_median": float(rawx["decoded_measurements"].median()) if len(rawx) else None,
        "observations_per_epoch_min": int(rawx["decoded_measurements"].min()) if len(rawx) else None,
        "observations_per_epoch_max": int(rawx["decoded_measurements"].max()) if len(rawx) else None,
        "pseudorange_valid_fraction": float(rawx["pseudorange_valid"].sum() / total_observations) if total_observations else None,
        "carrier_phase_valid_fraction": float(rawx["carrier_phase_valid"].sum() / total_observations) if total_observations else None,
        "cno_median_dbhz": float(np.median(data["rawx_cno"])) if len(data["rawx_cno"]) else None,
        "cno_p05_dbhz": float(np.quantile(data["rawx_cno"], 0.05)) if len(data["rawx_cno"]) else None,
        "cno_p95_dbhz": float(np.quantile(data["rawx_cno"], 0.95)) if len(data["rawx_cno"]) else None,
    }
    pd.DataFrame([raw_summary]).to_csv(tables / "raw_gnss_summary.csv", index=False)
    plot_raw_gnss(rawx, data["rawx_cno"], figures / "raw_gnss_quality")

    stationary, imu_axes, stationary_samples = stationary_quality(
        data["vel"],
        imu_streams[64],
        args.stationary_speed_threshold,
        args.stationary_min_duration,
        reference_gravity_mps2,
        gravity_reference,
    )
    write_json(tables / "imu_stationary_summary.json", json_safe(stationary))
    imu_axes.to_csv(tables / "imu_stationary_axis_statistics.csv", index=False)
    if len(stationary_samples):
        source_columns = [
            "t_unix_ns", "gyro_x", "gyro_y", "gyro_z",
            "acc_x", "acc_y", "acc_z", "acceleration_norm_mps2",
        ]
        stationary_samples[source_columns].to_csv(
            tables / "imu_stationary_figure_data.csv", index=False
        )
        plot_imu_stationary(
            stationary_samples,
            imu_axes,
            reference_gravity_mps2,
            figures / "imu_stationary_quality",
        )

    course = build_course_rate(
        data["vel"], args.course_min_speed, args.course_max_accuracy
    )
    if not len(course):
        raise RuntimeError("No valid GNSS course-rate samples after quality gates")
    curve, best_lag, best_corr = lag_correlation(
        course,
        imu_streams[64],
        args.max_lag,
        args.lag_step,
        args.turn_rate_threshold,
    )
    selected, turn_summary = select_turn(course, args.turn_duration)
    turn_summary.update({
        "effective_lag_s": best_lag,
        "peak_normalized_correlation": best_corr,
        "lag_definition": (
            "Positive lag means IMU samples at t + lag best match the GNSS "
            "course-rate sample at t. This is an effective temporal offset, "
            "not a direct clock-offset measurement."
        ),
        "course_source": "/ubx_nav_vel_ned",
        "imu_source": "dtype64 raw gyro_z converted from native FRD to ROS FLU as -gyro_z",
        "minimum_speed_mps": args.course_min_speed,
        "maximum_course_accuracy_deg": args.course_max_accuracy,
    })
    course_turn, imu_turn = build_turn_signals(selected, imu_streams[64], best_lag)
    course.to_csv(tables / "gnss_course_rate.csv", index=False)
    curve.to_csv(tables / "temporal_lag_curve.csv", index=False)
    course_turn.to_csv(tables / "representative_turn_gnss.csv", index=False)
    imu_turn[[
        "t_unix_ns", "aligned_time_s", "time_from_turn_start_s",
        "imu_yaw_rate_flu_rad_s", "integrated_imu_yaw_change_deg",
    ]].to_csv(tables / "representative_turn_imu.csv", index=False)
    write_json(tables / "representative_turn.json", json_safe(turn_summary))
    plot_temporal_alignment(
        curve,
        course_turn,
        imu_turn,
        best_lag,
        best_corr,
        figures / "temporal_alignment_and_complementarity",
    )

    captions = """stream_completeness_and_timing: Effective sampling rate, P99 interval, longest gap, valid-frame fraction, and temporal coverage for GNSS and IMU streams.\nrtk_status_and_hacc: RTK carrier solution together with the receiver-reported horizontal accuracy estimate. hAcc is not measured position error.\ngnss_route_by_rtk_status: Local ENU trajectory coloured by RTK carrier solution.\ngnss_quality_distributions: (a) GNSS trajectory coloured by RTK carrier solution; (b) RTK carrier solution and receiver-reported hAcc over time; (c) satellites used; (d) PDOP. hAcc is a receiver-reported estimate, not measured position error.\nraw_gnss_quality: Raw observation count, carrier-phase validity, and C/N0 distribution from UBX-RXM-RAWX.\nimu_stationary_quality: Native-frame raw IMU statistics in the predefined stationary-analysis interval, referenced to a model-derived Dresden normal-gravity estimate. Allan deviation is not reported for a short stationary interval.\ntemporal_alignment_and_complementarity: Exploratory effective-lag analysis; not a clock-synchronization validation.\n"""
    (temporary / "figure_captions.txt").write_text(captions, encoding="utf-8")

    repo_root = SCRIPT_DIR.parents[1]
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "session_dir": str(session_dir),
        "inputs": {
            "bag_dir": str(bag_dir),
            "bag_sha256": sha256_path(bag_dir),
            "imu_csv": str(imu_csv),
            "imu_sha256": sha256_path(imu_csv),
        },
        "bag": metadata,
        "parameters": vars(args),
        "overlap": overlap,
        "gravity_reference": gravity_reference,
        "timestamp_audit": timestamp_audit,
        "gnss_quality_reference": (
            "Receiver-reported uncertainty and RTK status; no independent "
            "position ground truth is used."
        ),
        "stationary_analysis": stationary,
        "temporal_alignment": turn_summary,
        "raw_gnss": raw_summary,
        "git": git_info(repo_root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(temporary / "run_manifest.json", json_safe(manifest))
    write_checksums(temporary)
    os.rename(temporary, output)
    print(f"Technical-validation output: {output}")
    print(f"Representative turn: {turn_summary['start_ns']} to {turn_summary['end_ns']}")
    print(f"Effective lag: {best_lag:+.3f} s (r={best_corr:.3f})")


if __name__ == "__main__":
    apply_paper_style()
    main()
