#!/usr/bin/env python3
"""Generate P9 speed, sampling-interval, and closed-loop example figures.

The script reads the timestamped files in one session directory and can use a
separate rule-based cleaned steering table. All streams are first restricted
to the common interval from the first to the last GNSS speed sample above the
configured threshold. One 30 s window is then selected for the speed and
closed-loop figures. Existing outputs are never overwritten.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
import shlex
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gnss_imu_technical_validation as GNSS  # noqa: E402
from paper_style import COLORS, apply_paper_style, panel_label, save_figure  # noqa: E402


NS_PER_SECOND = 1_000_000_000
STEERING_LIMIT_DEG = 45.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
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


def write_json(path: Path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "t_unix_ns" in frame.columns:
        frame["t_unix_ns"] = pd.to_numeric(frame["t_unix_ns"], errors="coerce")
        frame = frame[frame["t_unix_ns"].notna() & (frame["t_unix_ns"] > 0)].copy()
        frame["t_unix_ns"] = frame["t_unix_ns"].astype("int64")
        frame = frame.sort_values("t_unix_ns").drop_duplicates("t_unix_ns")
    return frame.reset_index(drop=True)


def read_tobii_gaze(gaze_path: Path, created_ns: int) -> pd.DataFrame:
    """Read the raw Tobii gaze stream using the recording start time."""
    gaze_rows = []
    with gzip.open(gaze_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            timestamp_s = float(item["timestamp"])
            data = item.get("data", {})
            gaze_2d = data.get("gaze2d")
            gaze_3d = data.get("gaze3d")
            left_eye = data.get("eyeleft") or {}
            right_eye = data.get("eyeright") or {}
            left_origin = left_eye.get("gazeorigin")
            right_origin = right_eye.get("gazeorigin")
            left_direction = left_eye.get("gazedirection")
            right_direction = right_eye.get("gazedirection")
            valid = (
                isinstance(gaze_2d, list)
                and len(gaze_2d) >= 2
                and all(math.isfinite(float(value)) for value in gaze_2d[:2])
            )
            def vector_values(value):
                if isinstance(value, list) and len(value) >= 3:
                    try:
                        values = [float(part) for part in value[:3]]
                    except (TypeError, ValueError):
                        return [np.nan, np.nan, np.nan]
                    return values if all(math.isfinite(part) for part in values) else [np.nan, np.nan, np.nan]
                return [np.nan, np.nan, np.nan]

            gaze3d_values = vector_values(gaze_3d)
            left_origin_values = vector_values(left_origin)
            right_origin_values = vector_values(right_origin)
            left_direction_values = vector_values(left_direction)
            right_direction_values = vector_values(right_direction)
            gaze_rows.append({
                "timestamp_s": timestamp_s,
                "t_unix_ns": created_ns + int(round(timestamp_s * NS_PER_SECOND)),
                "gaze_x_norm": float(gaze_2d[0]) if valid else np.nan,
                "gaze_y_norm": float(gaze_2d[1]) if valid else np.nan,
                "gaze3d_x_mm": gaze3d_values[0],
                "gaze3d_y_mm": gaze3d_values[1],
                "gaze3d_z_mm": gaze3d_values[2],
                "left_origin_x_mm": left_origin_values[0],
                "left_origin_y_mm": left_origin_values[1],
                "left_origin_z_mm": left_origin_values[2],
                "right_origin_x_mm": right_origin_values[0],
                "right_origin_y_mm": right_origin_values[1],
                "right_origin_z_mm": right_origin_values[2],
                "left_direction_x": left_direction_values[0],
                "left_direction_y": left_direction_values[1],
                "left_direction_z": left_direction_values[2],
                "right_direction_x": right_direction_values[0],
                "right_direction_y": right_direction_values[1],
                "right_direction_z": right_direction_values[2],
                "valid": bool(valid),
            })
    return pd.DataFrame(gaze_rows)


def read_tobii_recording(recording_path: Path, gaze_path: Path, imu_path: Path):
    metadata = json.loads(recording_path.read_text(encoding="utf-8"))
    created = datetime.fromisoformat(metadata["created"].replace("Z", "+00:00"))
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))
    gaze_frame = read_tobii_gaze(gaze_path, created_ns)

    tobii_imu_rows = []
    with gzip.open(imu_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            timestamp_s = float(item["timestamp"])
            data = item.get("data", {})
            gyroscope = data.get("gyroscope")
            accelerometer = data.get("accelerometer")
            valid = (
                isinstance(gyroscope, list)
                and len(gyroscope) >= 3
                and isinstance(accelerometer, list)
                and len(accelerometer) >= 3
            )
            tobii_imu_rows.append({
                "timestamp_s": timestamp_s,
                "t_unix_ns": created_ns + int(round(timestamp_s * NS_PER_SECOND)),
                "valid": bool(valid),
            })
    return metadata, gaze_frame, pd.DataFrame(tobii_imu_rows)


def _normalized_column_name(value) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _find_column(frame: pd.DataFrame, candidates) -> str | None:
    normalized = {
        _normalized_column_name(column): column for column in frame.columns
    }
    for candidate in candidates:
        key = _normalized_column_name(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = _normalized_column_name(candidate)
        for normalized_name, column in normalized.items():
            if normalized_name.startswith(key):
                return column
    return None


def _xlsx_cell_value(cell, shared_strings):
    """Return one XLSX cell value without loading the complete worksheet."""
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(cell.itertext())
    value = None
    for child in cell:
        if child.tag.rsplit("}", 1)[-1] == "v":
            value = child.text
            break
    if value is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (IndexError, TypeError, ValueError):
            return None
    if cell_type == "b":
        return value == "1"
    return value


def _read_tobii_fixation_xlsx(table_path: Path) -> pd.DataFrame:
    """Stream only fixation-related columns from a large Tobii XLSX export.

    Tobii exports can contain eye-tracker and IMU records in one worksheet. A
    20 MB compressed file may expand to several hundred MB, so pandas/openpyxl
    loading of every exported column is unnecessarily slow and memory-heavy.
    """
    wanted_prefixes = {
        "tunixns",
        "recordingtimestamp",
        "sensor",
        "eyemovementtype",
        "eyemovementtypeindex",
        "eyemovementeventduration",
        "fixationpointx",
        "fixationpointy",
        "validityleft",
        "validityright",
    }
    with zipfile.ZipFile(table_path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                root = ElementTree.parse(stream).getroot()
            shared_strings = ["".join(item.itertext()) for item in root]
        worksheets = sorted(
            name for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheets:
            raise RuntimeError(f"No worksheet found in {table_path}")

        headers = None
        selected_indices = []
        selected_headers = []
        columns = None
        with archive.open(worksheets[0]) as stream:
            for _, row in ElementTree.iterparse(stream, events=("end",)):
                if row.tag.rsplit("}", 1)[-1] != "row":
                    continue
                cells = [
                    cell for cell in row
                    if cell.tag.rsplit("}", 1)[-1] == "c"
                ]
                if headers is None:
                    headers = [
                        _xlsx_cell_value(cell, shared_strings) for cell in cells
                    ]
                    for index, header in enumerate(headers):
                        normalized = _normalized_column_name(header)
                        if any(normalized.startswith(prefix) for prefix in wanted_prefixes):
                            selected_indices.append(index)
                            selected_headers.append(str(header))
                    columns = {header: [] for header in selected_headers}
                else:
                    for index, header in zip(selected_indices, selected_headers):
                        value = (
                            _xlsx_cell_value(cells[index], shared_strings)
                            if index < len(cells) else None
                        )
                        columns[header].append(value)
                row.clear()
    if headers is None or columns is None:
        raise RuntimeError(f"Empty worksheet in {table_path}")
    return pd.DataFrame(columns)


def _read_tobii_export_table(table_path: Path) -> pd.DataFrame:
    table_path = Path(table_path).resolve()
    if table_path.suffix.lower() in {".xlsx", ".xls", ".xlsm"}:
        if table_path.suffix.lower() == ".xlsx":
            return _read_tobii_fixation_xlsx(table_path)
        return pd.read_excel(table_path)
    return pd.read_csv(table_path, low_memory=False)


def _tobii_export_timestamp_ns(
    frame: pd.DataFrame,
    recording_created: str,
) -> tuple[pd.Series, str]:
    """Return absolute nanoseconds for a Tobii Pro Lab export."""
    unix_column = _find_column(frame, ["t_unix_ns"])
    recording_column = _find_column(
        frame,
        [
            "Recording timestamp [μs]",
            "Recording timestamp [µs]",
            "Recording timestamp [us]",
            "Recording timestamp",
        ],
    )
    if unix_column is not None:
        return pd.to_numeric(frame[unix_column], errors="coerce"), unix_column
    if recording_column is None:
        raise RuntimeError(
            "Tobii export has no t_unix_ns or Recording timestamp"
        )
    created = datetime.fromisoformat(recording_created.replace("Z", "+00:00"))
    created_ns = int(round(created.timestamp() * NS_PER_SECOND))
    recording_value = pd.to_numeric(frame[recording_column], errors="coerce")
    normalized_name = _normalized_column_name(recording_column)
    if normalized_name.endswith("ns"):
        factor = 1
    elif normalized_name.endswith("ms"):
        factor = 1_000_000
    else:
        factor = 1_000
    return (
        created_ns + recording_value * factor,
        f"recording.g3.created + {recording_column} * {factor} ns",
    )


def _tobii_fixation_intervals_from_frame(
    frame: pd.DataFrame,
    table_path: Path,
    recording_created: str,
) -> tuple[pd.DataFrame, dict]:

    type_column = _find_column(frame, ["Eye movement type"])
    x_column = _find_column(frame, ["Fixation point X"])
    y_column = _find_column(frame, ["Fixation point Y"])
    event_column = _find_column(frame, ["Eye movement type index"])
    duration_column = _find_column(frame, ["Eye movement event duration"])
    sensor_column = _find_column(frame, ["Sensor"])
    required = {
        "Eye movement type": type_column,
        "Fixation point X": x_column,
        "Fixation point Y": y_column,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise RuntimeError(
            f"Fixation export {table_path} is missing: {', '.join(missing)}"
        )

    timestamp_ns, timestamp_source = _tobii_export_timestamp_ns(
        frame, recording_created
    )

    fixation_mask = (
        frame[type_column].astype(str).str.strip().str.casefold().eq("fixation")
    )
    if sensor_column is not None:
        fixation_mask &= frame[sensor_column].astype(str).str.strip().str.casefold().eq(
            "eye tracker"
        )
    fixation_x = pd.to_numeric(frame[x_column], errors="coerce")
    fixation_y = pd.to_numeric(frame[y_column], errors="coerce")
    duration_values = (
        pd.to_numeric(frame[duration_column], errors="coerce")
        if duration_column is not None
        else pd.Series(np.nan, index=frame.index)
    )
    fixation_mask &= fixation_x.notna() & fixation_y.notna()
    fixation_mask &= pd.Series(timestamp_ns, index=frame.index).notna()

    selected = pd.DataFrame({
        "t_unix_ns": pd.Series(timestamp_ns, index=frame.index),
        "fixation_x": fixation_x,
        "fixation_y": fixation_y,
        "duration_ms": duration_values,
    }).loc[fixation_mask].copy()
    selected["t_unix_ns"] = selected["t_unix_ns"].round().astype("int64")
    if event_column is not None:
        selected["event_id"] = frame.loc[fixation_mask, event_column].astype(str).to_numpy()
    else:
        time = selected["t_unix_ns"].sort_values().to_numpy(dtype=np.int64)
        intervals = np.diff(time).astype(float) / NS_PER_SECOND
        positive = intervals[intervals > 0]
        gap = 2.5 * float(np.median(positive)) if len(positive) else 0.05
        ordered = selected.sort_values("t_unix_ns").copy()
        ordered["event_id"] = (
            ordered["t_unix_ns"].diff().fillna(0).gt(gap * NS_PER_SECOND).cumsum()
        ).astype(str)
        selected = ordered
    intervals = []
    for event_id, group in selected.groupby("event_id", sort=False):
        start_ns = int(group["t_unix_ns"].min())
        event_times = np.sort(group["t_unix_ns"].to_numpy(dtype=np.int64))
        event_steps = np.diff(event_times)
        event_steps = event_steps[event_steps > 0]
        tail_ns = int(np.median(event_steps)) if len(event_steps) else 20_000_000
        end_ns = int(group["t_unix_ns"].max()) + tail_ns
        duration_ms = pd.to_numeric(group["duration_ms"], errors="coerce").dropna()
        if len(duration_ms):
            exported_end = start_ns + int(round(float(duration_ms.median()) * 1.0e6))
            end_ns = max(end_ns, exported_end)
        intervals.append({
            "event_id": str(event_id),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_s": (end_ns - start_ns) / NS_PER_SECOND,
            "export_rows": int(len(group)),
        })

    interval_frame = pd.DataFrame(intervals)
    summary = {
        "source": str(table_path),
        "source_row_count": int(len(frame)),
        "fixation_row_count": int(len(selected)),
        "fixation_event_count": int(len(interval_frame)),
        "timestamp_source": timestamp_source,
        "selection": (
            "Tobii export Eye movement type == Fixation, finite Fixation point "
            "X/Y, grouped by Eye movement type index"
        ),
    }
    return interval_frame, summary


def _tobii_eye_validity_from_frame(
    frame: pd.DataFrame,
    table_path: Path,
    recording_created: str,
) -> tuple[pd.DataFrame, dict]:
    """Return the exported left/right Tobii validity flags at gaze timestamps."""
    left_column = _find_column(frame, ["Validity left"])
    right_column = _find_column(frame, ["Validity right"])
    sensor_column = _find_column(frame, ["Sensor"])
    if left_column is None or right_column is None:
        raise RuntimeError(
            f"Tobii export {table_path} has no left/right validity columns"
        )
    timestamp_ns, timestamp_source = _tobii_export_timestamp_ns(
        frame, recording_created
    )
    eye_mask = pd.Series(timestamp_ns, index=frame.index).notna()
    if sensor_column is not None:
        eye_mask &= frame[sensor_column].astype(str).str.strip().str.casefold().eq(
            "eye tracker"
        )
    validity = pd.DataFrame({
        "t_unix_ns": pd.Series(timestamp_ns, index=frame.index),
        "left_validity_text": frame[left_column].astype(str).str.strip(),
        "right_validity_text": frame[right_column].astype(str).str.strip(),
    }).loc[eye_mask].copy()
    validity["t_unix_ns"] = validity["t_unix_ns"].round().astype("int64")
    validity["left_eye_valid"] = validity["left_validity_text"].str.casefold().eq(
        "valid"
    )
    validity["right_eye_valid"] = validity["right_validity_text"].str.casefold().eq(
        "valid"
    )
    validity = validity.sort_values("t_unix_ns").drop_duplicates(
        "t_unix_ns", keep="last"
    ).reset_index(drop=True)
    summary = {
        "source": str(Path(table_path).resolve()),
        "eye_tracker_row_count": int(len(validity)),
        "left_valid_count": int(validity["left_eye_valid"].sum()),
        "right_valid_count": int(validity["right_eye_valid"].sum()),
        "both_valid_count": int(
            (validity["left_eye_valid"] & validity["right_eye_valid"]).sum()
        ),
        "timestamp_source": timestamp_source,
        "selection": "Sensor == Eye Tracker; validity text == Valid",
    }
    return validity, summary


def read_tobii_fixation_and_eye_validity(
    table_path: Path,
    recording_created: str,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Read fixation events and eye validity flags in one XLSX pass."""
    table_path = Path(table_path).resolve()
    frame = _read_tobii_export_table(table_path)
    fixation_intervals, fixation_summary = _tobii_fixation_intervals_from_frame(
        frame, table_path, recording_created
    )
    eye_validity, validity_summary = _tobii_eye_validity_from_frame(
        frame, table_path, recording_created
    )
    return fixation_intervals, fixation_summary, eye_validity, validity_summary


def read_tobii_fixation_intervals(
    table_path: Path,
    recording_created: str,
) -> tuple[pd.DataFrame, dict]:
    """Read Tobii-classified fixation events from a separate export table.

    The function does not run a new fixation detector. It selects rows already
    labelled ``Fixation`` by the Tobii export, requires finite fixation-point
    coordinates, and groups rows by the exported eye-movement event index.
    """
    table_path = Path(table_path).resolve()
    frame = _read_tobii_export_table(table_path)
    return _tobii_fixation_intervals_from_frame(
        frame, table_path, recording_created
    )


def valid_timestamps(frame: pd.DataFrame, time_column: str, valid=None) -> np.ndarray:
    timestamps = pd.to_numeric(frame[time_column], errors="coerce")
    mask = timestamps.notna() & (timestamps > 0)
    if valid is not None:
        mask &= pd.Series(valid, index=frame.index).fillna(False).astype(bool)
    result = timestamps.loc[mask].sort_values().drop_duplicates().to_numpy(dtype=np.int64)
    return result


def sampling_statistics(name: str, source: str, timestamps: np.ndarray, valid_fraction: float):
    timestamps = np.asarray(timestamps, dtype=np.int64)
    intervals_ms = np.diff(timestamps).astype(float) / 1.0e6
    intervals_ms = intervals_ms[np.isfinite(intervals_ms) & (intervals_ms > 0)]
    duration_s = (
        float((timestamps[-1] - timestamps[0]) / NS_PER_SECOND)
        if len(timestamps) >= 2 else 0.0
    )
    row = {
        "stream": name,
        "source": source,
        "n_timestamps": int(len(timestamps)),
        "valid_fraction": valid_fraction,
        "duration_s": duration_s,
        "effective_rate_hz": (
            float((len(timestamps) - 1) / duration_s)
            if len(timestamps) >= 2 and duration_s > 0 else None
        ),
        "median_interval_ms": float(np.median(intervals_ms)) if len(intervals_ms) else None,
        "q25_interval_ms": float(np.quantile(intervals_ms, 0.25)) if len(intervals_ms) else None,
        "q75_interval_ms": float(np.quantile(intervals_ms, 0.75)) if len(intervals_ms) else None,
        "p95_interval_ms": float(np.quantile(intervals_ms, 0.95)) if len(intervals_ms) else None,
        "p99_interval_ms": float(np.quantile(intervals_ms, 0.99)) if len(intervals_ms) else None,
        "maximum_interval_ms": float(np.max(intervals_ms)) if len(intervals_ms) else None,
    }
    return row, intervals_ms


def build_sampling_streams(
    bag_data,
    imu: pd.DataFrame,
    steering: pd.DataFrame,
    brake: pd.DataFrame,
    wheel: pd.DataFrame,
    power: pd.DataFrame,
    camera: pd.DataFrame,
    gaze: pd.DataFrame,
    tobii_imu: pd.DataFrame,
    lidar_streams=None,
):
    raw_imu = imu[imu["dtype"].eq(64)].copy()
    ahrs_imu = imu[imu["dtype"].eq(65)].copy()
    frame_checks = ["crc8_ok", "crc16_ok", "end_ok"]
    raw_valid = raw_imu[frame_checks].apply(pd.to_numeric, errors="coerce").eq(1).all(axis=1)
    ahrs_valid = ahrs_imu[frame_checks].apply(pd.to_numeric, errors="coerce").eq(1).all(axis=1)
    wheel_value_valid = pd.to_numeric(wheel["speed_mps"], errors="coerce").notna()
    steering_valid = (
        pd.to_numeric(steering["ok"], errors="coerce").eq(1)
        & pd.to_numeric(steering["angle_deg"], errors="coerce").notna()
    )
    brake_valid = (
        pd.to_numeric(brake["ok_left"], errors="coerce").eq(1)
        & pd.to_numeric(brake["ok_right"], errors="coerce").eq(1)
    )
    p10 = power[power["page_name"].eq("standard_power")].copy()
    p12 = power[power["page_name"].eq("standard_torque")].copy()

    definitions = [
        (
            "GNSS position (/fix)", "/fix header.stamp",
            valid_timestamps(bag_data["fix"], "t_ns"),
            float((bag_data["fix"]["status"] >= 0).mean()),
        ),
        (
            "GNSS velocity (/ubx_nav_vel_ned)", "/ubx_nav_vel_ned header.stamp",
            valid_timestamps(bag_data["vel"], "t_ns"),
            float(pd.to_numeric(bag_data["vel"]["ground_speed_mps"], errors="coerce").notna().mean()),
        ),
        (
            "Bike IMU raw (dtype 64)", "imu CSV t_unix_ns",
            valid_timestamps(raw_imu, "t_unix_ns"), float(raw_valid.mean()),
        ),
        (
            "Bike IMU AHRS (dtype 65)", "imu CSV t_unix_ns",
            valid_timestamps(ahrs_imu, "t_unix_ns"), float(ahrs_valid.mean()),
        ),
        (
            "Tobii gaze", "gazedata.gz timestamp",
            valid_timestamps(gaze, "t_unix_ns"), float(gaze["valid"].mean()),
        ),
        (
            "Tobii IMU", "imudata.gz timestamp",
            valid_timestamps(tobii_imu, "t_unix_ns"), float(tobii_imu["valid"].mean()),
        ),
        (
            "Rear camera", "camera timestamps.csv unix_ns",
            valid_timestamps(camera, "unix_ns"), 1.0,
        ),
        (
            "Steering", "steering CSV t_unix_ns",
            valid_timestamps(steering, "t_unix_ns"), float(steering_valid.mean()),
        ),
        (
            "Brake sensors", "brake CSV t_unix_ns",
            valid_timestamps(brake, "t_unix_ns"), float(brake_valid.mean()),
        ),
        (
            "Wheel-speed messages", "speed CSV t_unix_ns",
            valid_timestamps(wheel, "t_unix_ns"), float(wheel_value_valid.mean()),
        ),
        (
            "Power meter: standard power", "Rally page 0x10 t_unix_ns",
            valid_timestamps(p10, "t_unix_ns"),
            float(pd.to_numeric(p10["p10_instantaneous_power_w"], errors="coerce").notna().mean()),
        ),
        (
            "Power meter: standard torque", "Rally page 0x12 t_unix_ns",
            valid_timestamps(p12, "t_unix_ns"),
            float(pd.to_numeric(p12["p12_accumulated_torque_1_32nm"], errors="coerce").notna().mean()),
        ),
    ]
    for name, timestamps in (lidar_streams or {}).items():
        definitions.append((
            name,
            "PCAP MSOP azimuth-wrap scan timestamps",
            np.asarray(timestamps, dtype=np.int64),
            1.0,
        ))

    rows = []
    intervals = {}
    for name, source, timestamps, fraction in definitions:
        row, values = sampling_statistics(name, source, timestamps, fraction)
        if len(values):
            rows.append(row)
            intervals[name] = values
    return pd.DataFrame(rows), intervals


def plot_sampling_intervals(summary: pd.DataFrame, intervals: dict, output_base: Path):
    order = summary.sort_values("median_interval_ms")["stream"].tolist()
    values = [intervals[name] for name in order]
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    boxes = axis.boxplot(
        values,
        vert=False,
        labels=order,
        showfliers=False,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": COLORS["black"], "linewidth": 1.0},
        whiskerprops={"color": COLORS["blue"], "linewidth": 0.8},
        capprops={"color": COLORS["blue"], "linewidth": 0.8},
        boxprops={"edgecolor": COLORS["blue"], "linewidth": 0.8},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(COLORS["sky"])
        box.set_alpha(0.45)
    axis.set_xscale("log")
    axis.set_xlabel("Inter-message interval, $\\Delta t$ (ms; logarithmic scale)")
    axis.set_title("Sampling interval distributions across sensor streams")
    axis.grid(True, axis="x")
    fig.tight_layout()
    save_figure(fig, output_base)


def crop(frame: pd.DataFrame, start_ns: int, end_ns: int, column="t_unix_ns"):
    time = pd.to_numeric(frame[column], errors="coerce")
    return frame[(time >= start_ns) & (time <= end_ns)].copy().reset_index(drop=True)


def normalized_score(series: pd.Series) -> pd.Series:
    lower = float(series.quantile(0.05))
    upper = float(series.quantile(0.95))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        return pd.Series(0.0, index=series.index)
    return ((series - lower) / (upper - lower)).clip(0.0, 1.0)


def select_representative_window(
    velocity: pd.DataFrame,
    wheel_valid: pd.DataFrame,
    steering: pd.DataFrame,
    brake: pd.DataFrame,
    power_p10: pd.DataFrame,
    duration_s: float,
    step_s: float,
):
    start_ns = max(
        int(velocity["t_unix_ns"].min()),
        int(wheel_valid["t_unix_ns"].min()),
        int(steering["t_unix_ns"].min()),
        int(brake["t_unix_ns"].min()),
        int(power_p10["t_unix_ns"].min()),
    )
    end_ns = min(
        int(velocity["t_unix_ns"].max()),
        int(wheel_valid["t_unix_ns"].max()),
        int(steering["t_unix_ns"].max()),
        int(brake["t_unix_ns"].max()),
        int(power_p10["t_unix_ns"].max()),
    )
    duration_ns = int(round(duration_s * NS_PER_SECOND))
    step_ns = int(round(step_s * NS_PER_SECOND))
    rows = []
    for candidate_start in np.arange(
        start_ns, end_ns - duration_ns + 1, step_ns, dtype=np.int64
    ):
        candidate_end = int(candidate_start + duration_ns)
        gnss_part = crop(velocity, int(candidate_start), candidate_end)
        wheel_part = crop(wheel_valid, int(candidate_start), candidate_end)
        steering_part = crop(steering, int(candidate_start), candidate_end)
        brake_part = crop(brake, int(candidate_start), candidate_end)
        power_part = crop(power_p10, int(candidate_start), candidate_end)
        if len(gnss_part) < 30 or len(wheel_part) < 10 or len(steering_part) < 10:
            continue
        speed = pd.to_numeric(gnss_part["ground_speed_mps"], errors="coerce")
        left = pd.to_numeric(brake_part["left_force_n"], errors="coerce")
        right = pd.to_numeric(brake_part["right_force_n"], errors="coerce")
        angle = pd.to_numeric(steering_part["angle_deg"], errors="coerce")
        rows.append({
            "start_ns": int(candidate_start),
            "end_ns": candidate_end,
            "duration_s": duration_s,
            "gnss_speed_p05_p95_range_mps": float(speed.quantile(0.95) - speed.quantile(0.05)),
            "brake_force_p99_n": float(max(left.quantile(0.99), right.quantile(0.99))),
            "steering_p05_p95_range_deg": float(
                angle.quantile(0.95) - angle.quantile(0.05)
            ),
            "standard_power_event_count": int(len(power_part)),
            "wheel_valid_value_count": int(len(wheel_part)),
            "gnss_value_count": int(len(gnss_part)),
        })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No representative window satisfies the data-availability gates")
    candidates["selection_score"] = (
        normalized_score(candidates["gnss_speed_p05_p95_range_mps"])
        + normalized_score(candidates["brake_force_p99_n"])
        + normalized_score(candidates["steering_p05_p95_range_deg"])
        + normalized_score(candidates["standard_power_event_count"].astype(float))
    )
    selected_index = int(candidates["selection_score"].idxmax())
    candidates["selected"] = False
    candidates.loc[selected_index, "selected"] = True
    selected = candidates.loc[selected_index].to_dict()
    return selected, candidates.sort_values("start_ns").reset_index(drop=True)


def add_gap_aware_line(axis, time_s, values, *, gap_factor=2.5, **kwargs):
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    # Keep invalid values as NaNs so Matplotlib breaks the line at every
    # rejected sample. Only rows without a usable timestamp are removed.
    valid_time = np.isfinite(time_s)
    time_s = time_s[valid_time]
    values = values[valid_time]
    if not len(time_s):
        return None
    order = np.argsort(time_s)
    time_s = time_s[order]
    values = values[order]
    intervals = np.diff(time_s)
    positive = intervals[np.isfinite(intervals) & (intervals > 0)]
    if len(positive):
        gap_limit = gap_factor * float(np.median(positive))
        breaks = np.flatnonzero(intervals > gap_limit) + 1
        time_s = np.insert(time_s, breaks, np.nan)
        values = np.insert(values, breaks, np.nan)
    return axis.plot(time_s, values, **kwargs)[0]


def prepare_speed_comparison(
    velocity: pd.DataFrame,
    wheel_valid: pd.DataFrame,
    start_ns: int,
    end_ns: int,
    tolerance_s: float,
):
    gnss = crop(velocity, start_ns, end_ns)[
        ["t_unix_ns", "ground_speed_mps"]
    ].sort_values("t_unix_ns")
    wheel = crop(wheel_valid, start_ns, end_ns)[
        ["t_unix_ns", "wheel_speed_mps"]
    ].sort_values("t_unix_ns")
    paired = pd.merge_asof(
        wheel,
        gnss,
        on="t_unix_ns",
        direction="nearest",
        tolerance=int(round(tolerance_s * NS_PER_SECOND)),
    ).dropna()
    paired["wheel_minus_gnss_mps"] = (
        paired["wheel_speed_mps"] - paired["ground_speed_mps"]
    )
    error = paired["wheel_minus_gnss_mps"]
    summary = {
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "window_duration_s": (end_ns - start_ns) / NS_PER_SECOND,
        "wheel_unit_in_csv": "km/h despite the legacy column name speed_mps",
        "wheel_conversion": "wheel_speed_mps = CSV speed_mps / 3.6",
        "pairing_rule": f"Nearest GNSS velocity timestamp within {tolerance_s:.3f} s",
        "paired_sample_count": int(len(paired)),
        "mean_wheel_minus_gnss_mps": float(error.mean()),
        "median_wheel_minus_gnss_mps": float(error.median()),
        "mae_mps": float(error.abs().mean()),
        "rmse_mps": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_difference_mps": float(error.abs().quantile(0.95)),
        "pearson_correlation": float(
            paired[["wheel_speed_mps", "ground_speed_mps"]].corr().iloc[0, 1]
        ),
        "interpretation": "Agreement between two onboard sensors; GNSS is not independent ground truth.",
    }
    return gnss, wheel, paired, summary


def plot_speed_comparison(
    gnss: pd.DataFrame,
    wheel: pd.DataFrame,
    paired: pd.DataFrame,
    summary: dict,
    start_ns: int,
    output_base: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    gnss_time = (gnss["t_unix_ns"].to_numpy(dtype=float) - start_ns) / NS_PER_SECOND
    wheel_time = (wheel["t_unix_ns"].to_numpy(dtype=float) - start_ns) / NS_PER_SECOND
    add_gap_aware_line(
        axes[0], gnss_time, gnss["ground_speed_mps"],
        color=COLORS["blue"], linewidth=1.1, label="GNSS ground speed",
    )
    add_gap_aware_line(
        axes[0], wheel_time, wheel["wheel_speed_mps"],
        color=COLORS["orange"], linewidth=0.9, marker="o", markersize=2.2,
        label="Wheel speed",
    )
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Wheel-speed and GNSS-speed comparison")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.99, 0.98))

    paired_time = (
        paired["t_unix_ns"].to_numpy(dtype=float) - start_ns
    ) / NS_PER_SECOND
    add_gap_aware_line(
        axes[1], paired_time, paired["wheel_minus_gnss_mps"],
        color=COLORS["green"], linewidth=0.9, marker="o", markersize=2.2,
        label="Wheel minus GNSS",
    )
    axes[1].axhline(0.0, color=COLORS["black"], linewidth=0.7, linestyle="--")
    axes[1].axhline(
        summary["mean_wheel_minus_gnss_mps"], color=COLORS["green"],
        linewidth=0.8, linestyle=":", label="Mean difference",
    )
    axes[1].set_ylabel("Difference (m/s)")
    axes[1].set_xlabel("Time from comparison-window start (s)")
    axes[1].legend(loc="upper right", bbox_to_anchor=(0.99, 0.98))
    metrics = (
        f"n = {summary['paired_sample_count']}\n"
        f"MAE = {summary['mae_mps']:.2f} m/s\n"
        f"RMSE = {summary['rmse_mps']:.2f} m/s\n"
        f"r = {summary['pearson_correlation']:.3f}"
    )
    axes[1].text(
        0.52, 0.96, metrics, transform=axes[1].transAxes,
        ha="left", va="top",
        bbox={"facecolor": "white", "edgecolor": COLORS["light_grey"], "alpha": 0.9},
    )
    for axis, label in zip(axes, ["(a)", "(b)"]):
        panel_label(axis, label)
        axis.grid(True)
    fig.tight_layout()
    save_figure(fig, output_base)


def gnss_course_rate(velocity: pd.DataFrame) -> pd.DataFrame:
    frame = velocity.sort_values("t_unix_ns").copy()
    time = frame["t_unix_ns"].to_numpy(dtype=np.int64)
    course = pd.to_numeric(frame["course_deg"], errors="coerce").to_numpy(dtype=float)
    speed = pd.to_numeric(frame["ground_speed_mps"], errors="coerce").to_numpy(dtype=float)
    accuracy = pd.to_numeric(frame["course_accuracy_deg"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(time).astype(float) / NS_PER_SECOND
    change = (np.diff(course) + 180.0) % 360.0 - 180.0
    rate = np.full(len(frame), np.nan)
    valid = (
        np.isfinite(change)
        & np.isfinite(dt)
        & (dt > 0)
        & (dt <= 0.5)
        & (speed[1:] >= 2.0)
        & (accuracy[1:] <= 5.0)
    )
    rate[1:][valid] = np.deg2rad(change[valid] / dt[valid])
    return pd.DataFrame({"t_unix_ns": time, "gnss_course_rate_rad_s": rate})


def smooth_gaze(gaze: pd.DataFrame) -> pd.DataFrame:
    result = gaze.copy()
    x = pd.to_numeric(result["gaze_x_norm"], errors="coerce")
    x = x.where(x.between(0.0, 1.0))
    result["gaze_horizontal_centered"] = x.rolling(
        window=5, center=True, min_periods=3
    ).median() - 0.5
    return result


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def add_gaze_angle_methods(gaze: pd.DataFrame, extrinsics_path: Path | None = None) -> pd.DataFrame:
    """Add static-frame gaze diagnostics A/B/C and the former ego-angle formula.

    ``gaze2d`` is Tobii's normalized image coordinate, not an angle.  Method A
    uses the Tobii 3-D gaze point, B averages the two eye gaze directions, and
    C back-projects the 2-D point using the scene-camera intrinsics.  For P8,
    the recording calibration provides the Tobii HUCS -> scene-camera rotation;
    A and B use this same rotation before computing ``atan2(x, z)`` as C.  This
    puts all three methods in one right-handed camera-optical convention
    (x-right, y-down, z-forward).  The dynamic head-pose transform is not
    available in this raw P8 export, so these remain static-frame diagnostics.
    """
    result = gaze.copy()
    x_norm = pd.to_numeric(result.get("gaze_x_norm"), errors="coerce")
    y_norm = pd.to_numeric(result.get("gaze_y_norm"), errors="coerce")
    gaze_2d_finite = x_norm.notna() & y_norm.notna()
    gaze_2d_in_image = gaze_2d_finite & x_norm.between(0.0, 1.0) & y_norm.between(0.0, 1.0)
    result["gaze_2d_fields_valid"] = gaze_2d_finite
    result["gaze_2d_image_valid"] = gaze_2d_in_image
    result["gaze_x_relative_norm"] = (x_norm - 0.5).where(gaze_2d_in_image)
    # Former calculation retained for audit: treating the raw Tobii 3-D point
    # as an x-forward/y-left ego vector without a frame or front-facing gate.
    x3 = pd.to_numeric(result.get("gaze3d_x_mm"), errors="coerce")
    y3 = pd.to_numeric(result.get("gaze3d_y_mm"), errors="coerce")
    result["former_ego_angle_deg"] = np.degrees(np.arctan2(y3, x3))
    lx = pd.to_numeric(result.get("left_direction_x"), errors="coerce")
    ly = pd.to_numeric(result.get("left_direction_y"), errors="coerce")
    lz = pd.to_numeric(result.get("left_direction_z"), errors="coerce")
    rx = pd.to_numeric(result.get("right_direction_x"), errors="coerce")
    ry = pd.to_numeric(result.get("right_direction_y"), errors="coerce")
    rz = pd.to_numeric(result.get("right_direction_z"), errors="coerce")
    dx = (lx + rx) / 2.0
    dy = (ly + ry) / 2.0
    dz = (lz + rz) / 2.0
    # The former implementation treated the Tobii HUCS x axis as the bicycle
    # forward axis.  That is not valid for this recording.  When the P8
    # recording calibration is present, transform HUCS vectors to the scene
    # camera frame and use the same horizontal-angle definition as the 2-D ray.

    # P8 camera calibration from the recording metadata.  The optical-frame
    # rotation is changed only through a new P8 extrinsics file; the original
    # calibration remains untouched.
    fx, fy = 916.0458, 915.7009
    cx, cy = 954.0568, 514.4645
    u = x_norm * 1920.0
    v = y_norm * 1080.0
    ray_optical = np.column_stack(((u - cx) / fx, (v - cy) / fy, np.ones(len(result))))
    centre_ray_angle = math.atan2(0.5 * 1920.0 - cx, fx)
    result["gaze_2d_relative_angle_deg"] = np.degrees(
        np.arctan2(u - cx, fx) - centre_ray_angle
    )
    result.loc[~gaze_2d_in_image, "gaze_2d_relative_angle_deg"] = np.nan
    hucs_to_scene = None
    angle_frame = "base_link"
    optical_to_link = _rpy_matrix(-math.pi / 2.0, 0.0, -math.pi / 2.0).T
    link_to_base = np.eye(3)
    if extrinsics_path is not None and Path(extrinsics_path).is_file():
        try:
            data = json.loads(Path(extrinsics_path).read_text(encoding="utf-8"))
            if "tobii_hucs_to_camera_optical_rotation" in data:
                hucs_to_scene = np.asarray(
                    data["tobii_hucs_to_camera_optical_rotation"], dtype=float
                ).reshape(3, 3)
                angle_frame = "scene_camera_optical"
            rpy = data["transforms"]["camera_link_to_camera_optical_frame"]["rpy_rad"]
            optical_to_link = _rpy_matrix(*rpy)
            link_to_base = _rpy_matrix(*data["transforms"]["base_link_to_camera_link"]["rpy_rad"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    gaze3d_optical = np.column_stack((x3.to_numpy(float), y3.to_numpy(float), pd.to_numeric(result.get("gaze3d_z_mm"), errors="coerce").to_numpy(float)))
    if hucs_to_scene is not None:
        gaze3d_frame = (hucs_to_scene @ gaze3d_optical.T).T
    else:
        gaze3d_frame = (link_to_base @ optical_to_link @ gaze3d_optical.T).T
    directions_optical = np.column_stack((dx.to_numpy(float), dy.to_numpy(float), dz.to_numpy(float)))
    if hucs_to_scene is not None:
        directions_frame = (hucs_to_scene @ directions_optical.T).T
        ray_frame = ray_optical
        result["method_a_ego_angle_deg"] = np.degrees(
            np.arctan2(gaze3d_frame[:, 0], gaze3d_frame[:, 2])
        )
        result["method_b_ego_angle_deg"] = np.degrees(
            np.arctan2(directions_frame[:, 0], directions_frame[:, 2])
        )
        result["method_c_ego_angle_deg"] = np.degrees(
            np.arctan2(ray_frame[:, 0], ray_frame[:, 2])
        )
        method_a_fields = np.isfinite(gaze3d_frame).all(axis=1)
        method_b_fields = np.isfinite(directions_frame).all(axis=1)
        method_a_forward = method_a_fields & (gaze3d_frame[:, 2] > 1.0e-6)
        method_b_forward = method_b_fields & (directions_frame[:, 2] > 1.0e-6)
        method_c_forward = gaze_2d_in_image.to_numpy(dtype=bool) & (ray_frame[:, 2] > 1.0e-6)
        result.loc[~method_a_forward, "method_a_ego_angle_deg"] = np.nan
        result.loc[~method_b_forward, "method_b_ego_angle_deg"] = np.nan
        result.loc[~method_c_forward, "method_c_ego_angle_deg"] = np.nan
    else:
        directions_base = (link_to_base @ optical_to_link @ directions_optical.T).T
        result["method_a_ego_angle_deg"] = np.degrees(np.arctan2(gaze3d_frame[:, 1], gaze3d_frame[:, 0]))
        result["method_b_ego_angle_deg"] = np.degrees(np.arctan2(directions_base[:, 1], directions_base[:, 0]))
        ray_base = (link_to_base @ optical_to_link @ ray_optical.T).T
        result["method_c_ego_angle_deg"] = np.degrees(np.arctan2(ray_base[:, 1], ray_base[:, 0]))
        method_a_fields = np.isfinite(gaze3d_frame).all(axis=1)
        method_b_fields = np.isfinite(directions_base).all(axis=1)
        method_a_forward = method_a_fields & (gaze3d_frame[:, 0] > 1.0e-6)
        method_b_forward = method_b_fields & (directions_base[:, 0] > 1.0e-6)
        method_c_forward = gaze_2d_in_image.to_numpy(dtype=bool) & (ray_base[:, 0] > 1.0e-6)
        result.loc[~method_a_forward, "method_a_ego_angle_deg"] = np.nan
        result.loc[~method_b_forward, "method_b_ego_angle_deg"] = np.nan
        result.loc[~method_c_forward, "method_c_ego_angle_deg"] = np.nan
    result["method_a_fields_valid"] = method_a_fields
    result["method_b_fields_valid"] = method_b_fields
    result["method_c_fields_valid"] = gaze_2d_finite
    result["method_a_forward_valid"] = method_a_forward
    result["method_b_forward_valid"] = method_b_forward
    result["method_c_forward_valid"] = method_c_forward
    result["method_a_valid"] = pd.to_numeric(
        result["method_a_ego_angle_deg"], errors="coerce"
    ).notna()
    result["method_b_valid"] = pd.to_numeric(
        result["method_b_ego_angle_deg"], errors="coerce"
    ).notna()
    result["method_c_valid"] = pd.to_numeric(
        result["method_c_ego_angle_deg"], errors="coerce"
    ).notna()
    result.attrs["gaze_angle_frame"] = angle_frame
    return result


def plot_closed_loop(
    gaze: pd.DataFrame,
    steering: pd.DataFrame,
    brake: pd.DataFrame,
    power_p10: pd.DataFrame,
    raw_imu: pd.DataFrame,
    course_rate: pd.DataFrame,
    gnss_speed: pd.DataFrame,
    wheel_speed: pd.DataFrame,
    start_ns: int,
    end_ns: int,
    output_base: Path,
    extrinsics_path: Path | None = None,
    steering_neutral: dict | None = None,
    brake_bands: dict[str, tuple[float, float]] | None = None,
    fixation_intervals: pd.DataFrame | None = None,
    fixation_label: str = "Tobii-classified fixation",
    precomputed_gaze_angles: pd.DataFrame | None = None,
    gaze_axis_label: str = "Static camera-frame\ngaze angle (deg)",
    trajectory: pd.DataFrame | None = None,
    trajectory_event_label: str | None = None,
    trajectory_event_time_ns: int | None = None,
    trajectory_layout: str = "spatial",
    figure_title: str | None = None,
):
    if trajectory_layout not in {"spatial", "components", "components_last"}:
        raise ValueError(
            "trajectory_layout must be 'spatial', 'components', or "
            "'components_last'"
        )
    if precomputed_gaze_angles is None:
        gaze_part = add_gaze_angle_methods(
            smooth_gaze(crop(gaze, start_ns, end_ns)), extrinsics_path
        )
    else:
        gaze_part = crop(precomputed_gaze_angles, start_ns, end_ns)
    steering_part = crop(steering, start_ns, end_ns)
    brake_part = crop(brake, start_ns, end_ns)
    power_part = crop(power_p10, start_ns, end_ns)
    imu_part = crop(raw_imu, start_ns, end_ns)
    course_part = crop(course_rate, start_ns, end_ns)
    gnss_part = crop(gnss_speed, start_ns, end_ns)
    wheel_part = crop(wheel_speed, start_ns, end_ns)

    has_trajectory = (
        trajectory is not None
        and len(trajectory) >= 2
        and {"t_unix_ns", "east_m", "north_m"}.issubset(trajectory.columns)
    )
    trajectory_axes = []
    if has_trajectory and trajectory_layout == "spatial":
        fig = plt.figure(figsize=(7.2, 11.4))
        grid = fig.add_gridspec(
            7, 1, height_ratios=[1.55, 1, 1, 1, 1, 1, 1]
        )
        trajectory_axis = fig.add_subplot(grid[0])
        axes = [fig.add_subplot(grid[1])]
        axes.extend(fig.add_subplot(grid[index], sharex=axes[0]) for index in range(2, 7))
    elif has_trajectory and trajectory_layout == "components":
        fig = plt.figure(figsize=(7.2, 12.0))
        grid = fig.add_gridspec(
            8, 1, height_ratios=[0.8, 0.8, 1, 1, 1, 1, 1, 1]
        )
        trajectory_axes = [fig.add_subplot(grid[0])]
        trajectory_axes.append(
            fig.add_subplot(grid[1], sharex=trajectory_axes[0])
        )
        axes = [
            fig.add_subplot(grid[2], sharex=trajectory_axes[0])
        ]
        axes.extend(
            fig.add_subplot(grid[index], sharex=trajectory_axes[0])
            for index in range(3, 8)
        )
        trajectory_axis = None
    elif has_trajectory:
        fig = plt.figure(figsize=(7.2, 12.0))
        grid = fig.add_gridspec(
            8, 1, height_ratios=[1, 1, 1, 1, 1, 1, 0.8, 0.8]
        )
        axes = [fig.add_subplot(grid[0])]
        axes.extend(
            fig.add_subplot(grid[index], sharex=axes[0])
            for index in range(1, 6)
        )
        trajectory_axes = [
            fig.add_subplot(grid[6], sharex=axes[0])
        ]
        trajectory_axes.append(
            fig.add_subplot(grid[7], sharex=axes[0])
        )
        trajectory_axis = None
    else:
        fig, axes_array = plt.subplots(6, 1, figsize=(7.2, 9.6), sharex=True)
        axes = list(axes_array)
        trajectory_axis = None
    relative = lambda values: (
        np.asarray(values, dtype=float) - start_ns
    ) / NS_PER_SECOND

    trajectory_event_time_s = None
    if has_trajectory:
        route = crop(trajectory, start_ns, end_ns).copy()
        route["east_m"] = pd.to_numeric(route["east_m"], errors="coerce")
        route["north_m"] = pd.to_numeric(route["north_m"], errors="coerce")
        route = route.dropna(subset=["east_m", "north_m"]).sort_values(
            "t_unix_ns"
        )
        route_east = route["east_m"].to_numpy(dtype=float)
        route_north = route["north_m"].to_numpy(dtype=float)
        route_east -= route_east[0]
        route_north -= route_north[0]
        route_time = relative(route["t_unix_ns"])
        duration_s = (end_ns - start_ns) / NS_PER_SECOND
        route_lines = None
        if trajectory_layout == "spatial":
            points = np.column_stack((route_east, route_north))
            segments = np.stack((points[:-1], points[1:]), axis=1)
            time_norm = Normalize(vmin=0.0, vmax=duration_s)
            route_lines = LineCollection(
                segments,
                cmap="cividis",
                norm=time_norm,
                linewidths=2.0,
                zorder=2,
            )
            route_lines.set_array(
                0.5 * (route_time[:-1] + route_time[1:])
            )
            trajectory_axis.add_collection(route_lines)
            trajectory_axis.scatter(
                route_east[0], route_north[0], marker="o", s=34,
                facecolor="white", edgecolor=COLORS["black"], linewidth=1.0,
                zorder=5, label="Start",
            )
            trajectory_axis.scatter(
                route_east[-1], route_north[-1], marker="s", s=32,
                facecolor=COLORS["black"], edgecolor=COLORS["black"],
                linewidth=0.8, zorder=5, label="End",
            )
            for target_s in np.arange(10.0, duration_s, 10.0):
                nearest = int(np.argmin(np.abs(route_time - target_s)))
                trajectory_axis.scatter(
                    route_east[nearest], route_north[nearest],
                    marker="o", s=15,
                    facecolor=plt.get_cmap("cividis")(
                        time_norm(route_time[nearest])
                    ),
                    edgecolor="white", linewidth=0.5, zorder=4,
                )
                trajectory_axis.annotate(
                    f"{target_s:.0f} s",
                    (route_east[nearest], route_north[nearest]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.5, color=COLORS["black"],
                )
        event_ns = None
        if trajectory_event_label:
            if trajectory_event_time_ns is not None:
                event_ns = int(trajectory_event_time_ns)
                if not start_ns <= event_ns <= end_ns:
                    raise ValueError(
                        "trajectory_event_time_ns is outside the plot window"
                    )
            elif "method_a_ego_angle_deg" in gaze_part:
                method_a = pd.to_numeric(
                    gaze_part["method_a_ego_angle_deg"], errors="coerce"
                )
                if method_a.notna().any():
                    event_index = method_a.abs().idxmax()
                    event_ns = int(gaze_part.loc[event_index, "t_unix_ns"])
            if event_ns is not None:
                trajectory_event_time_s = (event_ns - start_ns) / NS_PER_SECOND
                route_index = int(
                    np.argmin(
                        np.abs(
                            route["t_unix_ns"].to_numpy(dtype=np.int64)
                            - event_ns
                        )
                    )
                )
                if trajectory_layout == "spatial":
                    trajectory_axis.scatter(
                        route_east[route_index], route_north[route_index],
                        marker="*", s=82, facecolor=COLORS["purple"],
                        edgecolor="white", linewidth=0.7, zorder=6,
                        label=(
                            f"{trajectory_event_label} "
                            f"({trajectory_event_time_s:.1f} s)"
                        ),
                    )
        if trajectory_layout == "spatial":
            trajectory_axis.autoscale()
            trajectory_axis.margins(0.08)
            trajectory_axis.set_aspect("equal", adjustable="datalim")
            trajectory_axis.set_xlabel("Relative east (m)")
            trajectory_axis.set_ylabel("Relative north (m)")
            trajectory_axis.set_title(
                "GNSS RTK position path during the selected interval"
            )
            trajectory_axis.grid(True)
            trajectory_axis.legend(
                loc="upper left", bbox_to_anchor=(0.055, 0.99), fontsize=7
            )
            colorbar = fig.colorbar(
                route_lines, ax=trajectory_axis, pad=0.015, fraction=0.028
            )
            colorbar.set_label("Elapsed acquisition time (s)")
        else:
            add_gap_aware_line(
                trajectory_axes[0], route_time, route_east,
                color=COLORS["blue"], linewidth=1.1,
                label="GNSS RTK position — east",
            )
            add_gap_aware_line(
                trajectory_axes[1], route_time, route_north,
                color=COLORS["orange"], linewidth=1.1,
                label="GNSS RTK position — north",
            )
            trajectory_axes[0].set_ylabel(
                "Relative east\nposition (m)"
            )
            trajectory_axes[1].set_ylabel(
                "Relative north\nposition (m)"
            )
            trajectory_axes[0].legend(loc="upper right", fontsize=7)
            trajectory_axes[1].legend(loc="upper right", fontsize=7)

    # A/B/C are calculated from the raw gaze stream. Fixation intervals, when
    # supplied, come from the separate Tobii-classified fixation export.
    angle_axis = axes[0]
    if fixation_intervals is not None and len(fixation_intervals):
        first_fixation = True
        for _, fixation in fixation_intervals.iterrows():
            fixation_start = max(int(fixation["start_ns"]), start_ns)
            fixation_end = min(int(fixation["end_ns"]), end_ns)
            if fixation_end <= fixation_start:
                continue
            angle_axis.axvspan(
                (fixation_start - start_ns) / NS_PER_SECOND,
                (fixation_end - start_ns) / NS_PER_SECOND,
                color=COLORS["sky"], alpha=0.13, linewidth=0.0, zorder=0,
                label=fixation_label if first_fixation else None,
            )
            first_fixation = False
    angle_colors = [COLORS["orange"], COLORS["purple"], COLORS["green"]]
    angle_labels = [
        ("method_a_ego_angle_deg", "Method A: 3-D gaze-point ray", "-", 0.95),
        ("method_b_ego_angle_deg", "Method B: eye-direction ray", "--", 0.75),
        ("method_c_ego_angle_deg", "Method C: 2-D back-projected ray", "-.", 0.85),
    ]
    for (column, label, linestyle, linewidth), color in zip(angle_labels, angle_colors):
        add_gap_aware_line(
            angle_axis, relative(gaze_part["t_unix_ns"]), gaze_part[column],
            color=color, linewidth=linewidth, linestyle=linestyle, label=label,
        )
    angle_axis.set_ylabel(gaze_axis_label)
    angle_axis.legend(loc="upper right", fontsize=7)

    angle = pd.to_numeric(steering_part["angle_deg"], errors="coerce")
    steering_label = "Steering angle"
    steering_ylabel = "Steering\n(deg)"
    if steering_neutral is not None:
        angle = angle - float(steering_neutral["neutral_angle_deg"])
        steering_label = "Relative steering angle"
        steering_ylabel = "Relative steering\nangle (deg)"
    steering_valid = angle.notna()
    add_gap_aware_line(
        axes[1], relative(steering_part.loc[steering_valid, "t_unix_ns"]),
        angle.loc[steering_valid], color=COLORS["blue"], linewidth=0.9,
        label=steering_label,
    )
    if steering_neutral is not None:
        axes[1].axhspan(
            float(steering_neutral["neutral_band_lower_relative_deg"]),
            float(steering_neutral["neutral_band_upper_relative_deg"]),
            color=COLORS["blue"], alpha=0.16, zorder=0,
            label="Estimated neutral band",
        )
        axes[1].axhline(0.0, color=COLORS["grey"], linestyle="--", linewidth=0.7, zorder=1)
    axes[1].set_ylabel(steering_ylabel)
    axes[1].legend(loc="upper right")

    power_values = pd.to_numeric(
        power_part.get("p10_instantaneous_power_w", pd.Series(dtype=float)),
        errors="coerce",
    )
    power_valid = power_values.notna()
    if power_valid.any() and not np.allclose(
        power_values.loc[power_valid].to_numpy(dtype=float), 0.0
    ):
        add_gap_aware_line(
            axes[3], relative(power_part.loc[power_valid, "t_unix_ns"]),
            power_values.loc[power_valid], color=COLORS["blue"],
            linewidth=1.0, drawstyle="steps-post",
            label="Reported instantaneous power",
        )
    else:
        axes[3].text(
            0.5, 0.5, "No non-zero standard-power output in this interval",
            transform=axes[3].transAxes, ha="center", va="center",
            color=COLORS["grey"], fontsize=8,
        )
    axes[3].set_ylabel("Power (W)")
    if power_valid.any() and not np.allclose(
        power_values.loc[power_valid].to_numpy(dtype=float), 0.0
    ):
        axes[3].legend(loc="upper right")

    add_gap_aware_line(
        axes[4], relative(brake_part["t_unix_ns"]), brake_part["left_force_n"],
        color=COLORS["blue"], linewidth=1.0, label="Left brake",
    )
    add_gap_aware_line(
        axes[4], relative(brake_part["t_unix_ns"]), brake_part["right_force_n"],
        color=COLORS["orange"], linewidth=0.75, linestyle="--", label="Right brake",
    )
    axes[4].set_ylabel("Brake\nforce (N)")
    if brake_bands:
        for column, color, label in [
            ("left_force_n", COLORS["blue"], "Left zero-input band"),
            ("right_force_n", COLORS["orange"], "Right zero-input band"),
        ]:
            lower, upper = brake_bands[column]
            axes[4].axhspan(
                lower, upper, color=color, alpha=0.18, zorder=10, label=label,
            )
            axes[4].axhline(upper, color=color, linewidth=0.75, linestyle=":", zorder=11)
    axes[4].legend(loc="upper right", ncol=1)

    add_gap_aware_line(
        axes[2], relative(imu_part["t_unix_ns"]), imu_part["gyro_z"],
        color=COLORS["blue"], linewidth=0.8, label="Raw gyroscope $z$",
    )
    axes[2].set_ylabel("Yaw rate\n(rad/s)")
    axes[2].legend(loc="upper right")

    add_gap_aware_line(
        axes[5], relative(gnss_part["t_unix_ns"]), gnss_part["ground_speed_mps"],
        color=COLORS["blue"], linewidth=1.1, label="GNSS ground speed",
    )
    add_gap_aware_line(
        axes[5], relative(wheel_part["t_unix_ns"]), wheel_part["wheel_speed_mps"],
        color=COLORS["orange"], linewidth=0.9, marker="o", markersize=2.0,
        label="Wheel speed",
    )
    axes[5].set_ylabel("Speed\n(m/s)")
    axes[5].legend(loc="upper right", ncol=1)

    if has_trajectory and trajectory_layout == "spatial":
        panel_labels = ["(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
        panel_label(trajectory_axis, "(a)")
    elif has_trajectory:
        if trajectory_layout == "components":
            panel_labels = ["(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
            panel_label(trajectory_axes[0], "(a)")
            panel_label(trajectory_axes[1], "(b)")
        else:
            panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
            panel_label(trajectory_axes[0], "(g)")
            panel_label(trajectory_axes[1], "(h)")
    else:
        panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for axis, label in zip(axes, panel_labels):
        panel_label(axis, label)
        axis.grid(True)
        axis.set_xlim(0.0, (end_ns - start_ns) / NS_PER_SECOND)
        if trajectory_event_time_s is not None:
            axis.axvline(
                trajectory_event_time_s, color=COLORS["purple"],
                linewidth=0.6, linestyle=":", alpha=0.75, zorder=1,
            )
    if trajectory_axes:
        for axis in trajectory_axes:
            axis.grid(True)
            axis.set_xlim(0.0, (end_ns - start_ns) / NS_PER_SECOND)
            if trajectory_event_time_s is not None:
                axis.axvline(
                    trajectory_event_time_s, color=COLORS["purple"],
                    linewidth=0.6, linestyle=":", alpha=0.75, zorder=1,
                )
        if trajectory_layout == "components_last":
            for axis in axes:
                axis.tick_params(labelbottom=False)
            trajectory_axes[0].tick_params(labelbottom=False)
            trajectory_axes[1].set_xlabel("Elapsed acquisition time (s)")
        else:
            for axis in trajectory_axes:
                axis.tick_params(labelbottom=False)
            for axis in axes[:-1]:
                axis.tick_params(labelbottom=False)
            axes[-1].set_xlabel("Elapsed acquisition time (s)")
    else:
        for axis in axes[:-1]:
            axis.tick_params(labelbottom=False)
        axes[-1].set_xlabel("Elapsed acquisition time (s)")
    if figure_title:
        fig.suptitle(figure_title, y=0.995)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    else:
        fig.tight_layout()
    save_figure(fig, output_base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="")
    parser.add_argument(
        "--steering-file",
        default="",
        help="Optional cleaned steering CSV with the original six columns",
    )
    parser.add_argument("--speed-threshold-mps", type=float, default=0.1)
    parser.add_argument("--window-duration-s", type=float, default=30.0)
    parser.add_argument("--window-step-s", type=float, default=5.0)
    parser.add_argument("--speed-pair-tolerance-s", type=float, default=0.2)
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory does not exist: {session_dir}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    paths = {
        "imu": session_dir / "imu_20260603_134654.csv",
        "steering": session_dir / "steering_angle_20260603_134654.csv",
        "brake": session_dir / "brake_sensors_force_20260603_134654.csv",
        "wheel": session_dir / "speed_decoded_20260603_134656.csv",
        "power": session_dir / "rally_payload_decoded_20260603_134654.csv",
        "camera_timestamps": session_dir / "camera_20260603_135432" / "timestamps.csv",
        "recording": session_dir / "20260603T115600Z" / "recording.g3",
        "gaze": session_dir / "20260603T115600Z" / "gazedata.gz",
        "tobii_imu": session_dir / "20260603T115600Z" / "imudata.gz",
    }
    if args.steering_file:
        paths["steering"] = Path(args.steering_file).resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))
    bag_matches = sorted(session_dir.glob("rosbag2_*/metadata.yaml"))
    if len(bag_matches) != 1:
        raise SystemExit("Expected exactly one rosbag2 metadata.yaml")
    bag_dir = bag_matches[0].parent

    imu = read_csv(paths["imu"])
    steering = read_csv(paths["steering"])
    brake = read_csv(paths["brake"])
    wheel = read_csv(paths["wheel"])
    power = read_csv(paths["power"])
    camera = pd.read_csv(paths["camera_timestamps"])
    metadata, gaze, tobii_imu = read_tobii_recording(
        paths["recording"], paths["gaze"], paths["tobii_imu"]
    )

    bag_metadata = GNSS.load_bag_metadata(bag_dir)
    bag_data = GNSS.read_bag_topics(bag_dir, bag_metadata["storage_identifier"])
    velocity_full = bag_data["vel"].rename(
        columns={"t_ns": "t_unix_ns"}
    ).copy()
    moving = (
        pd.to_numeric(velocity_full["ground_speed_mps"], errors="coerce")
        > args.speed_threshold_mps
    )
    if not moving.any():
        raise RuntimeError(
            f"No GNSS speed samples above {args.speed_threshold_mps} m/s"
        )
    common_start_ns = int(velocity_full.loc[moving, "t_unix_ns"].min())
    common_end_ns = int(velocity_full.loc[moving, "t_unix_ns"].max())
    common_duration_s = (common_end_ns - common_start_ns) / NS_PER_SECOND

    imu = crop(imu, common_start_ns, common_end_ns)
    steering = crop(steering, common_start_ns, common_end_ns)
    brake = crop(brake, common_start_ns, common_end_ns)
    wheel = crop(wheel, common_start_ns, common_end_ns)
    power = crop(power, common_start_ns, common_end_ns)
    camera = crop(
        camera, common_start_ns, common_end_ns, column="unix_ns"
    )
    gaze = crop(gaze, common_start_ns, common_end_ns)
    tobii_imu = crop(tobii_imu, common_start_ns, common_end_ns)
    for name, frame in list(bag_data.items()):
        if isinstance(frame, pd.DataFrame) and "t_ns" in frame.columns:
            bag_data[name] = crop(
                frame.rename(columns={"t_ns": "t_unix_ns"}),
                common_start_ns,
                common_end_ns,
            ).rename(columns={"t_unix_ns": "t_ns"})
    velocity = crop(velocity_full, common_start_ns, common_end_ns)

    wheel["wheel_speed_mps"] = pd.to_numeric(
        wheel["speed_mps"], errors="coerce"
    ) / 3.6
    wheel_valid = wheel[wheel["wheel_speed_mps"].notna()].copy()
    power_p10 = power[
        power["page_name"].eq("standard_power")
        & pd.to_numeric(power["p10_instantaneous_power_w"], errors="coerce").notna()
    ].copy()
    raw_imu = imu[imu["dtype"].eq(64)].copy()
    for column in ["gyro_z", "left_force_n", "right_force_n", "angle_deg"]:
        for frame in [raw_imu, brake, steering]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sampling_summary, sampling_intervals = build_sampling_streams(
        bag_data, imu, steering, brake, wheel, power, camera, gaze, tobii_imu
    )
    selected, candidates = select_representative_window(
        velocity,
        wheel_valid,
        steering,
        brake,
        power_p10,
        args.window_duration_s,
        args.window_step_s,
    )
    selected_start_ns = int(selected["start_ns"])
    selected_end_ns = int(selected["end_ns"])
    gnss_speed, wheel_speed, paired, speed_summary = prepare_speed_comparison(
        velocity,
        wheel_valid,
        selected_start_ns,
        selected_end_ns,
        args.speed_pair_tolerance_s,
    )
    course_rate = gnss_course_rate(velocity)

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True)
    tables.mkdir(parents=True)

    plot_sampling_intervals(
        sampling_summary,
        sampling_intervals,
        figures / "F2_sampling_interval_boxplot",
    )
    plot_speed_comparison(
        gnss_speed,
        wheel_speed,
        paired,
        speed_summary,
        selected_start_ns,
        figures / "F6_speed_compare_overall",
    )
    plot_closed_loop(
        gaze,
        steering,
        brake,
        power_p10,
        raw_imu,
        course_rate,
        velocity,
        wheel_valid,
        selected_start_ns,
        selected_end_ns,
        figures / "P9_representative_closed_loop",
    )

    sampling_summary.sort_values("median_interval_ms").to_csv(
        tables / "sampling_interval_summary.csv", index=False
    )
    candidates.to_csv(tables / "representative_window_candidates.csv", index=False)
    paired.to_csv(tables / "F6_speed_paired_samples.csv", index=False)
    pd.DataFrame([speed_summary]).to_csv(
        tables / "F6_speed_comparison_summary.csv", index=False
    )
    write_json(tables / "selected_window.json", selected)

    captions = (
        "F2 sampling intervals. Inter-message interval distributions for timestamped P9 sensor "
        f"streams within the common {common_duration_s:.2f} s interval defined by GNSS ground "
        f"speed above {args.speed_threshold_mps:g} m/s. Boxes show the median and interquartile range, whiskers extend to 1.5 times "
        "the interquartile range, and outliers are hidden. The logarithmic axis accommodates "
        "both periodic and event-driven streams.\n\n"
        "F6 speed comparison. (a) Wheel speed and GNSS receiver ground speed in the selected "
        "30 s dynamic window. Lines are broken across acquisition gaps. (b) Wheel speed minus "
        "GNSS ground speed for nearest timestamp pairs within 0.2 s. GNSS is used as an onboard "
        "comparison signal, not as independent ground truth.\n\n"
        "Closed-loop workflow example. Time-aligned visual attention, steering, brake force, "
        "pedal power, yaw-rate signals, and bicycle speed in the same selected window. Horizontal "
        "gaze is the normalized scene-image coordinate after a five-sample median filter; it is "
        "not an ego-frame gaze angle. Steering uses the rule-based cleaned table without "
        "interpolation; plausible +/-45 degree boundary measurements are retained. GNSS course "
        "rate is shown only above 2 m/s with course accuracy <=5 degrees.\n"
    )
    (output / "figure_captions.txt").write_text(captions, encoding="utf-8")
    readme = (
        "This report uses only the P9 session files listed in run_manifest.json. It does not use "
        "the critical_scenarios.csv from the older validation dataset. All timestamped streams "
        f"are first restricted to {common_start_ns}--{common_end_ns}, defined by GNSS speed "
        f"above {args.speed_threshold_mps:g} m/s. One 30 s window is selected "
        "for visualization by equally weighting normalized GNSS speed range, brake-force P99, "
        "the steering-angle P05--P95 range, and the number of standard-power "
        "events. This is a workflow example, not a manually labelled scenario or an independent "
        "accuracy test. The legacy wheel-speed CSV column named speed_mps is interpreted as km/h "
        "and divided by 3.6, consistent with the older script configuration and the GNSS scale. "
        "Scene-video and lidar intervals are excluded because this session does not provide decoded "
        "per-frame timestamps for them in the inputs used here. Raw data are not changed.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    bag_files = sorted(item for item in bag_dir.iterdir() if item.is_file())
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id or session_dir.name,
        "session_dir": str(session_dir),
        "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in paths.items()
            },
            "rosbag_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in bag_files
            ],
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "selected_window": selected,
        "common_interval": {
            "selection": "First to last GNSS ground-speed sample above threshold",
            "speed_threshold_mps": args.speed_threshold_mps,
            "start_ns": common_start_ns,
            "end_ns": common_end_ns,
            "duration_s": common_duration_s,
        },
        "speed_comparison": speed_summary,
        "tobii_recording": {
            "created": metadata["created"],
            "duration_s": metadata["duration"],
            "gaze_samples_reported": metadata["gaze"]["samples"],
            "gaze_valid_samples_reported": metadata["gaze"]["valid-samples"],
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    write_json(output / "run_manifest.json", manifest)

    checksum_lines = []
    for path in sorted(
        item for item in output.rglob("*")
        if item.is_file() and item.name != "CHECKSUMS.sha256"
    ):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote P9 validation figures to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
