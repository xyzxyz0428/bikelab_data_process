#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset paper validation suite

Inputs:
  - merged sensor Excel (bike_interface_merged.xlsx)
  - camera frame timestamp CSV (timestamps.csv)
  - LiDAR packet CSVs extracted from PCAP (one CSV per LiDAR/IP)
  - LiDAR frame CSVs exported from ROS PointCloud2 (one CSV per LiDAR)

Outputs:
  Main stream health:
    - T1_topic_health_summary.csv
    - F1_sensor_availability_timeline.png
    - F2_inter_message_interval_boxplot.png
    - F3_gap_statistics.png
    - F4_relative_start_end_alignment.png

  Physics-based temporal validation:
    - T3_steering_imu_residual_summary.csv
    - F5_steering_imu_temporal_consistency.png
    - T4_speed_gnss_residual_summary.csv
    - F6_speed_gnss_temporal_consistency.png

  LiDAR-specific health:
    - T6_lidar_packet_health_summary.csv
    - F7_lidar_packet_interval_boxplot.png
    - T7_lidar_frame_health_summary.csv
    - F8_lidar_frame_health.png

  Behavioral sanity check:
    - F16_candidate_segments.csv
    - F16_behavior_<name>.png (if segments provided)

Example:
  python dataset_paper_validation_suite.py \
      --excel bike_interface_merged.xlsx \
      --camera_csv timestamps.csv \
      --lidar_packet_csvs lidar_packets_192_168_1_200.csv lidar_packets_192_168_1_201.csv lidar_packets_192_168_1_202.csv \
      --lidar_frame_csvs lidar_200_frames.csv lidar_201_frames.csv lidar_202_frames.csv \
      --outdir validation_suite_outputs
"""

import os
import re
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration for known Excel file structure
# -----------------------------
STREAM_CONFIG = [
    {
        "prefix": "gps",
        "sheet": "gps",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 10.0,
        "message_type": "ubx_nav",
    },
    {
        "prefix": "steering",
        "sheet": "potentiometer",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 50.0,
        "message_type": "potentiometer",
    },
    {
        "prefix": "wheel_speed",
        "sheet": "wheel speed",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 4.0,
        "message_type": "garmin_speed",
    },
    {
        "prefix": "imu",
        "sheet": "imu",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 100.0,
        "message_type": "imu",
    },
    {
        "prefix": "powermeter",
        "sheet": "powermeter",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 4.0,
        "message_type": "garmin_powermeter",
    },
    {
        "prefix": "eyetracker",
        "sheet": "eyetracker",
        "time_col": "t_unix_ns",
        "nominal_rate_hz": 50.0,
        "message_type": "tobii_gaze",
    },
]

# -----------------------------
# Helpers
# -----------------------------
@dataclass
class StreamData:
    prefix: str
    source_name: str
    message_type: str
    nominal_rate_hz: float
    timestamps_ns: np.ndarray
    df: Optional[pd.DataFrame] = None

def drop_mixed_timebase_outliers(timestamps_ns: np.ndarray) -> np.ndarray:
    """
    Remove clearly invalid timestamps when a stream contains a few startup
    zeros / tiny values mixed with valid Unix epoch ns timestamps.
    """
    if timestamps_ns.size == 0:
        return timestamps_ns

    ts = np.asarray(timestamps_ns, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    ts = ts[ts > 0]

    if ts.size == 0:
        return ts

    # Use positive timestamps median as reference
    med = np.median(ts)

    # If the stream is basically epoch-ns scale (~1e18), remove tiny outliers
    if med > 1e17:
        lower = med * 0.1
        upper = med * 10.0
        ts = ts[(ts >= lower) & (ts <= upper)]

    ts = np.sort(ts)
    ts = ts[np.insert(np.diff(ts) > 0, 0, True)]
    return ts

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_timestamps_ns(s: pd.Series) -> np.ndarray:
    arr = ensure_numeric(s).dropna().values.astype(np.float64)
    if len(arr) == 0:
        return np.array([], dtype=np.float64)
    arr = np.sort(arr)
    keep = np.insert(np.diff(arr) > 0, 0, True)
    return arr[keep]


def diff_sec(ts_ns: np.ndarray) -> np.ndarray:
    if len(ts_ns) < 2:
        return np.array([], dtype=float)
    return np.diff(ts_ns) / 1e9


def infer_nominal_rate_from_ts(ts_ns: np.ndarray) -> float:
    d = diff_sec(ts_ns)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 0.0
    med = float(np.median(d))
    if med <= 0:
        return 0.0
    return 1.0 / med


def compute_gap_stats(ts_ns: np.ndarray, nominal_rate_hz: float, gap_factor: float = 2.0):
    d = diff_sec(ts_ns)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 0, 0, np.nan
    nominal_dt = 1.0 / nominal_rate_hz if nominal_rate_hz > 0 else np.median(d)
    robust_dt = max(gap_factor * nominal_dt, 10.0 * float(np.median(d)), 2.0 * float(np.percentile(d, 95)))
    gap_mask = d > robust_dt
    gap_count = int(np.sum(gap_mask))
    longest_gap = float(np.max(d)) if len(d) > 0 else np.nan
    missing_count = 0
    for dt in d[gap_mask]:
        missing_count += max(0, int(round(dt / nominal_dt)) - 1)
    return gap_count, missing_count, longest_gap


def compute_topic_health(stream: StreamData, session_start_ns: float, session_end_ns: float, gap_factor: float = 2.0) -> Dict:
    ts = stream.timestamps_ns
    n = len(ts)
    if n == 0:
        return {
            "topic_name": stream.prefix,
            "source_name": stream.source_name,
            "message_type": stream.message_type,
            "nominal_rate_hz": stream.nominal_rate_hz,
            "n_messages": 0,
            "observed_mean_rate_hz": np.nan,
            "median_dt_ms": np.nan,
            "jitter_std_dt_ms": np.nan,
            "gap_count": np.nan,
            "missing_ratio_pct": np.nan,
            "longest_gap_s": np.nan,
            "session_coverage_pct": 0.0,
            "start_offset_s": np.nan,
            "end_offset_s": np.nan,
        }
    duration_sec = max((ts[-1] - ts[0]) / 1e9, 1e-12)
    observed_rate = (n - 1) / duration_sec if n > 1 else 0.0
    d = diff_sec(ts)
    med_dt_ms = np.median(d) * 1e3 if len(d) else np.nan
    jitter_std_ms = np.std(d) * 1e3 if len(d) else np.nan
    gap_count, missing_count, longest_gap = compute_gap_stats(ts, stream.nominal_rate_hz, gap_factor)
    denom = n + missing_count
    missing_ratio_pct = 100.0 * missing_count / denom if denom > 0 else np.nan
    total_session_sec = max((session_end_ns - session_start_ns) / 1e9, 1e-12)
    coverage_pct = 100.0 * ((ts[-1] - ts[0]) / 1e9) / total_session_sec
    return {
        "topic_name": stream.prefix,
        "source_name": stream.source_name,
        "message_type": stream.message_type,
        "nominal_rate_hz": stream.nominal_rate_hz,
        "n_messages": int(n),
        "observed_mean_rate_hz": float(observed_rate),
        "median_dt_ms": float(med_dt_ms) if pd.notna(med_dt_ms) else np.nan,
        "jitter_std_dt_ms": float(jitter_std_ms) if pd.notna(jitter_std_ms) else np.nan,
        "gap_count": int(gap_count) if pd.notna(gap_count) else np.nan,
        "missing_ratio_pct": float(missing_ratio_pct) if pd.notna(missing_ratio_pct) else np.nan,
        "longest_gap_s": float(longest_gap) if pd.notna(longest_gap) else np.nan,
        "session_coverage_pct": float(coverage_pct),
        "start_offset_s": float((ts[0] - session_start_ns) / 1e9),
        "end_offset_s": float((session_end_ns - ts[-1]) / 1e9),
    }


def contiguous_segments(ts_ns: np.ndarray, nominal_rate_hz: float, gap_factor: float = 2.0):
    if len(ts_ns) == 0:
        return []
    if len(ts_ns) == 1:
        return [(ts_ns[0], ts_ns[0])]
    d = diff_sec(ts_ns)
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return [(ts_ns[0], ts_ns[-1])]
    nominal_dt = 1.0 / nominal_rate_hz if nominal_rate_hz > 0 else np.median(d)
    robust_dt = max(gap_factor * nominal_dt, 10.0 * float(np.median(d)), 2.0 * float(np.percentile(d, 95)))
    gap_idx = np.where(np.diff(ts_ns) / 1e9 > robust_dt)[0]
    segs = []
    start = ts_ns[0]
    for idx in gap_idx:
        segs.append((start, ts_ns[idx]))
        start = ts_ns[idx + 1]
    segs.append((start, ts_ns[-1]))
    return segs


def resample_to_grid(t_ns: np.ndarray, y: np.ndarray, fs_hz: float, t_start_ns=None, t_end_ns=None):
    mask = np.isfinite(t_ns) & np.isfinite(y)
    t_ns = t_ns[mask]
    y = y[mask]
    if len(t_ns) < 2:
        return np.array([]), np.array([])
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    y = y[order]
    t_s = t_ns / 1e9
    if t_start_ns is None:
        t0 = t_ns[0]
    else:
        t0 = t_start_ns
    if t_end_ns is None:
        t1 = t_ns[-1]
    else:
        t1 = t_end_ns
    t_grid_s = np.arange(t0 / 1e9, t1 / 1e9, 1.0 / fs_hz)
    if len(t_grid_s) < 2:
        return np.array([]), np.array([])
    y_grid = np.interp(t_grid_s, t_s, y)
    return t_grid_s, y_grid


def smooth_series(y: np.ndarray, window: int = 9) -> np.ndarray:
    if len(y) == 0:
        return y
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    s = pd.Series(y)
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def maxcorr_lag_seconds(x: np.ndarray, y: np.ndarray, fs_hz: float, max_lag_s: float = 1.0):
    if len(x) != len(y) or len(x) < 5:
        return np.nan, np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if sx == 0 or sy == 0:
        return np.nan, np.nan
    x = x / sx
    y = y / sy
    max_lag = int(round(max_lag_s * fs_hz))
    lags = np.arange(-max_lag, max_lag + 1)
    corr_vals = []
    for lag in lags:
        if lag < 0:
            a = x[:lag]
            b = y[-lag:]
        elif lag > 0:
            a = x[lag:]
            b = y[:-lag]
        else:
            a = x
            b = y
        if len(a) < 5:
            corr_vals.append(np.nan)
            continue
        corr_vals.append(np.nanmean(a * b))
    corr_vals = np.array(corr_vals, dtype=float)
    if np.all(~np.isfinite(corr_vals)):
        return np.nan, np.nan
    idx = int(np.nanargmax(corr_vals))
    return lags[idx] / fs_hz, float(corr_vals[idx])


def sliding_window_lags(t_s: np.ndarray, x: np.ndarray, y: np.ndarray,
                        window_s: float = 10.0, step_s: float = 2.0,
                        fs_hz: float = 50.0, max_lag_s: float = 1.0):
    if len(t_s) < 2:
        return pd.DataFrame(columns=["center_time_s", "lag_s", "corr"])
    t0 = t_s[0]
    t1 = t_s[-1]
    rows = []
    cur = t0
    while cur + window_s <= t1:
        mask = (t_s >= cur) & (t_s < cur + window_s)
        if np.sum(mask) >= max(20, int(window_s * fs_hz * 0.5)):
            lag_s, corr = maxcorr_lag_seconds(x[mask], y[mask], fs_hz=fs_hz, max_lag_s=max_lag_s)
            rows.append({"center_time_s": cur + 0.5 * window_s, "lag_s": lag_s, "corr": corr})
        cur += step_s
    return pd.DataFrame(rows)


def choose_best_overlay_window(df_lags: pd.DataFrame, desired_window_s: float = 12.0):
    if df_lags.empty:
        return None
    valid = df_lags.dropna(subset=["corr"]).copy()
    if valid.empty:
        return None
    row = valid.iloc[valid["corr"].idxmax()] if valid.index.dtype != 'int64' else valid.loc[valid["corr"].idxmax()]
    center = float(row["center_time_s"])
    return center - desired_window_s / 2.0, center + desired_window_s / 2.0


def infer_gnss_speed_mps(gps_df: pd.DataFrame) -> pd.Series:
    # ubx g_speed is commonly mm/s; use heuristic based on percentile
    s = ensure_numeric(gps_df["g_speed"])
    q95 = np.nanpercentile(s.dropna(), 95) if s.dropna().shape[0] > 0 else np.nan
    if pd.notna(q95) and q95 > 100:
        return s / 1000.0
    return s.copy()


def infer_gaze_yaw_deg(eye_df: pd.DataFrame) -> pd.Series:
    lx = ensure_numeric(eye_df.get("Gaze direction left X [HUCS norm]", pd.Series(index=eye_df.index, dtype=float)))
    lz = ensure_numeric(eye_df.get("Gaze direction left Z [HUCS norm]", pd.Series(index=eye_df.index, dtype=float)))
    rx = ensure_numeric(eye_df.get("Gaze direction right X [HUCS norm]", pd.Series(index=eye_df.index, dtype=float)))
    rz = ensure_numeric(eye_df.get("Gaze direction right Z [HUCS norm]", pd.Series(index=eye_df.index, dtype=float)))
    x = np.nanmean(np.vstack([lx.values, rx.values]), axis=0)
    z = np.nanmean(np.vstack([lz.values, rz.values]), axis=0)
    yaw = np.degrees(np.arctan2(x, z))
    return pd.Series(yaw, index=eye_df.index)


def parse_behavior_segments(arg: Optional[str]):
    if arg is None:
        return []
    if os.path.exists(arg):
        with open(arg, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(arg)


def make_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Loading
# -----------------------------
def load_excel_streams(excel_path: str) -> Tuple[List[StreamData], Dict[str, pd.DataFrame]]:
    dfs = {}
    streams = []
    for cfg in STREAM_CONFIG:
        df = pd.read_excel(excel_path, sheet_name=cfg["sheet"])
        dfs[cfg["prefix"]] = df.copy()
        ts = clean_timestamps_ns(df[cfg["time_col"]]) if cfg["time_col"] in df.columns else np.array([])
        if "lidar_frame" in cfg["prefix"]:
            ts = drop_mixed_timebase_outliers(ts)
        streams.append(StreamData(
            prefix=cfg["prefix"],
            source_name=f"Excel:{cfg['sheet']}",
            message_type=cfg["message_type"],
            nominal_rate_hz=cfg["nominal_rate_hz"],
            timestamps_ns=ts,
            df=df,
        ))
    return streams, dfs


def load_camera_stream(camera_csv: str) -> StreamData:
    df = pd.read_csv(camera_csv)
    time_col = None
    for c in ["unix_ns", "t_unix_ns", "timestamp_ns", "time_ns"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError("Camera CSV must contain one of: unix_ns, t_unix_ns, timestamp_ns, time_ns")
    ts = clean_timestamps_ns(df[time_col])
    if "lidar_frame" in prefix:
        ts = drop_mixed_timebase_outliers(ts)
    nominal = infer_nominal_rate_from_ts(ts)
    return StreamData(
        prefix="camera_frames",
        source_name=f"CSV:{os.path.basename(camera_csv)}",
        message_type="camera/frame",
        nominal_rate_hz=nominal if nominal > 0 else 30.0,
        timestamps_ns=ts,
        df=df,
    )


def packet_prefix_from_path(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(\d+_\d+_\d+_\d+)", base)
    if m:
        return f"lidar_packet_{m.group(1)}"
    return os.path.splitext(base)[0]


def frame_prefix_from_df_or_path(df: pd.DataFrame, path: str) -> str:
    if "frame_id" in df.columns and df["frame_id"].dropna().shape[0] > 0:
        fid = str(df["frame_id"].dropna().iloc[0]).strip().replace("/", "_")
        fid = re.sub(r"[^A-Za-z0-9_]+", "_", fid)
        return f"lidar_frame_{fid}"
    return os.path.splitext(os.path.basename(path))[0]


def load_lidar_packet_streams(paths: List[str]) -> List[StreamData]:
    streams = []
    for p in paths:
        df = pd.read_csv(p)
        # Packet script provides frame.time_epoch in seconds
        if "frame.time_epoch" in df.columns:
            ts_ns = clean_timestamps_ns(pd.to_numeric(df["frame.time_epoch"], errors="coerce") * 1e9)
            if "lidar_frame" in prefix:
                ts_ns = drop_mixed_timebase_outliers(ts_ns)
        else:
            # fallback for user-modified csv
            time_col = None
            for c in ["t_unix_ns", "timestamp_ns", "unix_ns", "time_ns"]:
                if c in df.columns:
                    time_col = c
                    break
            if time_col is None:
                raise ValueError(f"No timestamp column found in packet CSV: {p}")
            ts_ns = clean_timestamps_ns(df[time_col])
            if "lidar_frame" in prefix:
                ts_ns = drop_mixed_timebase_outliers(ts_ns)
            
        nominal = infer_nominal_rate_from_ts(ts_ns)
        streams.append(StreamData(
            prefix=packet_prefix_from_path(p),
            source_name=f"CSV:{os.path.basename(p)}",
            message_type="lidar/packet",
            nominal_rate_hz=nominal,
            timestamps_ns=ts_ns,
            df=df,
        ))
    return streams


def load_lidar_frame_streams(paths: List[str]) -> List[StreamData]:
    streams = []
    for p in paths:
        df = pd.read_csv(p)
        if "t_unix_ns" not in df.columns:
            raise ValueError(f"Frame CSV must contain t_unix_ns: {p}")
        ts_ns = clean_timestamps_ns(df["t_unix_ns"])
        if "lidar_frame" in prefix:
                ts_ns = drop_mixed_timebase_outliers(ts_ns)
        nominal = infer_nominal_rate_from_ts(ts_ns)
        streams.append(StreamData(
            prefix=frame_prefix_from_df_or_path(df, p),
            source_name=f"CSV:{os.path.basename(p)}",
            message_type="lidar/frame",
            nominal_rate_hz=nominal,
            timestamps_ns=ts_ns,
            df=df,
        ))
    return streams

# -----------------------------
# Plotting: stream health
# -----------------------------
def plot_f1_availability_timeline(streams: List[StreamData], session_start_ns: float, session_end_ns: float, out_png: str):
    fig, ax = plt.subplots(figsize=(13, max(4.2, 0.6 * len(streams))))
    y = np.arange(len(streams))[::-1]
    for yi, s in zip(y, streams):
        segs = contiguous_segments(s.timestamps_ns, s.nominal_rate_hz)
        for a, b in segs:
            x0 = (a - session_start_ns) / 1e9
            x1 = (b - session_start_ns) / 1e9
            ax.broken_barh([(x0, max(x1 - x0, 0.03))], (yi - 0.35, 0.7))
    ax.set_yticks(y)
    ax.set_yticklabels([s.prefix for s in streams])
    ax.set_xlim(0, (session_end_ns - session_start_ns) / 1e9)
    ax.set_xlabel("Time since session start (s)")
    ax.set_title("F1. Sensor availability timeline across the recording session")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_f2_interval_boxplot(streams: List[StreamData], out_png: str):
    labels, data = [], []
    for s in streams:
        d_ms = diff_sec(s.timestamps_ns) * 1e3
        d_ms = d_ms[np.isfinite(d_ms) & (d_ms > 0)]
        if len(d_ms) > 0:
            labels.append(s.prefix)
            data.append(d_ms)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.42 * len(labels))))
    ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inter-message interval Δt (ms)")
    ax.set_title("F2. Distribution of inter-message intervals for key sensor streams")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_f3_gap_statistics(stats_df: pd.DataFrame, out_png: str):
    plot_df = stats_df.sort_values("topic_name").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.42 * len(plot_df))))
    axes[0].barh(plot_df["topic_name"], plot_df["gap_count"])
    axes[0].set_title("Gap count")
    axes[0].set_xlabel("Count")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.3)
    axes[1].barh(plot_df["topic_name"], plot_df["longest_gap_s"])
    axes[1].set_title("Longest gap")
    axes[1].set_xlabel("Seconds")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.suptitle("F3. Gap statistics across sensor streams", y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f4_relative_alignment(stats_df: pd.DataFrame, out_png: str):
    plot_df = stats_df.sort_values("start_offset_s").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.42 * len(plot_df))))
    y = np.arange(len(plot_df))[::-1]
    for yi, (_, row) in zip(y, plot_df.iterrows()):
        start = float(row["start_offset_s"])
        end = float(row["end_offset_s"])
        # plot from start offset to session end-end_offset
        full = plot_df["start_offset_s"].min()  # just for reference not used
        session_len = start + (max(plot_df["start_offset_s"]) + max(plot_df["end_offset_s"]) + 1)
        # actual bar length from start to session_duration - end_offset unknown here; use coverage-based proxy
        bar_len = float(row["session_coverage_pct"]) / 100.0
        # better reconstruct relative end using max session span from offsets and coverage impossible; use normalized session [0,1]
    plt.close(fig)


def plot_f4_relative_alignment(streams: List[StreamData], session_start_ns: float, session_end_ns: float, out_png: str):
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.42 * len(streams))))
    plot_streams = sorted(streams, key=lambda s: s.timestamps_ns[0] if len(s.timestamps_ns) else np.inf)
    y = np.arange(len(plot_streams))[::-1]
    for yi, s in zip(y, plot_streams):
        if len(s.timestamps_ns) == 0:
            continue
        x0 = (s.timestamps_ns[0] - session_start_ns) / 1e9
        x1 = (s.timestamps_ns[-1] - session_start_ns) / 1e9
        ax.broken_barh([(x0, max(x1 - x0, 0.03))], (yi - 0.35, 0.7))
    ax.set_yticks(y)
    ax.set_yticklabels([s.prefix for s in plot_streams])
    ax.set_xlabel("Time since session start (s)")
    ax.set_title("F4. Relative start and end times of all streams within the session")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)

# -----------------------------
# Plotting: F5 steering vs IMU
# -----------------------------
def run_f5_steering_vs_imu(dfs: Dict[str, pd.DataFrame], outdir: str):
    if "steering" not in dfs or "imu" not in dfs:
        print("[WARN] Missing steering or imu sheet; skip F5.")
        return
    s_df = dfs["steering"].copy()
    i_df = dfs["imu"].copy()
    if "angle_deg_clamped" not in s_df.columns:
        print("[WARN] Missing angle_deg_clamped; skip F5.")
        return
    s_t = ensure_numeric(s_df["t_unix_ns"]).values
    s_angle = ensure_numeric(s_df["angle_deg_clamped"]).values
    # steering rate from derivative
    valid = np.isfinite(s_t) & np.isfinite(s_angle)
    s_t, s_angle = s_t[valid], s_angle[valid]
    order = np.argsort(s_t)
    s_t, s_angle = s_t[order], s_angle[order]
    if len(s_t) < 5:
        print("[WARN] Too few steering samples; skip F5.")
        return
    s_rate = np.gradient(s_angle, s_t / 1e9)
    s_rate = smooth_series(s_rate, 11)

    yaw_col = "headingspeed" if "headingspeed" in i_df.columns else ("gyro_z" if "gyro_z" in i_df.columns else None)
    if yaw_col is None:
        print("[WARN] No headingspeed or gyro_z in imu; skip F5.")
        return
    i_t = ensure_numeric(i_df["t_unix_ns"]).values
    i_yaw = ensure_numeric(i_df[yaw_col]).values
    valid = np.isfinite(i_t) & np.isfinite(i_yaw)
    i_t, i_yaw = i_t[valid], i_yaw[valid]
    order = np.argsort(i_t)
    i_t, i_yaw = i_t[order], i_yaw[order]
    if len(i_t) < 5:
        print("[WARN] Too few imu samples; skip F5.")
        return
    i_yaw = smooth_series(i_yaw, 11)

    t0 = max(s_t[0], i_t[0])
    t1 = min(s_t[-1], i_t[-1])
    if t1 <= t0:
        print("[WARN] No overlap between steering and imu; skip F5.")
        return
    fs = 50.0
    tg, s_res = resample_to_grid(s_t, s_rate, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    _, i_res = resample_to_grid(i_t, i_yaw, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    if len(tg) < 20:
        print("[WARN] Too little overlap after resampling; skip F5.")
        return

    lag_df = sliding_window_lags(tg, s_res, i_res, window_s=10.0, step_s=2.0, fs_hz=fs, max_lag_s=1.0)
    if lag_df.empty:
        print("[WARN] Unable to compute lag windows for F5.")
        return
    summary = {
        "signal_a": "steering_rate_deg_per_s",
        "signal_b": yaw_col,
        "n_windows": int(len(lag_df)),
        "median_lag_s": float(lag_df["lag_s"].median()),
        "p95_abs_lag_s": float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95)),
        "median_corr": float(lag_df["corr"].median()),
        "p95_corr": float(np.nanpercentile(lag_df["corr"], 95)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(outdir, "T3_steering_imu_residual_summary.csv"), index=False)

    best = choose_best_overlay_window(lag_df, desired_window_s=12.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    if best is not None:
        a, b = best
        mask = (tg >= a) & (tg <= b)
        t_local = tg[mask] - a
        s_norm = (s_res[mask] - np.mean(s_res[mask])) / (np.std(s_res[mask]) + 1e-9)
        i_norm = (i_res[mask] - np.mean(i_res[mask])) / (np.std(i_res[mask]) + 1e-9)
        axes[0].plot(t_local, s_norm, label="Steering rate", linewidth=1.8)
        axes[0].plot(t_local, i_norm, label=f"IMU {yaw_col}", linewidth=1.8)
        axes[0].set_xlabel("Time within representative window (s)")
        axes[0].set_ylabel("Normalized signal")
        axes[0].set_title("Representative overlay")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    axes[1].hist(lag_df["lag_s"].dropna(), bins=30)
    axes[1].axvline(summary["median_lag_s"], linestyle="--", linewidth=1.5, label=f"Median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Peak lag Δt (s)")
    axes[1].set_ylabel("Window count")
    axes[1].set_title("Lag distribution across sliding windows")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("F5. Steering-rate vs IMU yaw-rate temporal consistency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "F5_steering_imu_temporal_consistency.png"), dpi=300)
    plt.close(fig)

# -----------------------------
# Plotting: F6 speed vs GNSS
# -----------------------------
def run_f6_speed_vs_gnss(dfs: Dict[str, pd.DataFrame], outdir: str):
    if "wheel_speed" not in dfs or "gps" not in dfs:
        print("[WARN] Missing wheel_speed or gps sheet; skip F6.")
        return
    w_df = dfs["wheel_speed"].copy()
    g_df = dfs["gps"].copy()
    if "speed_mps" not in w_df.columns or "g_speed" not in g_df.columns:
        print("[WARN] Missing speed columns; skip F6.")
        return
    w_t = ensure_numeric(w_df["t_unix_ns"]).values
    w_v = ensure_numeric(w_df["speed_mps"]).values
    valid = np.isfinite(w_t) & np.isfinite(w_v)
    w_t, w_v = w_t[valid], w_v[valid]
    order = np.argsort(w_t)
    w_t, w_v = w_t[order], w_v[order]

    g_t = ensure_numeric(g_df["t_unix_ns"]).values
    g_v = infer_gnss_speed_mps(g_df).values
    valid = np.isfinite(g_t) & np.isfinite(g_v)
    g_t, g_v = g_t[valid], g_v[valid]
    order = np.argsort(g_t)
    g_t, g_v = g_t[order], g_v[order]

    if len(w_t) < 5 or len(g_t) < 5:
        print("[WARN] Too few speed samples; skip F6.")
        return

    t0 = max(w_t[0], g_t[0])
    t1 = min(w_t[-1], g_t[-1])
    if t1 <= t0:
        print("[WARN] No overlap between wheel speed and gps; skip F6.")
        return
    fs = 10.0
    tg, w_res = resample_to_grid(w_t, w_v, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    _, g_res = resample_to_grid(g_t, g_v, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    if len(tg) < 20:
        print("[WARN] Too little overlap after resampling; skip F6.")
        return

    w_res = smooth_series(w_res, 5)
    g_res = smooth_series(g_res, 5)
    lag_df = sliding_window_lags(tg, w_res, g_res, window_s=20.0, step_s=5.0, fs_hz=fs, max_lag_s=2.0)
    if lag_df.empty:
        print("[WARN] Unable to compute lag windows for F6.")
        return
    summary = {
        "signal_a": "wheel_speed_mps",
        "signal_b": "gnss_speed_mps",
        "n_windows": int(len(lag_df)),
        "median_lag_s": float(lag_df["lag_s"].median()),
        "p95_abs_lag_s": float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95)),
        "median_corr": float(lag_df["corr"].median()),
        "p95_corr": float(np.nanpercentile(lag_df["corr"], 95)),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(outdir, "T4_speed_gnss_residual_summary.csv"), index=False)

    best = choose_best_overlay_window(lag_df, desired_window_s=20.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    if best is not None:
        a, b = best
        mask = (tg >= a) & (tg <= b)
        t_local = tg[mask] - a
        axes[0].plot(t_local, w_res[mask], label="Wheel speed", linewidth=1.8)
        axes[0].plot(t_local, g_res[mask], label="GNSS speed", linewidth=1.8)
        axes[0].set_xlabel("Time within representative window (s)")
        axes[0].set_ylabel("Speed (m/s)")
        axes[0].set_title("Representative speed overlay")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    axes[1].hist(lag_df["lag_s"].dropna(), bins=30)
    axes[1].axvline(summary["median_lag_s"], linestyle="--", linewidth=1.5, label=f"Median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Peak lag Δt (s)")
    axes[1].set_ylabel("Window count")
    axes[1].set_title("Lag distribution across sliding windows")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("F6. Wheel/GNSS speed temporal consistency")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "F6_speed_gnss_temporal_consistency.png"), dpi=300)
    plt.close(fig)

# -----------------------------
# LiDAR-specific health
# -----------------------------
def run_lidar_health(packet_streams: List[StreamData], frame_streams: List[StreamData], outdir: str, session_start_ns: float, session_end_ns: float):
    if packet_streams:
        records = [compute_topic_health(s, session_start_ns, session_end_ns) for s in packet_streams]
        pdf = pd.DataFrame(records).sort_values("topic_name")
        pdf.to_csv(os.path.join(outdir, "T6_lidar_packet_health_summary.csv"), index=False)

        fig, ax = plt.subplots(figsize=(12, max(4.2, 0.45 * len(packet_streams))))
        labels, data = [], []
        for s in packet_streams:
            d_ms = diff_sec(s.timestamps_ns) * 1e3
            d_ms = d_ms[np.isfinite(d_ms) & (d_ms > 0)]
            if len(d_ms):
                labels.append(s.prefix)
                data.append(d_ms)
        ax.boxplot(data, vert=False, showfliers=False)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Packet inter-arrival interval Δt (ms)")
        ax.set_title("F7. LiDAR packet inter-arrival interval distribution")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "F7_lidar_packet_interval_boxplot.png"), dpi=300)
        plt.close(fig)

    if frame_streams:
        records = [compute_topic_health(s, session_start_ns, session_end_ns) for s in frame_streams]
        fdf = pd.DataFrame(records).sort_values("topic_name")
        # add point count statistics if available
        add = []
        for s in frame_streams:
            row = {"topic_name": s.prefix}
            if s.df is not None and "point_count" in s.df.columns:
                pc = ensure_numeric(s.df["point_count"]).dropna()
                row.update({
                    "median_point_count": float(pc.median()) if len(pc) else np.nan,
                    "p05_point_count": float(np.nanpercentile(pc, 5)) if len(pc) else np.nan,
                    "p95_point_count": float(np.nanpercentile(pc, 95)) if len(pc) else np.nan,
                })
            add.append(row)
        add_df = pd.DataFrame(add)
        fdf = fdf.merge(add_df, on="topic_name", how="left")
        fdf.to_csv(os.path.join(outdir, "T7_lidar_frame_health_summary.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4.8, 0.42 * len(frame_streams))))
        labels_i, data_i = [], []
        labels_p, data_p = [], []
        for s in frame_streams:
            d_ms = diff_sec(s.timestamps_ns) * 1e3
            d_ms = d_ms[np.isfinite(d_ms) & (d_ms > 0)]
            if len(d_ms):
                labels_i.append(s.prefix)
                data_i.append(d_ms)
            if s.df is not None and "point_count" in s.df.columns:
                pc = ensure_numeric(s.df["point_count"]).dropna().values
                if len(pc):
                    labels_p.append(s.prefix)
                    data_p.append(pc)
        if data_i:
            axes[0].boxplot(data_i, vert=False, showfliers=False)
            axes[0].set_yticklabels(labels_i)
            axes[0].set_xlabel("Frame interval Δt (ms)")
            axes[0].set_title("Frame interval stability")
            axes[0].grid(True, axis="x", linestyle="--", alpha=0.3)
        if data_p:
            axes[1].boxplot(data_p, vert=False, showfliers=False)
            axes[1].set_yticklabels(labels_p)
            axes[1].set_xlabel("Points per frame")
            axes[1].set_title("Points-per-frame distribution")
            axes[1].grid(True, axis="x", linestyle="--", alpha=0.3)
        fig.suptitle("F8. LiDAR frame health")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "F8_lidar_frame_health.png"), dpi=300)
        plt.close(fig)

# -----------------------------
# Behavioral sanity check (F16)
# -----------------------------
def suggest_f16_candidates(dfs: Dict[str, pd.DataFrame], outdir: str):
    needed = {"wheel_speed", "steering", "eyetracker"}
    if not needed.issubset(dfs.keys()):
        return None
    w_df = dfs["wheel_speed"].copy()
    s_df = dfs["steering"].copy()
    e_df = dfs["eyetracker"].copy()
    if "speed_mps" not in w_df.columns or "angle_deg_clamped" not in s_df.columns:
        return None

    w_t = ensure_numeric(w_df["t_unix_ns"]).values
    w_v = ensure_numeric(w_df["speed_mps"]).values
    s_t = ensure_numeric(s_df["t_unix_ns"]).values
    s_a = ensure_numeric(s_df["angle_deg_clamped"]).values
    gaze_yaw = infer_gaze_yaw_deg(e_df).values
    e_t = ensure_numeric(e_df["t_unix_ns"]).values

    valid = np.isfinite(w_t) & np.isfinite(w_v)
    w_t, w_v = w_t[valid], w_v[valid]
    valid = np.isfinite(s_t) & np.isfinite(s_a)
    s_t, s_a = s_t[valid], s_a[valid]
    valid = np.isfinite(e_t) & np.isfinite(gaze_yaw)
    e_t, gaze_yaw = e_t[valid], gaze_yaw[valid]
    if len(w_t) < 10 or len(s_t) < 10 or len(e_t) < 10:
        return None

    t0 = max(w_t.min(), s_t.min(), e_t.min())
    t1 = min(w_t.max(), s_t.max(), e_t.max())
    if t1 <= t0:
        return None
    fs = 10.0
    tg, w_res = resample_to_grid(w_t, w_v, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    _, s_res = resample_to_grid(s_t, s_a, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    _, g_res = resample_to_grid(e_t, gaze_yaw, fs_hz=fs, t_start_ns=t0, t_end_ns=t1)
    if len(tg) < 100:
        return None

    s_rate = np.gradient(s_res, tg)
    win_s = 12.0
    step_s = 3.0
    rows = []
    cur = tg[0]
    while cur + win_s <= tg[-1]:
        mask = (tg >= cur) & (tg < cur + win_s)
        if np.sum(mask) < 20:
            cur += step_s
            continue
        speed_range = np.nanpercentile(w_res[mask], 90) - np.nanpercentile(w_res[mask], 10)
        steer_activity = np.nanpercentile(np.abs(s_rate[mask]), 90)
        gaze_var = np.nanstd(g_res[mask])
        score = 1.2 * steer_activity + 0.8 * speed_range + 0.6 * gaze_var
        rows.append({
            "start_s": float(cur - tg[0]),
            "end_s": float(cur + win_s - tg[0]),
            "score": float(score),
            "speed_range_mps": float(speed_range),
            "steer_activity": float(steer_activity),
            "gaze_yaw_std_deg": float(gaze_var),
        })
        cur += step_s
    cand = pd.DataFrame(rows).sort_values("score", ascending=False).head(20)
    cand.to_csv(os.path.join(outdir, "F16_candidate_segments.csv"), index=False)
    return cand


def make_f16_plots(dfs: Dict[str, pd.DataFrame], outdir: str, segments: List[Dict]):
    needed = {"wheel_speed", "steering", "eyetracker"}
    if not needed.issubset(dfs.keys()):
        print("[WARN] Missing required streams for F16; skip.")
        return
    w_df = dfs["wheel_speed"].copy()
    s_df = dfs["steering"].copy()
    e_df = dfs["eyetracker"].copy()
    w_t = ensure_numeric(w_df["t_unix_ns"]).values / 1e9
    w_v = ensure_numeric(w_df["speed_mps"]).values
    s_t = ensure_numeric(s_df["t_unix_ns"]).values / 1e9
    s_a = ensure_numeric(s_df["angle_deg_clamped"]).values
    e_t = ensure_numeric(e_df["t_unix_ns"]).values / 1e9
    gaze_yaw = infer_gaze_yaw_deg(e_df).values

    # establish common zero = min across all three
    t_zero = np.nanmin([np.nanmin(w_t), np.nanmin(s_t), np.nanmin(e_t)])
    w_t -= t_zero; s_t -= t_zero; e_t -= t_zero

    for seg in segments:
        name = seg["name"]
        a = float(seg["start_s"])
        b = float(seg["end_s"])
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

        mask = (w_t >= a) & (w_t <= b) & np.isfinite(w_v)
        axes[0].plot(w_t[mask] - a, w_v[mask], linewidth=1.5)
        axes[0].set_ylabel("Speed (m/s)")
        axes[0].set_title(f"F16. Behavioral sanity check: {name}")
        axes[0].grid(True, alpha=0.3)

        mask = (s_t >= a) & (s_t <= b) & np.isfinite(s_a)
        axes[1].plot(s_t[mask] - a, s_a[mask], linewidth=1.5)
        axes[1].set_ylabel("Steering (deg)")
        axes[1].grid(True, alpha=0.3)

        mask = (e_t >= a) & (e_t <= b) & np.isfinite(gaze_yaw)
        tt = e_t[mask] - a
        gy = gaze_yaw[mask]
        axes[2].plot(tt, gy, linewidth=1.2, label="Gaze yaw")
        left_idx = gy > 20
        right_idx = gy < -20
        axes[2].scatter(tt[left_idx], gy[left_idx], s=10, label="look-left")
        axes[2].scatter(tt[right_idx], gy[right_idx], s=10, label="look-right")
        axes[2].set_ylabel("Gaze yaw (deg)")
        axes[2].set_xlabel("Time within segment (s)")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", ncol=3)

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"F16_behavior_{name}.png"), dpi=300)
        plt.close(fig)

# -----------------------------
# Future work manifest
# -----------------------------
def write_future_work_manifest(outdir: str):
    items = [
        {
            "future_item": "T2 / offset-vs-time / offset distribution for NTP/PTP",
            "what_is_needed": "chrony/ptp4l/phc2sys offset logs recorded during the same acquisition sessions",
            "status": "not derivable from current sensor timestamps alone"
        },
        {
            "future_item": "F9 / F10 / T5 ego-motion validation",
            "what_is_needed": "fused odometry output such as /odometry/filtered or a fused CSV; optional RTK-fixed reference",
            "status": "future"
        },
        {
            "future_item": "gaze-object association and interaction metrics (TTC/PET/min lateral distance)",
            "what_is_needed": "LiDAR detection/tracking outputs or manually curated object trajectories",
            "status": "future"
        },
        {
            "future_item": "LiDAR spatial consistency across the three units",
            "what_is_needed": "decoded point clouds + stable extrinsics + static-scene overlap analysis",
            "status": "future"
        },
    ]
    pd.DataFrame(items).to_csv(os.path.join(outdir, "future_analyses_manifest.csv"), index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Merged sensor Excel file")
    parser.add_argument("--camera_csv", required=True, help="Camera frame timestamp CSV")
    parser.add_argument("--lidar_packet_csvs", nargs="*", default=[], help="LiDAR packet CSVs extracted from PCAP")
    parser.add_argument("--lidar_frame_csvs", nargs="*", default=[], help="LiDAR frame CSVs exported from ROS PointCloud2")
    parser.add_argument("--outdir", default="validation_suite_outputs", help="Output directory")
    parser.add_argument("--behavior_segments", default=None,
                        help='JSON file path or inline JSON list, e.g. "[{\"name\":\"urban_turn\",\"start_s\":120,\"end_s\":135}]"')
    args = parser.parse_args()

    make_outdir(args.outdir)

    excel_streams, dfs = load_excel_streams(args.excel)
    camera_stream = load_camera_stream(args.camera_csv)
    packet_streams = load_lidar_packet_streams(args.lidar_packet_csvs) if args.lidar_packet_csvs else []
    frame_streams = load_lidar_frame_streams(args.lidar_frame_csvs) if args.lidar_frame_csvs else []

    all_streams = excel_streams + [camera_stream] + packet_streams + frame_streams
    all_streams = [s for s in all_streams if len(s.timestamps_ns) > 0]
    if not all_streams:
        raise RuntimeError("No valid streams loaded.")

    session_start_ns = min(s.timestamps_ns[0] for s in all_streams)
    session_end_ns = max(s.timestamps_ns[-1] for s in all_streams)

    # T1
    records = [compute_topic_health(s, session_start_ns, session_end_ns) for s in all_streams]
    stats_df = pd.DataFrame(records).sort_values("topic_name").reset_index(drop=True)
    stats_df.to_csv(os.path.join(args.outdir, "T1_topic_health_summary.csv"), index=False)

    # F1-F4
    plot_f1_availability_timeline(all_streams, session_start_ns, session_end_ns,
                                  os.path.join(args.outdir, "F1_sensor_availability_timeline.png"))
    plot_f2_interval_boxplot(all_streams,
                             os.path.join(args.outdir, "F2_inter_message_interval_boxplot.png"))
    plot_f3_gap_statistics(stats_df,
                           os.path.join(args.outdir, "F3_gap_statistics.png"))
    plot_f4_relative_alignment(all_streams, session_start_ns, session_end_ns,
                               os.path.join(args.outdir, "F4_relative_start_end_alignment.png"))

    # F5/F6
    run_f5_steering_vs_imu(dfs, args.outdir)
    run_f6_speed_vs_gnss(dfs, args.outdir)

    # LiDAR-specific
    run_lidar_health(packet_streams, frame_streams, args.outdir, session_start_ns, session_end_ns)

    # F16 candidates and plots
    suggest_f16_candidates(dfs, args.outdir)
    segments = parse_behavior_segments(args.behavior_segments)
    if segments:
        make_f16_plots(dfs, args.outdir, segments)

    # future work manifest
    write_future_work_manifest(args.outdir)

    print("Done. Outputs saved to:")
    print(args.outdir)


if __name__ == "__main__":
    main()
