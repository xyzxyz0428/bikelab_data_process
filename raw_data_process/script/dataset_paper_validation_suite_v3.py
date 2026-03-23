#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_paper_validation_suite.py

Read the merged Bikelab XLSX and generate validation tables/figures for a dataset paper.
Main outputs:
  T1_topic_health_summary.csv
  F1_sensor_availability_timeline.png
  F2_inter_message_interval_boxplot.png
  F3_gap_statistics.png
  F4_relative_start_end_alignment.png
  T3_steering_imu_residual_summary.csv
  F5_steering_imu_temporal_consistency.png
  T4_speed_gnss_residual_summary.csv
  F6_speed_gnss_temporal_consistency.png
  T6_lidar_frame_health_summary.csv
  F8_lidar_frame_health.png
  F16_candidate_segments.csv
Optional:
  packet-level LiDAR health from external CSVs filtered to the same session window.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STREAM_CONFIG = [
    {"sheet": "gps",         "name": "gps",          "nominal_rate_hz": 10.0},
    {"sheet": "potentiometer", "name": "steering",   "nominal_rate_hz": 50.0},
    {"sheet": "wheel_speed", "name": "wheel_speed",  "nominal_rate_hz": 4.0},
    {"sheet": "imu",         "name": "imu",          "nominal_rate_hz": 100.0},
    {"sheet": "powermeter",  "name": "powermeter",   "nominal_rate_hz": 4.0},
    {"sheet": "eyetracker",  "name": "eyetracker",   "nominal_rate_hz": 50.0},
    {"sheet": "camera",      "name": "camera_frames","nominal_rate_hz": 25.0},
    {"sheet": "lidar_f_200", "name": "lidar_frame_200", "nominal_rate_hz": 10},
    {"sheet": "lidar_f_201", "name": "lidar_frame_201", "nominal_rate_hz": 10},
    {"sheet": "lidar_f_202", "name": "lidar_frame_202", "nominal_rate_hz": 10},
]

GAP_FACTOR = 2.0
DEFAULT_LIDAR_FRAME_RATE_HZ = 10.0
DEFAULT_PACKET_RATE_HZ = None  # packet-level usually used only for interval/gap sanity, not missing-ratio


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_time_ns(series: pd.Series) -> np.ndarray:
    s = ensure_numeric(series).dropna().astype(np.float64)
    if s.empty:
        return np.array([], dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return np.array([], dtype=np.float64)
    med = np.median(s)
    if med > 1e17:
        s = s[(s >= med * 0.1) & (s <= med * 10.0)]
    s = np.sort(s)
    if s.size > 1:
        s = s[np.insert(np.diff(s) > 0, 0, True)]
    return s


def compute_intervals_sec(ts_ns: np.ndarray) -> np.ndarray:
    if ts_ns.size < 2:
        return np.array([], dtype=float)
    return np.diff(ts_ns) / 1e9


def infer_nominal_rate_hz(ts_ns: np.ndarray, fallback: Optional[float] = None) -> Optional[float]:
    dt = compute_intervals_sec(ts_ns)
    if dt.size == 0:
        return fallback
    med = np.median(dt)
    if med <= 0:
        return fallback
    return float(1.0 / med)


def count_missing_messages(dt_sec: np.ndarray, nominal_rate_hz: Optional[float]) -> Tuple[int, int, float]:
    if nominal_rate_hz is None or nominal_rate_hz <= 0 or dt_sec.size == 0:
        return 0, 0, float(np.max(dt_sec)) if dt_sec.size > 0 else 0.0

    nominal_dt = 1.0 / nominal_rate_hz
    gap_mask = dt_sec > GAP_FACTOR * nominal_dt
    gap_count = int(np.sum(gap_mask))
    longest_gap_s = float(np.max(dt_sec)) if dt_sec.size > 0 else 0.0

    missing_msg_count = 0
    for dt in dt_sec[gap_mask]:
        miss = max(0, int(round(dt / nominal_dt)) - 1)
        missing_msg_count += miss

    return gap_count, missing_msg_count, longest_gap_s


def compute_health_row(name: str, ts_ns: np.ndarray, nominal_rate_hz: Optional[float],
                       session_start_ns: float, session_end_ns: float) -> Dict:
    if ts_ns.size == 0:
        return {
            "topic_name": name,
            "nominal_rate_hz": nominal_rate_hz,
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

    duration_sec = max((ts_ns[-1] - ts_ns[0]) / 1e9, 1e-12)
    dt_sec = compute_intervals_sec(ts_ns)
    observed_rate = (len(ts_ns) - 1) / duration_sec if len(ts_ns) > 1 else 0.0
    median_dt_ms = np.median(dt_sec) * 1e3 if dt_sec.size else np.nan
    jitter_std_dt_ms = np.std(dt_sec) * 1e3 if dt_sec.size else np.nan
    gap_count, missing_msg_count, longest_gap_s = count_missing_messages(dt_sec, nominal_rate_hz)
    denom = len(ts_ns) + missing_msg_count
    missing_ratio_pct = 100.0 * missing_msg_count / denom if denom > 0 else np.nan
    total_session_sec = max((session_end_ns - session_start_ns) / 1e9, 1e-12)
    session_coverage_pct = 100.0 * ((ts_ns[-1] - ts_ns[0]) / 1e9) / total_session_sec

    return {
        "topic_name": name,
        "nominal_rate_hz": nominal_rate_hz,
        "n_messages": len(ts_ns),
        "observed_mean_rate_hz": observed_rate,
        "median_dt_ms": median_dt_ms,
        "jitter_std_dt_ms": jitter_std_dt_ms,
        "gap_count": gap_count,
        "missing_ratio_pct": missing_ratio_pct,
        "longest_gap_s": longest_gap_s,
        "session_coverage_pct": session_coverage_pct,
        "start_offset_s": (ts_ns[0] - session_start_ns) / 1e9,
        "end_offset_s": (session_end_ns - ts_ns[-1]) / 1e9,
    }


def build_availability_segments(ts_ns: np.ndarray, nominal_rate_hz: Optional[float]) -> List[Tuple[float, float]]:
    if ts_ns.size == 0:
        return []
    if ts_ns.size == 1:
        return [(ts_ns[0], ts_ns[0])]

    if nominal_rate_hz is None or nominal_rate_hz <= 0:
        nominal_rate_hz = infer_nominal_rate_hz(ts_ns, fallback=1.0)

    dt_sec = compute_intervals_sec(ts_ns)
    nominal_dt = 1.0 / nominal_rate_hz if nominal_rate_hz and nominal_rate_hz > 0 else np.inf
    gap_mask = dt_sec > GAP_FACTOR * nominal_dt

    segments = []
    seg_start = ts_ns[0]
    for i, is_gap in enumerate(gap_mask):
        if is_gap:
            segments.append((seg_start, ts_ns[i]))
            seg_start = ts_ns[i + 1]
    segments.append((seg_start, ts_ns[-1]))
    return segments


def resample_series(t_ns: np.ndarray, y: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(t_ns) & np.isfinite(y)
    t_ns = t_ns[mask]
    y = y[mask]
    if t_ns.size < 2:
        return np.array([]), np.array([])
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    y = y[order]
    t0 = t_ns[0]
    t1 = t_ns[-1]
    step_ns = int(round(1e9 / fs_hz))
    grid_ns = np.arange(int(t0), int(t1) + 1, step_ns, dtype=np.int64)
    y_grid = np.interp(grid_ns.astype(np.float64), t_ns.astype(np.float64), y.astype(np.float64))
    return grid_ns, y_grid


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    std = np.nanstd(x)
    return x / std if std > 1e-12 else x


def crosscorr_lag_seconds(x: np.ndarray, y: np.ndarray, fs_hz: float, max_lag_s: float = 2.0) -> Tuple[float, float]:
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan, np.nan
    x = normalized(x)
    y = normalized(y)
    max_lag = int(round(max_lag_s * fs_hz))
    corr_full = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))
    center_mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr_full[center_mask]
    lags = lags[center_mask]
    if corr.size == 0:
        return np.nan, np.nan
    idx = int(np.argmax(corr))
    lag_s = lags[idx] / fs_hz
    # Pearson-like normalized score over equal-length vectors
    corr_score = corr[idx] / max(len(x), 1)
    return lag_s, corr_score


def sliding_window_lag(t_ns: np.ndarray, x: np.ndarray, y: np.ndarray,
                       fs_hz: float, window_s: float = 10.0, step_s: float = 2.0,
                       max_lag_s: float = 2.0) -> pd.DataFrame:
    if t_ns.size == 0 or x.size == 0 or y.size == 0:
        return pd.DataFrame(columns=["window_center_s", "lag_s", "corr_score"])
    n_win = int(round(window_s * fs_hz))
    n_step = int(round(step_s * fs_hz))
    rows = []
    for start in range(0, max(0, len(x) - n_win + 1), max(1, n_step)):
        stop = start + n_win
        lag_s, corr_score = crosscorr_lag_seconds(x[start:stop], y[start:stop], fs_hz, max_lag_s=max_lag_s)
        center_s = (t_ns[start] + t_ns[min(stop - 1, len(t_ns) - 1)]) / 2 / 1e9
        rows.append({"window_center_s": center_s, "lag_s": lag_s, "corr_score": corr_score})
    return pd.DataFrame(rows)


def find_sheet_or_none(xl: pd.ExcelFile, sheet: str) -> Optional[pd.DataFrame]:
    if sheet not in xl.sheet_names:
        return None
    return pd.read_excel(xl, sheet_name=sheet)


def compute_gaze_yaw_deg(df_eye: pd.DataFrame) -> pd.DataFrame:
    req = [
        "Gaze direction left X [HUCS norm]", "Gaze direction left Z [HUCS norm]",
        "Gaze direction right X [HUCS norm]", "Gaze direction right Z [HUCS norm]",
        "Validity left", "Validity right",
    ]
    missing = [c for c in req if c not in df_eye.columns]
    if missing:
        return pd.DataFrame(columns=["t_unix_ns", "gaze_yaw_deg", "glance_left", "glance_right"])

    lx = ensure_numeric(df_eye["Gaze direction left X [HUCS norm]"]).to_numpy()
    lz = ensure_numeric(df_eye["Gaze direction left Z [HUCS norm]"]).to_numpy()
    rx = ensure_numeric(df_eye["Gaze direction right X [HUCS norm]"]).to_numpy()
    rz = ensure_numeric(df_eye["Gaze direction right Z [HUCS norm]"]).to_numpy()
    vl = ensure_numeric(df_eye["Validity left"]).fillna(0).to_numpy()
    vr = ensure_numeric(df_eye["Validity right"]).fillna(0).to_numpy()
    t = ensure_numeric(df_eye["t_unix_ns"]).to_numpy()

    # Prefer average of valid eyes
    x = np.where((vl > 0) & (vr > 0), (lx + rx) / 2.0,
                 np.where(vl > 0, lx, np.where(vr > 0, rx, np.nan)))
    z = np.where((vl > 0) & (vr > 0), (lz + rz) / 2.0,
                 np.where(vl > 0, lz, np.where(vr > 0, rz, np.nan)))

    yaw_deg = np.degrees(np.arctan2(x, z))
    out = pd.DataFrame({"t_unix_ns": t, "gaze_yaw_deg": yaw_deg})
    out["glance_left"] = (out["gaze_yaw_deg"] > 25).astype(int)
    out["glance_right"] = (out["gaze_yaw_deg"] < -25).astype(int)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["t_unix_ns"])
    return out


def plot_f1_timeline(streams: List[Tuple[str, np.ndarray, Optional[float]]],
                     session_start_ns: float, session_end_ns: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.6 * len(streams))))
    y_positions = np.arange(len(streams))[::-1]

    for y, (name, ts, nominal_rate) in zip(y_positions, streams):
        segments = build_availability_segments(ts, nominal_rate)
        for a, b in segments:
            x0 = (a - session_start_ns) / 1e9
            x1 = (b - session_start_ns) / 1e9
            width = max(x1 - x0, 0.02)
            ax.broken_barh([(x0, width)], (y - 0.35, 0.7))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([x[0] for x in streams])
    ax.set_xlim(0, max((session_end_ns - session_start_ns) / 1e9, 1.0))
    ax.set_xlabel("Time since session start (s)")
    ax.set_title("F1. Sensor availability timeline across the recording session")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_f2_boxplot(streams: List[Tuple[str, np.ndarray, Optional[float]]], out_path: Path):
    labels, data = [], []
    for name, ts, _ in streams:
        dt_ms = compute_intervals_sec(ts) * 1e3
        dt_ms = dt_ms[np.isfinite(dt_ms)]
        if dt_ms.size > 0:
            labels.append(name)
            data.append(dt_ms)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(labels))))
    ax.boxplot(data, vert=False, showfliers=False)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Inter-message interval Δt (ms)")
    ax.set_title("F2. Distribution of inter-message intervals for key sensor streams")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_f3_gap(stats_df: pd.DataFrame, out_path: Path):
    plot_df = stats_df.sort_values("topic_name").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, 0.45 * len(plot_df))))
    axes[0].barh(plot_df["topic_name"], plot_df["gap_count"])
    axes[0].set_title("Gap count")
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.3)
    axes[1].barh(plot_df["topic_name"], plot_df["longest_gap_s"])
    axes[1].set_title("Longest gap (s)")
    axes[1].grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.suptitle("F3. Gap statistics across sensor streams")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f4_alignment(stats_df: pd.DataFrame, out_path: Path):
    plot_df = stats_df.sort_values("start_offset_s").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(plot_df))))
    y = np.arange(len(plot_df))[::-1]
    start = plot_df["start_offset_s"].to_numpy()
    end = (plot_df["start_offset_s"] + (plot_df["session_coverage_pct"] / 100.0) *
           (plot_df["start_offset_s"] + plot_df["end_offset_s"] + 1e-9))
    # better use actual duration from offsets
    duration = []
    for _, row in plot_df.iterrows():
        total = row["start_offset_s"] + row["end_offset_s"]
        # total is not session duration; use coverage and offsets only as display proxies
        duration.append(max(0.0, row["session_coverage_pct"] / 100.0))
    # Use real stream duration:
    real_duration = []
    for _, row in plot_df.iterrows():
        total_session = row["start_offset_s"] + row["end_offset_s"]
        # placeholder, fixed below by reading existing columns if present
        real_duration.append(np.nan)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        x0 = row["start_offset_s"]
        # stream duration = total_session - start_offset - end_offset
        # Need total session duration from max offsets; infer from dataframe
        # Use global maximum of start+duration+end approximation from all streams
        width = max(0.01, 0.0)
        ax.broken_barh([(x0, row["_stream_duration_s"])], (yi - 0.35, 0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["topic_name"])
    ax.set_xlabel("Time since session start (s)")
    ax.set_title("F4. Relative start and end alignment of sensor streams")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def pick_representative_time_window(t_ns: np.ndarray, score_signal: np.ndarray, window_s: float, fs_hz: float) -> Tuple[float, float]:
    if t_ns.size == 0 or score_signal.size == 0:
        return 0.0, 0.0
    n_win = max(2, int(round(window_s * fs_hz)))
    if len(score_signal) <= n_win:
        return t_ns[0] / 1e9, t_ns[-1] / 1e9
    kernel = np.ones(n_win)
    score = np.convolve(np.abs(score_signal), kernel, mode="valid")
    idx = int(np.argmax(score))
    return t_ns[idx] / 1e9, t_ns[idx + n_win - 1] / 1e9


def analyze_steering_imu(df_steer: pd.DataFrame, df_imu: pd.DataFrame, outdir: Path):
    steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
    imu_col = "headingspeed" if "headingspeed" in df_imu.columns else "gyro_z"
    if steer_col not in df_steer.columns or imu_col not in df_imu.columns:
        return None

    t1 = clean_time_ns(df_steer["t_unix_ns"])
    y1_raw = ensure_numeric(df_steer.set_index("t_unix_ns").loc[t1.astype(np.int64), steer_col]).to_numpy()
    t2 = clean_time_ns(df_imu["t_unix_ns"])
    y2_raw = ensure_numeric(df_imu.set_index("t_unix_ns").loc[t2.astype(np.int64), imu_col]).to_numpy()

    fs = 50.0
    g1, s = resample_series(t1, y1_raw, fs)
    g2, imu = resample_series(t2, y2_raw, fs)
    if g1.size == 0 or g2.size == 0:
        return None

    t0 = max(g1[0], g2[0])
    t1e = min(g1[-1], g2[-1])
    if t1e <= t0:
        return None

    common = np.arange(t0, t1e + 1, int(round(1e9 / fs)), dtype=np.int64)
    s_common = np.interp(common.astype(float), g1.astype(float), s.astype(float))
    imu_common = np.interp(common.astype(float), g2.astype(float), imu.astype(float))

    steering_rate = np.gradient(s_common) * fs
    lag_df = sliding_window_lag(common, steering_rate, imu_common, fs_hz=fs, window_s=10.0, step_s=2.0, max_lag_s=2.0)

    summary = {
        "n_windows": len(lag_df),
        "median_lag_s": float(lag_df["lag_s"].median()) if len(lag_df) else np.nan,
        "p95_abs_lag_s": float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95)) if len(lag_df) else np.nan,
        "median_corr_score": float(lag_df["corr_score"].median()) if len(lag_df) else np.nan,
        "steering_signal": steer_col,
        "imu_signal": imu_col,
    }
    pd.DataFrame([summary]).to_csv(outdir / "T3_steering_imu_residual_summary.csv", index=False)

    win_start_s, win_end_s = pick_representative_time_window(common, steering_rate, 20.0, fs)
    mask = (common / 1e9 >= win_start_s) & (common / 1e9 <= win_end_s)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(common[mask] / 1e9 - win_start_s, normalized(steering_rate[mask]), label="Steering rate")
    axes[0].plot(common[mask] / 1e9 - win_start_s, normalized(imu_common[mask]), label="IMU yaw-related")
    axes[0].set_xlabel("Time within representative window (s)")
    axes[0].set_ylabel("Normalized signal")
    axes[0].set_title("Representative signal overlay")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
    axes[1].axvline(summary["median_lag_s"], linestyle="--", label=f"median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("F5. Steering-rate vs IMU yaw-rate temporal consistency")
    plt.tight_layout()
    plt.savefig(outdir / "F5_steering_imu_temporal_consistency.png", dpi=300)
    plt.close(fig)
    return summary


def analyze_speed_gnss(df_speed: pd.DataFrame, df_gps: pd.DataFrame, outdir: Path):
    if "speed_mps" not in df_speed.columns or "g_speed" not in df_gps.columns:
        return None

    t1 = clean_time_ns(df_speed["t_unix_ns"])
    v1_raw = ensure_numeric(df_speed.set_index("t_unix_ns").loc[t1.astype(np.int64), "speed_mps"]).to_numpy()
    t2 = clean_time_ns(df_gps["t_unix_ns"])
    v2_raw = ensure_numeric(df_gps.set_index("t_unix_ns").loc[t2.astype(np.int64), "g_speed"]).to_numpy()

    v1_raw = v1_raw / 3.6
    # Heuristic: ubx g_speed is often mm/s
    med_g = np.nanmedian(v2_raw) if np.isfinite(v2_raw).any() else np.nan
    if np.isfinite(med_g) and med_g > 50:
        v2_raw = v2_raw / 1000.0

    fs = 10.0
    g1, w = resample_series(t1, v1_raw, fs)
    g2, g = resample_series(t2, v2_raw, fs)
    if g1.size == 0 or g2.size == 0:
        return None

    t0 = max(g1[0], g2[0])
    t1e = min(g1[-1], g2[-1])
    if t1e <= t0:
        return None

    common = np.arange(t0, t1e + 1, int(round(1e9 / fs)), dtype=np.int64)
    w_common = np.interp(common.astype(float), g1.astype(float), w.astype(float))
    g_common = np.interp(common.astype(float), g2.astype(float), g.astype(float))
    lag_df = sliding_window_lag(common, w_common, g_common, fs_hz=fs, window_s=15.0, step_s=3.0, max_lag_s=3.0)

    err = w_common - g_common
    summary = {
        "n_windows": len(lag_df),
        "median_lag_s": float(lag_df["lag_s"].median()) if len(lag_df) else np.nan,
        "p95_abs_lag_s": float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95)) if len(lag_df) else np.nan,
        "median_corr_score": float(lag_df["corr_score"].median()) if len(lag_df) else np.nan,
        "speed_rmse_mps": float(np.sqrt(np.nanmean(err ** 2))),
        "speed_mae_mps": float(np.nanmean(np.abs(err))),
    }
    pd.DataFrame([summary]).to_csv(outdir / "T4_speed_gnss_residual_summary.csv", index=False)

    win_start_s, win_end_s = pick_representative_time_window(common, w_common - g_common, 20.0, fs)
    mask = (common / 1e9 >= win_start_s) & (common / 1e9 <= win_end_s)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(common[mask] / 1e9 - win_start_s, w_common[mask], label="Wheel speed")
    axes[0].plot(common[mask] / 1e9 - win_start_s, g_common[mask], label="GNSS speed")
    axes[0].set_xlabel("Time within representative window (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Representative speed overlay")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
    axes[1].axvline(summary["median_lag_s"], linestyle="--", label=f"median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("F6. Wheel speed vs GNSS speed temporal consistency")
    plt.tight_layout()
    plt.savefig(outdir / "F6_speed_gnss_temporal_consistency.png", dpi=300)
    plt.close(fig)
    return summary


def analyze_lidar_frames(lidar_dfs: Dict[str, pd.DataFrame], outdir: Path, default_rate_hz: float):
    rows = []
    interval_box = []
    interval_labels = []
    points_box = []
    points_labels = []

    for name, df in lidar_dfs.items():
        if df is None or len(df) == 0 or "t_unix_ns" not in df.columns:
            continue
        ts = clean_time_ns(df["t_unix_ns"])
        rate = infer_nominal_rate_hz(ts, fallback=default_rate_hz)
        row = compute_health_row(name, ts, rate, ts[0], ts[-1])
        rows.append(row)

        dt_ms = compute_intervals_sec(ts) * 1e3
        if dt_ms.size > 0:
            interval_box.append(dt_ms)
            interval_labels.append(name)

        if "point_count" in df.columns:
            pts = ensure_numeric(df["point_count"]).dropna().to_numpy()
            if pts.size > 0:
                points_box.append(pts)
                points_labels.append(name)

    if not rows:
        return None

    out_df = pd.DataFrame(rows).sort_values("topic_name").reset_index(drop=True)
    out_df.to_csv(outdir / "T6_lidar_frame_health_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if interval_box:
        axes[0].boxplot(interval_box, vert=False, showfliers=False)
        axes[0].set_yticklabels(interval_labels)
        axes[0].set_xlabel("Frame interval Δt (ms)")
        axes[0].set_title("Frame interval stability")
        axes[0].grid(True, axis="x", linestyle="--", alpha=0.3)
    else:
        axes[0].set_visible(False)

    if points_box:
        axes[1].boxplot(points_box, vert=False, showfliers=False)
        axes[1].set_yticklabels(points_labels)
        axes[1].set_xlabel("Points per frame")
        axes[1].set_title("Points-per-frame distribution")
        axes[1].grid(True, axis="x", linestyle="--", alpha=0.3)
    else:
        axes[1].set_visible(False)

    fig.suptitle("F8. LiDAR frame health")
    plt.tight_layout()
    plt.savefig(outdir / "F8_lidar_frame_health.png", dpi=300)
    plt.close(fig)
    return out_df


def read_packet_csv(path: Path) -> Tuple[str, np.ndarray]:
    df = pd.read_csv(path)
    if "frame.time_epoch" not in df.columns:
        raise ValueError(f"{path} missing column frame.time_epoch")
    t_ns = pd.to_numeric(df["frame.time_epoch"], errors="coerce") * 1e9
    ts = clean_time_ns(t_ns)
    name = path.stem
    return name, ts


def analyze_lidar_packets(packet_paths: List[Path], session_start_ns: float, session_end_ns: float, outdir: Path):
    rows = []
    labels, box = [], []
    for path in packet_paths:
        name, ts = read_packet_csv(path)
        ts = ts[(ts >= session_start_ns) & (ts <= session_end_ns)]
        if ts.size == 0:
            continue
        row = compute_health_row(name, ts, None, session_start_ns, session_end_ns)
        rows.append(row)
        dt_ms = compute_intervals_sec(ts) * 1e3
        if dt_ms.size > 0:
            labels.append(name)
            box.append(dt_ms)
    if not rows:
        return None
    out_df = pd.DataFrame(rows).sort_values("topic_name").reset_index(drop=True)
    out_df.to_csv(outdir / "T7_lidar_packet_health_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.5 * len(labels))))
    ax.boxplot(box, vert=False, showfliers=False)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Packet inter-arrival interval Δt (ms)")
    ax.set_title("F7. LiDAR packet interval stability")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "F7_lidar_packet_interval_boxplot.png", dpi=300)
    plt.close(fig)
    return out_df


def find_behavior_candidates(df_speed: Optional[pd.DataFrame], df_steer: Optional[pd.DataFrame],
                             df_eye: Optional[pd.DataFrame], outdir: Path):
    if df_speed is None or df_steer is None or df_eye is None:
        pd.DataFrame(columns=["name", "start_s", "end_s", "score"]).to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    if "speed_mps" not in df_speed.columns:
        pd.DataFrame(columns=["name", "start_s", "end_s", "score"]).to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
    if steer_col not in df_steer.columns:
        pd.DataFrame(columns=["name", "start_s", "end_s", "score"]).to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    gaze_df = compute_gaze_yaw_deg(df_eye)
    if len(gaze_df) == 0:
        pd.DataFrame(columns=["name", "start_s", "end_s", "score"]).to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    # Resample all to 10 Hz over common overlap
    t_speed = clean_time_ns(df_speed["t_unix_ns"])
    y_speed = ensure_numeric(df_speed.set_index("t_unix_ns").loc[t_speed.astype(np.int64), "speed_mps"]).to_numpy()

    t_steer = clean_time_ns(df_steer["t_unix_ns"])
    y_steer = ensure_numeric(df_steer.set_index("t_unix_ns").loc[t_steer.astype(np.int64), steer_col]).to_numpy()

    t_gaze = clean_time_ns(gaze_df["t_unix_ns"])
    y_gaze = ensure_numeric(gaze_df.set_index("t_unix_ns").loc[t_gaze.astype(np.int64), "gaze_yaw_deg"]).to_numpy()

    fs = 10.0
    gs, spd = resample_series(t_speed, y_speed, fs)
    gt, steer = resample_series(t_steer, y_steer, fs)
    gg, gaze = resample_series(t_gaze, y_gaze, fs)
    if gs.size == 0 or gt.size == 0 or gg.size == 0:
        pd.DataFrame(columns=["name", "start_s", "end_s", "score"]).to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    t0 = max(gs[0], gt[0], gg[0])
    t1 = min(gs[-1], gt[-1], gg[-1])
    common = np.arange(t0, t1 + 1, int(round(1e9 / fs)), dtype=np.int64)

    spd_c = np.interp(common.astype(float), gs.astype(float), spd.astype(float))
    steer_c = np.interp(common.astype(float), gt.astype(float), steer.astype(float))
    gaze_c = np.interp(common.astype(float), gg.astype(float), gaze.astype(float))

    speed_drop = np.maximum(0, -np.gradient(spd_c) * fs)
    steer_mag = np.abs(np.gradient(steer_c) * fs) + 0.2 * np.abs(steer_c)
    gaze_activity = np.abs(np.gradient(gaze_c) * fs) + 0.2 * np.abs(gaze_c)

    score_signal = normalized(speed_drop) + normalized(steer_mag) + normalized(gaze_activity)
    score_signal = np.nan_to_num(score_signal, nan=0.0)

    win_s = 12.0
    n_win = int(round(win_s * fs))
    if len(score_signal) < n_win:
        cand = pd.DataFrame([{
            "name": "candidate_01",
            "start_s": common[0] / 1e9,
            "end_s": common[-1] / 1e9,
            "score": float(np.nanmean(score_signal))
        }])
        cand.to_csv(outdir / "F16_candidate_segments.csv", index=False)
        return

    rows = []
    for i in range(0, len(score_signal) - n_win + 1, int(2 * fs)):
        j = i + n_win
        rows.append({
            "name": f"candidate_{len(rows)+1:02d}",
            "start_s": common[i] / 1e9,
            "end_s": common[j - 1] / 1e9,
            "score": float(np.nanmean(score_signal[i:j]))
        })

    cand = pd.DataFrame(rows).sort_values("score", ascending=False).head(10).reset_index(drop=True)
    cand.to_csv(outdir / "F16_candidate_segments.csv", index=False)


def plot_behavior_segment(df_speed: pd.DataFrame, df_steer: pd.DataFrame, df_eye: pd.DataFrame,
                          segment: Dict, outdir: Path):
    name = segment["name"]
    start_s = float(segment["start_s"])
    end_s = float(segment["end_s"])

    steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
    gaze_df = compute_gaze_yaw_deg(df_eye)

    def crop(df):
        if df is None or len(df) == 0:
            return df
        t = ensure_numeric(df["t_unix_ns"]) / 1e9
        return df[(t >= start_s) & (t <= end_s)].copy()

    ds = crop(df_speed)
    dt = crop(df_steer)
    dg = crop(gaze_df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    if ds is not None and len(ds) and "speed_mps" in ds.columns:
        t = ensure_numeric(ds["t_unix_ns"]) / 1e9 - start_s
        axes[0].plot(t, ensure_numeric(ds["speed_mps"]), label="speed")
        axes[0].set_ylabel("Speed (m/s)")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.3)

    if dt is not None and len(dt) and steer_col in dt.columns:
        t = ensure_numeric(dt["t_unix_ns"]) / 1e9 - start_s
        axes[1].plot(t, ensure_numeric(dt[steer_col]), label="steering")
        axes[1].set_ylabel("Steering (deg)")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.3)

    if dg is not None and len(dg):
        t = ensure_numeric(dg["t_unix_ns"]) / 1e9 - start_s
        yaw = ensure_numeric(dg["gaze_yaw_deg"])
        axes[2].plot(t, yaw, label="gaze yaw")
        left_t = t[dg["glance_left"] > 0]
        right_t = t[dg["glance_right"] > 0]
        if len(left_t):
            axes[2].scatter(left_t, np.full_like(left_t, 35.0), marker="|", label="look-left")
        if len(right_t):
            axes[2].scatter(right_t, np.full_like(right_t, -35.0), marker="|", label="look-right")
        axes[2].set_ylabel("Gaze yaw (deg)")
        axes[2].set_xlabel("Time within segment (s)")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(f"F16. Behavioral sanity check: {name}")
    plt.tight_layout()
    plt.savefig(outdir / f"F16_behavior_{name}.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Merged XLSX from merge_bikelab_csvs_to_xlsx.py")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--lidar-packet-csvs", nargs="*", default=[], help="Optional LiDAR packet CSVs")
    ap.add_argument("--lidar-frame-rate-hz", type=float, default=DEFAULT_LIDAR_FRAME_RATE_HZ,
                    help="Fallback nominal frame rate for LiDAR frame streams")
    ap.add_argument("--behavior-segments", default="[]",
                    help='Optional JSON list, e.g. \'[{"name":"urban_turn","start_s":120,"end_s":135}]\'')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(args.xlsx)
    stream_rows = []
    stream_health_rows = []
    sheet_data = {}

    # Load available sheets
    for cfg in STREAM_CONFIG:
        df = find_sheet_or_none(xl, cfg["sheet"])
        sheet_data[cfg["sheet"]] = df
        if df is None or "t_unix_ns" not in df.columns:
            continue
        ts = clean_time_ns(df["t_unix_ns"])
        if ts.size == 0:
            continue
        nominal = cfg["nominal_rate_hz"]
        if nominal is None and cfg["sheet"].startswith("lidar_f_"):
            nominal = infer_nominal_rate_hz(ts, fallback=args.lidar_frame_rate_hz)
        elif nominal is None:
            nominal = infer_nominal_rate_hz(ts, fallback=None)
        stream_rows.append((cfg["name"], ts, nominal))

    if not stream_rows:
        raise RuntimeError("No valid streams found in XLSX.")

    session_start_ns = min(ts[0] for _, ts, _ in stream_rows)
    session_end_ns = max(ts[-1] for _, ts, _ in stream_rows)
    total_session_s = (session_end_ns - session_start_ns) / 1e9

    for name, ts, nominal in stream_rows:
        row = compute_health_row(name, ts, nominal, session_start_ns, session_end_ns)
        row["_stream_duration_s"] = (ts[-1] - ts[0]) / 1e9
        stream_health_rows.append(row)

    stats_df = pd.DataFrame(stream_health_rows).sort_values("topic_name").reset_index(drop=True)
    stats_df.to_csv(outdir / "T1_topic_health_summary.csv", index=False)

    # Main figures
    plot_f1_timeline(stream_rows, session_start_ns, session_end_ns, outdir / "F1_sensor_availability_timeline.png")
    plot_f2_boxplot(stream_rows, outdir / "F2_inter_message_interval_boxplot.png")
    plot_f3_gap(stats_df, outdir / "F3_gap_statistics.png")
    plot_f4_alignment(stats_df, outdir / "F4_relative_start_end_alignment.png")

    # Physics-based checks
    analyze_steering_imu(sheet_data.get("potentiometer"), sheet_data.get("imu"), outdir)
    analyze_speed_gnss(sheet_data.get("wheel_speed"), sheet_data.get("gps"), outdir)

    # LiDAR frame health
    lidar_dfs = {}
    for sheet_name, lidar_name in [("lidar_f_200", "lidar_frame_200"),
                                   ("lidar_f_201", "lidar_frame_201"),
                                   ("lidar_f_202", "lidar_frame_202")]:
        if sheet_data.get(sheet_name) is not None:
            lidar_dfs[lidar_name] = sheet_data[sheet_name]
    analyze_lidar_frames(lidar_dfs, outdir, args.lidar_frame_rate_hz)

    # Optional LiDAR packet health (same session window as merged XLSX)
    packet_paths = [Path(p) for p in args.lidar_packet_csvs]
    if packet_paths:
        analyze_lidar_packets(packet_paths, session_start_ns, session_end_ns, outdir)

    # F16 candidates
    try:
        find_behavior_candidates(
            sheet_data.get("wheel_speed"),
            sheet_data.get("potentiometer"),
            sheet_data.get("eyetracker"),
            outdir,
        )
    except Exception as e:
        print(f"[WARN] F16 candidate generation skipped: {e}")

    # Optional F16 explicit plots
    try:
        segments = json.loads(args.behavior_segments)
    except Exception:
        segments = []

    if segments:
        for seg in segments:
            plot_behavior_segment(
                sheet_data.get("wheel_speed"),
                sheet_data.get("potentiometer"),
                sheet_data.get("eyetracker"),
                seg,
                outdir,
            )

    # Future analyses manifest
    future = pd.DataFrame([
        {
            "analysis": "NTP/PTP synchronization quality",
            "status": "future",
            "required_inputs": "chrony / ptp4l / phc2sys offset logs",
            "possible_outputs": "T2 offset summary, offset-vs-time, offset distribution"
        },
        {
            "analysis": "Ego-motion validation against fused odometry",
            "status": "future",
            "required_inputs": "/odometry/filtered or fused odometry CSV, plus RTK reference",
            "possible_outputs": "F9 trajectory comparison, F10 speed consistency with fused, T5 ego-motion accuracy"
        },
        {
            "analysis": "Interaction metrics and gaze-object association",
            "status": "future",
            "required_inputs": "LiDAR detections / tracking results",
            "possible_outputs": "minimum lateral distance, TTC/PET, gaze-on-threat, maneuver anticipation"
        },
        {
            "analysis": "Multi-LiDAR spatial consistency",
            "status": "future",
            "required_inputs": "decoded point clouds and finalized extrinsics",
            "possible_outputs": "point-cloud overlay error, ICP residual, ground-plane consistency"
        },
    ])
    future.to_csv(outdir / "future_analyses_manifest.csv", index=False)

    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
