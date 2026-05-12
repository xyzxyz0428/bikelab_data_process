#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOMINAL_RATES = {
    "break_sensor_right": 10.0,
    "gps": 10.0,
    "powermeter": 4.0,
    "steering": 10.0,
    "wheel_speed": 4.0,
    "imu": None,
}

EYE_SENSOR_NOMINAL_RATES = {
    "Eye Tracker": 50.0,
    "Gyroscope": 100.0,
    "Accelerometer": 100.0,
    "Magnetometer": 10.0,
    "Scene camera": 25.0,
}


def normalize_col_name(name: str) -> str:
    return str(name).strip().replace("\n", " ").replace("\r", " ")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def find_sheet(xl: pd.ExcelFile, candidates: List[str]) -> Optional[pd.DataFrame]:
    normalized_map = {normalize_col_name(s).lower(): s for s in xl.sheet_names}
    for c in candidates:
        key = normalize_col_name(c).lower()
        if key in normalized_map:
            return clean_columns(pd.read_excel(xl, sheet_name=normalized_map[key]))
    return None


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_unix_ns_scalar(x) -> Optional[int]:
    try:
        val = float(x)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    if abs(val) > 1e17:  # ns
        return int(round(val))
    if abs(val) > 1e14:  # us
        return int(round(val * 1e3))
    if abs(val) > 1e11:  # ms
        return int(round(val * 1e6))
    if abs(val) > 1e8:   # s
        return int(round(val * 1e9))
    return None


def clean_time_ns(series: pd.Series) -> np.ndarray:
    t = ensure_numeric(series).to_numpy(dtype=np.float64)
    t = t[np.isfinite(t) & (t > 0)]
    if t.size == 0:
        return np.array([], dtype=np.float64)
    med = np.median(t)
    if med > 1e17:
        t = t[(t >= med * 0.1) & (t <= med * 10.0)]
    t = np.sort(t)
    if t.size > 1:
        t = t[np.insert(np.diff(t) > 0, 0, True)]
    return t


def clean_time_and_values(t_series: pd.Series, y_series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    t = ensure_numeric(t_series).to_numpy(dtype=np.float64)
    y = ensure_numeric(y_series).to_numpy(dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y) & (t > 0)
    t = t[mask]
    y = y[mask]

    if t.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    med = np.median(t)
    if med > 1e17:
        mask = (t >= med * 0.1) & (t <= med * 10.0)
        t = t[mask]
        y = y[mask]

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    if t.size > 1:
        keep = np.insert(np.diff(t) > 0, 0, True)
        t = t[keep]
        y = y[keep]

    return t, y


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


def count_missing_messages(dt_sec: np.ndarray, nominal_rate_hz: Optional[float] = None, gap_factor: float = 2.0):
    # Robust missing estimation based on actual median dt
    if dt_sec.size == 0:
        return 0, 0, 0.0

    ref_dt = np.median(dt_sec)
    if not np.isfinite(ref_dt) or ref_dt <= 0:
        return 0, 0, float(np.max(dt_sec))

    gap_mask = dt_sec > gap_factor * ref_dt
    gap_count = int(np.sum(gap_mask))
    longest_gap_s = float(np.max(dt_sec))

    missing_msg_count = 0
    for dt in dt_sec[gap_mask]:
        missing_msg_count += max(0, int(round(dt / ref_dt)) - 1)

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
            "_stream_duration_s": 0.0,
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
    stream_duration_s = (ts_ns[-1] - ts_ns[0]) / 1e9
    session_coverage_pct = 100.0 * stream_duration_s / total_session_sec

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
        "_stream_duration_s": stream_duration_s,
    }


def resample_series(t_ns: np.ndarray, y: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if t_ns.size == 0 or y.size == 0:
        return np.array([]), np.array([])
    mask = np.isfinite(t_ns) & np.isfinite(y)
    t_ns = t_ns[mask]
    y = y[mask]
    if t_ns.size < 2:
        return np.array([]), np.array([])
    order = np.argsort(t_ns)
    t_ns = t_ns[order]
    y = y[order]
    step_ns = int(round(1e9 / fs_hz))
    grid_ns = np.arange(int(t_ns[0]), int(t_ns[-1]) + 1, step_ns, dtype=np.int64)
    y_grid = np.interp(grid_ns.astype(float), t_ns.astype(float), y.astype(float))
    return grid_ns, y_grid


def normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmean(x)
    std = np.nanstd(x)
    return x / std if std > 1e-12 else x


def crosscorr_lag_seconds(x: np.ndarray, y: np.ndarray, fs_hz: float, max_lag_s: float = 2.0):
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return np.nan, np.nan
    x = normalized(x)
    y = normalized(y)
    max_lag = int(round(max_lag_s * fs_hz))
    corr_full = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))
    keep = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr_full[keep]
    lags = lags[keep]
    if corr.size == 0:
        return np.nan, np.nan
    idx = int(np.argmax(corr))
    return lags[idx] / fs_hz, corr[idx] / max(len(x), 1)


def sliding_window_lag(t_ns: np.ndarray, x: np.ndarray, y: np.ndarray,
                       fs_hz: float, window_s: float = 10.0, step_s: float = 2.0,
                       max_lag_s: float = 2.0):
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


def convert_speed_to_mps(values: np.ndarray, unit: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    unit = unit.lower()
    if unit == "mps":
        return values
    if unit == "mmps":
        return values / 1000.0
    if unit == "kmph":
        return values / 3.6
    if unit == "mph":
        return values * 0.44704
    raise ValueError(f"Unsupported speed unit: {unit}")


def gps_heading_acc_to_deg(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * 1e-5


def compute_gaze_yaw_deg(df_eye: pd.DataFrame) -> pd.DataFrame:
    req = [
        "Gaze direction left X", "Gaze direction left Z",
        "Gaze direction right X", "Gaze direction right Z",
        "Validity left", "Validity right",
    ]
    if any(c not in df_eye.columns for c in req):
        return pd.DataFrame(columns=["t_unix_ns", "gaze_yaw_deg", "glance_left", "glance_right"])

    lx = ensure_numeric(df_eye["Gaze direction left X"]).to_numpy()
    lz = ensure_numeric(df_eye["Gaze direction left Z"]).to_numpy()
    rx = ensure_numeric(df_eye["Gaze direction right X"]).to_numpy()
    rz = ensure_numeric(df_eye["Gaze direction right Z"]).to_numpy()
    vl = ensure_numeric(df_eye["Validity left"]).fillna(0).to_numpy()
    vr = ensure_numeric(df_eye["Validity right"]).fillna(0).to_numpy()
    t = ensure_numeric(df_eye["t_unix_ns"]).to_numpy()

    x = np.where((vl > 0) & (vr > 0), (lx + rx) / 2.0,
                 np.where(vl > 0, lx, np.where(vr > 0, rx, np.nan)))
    z = np.where((vl > 0) & (vr > 0), (lz + rz) / 2.0,
                 np.where(vl > 0, lz, np.where(vr > 0, rz, np.nan)))

    yaw_deg = np.degrees(np.arctan2(x, z))
    out = pd.DataFrame({"t_unix_ns": t, "gaze_yaw_deg": yaw_deg})
    out["glance_left"] = (out["gaze_yaw_deg"] > 25).astype(int)
    out["glance_right"] = (out["gaze_yaw_deg"] < -25).astype(int)
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["t_unix_ns"])


def crop_df_by_ns(df: Optional[pd.DataFrame], start_ns: int, end_ns: int) -> Optional[pd.DataFrame]:
    if df is None or "t_unix_ns" not in df.columns:
        return None
    t = ensure_numeric(df["t_unix_ns"])
    return df[(t >= start_ns) & (t <= end_ns)].copy()


def plot_f2_boxplot(streams, out_path):
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
    ax.set_title("F2. Sampling interval stability across key sensor streams")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_f4_alignment(stats_df, out_path):
    plot_df = stats_df.sort_values("start_offset_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(plot_df))))
    y = np.arange(len(plot_df))[::-1]

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.broken_barh([(row["start_offset_s"], max(0.01, row["_stream_duration_s"]))], (yi - 0.35, 0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["topic_name"])
    ax.set_xlabel("Time since session start (s)")
    ax.set_title("F4. Relative start and end alignment of sensor streams")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_gps_accuracy(df_gps, out_path, title):
    if df_gps is None or len(df_gps) == 0:
        return

    t = ensure_numeric(df_gps["t_unix_ns"]) / 1e9
    t = t - np.nanmin(t)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    if "h_acc" in df_gps.columns:
        axes[0].plot(t, ensure_numeric(df_gps["h_acc"]) / 1000.0)
        axes[0].set_ylabel("hAcc (m)")
        axes[0].grid(True, linestyle="--", alpha=0.3)

    if "v_acc" in df_gps.columns:
        axes[1].plot(t, ensure_numeric(df_gps["v_acc"]) / 1000.0)
        axes[1].set_ylabel("vAcc (m)")
        axes[1].grid(True, linestyle="--", alpha=0.3)

    if "s_acc" in df_gps.columns:
        axes[2].plot(t, ensure_numeric(df_gps["s_acc"]) / 1000.0)
        axes[2].set_ylabel("sAcc (m/s)")
        axes[2].grid(True, linestyle="--", alpha=0.3)

    if "head_acc" in df_gps.columns:
        axes[3].plot(t, gps_heading_acc_to_deg(ensure_numeric(df_gps["head_acc"])))
        axes[3].set_ylabel("headAcc (deg)")
        axes[3].set_xlabel("Time since segment start (s)")
        axes[3].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze_steering_imu(df_steer, df_imu, out_csv, out_fig, title):
    if df_steer is None or df_imu is None:
        return

    steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
    imu_col = "headingspeed" if "headingspeed" in df_imu.columns else ("gyro_z" if "gyro_z" in df_imu.columns else None)

    if steer_col not in df_steer.columns or imu_col is None:
        return

    t1, y1 = clean_time_and_values(df_steer["t_unix_ns"], df_steer[steer_col])
    t2, y2 = clean_time_and_values(df_imu["t_unix_ns"], df_imu[imu_col])

    fs = 50.0
    g1, s = resample_series(t1, y1, fs)
    g2, imu = resample_series(t2, y2, fs)

    if g1.size == 0 or g2.size == 0:
        return

    t0 = max(g1[0], g2[0])
    t1e = min(g1[-1], g2[-1])
    if t1e <= t0:
        return

    common = np.arange(t0, t1e + 1, int(round(1e9 / fs)), dtype=np.int64)
    s_common = np.interp(common.astype(float), g1.astype(float), s.astype(float))
    imu_common = np.interp(common.astype(float), g2.astype(float), imu.astype(float))
    steering_rate = np.gradient(s_common) * fs
    lag_df = sliding_window_lag(common, steering_rate, imu_common, fs_hz=fs)

    summary = {
        "n_windows": len(lag_df),
        "median_lag_s": float(lag_df["lag_s"].median()) if len(lag_df) else np.nan,
        "p95_abs_lag_s": float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95)) if len(lag_df) else np.nan,
        "median_corr_score": float(lag_df["corr_score"].median()) if len(lag_df) else np.nan,
        "steering_signal": steer_col,
        "imu_signal": imu_col,
    }
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    score = np.abs(steering_rate)
    n_win = int(round(20.0 * fs))
    if len(score) > n_win:
        idx = int(np.argmax(np.convolve(score, np.ones(n_win), mode="valid")))
        mask = np.zeros_like(score, dtype=bool)
        mask[idx:idx + n_win] = True
    else:
        mask = np.ones_like(score, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    rel_t = common[mask] / 1e9 - common[mask][0] / 1e9
    axes[0].plot(rel_t, normalized(steering_rate[mask]), label="Steering rate")
    axes[0].plot(rel_t, normalized(imu_common[mask]), label="IMU yaw-related")
    axes[0].set_xlabel("Time within representative window (s)")
    axes[0].set_ylabel("Normalized signal")
    axes[0].set_title("Representative signal overlay")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
    if len(lag_df):
        axes[1].axvline(summary["median_lag_s"], linestyle="--", label=f"median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def analyze_speed_compare(df_speed, df_gps, out_csv, out_fig, title, wheel_speed_unit, gps_speed_unit):
    if df_speed is None or df_gps is None or "speed_mps" not in df_speed.columns or "g_speed" not in df_gps.columns:
        return

    t1, v1 = clean_time_and_values(df_speed["t_unix_ns"], df_speed["speed_mps"])
    t2, v2 = clean_time_and_values(df_gps["t_unix_ns"], df_gps["g_speed"])

    v1 = convert_speed_to_mps(v1, wheel_speed_unit)
    v2 = convert_speed_to_mps(v2, gps_speed_unit)

    fs = 10.0
    g1, w = resample_series(t1, v1, fs)
    g2, g = resample_series(t2, v2, fs)

    if g1.size == 0 or g2.size == 0:
        return

    t0 = max(g1[0], g2[0])
    t1e = min(g1[-1], g2[-1])
    if t1e <= t0:
        return

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
        "wheel_speed_unit_in": wheel_speed_unit,
        "gps_speed_unit_in": gps_speed_unit,
    }
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    score = np.abs(np.gradient(g_common) * fs) + np.abs(np.gradient(w_common) * fs)
    n_win = int(round(20.0 * fs))
    if len(score) > n_win:
        idx = int(np.argmax(np.convolve(score, np.ones(n_win), mode="valid")))
        mask = np.zeros_like(score, dtype=bool)
        mask[idx:idx + n_win] = True
    else:
        mask = np.ones_like(score, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    rel_t = common[mask] / 1e9 - common[mask][0] / 1e9
    axes[0].plot(rel_t, w_common[mask], label="Wheel speed")
    axes[0].plot(rel_t, g_common[mask], label="GNSS speed")
    axes[0].set_xlabel("Time within representative window (s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Representative speed overlay")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
    if len(lag_df):
        axes[1].axvline(summary["median_lag_s"], linestyle="--", label=f"median={summary['median_lag_s']:.3f}s")
    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def plot_lidar_frame_health(lidar_frames, out_csv, out_fig):
    rows = []
    interval_box = []
    interval_labels = []
    points_box = []
    points_labels = []

    for name, df in lidar_frames.items():
        if df is None or "t_unix_ns" not in df.columns or len(df) == 0:
            continue

        ts = clean_time_ns(df["t_unix_ns"])
        if ts.size == 0:
            continue

        rate = infer_nominal_rate_hz(ts, fallback=10.0)
        rows.append(compute_health_row(name, ts, rate, ts[0], ts[-1]))

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
        return

    pd.DataFrame(rows).to_csv(out_csv, index=False)

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
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def compute_t7_eye_quality(df_eye, out_csv):
    if df_eye is None or len(df_eye) == 0:
        return

    if "Sensor" in df_eye.columns:
        sub = df_eye[df_eye["Sensor"].astype(str).str.strip() == "Eye Tracker"].copy()
        if len(sub) > 0:
            df_eye = sub

    row = {}

    if "Validity left" in df_eye.columns:
        vl = ensure_numeric(df_eye["Validity left"])
        row["valid_left_ratio"] = float((vl > 0).mean())

    if "Validity right" in df_eye.columns:
        vr = ensure_numeric(df_eye["Validity right"])
        row["valid_right_ratio"] = float((vr > 0).mean())

    if "Validity left" in df_eye.columns and "Validity right" in df_eye.columns:
        vl = ensure_numeric(df_eye["Validity left"])
        vr = ensure_numeric(df_eye["Validity right"])
        row["both_valid_ratio"] = float(((vl > 0) & (vr > 0)).mean())

    if "Eye movement type" in df_eye.columns:
        eye_type = df_eye["Eye movement type"].astype(str).str.lower()
        row["fixation_ratio"] = float(eye_type.str.contains("fix").mean())
        row["saccade_ratio"] = float(eye_type.str.contains("sacc").mean())

    if "Eye movement event duration" in df_eye.columns:
        dur = ensure_numeric(df_eye["Eye movement event duration"])
        if dur.notna().any():
            row["eye_event_duration_median_ms"] = float(dur.median())
            row["eye_event_duration_p95_ms"] = float(np.nanpercentile(dur.dropna(), 95))

    pd.DataFrame([row]).to_csv(out_csv, index=False)


def plot_f15_gaze_stability(df_eye, out_fig):
    if df_eye is None or len(df_eye) == 0:
        return

    if "Sensor" in df_eye.columns:
        sub = df_eye[df_eye["Sensor"].astype(str).str.strip() == "Eye Tracker"].copy()
        if len(sub) > 0:
            df_eye = sub

    gaze_df = compute_gaze_yaw_deg(df_eye)
    if len(gaze_df) == 0:
        return

    t, yaw = clean_time_and_values(gaze_df["t_unix_ns"], gaze_df["gaze_yaw_deg"])
    if t.size == 0:
        return

    dt_s = np.diff(t) / 1e9
    yaw_rate = np.diff(yaw) / np.where(dt_s > 1e-6, dt_s, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(yaw[np.isfinite(yaw)], bins=60)
    axes[0].set_xlabel("Gaze yaw (deg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of gaze yaw")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(yaw_rate[np.isfinite(yaw_rate)], bins=60)
    axes[1].set_xlabel("Gaze yaw rate (deg/s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of gaze angular velocity")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("F15. Eye-tracking / head-pose signal stability")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def plot_scenario_turning_control(df_gps, df_steer, df_imu, df_brake, out_fig, title, gps_speed_unit):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    if df_gps is not None and "g_speed" in df_gps.columns:
        t, spd = clean_time_and_values(df_gps["t_unix_ns"], df_gps["g_speed"])
        spd = convert_speed_to_mps(spd, gps_speed_unit)
        if t.size:
            axes[0].plot(t / 1e9 - t[0] / 1e9, spd, label="GNSS speed")
            axes[0].set_ylabel("Speed (m/s)")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.3)

    if df_steer is not None:
        steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
        if steer_col in df_steer.columns:
            t, steer = clean_time_and_values(df_steer["t_unix_ns"], df_steer[steer_col])
            if t.size:
                axes[1].plot(t / 1e9 - t[0] / 1e9, steer, label="Steering angle")
                axes[1].set_ylabel("Steer (deg)")
                axes[1].legend()
                axes[1].grid(True, linestyle="--", alpha=0.3)

    if df_imu is not None:
        imu_col = "headingspeed" if "headingspeed" in df_imu.columns else ("gyro_z" if "gyro_z" in df_imu.columns else None)
        if imu_col is not None:
            t, imu = clean_time_and_values(df_imu["t_unix_ns"], df_imu[imu_col])
            if t.size:
                axes[2].plot(t / 1e9 - t[0] / 1e9, imu, label=imu_col)
                axes[2].set_ylabel("Yaw-related")
                axes[2].legend()
                axes[2].grid(True, linestyle="--", alpha=0.3)

    if df_brake is not None and "force_total_n" in df_brake.columns:
        t, brake = clean_time_and_values(df_brake["t_unix_ns"], df_brake["force_total_n"])
        if t.size:
            axes[3].plot(t / 1e9 - t[0] / 1e9, brake, label="Brake force")
            axes[3].set_ylabel("Brake (N)")
            axes[3].set_xlabel("Time since scenario start (s)")
            axes[3].legend()
            axes[3].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def plot_scenario_closed_loop(df_eye, df_steer, df_brake, df_gps, df_power, out_fig, title, gps_speed_unit):
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    if df_eye is not None:
        if "Sensor" in df_eye.columns:
            sub = df_eye[df_eye["Sensor"].astype(str).str.strip() == "Eye Tracker"].copy()
            if len(sub) > 0:
                df_eye = sub

        gaze_df = compute_gaze_yaw_deg(df_eye)
        if len(gaze_df):
            t, yaw = clean_time_and_values(gaze_df["t_unix_ns"], gaze_df["gaze_yaw_deg"])
            if t.size:
                rel_t = t / 1e9 - t[0] / 1e9
                axes[0].plot(rel_t, yaw, label="Gaze yaw")
                left_mask = yaw > 25
                right_mask = yaw < -25
                axes[0].scatter(rel_t[left_mask], np.full(np.sum(left_mask), 35.0), marker="|", label="look-left")
                axes[0].scatter(rel_t[right_mask], np.full(np.sum(right_mask), -35.0), marker="|", label="look-right")
                axes[0].set_ylabel("Gaze yaw (deg)")
                axes[0].legend()
                axes[0].grid(True, linestyle="--", alpha=0.3)

    if df_steer is not None:
        steer_col = "angle_deg_clamped" if "angle_deg_clamped" in df_steer.columns else "angle_deg"
        if steer_col in df_steer.columns:
            t, steer = clean_time_and_values(df_steer["t_unix_ns"], df_steer[steer_col])
            if t.size:
                axes[1].plot(t / 1e9 - t[0] / 1e9, steer, label="Steering")
                axes[1].set_ylabel("Steer (deg)")
                axes[1].legend()
                axes[1].grid(True, linestyle="--", alpha=0.3)

    if df_brake is not None and "force_total_n" in df_brake.columns:
        t, brake = clean_time_and_values(df_brake["t_unix_ns"], df_brake["force_total_n"])
        if t.size:
            axes[2].plot(t / 1e9 - t[0] / 1e9, brake, label="Brake force")
            axes[2].set_ylabel("Brake (N)")
            axes[2].legend()
            axes[2].grid(True, linestyle="--", alpha=0.3)

    if df_power is not None and "p10_instantaneous_power_w" in df_power.columns:
        t, pwr = clean_time_and_values(df_power["t_unix_ns"], df_power["p10_instantaneous_power_w"])
        if t.size:
            axes[3].plot(t / 1e9 - t[0] / 1e9, pwr, label="Pedal power")
            axes[3].set_ylabel("Power (W)")
            axes[3].legend()
            axes[3].grid(True, linestyle="--", alpha=0.3)

    if df_gps is not None and "g_speed" in df_gps.columns:
        t, spd = clean_time_and_values(df_gps["t_unix_ns"], df_gps["g_speed"])
        spd = convert_speed_to_mps(spd, gps_speed_unit)
        if t.size:
            axes[4].plot(t / 1e9 - t[0] / 1e9, spd, label="GNSS speed")
            axes[4].set_ylabel("Speed (m/s)")
            axes[4].set_xlabel("Time since scenario start (s)")
            axes[4].legend()
            axes[4].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Merged XLSX")
    ap.add_argument("--scenarios", required=True, help="critical_scenarios.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--gps-speed-unit", default="mmps", choices=["mps", "mmps", "kmph", "mph"])
    ap.add_argument("--wheel-speed-unit", default="mps", choices=["mps", "mmps", "kmph", "mph"])
    ap.add_argument("--scenario-ids", nargs="*", default=[], help="Optional list of scenario ids")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(args.xlsx)

    gps = find_sheet(xl, ["gps"])
    steer = find_sheet(xl, ["potentiometer"])
    wheel = find_sheet(xl, ["wheel_speed", "wheel speed"])
    imu = find_sheet(xl, ["imu"])
    power = find_sheet(xl, ["powermeter"])
    eye = find_sheet(xl, ["eyetracker"])
    brake = find_sheet(xl, ["break_sensor_right", "brake_sensor_right"])

    lidar_frames = {
        "lidar_frame_200": find_sheet(xl, ["lidar_f_200"]),
        "lidar_frame_201": find_sheet(xl, ["lidar_f_201"]),
        "lidar_frame_202": find_sheet(xl, ["lidar_f_202"]),
    }

    streams = []

    main_stream_defs = [
        ("gps", gps, NOMINAL_RATES["gps"]),
        ("steering", steer, NOMINAL_RATES["steering"]),
        ("wheel_speed", wheel, NOMINAL_RATES["wheel_speed"]),
        ("imu", imu, NOMINAL_RATES["imu"]),
        ("powermeter", power, NOMINAL_RATES["powermeter"]),
        ("break_sensor_right", brake, NOMINAL_RATES["break_sensor_right"]),
    ]

    for name, df, nominal in main_stream_defs:
        if df is None or "t_unix_ns" not in df.columns:
            continue
        ts = clean_time_ns(df["t_unix_ns"])
        if ts.size == 0:
            continue
        streams.append((name, ts, nominal))

    if eye is not None and "t_unix_ns" in eye.columns and "Sensor" in eye.columns:
        eye_tmp = eye.copy()
        eye_tmp["Sensor"] = eye_tmp["Sensor"].astype(str).str.strip()

        for sensor_name, nominal in EYE_SENSOR_NOMINAL_RATES.items():
            sub = eye_tmp[eye_tmp["Sensor"] == sensor_name]
            if len(sub) == 0:
                continue
            ts = clean_time_ns(sub["t_unix_ns"])
            if ts.size == 0:
                continue
            stream_name = f"eyetracker_{sensor_name.lower().replace(' ', '_')}"
            streams.append((stream_name, ts, nominal))
    elif eye is not None and "t_unix_ns" in eye.columns:
        ts = clean_time_ns(eye["t_unix_ns"])
        if ts.size > 0:
            streams.append(("eyetracker", ts, 50.0))

    for name, df in lidar_frames.items():
        if df is None or "t_unix_ns" not in df.columns:
            continue
        ts = clean_time_ns(df["t_unix_ns"])
        if ts.size == 0:
            continue
        nominal = infer_nominal_rate_hz(ts, fallback=10.0)
        streams.append((name, ts, nominal))

    if not streams:
        raise RuntimeError("No valid streams found in XLSX.")

    session_start_ns = min(ts[0] for _, ts, _ in streams)
    session_end_ns = max(ts[-1] for _, ts, _ in streams)

    t1 = pd.DataFrame(
        [compute_health_row(name, ts, nominal, session_start_ns, session_end_ns) for name, ts, nominal in streams]
    ).sort_values("topic_name").reset_index(drop=True)

    t1.to_csv(outdir / "T1_topic_health_summary.csv", index=False)

    plot_f2_boxplot(streams, outdir / "F2_sampling_interval_boxplot.png")
    plot_f4_alignment(t1, outdir / "F4_relative_stream_alignment.png")

    analyze_steering_imu(
        steer, imu,
        outdir / "T3_steering_imu_residual_summary.csv",
        outdir / "F5_steering_imu_overall.png",
        "F5. Steering-rate vs IMU yaw-rate temporal consistency"
    )

    analyze_speed_compare(
        wheel, gps,
        outdir / "T4_speed_gnss_residual_summary.csv",
        outdir / "F6_speed_compare_overall.png",
        "F6. Wheel speed vs GNSS speed temporal consistency",
        args.wheel_speed_unit,
        args.gps_speed_unit,
    )

    plot_gps_accuracy(gps, outdir / "GPS_accuracy_overall.png", "GPS accuracy indicators from ubx_nav_pvt estimates")
    compute_t7_eye_quality(eye, outdir / "T7_eye_tracking_quality.csv")
    plot_f15_gaze_stability(eye, outdir / "F15_gaze_head_stability.png")
    plot_lidar_frame_health(
        {k: v for k, v in lidar_frames.items() if v is not None},
        outdir / "T6_lidar_frame_health_summary.csv",
        outdir / "F8_lidar_frame_health.png"
    )

    scenarios = clean_columns(pd.read_csv(args.scenarios))
    col_map = {}
    for c in scenarios.columns:
        cl = c.lower().strip()
        if cl == "scenario_id":
            col_map[c] = "scenario_id"
        elif cl == "scenario_type":
            col_map[c] = "scenario_type"
        elif cl.startswith("initial time"):
            col_map[c] = "initial_time"
        elif cl == "start":
            col_map[c] = "start"
        elif cl == "end":
            col_map[c] = "end"
        elif cl == "note":
            col_map[c] = "note"
    scenarios = scenarios.rename(columns=col_map)

    if args.scenario_ids:
        wanted = set(str(x) for x in args.scenario_ids)
        scenarios = scenarios[scenarios["scenario_id"].astype(str).isin(wanted)].copy()

    scenarios["start_ns"] = scenarios["start"].apply(to_unix_ns_scalar)
    scenarios["end_ns"] = scenarios["end"].apply(to_unix_ns_scalar)
    scenarios = scenarios.dropna(subset=["start_ns", "end_ns"]).copy()
    scenarios["start_ns"] = scenarios["start_ns"].astype(np.int64)
    scenarios["end_ns"] = scenarios["end_ns"].astype(np.int64)

    for _, row in scenarios.iterrows():
        sid = str(row["scenario_id"])
        sname = row["scenario_type"] if "scenario_type" in row else "scenario"
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])

        gps_seg = crop_df_by_ns(gps, start_ns, end_ns)
        steer_seg = crop_df_by_ns(steer, start_ns, end_ns)
        imu_seg = crop_df_by_ns(imu, start_ns, end_ns)
        brake_seg = crop_df_by_ns(brake, start_ns, end_ns)
        power_seg = crop_df_by_ns(power, start_ns, end_ns)
        eye_seg = crop_df_by_ns(eye, start_ns, end_ns)
        wheel_seg = crop_df_by_ns(wheel, start_ns, end_ns)

        plot_gps_accuracy(
            gps_seg,
            outdir / f"scenario_{sid}_gps_accuracy.png",
            f"Scenario {sid} ({sname}): GPS accuracy indicators"
        )

        analyze_speed_compare(
            wheel_seg, gps_seg,
            outdir / f"scenario_{sid}_speed_compare_summary.csv",
            outdir / f"scenario_{sid}_speed_compare.png",
            f"Scenario {sid} ({sname}): wheel speed vs GNSS speed",
            args.wheel_speed_unit,
            args.gps_speed_unit,
        )

        plot_scenario_turning_control(
            gps_seg, steer_seg, imu_seg, brake_seg,
            outdir / f"scenario_{sid}_turning_control_dynamics.png",
            f"Scenario {sid} ({sname}): speed, steering, IMU, brake",
            args.gps_speed_unit,
        )

        plot_scenario_closed_loop(
            eye_seg, steer_seg, brake_seg, gps_seg, power_seg,
            outdir / f"scenario_{sid}_closed_loop.png",
            f"Scenario {sid} ({sname}): gaze, control input, and dynamic response",
            args.gps_speed_unit,
        )

    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()