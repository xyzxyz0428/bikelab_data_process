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


# =========================================================
# Fixed nominal rates
# =========================================================
NOMINAL_RATES = {
    "break_sensor_right": 10.0,
    "gps": 10.0,
    "powermeter": 4.0,
    "steering": 10.0,
    "wheel_speed": 4.0,
    "imu": None,          # keep empty in T1; no gap/missing calculation
    "back_camera": 30.0,
    "scene_camera": 25.0,
}

EYE_SENSOR_NOMINAL_RATES = {
    "Eye Tracker": 50.0,
    "Gyroscope": 100.0,
    "Accelerometer": 100.0,
    "Magnetometer": 10.0,
    "Scene camera": 30.0,
}

ACADEMIC_RED = "#8B1E3F"
ACADEMIC_GREEN = "#2F6B3B"
ACADEMIC_GREY = "#B7B7B7"
ABC_COLORS = {
    "A": "#1f77b4",   # blue
    "B": "#ff7f0e",   # orange
    "C": "#9467bd",   # purple
}


# =========================================================
# Basic helpers
# =========================================================
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


def parse_validity_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return ensure_numeric(series).fillna(0) > 0
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["valid", "1", "true", "yes"])


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


def crop_df_by_ns(df: Optional[pd.DataFrame], start_ns: int, end_ns: int) -> Optional[pd.DataFrame]:
    if df is None or "t_unix_ns" not in df.columns:
        return None
    t = ensure_numeric(df["t_unix_ns"])
    return df[(t >= start_ns) & (t <= end_ns)].copy()


# =========================================================
# Time-series / rate helpers
# =========================================================
def compute_intervals_sec(ts_ns: np.ndarray) -> np.ndarray:
    if ts_ns.size < 2:
        return np.array([], dtype=float)
    return np.diff(ts_ns) / 1e9


def infer_rate_from_median(ts_ns: np.ndarray, fallback: Optional[float] = None) -> Optional[float]:
    dt = compute_intervals_sec(ts_ns)
    if dt.size == 0:
        return fallback
    med = np.median(dt)
    if med <= 0:
        return fallback
    return float(1.0 / med)


def count_missing_messages(dt_sec: np.ndarray, gap_factor: float = 2.0):
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
            "nominal_rate_hz": "" if nominal_rate_hz is None else nominal_rate_hz,
            "n_messages": 0,
            "observed_mean_rate_hz": np.nan,
            "median_dt_ms": np.nan,
            "jitter_std_dt_ms": np.nan,
            "gap_count": "" if nominal_rate_hz is None else np.nan,
            "missing_ratio_pct": "" if nominal_rate_hz is None else np.nan,
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
    longest_gap_s = float(np.max(dt_sec)) if dt_sec.size else np.nan

    if nominal_rate_hz is None:
        gap_count = ""
        missing_ratio_pct = ""
    else:
        gap_count, missing_msg_count, _ = count_missing_messages(dt_sec)
        denom = len(ts_ns) + missing_msg_count
        missing_ratio_pct = 100.0 * missing_msg_count / denom if denom > 0 else np.nan

    total_session_sec = max((session_end_ns - session_start_ns) / 1e9, 1e-12)
    stream_duration_s = (ts_ns[-1] - ts_ns[0]) / 1e9
    session_coverage_pct = 100.0 * stream_duration_s / total_session_sec

    return {
        "topic_name": name,
        "nominal_rate_hz": "" if nominal_rate_hz is None else nominal_rate_hz,
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
    empty = pd.DataFrame(columns=["window_center_s", "lag_s", "corr_score"])

    if t_ns.size == 0 or x.size == 0 or y.size == 0:
        return empty

    n_win = int(round(window_s * fs_hz))
    n_step = int(round(step_s * fs_hz))

    if len(x) < n_win or len(y) < n_win:
        return empty

    rows = []
    for start in range(0, max(0, len(x) - n_win + 1), max(1, n_step)):
        stop = start + n_win
        lag_s, corr_score = crosscorr_lag_seconds(
            x[start:stop], y[start:stop], fs_hz, max_lag_s=max_lag_s
        )
        center_s = (t_ns[start] + t_ns[min(stop - 1, len(t_ns) - 1)]) / 2 / 1e9
        rows.append({
            "window_center_s": center_s,
            "lag_s": lag_s,
            "corr_score": corr_score,
        })

    if not rows:
        return empty

    return pd.DataFrame(rows, columns=["window_center_s", "lag_s", "corr_score"])


# =========================================================
# Domain helpers
# =========================================================
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


def unwrap_heading_deg(heading_deg: np.ndarray) -> np.ndarray:
    if heading_deg.size == 0:
        return heading_deg
    return np.rad2deg(np.unwrap(np.deg2rad(heading_deg)))


def get_gps_heading_deg(df_gps: Optional[pd.DataFrame], unwrap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if df_gps is None or "t_unix_ns" not in df_gps.columns or "head_mot" not in df_gps.columns:
        return np.array([]), np.array([])

    t, heading = clean_time_and_values(df_gps["t_unix_ns"], df_gps["head_mot"])
    if t.size == 0:
        return np.array([]), np.array([])

    heading_deg = heading * 1e-5
    if unwrap:
        heading_deg = unwrap_heading_deg(heading_deg)

    return t, heading_deg


def get_gnss_heading_and_yawrate(df_gps: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, heading_deg = get_gps_heading_deg(df_gps, unwrap=True)
    if t.size < 2:
        return t, heading_deg, np.array([])

    dt_s = np.diff(t) / 1e9
    yawrate = np.full_like(heading_deg, np.nan, dtype=float)

    valid = dt_s > 1e-6
    dyaw = np.diff(heading_deg)
    tmp = np.full_like(dt_s, np.nan, dtype=float)
    tmp[valid] = dyaw[valid] / dt_s[valid]
    yawrate[1:] = tmp

    return t, heading_deg, yawrate


def get_imu_yawrate_series(df_imu: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    if df_imu is None or "t_unix_ns" not in df_imu.columns:
        return np.array([]), np.array([]), None

    for col in ["headingspeed", "gyro_z"]:
        if col in df_imu.columns:
            t, y = clean_time_and_values(df_imu["t_unix_ns"], df_imu[col])
            if t.size >= 2:
                return t, y, col
    return np.array([]), np.array([]), None


def get_best_steering_series(df_steer: Optional[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    if df_steer is None or "t_unix_ns" not in df_steer.columns:
        return np.array([]), np.array([]), None

    df = df_steer.copy()

    if "ok" in df.columns:
        ok_num = pd.to_numeric(df["ok"], errors="coerce")
        if ok_num.notna().any():
            df = df[(ok_num > 0) | ok_num.isna()].copy()

    for col in ["angle_deg_clamped", "angle_deg"]:
        if col in df.columns:
            t, y = clean_time_and_values(df["t_unix_ns"], df[col])
            if t.size >= 2:
                return t, y, col

    return np.array([]), np.array([]), None


def classify_glances_from_yaw(yaw_deg: np.ndarray, threshold_deg: float = 25.0):
    look_left = yaw_deg > threshold_deg
    look_right = yaw_deg < -threshold_deg
    return look_left, look_right


def compute_brake_ylim(brake_values: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(brake_values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 20.0)

    vmax = float(np.nanmax(vals))
    if vmax <= 20:
        return (0.0, 20.0)
    if vmax <= 50:
        return (0.0, 50.0)
    if vmax <= 100:
        return (0.0, 100.0)
    if vmax <= 200:
        return (0.0, 200.0)
    return (0.0, np.ceil(vmax / 50.0) * 50.0)


# =========================================================
# Final gaze representation helpers
# =========================================================
def load_gaze_repr_for_scenario(gaze_repr_root: Optional[Path], scenario_id: str) -> Optional[pd.DataFrame]:
    if gaze_repr_root is None:
        return None
    p = gaze_repr_root / f"scenario_{scenario_id}" / "gaze_repr.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return clean_columns(df)


def compute_final_gaze_methods_yaw_from_repr(df_gaze_repr: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_gaze_repr is None or len(df_gaze_repr) == 0 or "t_unix_ns" not in df_gaze_repr.columns:
        return pd.DataFrame(columns=[
            "t_unix_ns", "yaw_A_deg", "yaw_B_deg", "yaw_C_deg",
            "eyes_not_found", "headpose_parse_ok"
        ])

    out = pd.DataFrame({"t_unix_ns": ensure_numeric(df_gaze_repr["t_unix_ns"])})

    if all(c in df_gaze_repr.columns for c in ["gaze_A_point_M_x", "gaze_A_point_M_y"]):
        ax = ensure_numeric(df_gaze_repr["gaze_A_point_M_x"]).to_numpy(dtype=float)
        ay = ensure_numeric(df_gaze_repr["gaze_A_point_M_y"]).to_numpy(dtype=float)
        out["yaw_A_deg"] = np.degrees(np.arctan2(ay, ax))
    else:
        out["yaw_A_deg"] = np.nan

    if all(c in df_gaze_repr.columns for c in ["gaze_B_ray_dir_M_x", "gaze_B_ray_dir_M_y"]):
        bx = ensure_numeric(df_gaze_repr["gaze_B_ray_dir_M_x"]).to_numpy(dtype=float)
        by = ensure_numeric(df_gaze_repr["gaze_B_ray_dir_M_y"]).to_numpy(dtype=float)
        out["yaw_B_deg"] = np.degrees(np.arctan2(by, bx))
    else:
        out["yaw_B_deg"] = np.nan

    if all(c in df_gaze_repr.columns for c in ["gaze_C_ray_dir_M_x", "gaze_C_ray_dir_M_y"]):
        cx = ensure_numeric(df_gaze_repr["gaze_C_ray_dir_M_x"]).to_numpy(dtype=float)
        cy = ensure_numeric(df_gaze_repr["gaze_C_ray_dir_M_y"]).to_numpy(dtype=float)
        out["yaw_C_deg"] = np.degrees(np.arctan2(cy, cx))
    else:
        out["yaw_C_deg"] = np.nan

    if "Eye movement type" in df_gaze_repr.columns:
        eye_type = df_gaze_repr["Eye movement type"].astype(str).str.strip().str.lower()
        out["eyes_not_found"] = eye_type.eq("eyesnotfound")
    else:
        out["eyes_not_found"] = False

    if "headpose_parse_ok" in df_gaze_repr.columns:
        hp_ok = df_gaze_repr["headpose_parse_ok"]
        if pd.api.types.is_numeric_dtype(hp_ok):
            out["headpose_parse_ok"] = pd.to_numeric(hp_ok, errors="coerce").fillna(0) > 0
        else:
            out["headpose_parse_ok"] = hp_ok.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "ok"])
    else:
        out["headpose_parse_ok"] = True

    for col in ["yaw_A_deg", "yaw_B_deg", "yaw_C_deg"]:
        out.loc[~out["headpose_parse_ok"], col] = np.nan

    return out.dropna(subset=["t_unix_ns"]).copy()


def extract_true_intervals(t_s: np.ndarray, mask: np.ndarray):
    intervals = []
    if t_s.size == 0 or mask.size == 0:
        return intervals

    t_s = np.asarray(t_s, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if t_s.size == 1:
        if mask[0]:
            intervals.append((t_s[0] - 0.05, t_s[0] + 0.05))
        return intervals

    dt = np.diff(t_s)
    med_dt = np.nanmedian(dt[np.isfinite(dt)]) if np.isfinite(dt).any() else 0.05
    pad = max(0.02, 0.5 * med_dt)

    start = None
    prev_t = None
    for ti, mi in zip(t_s, mask):
        if mi and start is None:
            start = ti
        if mi:
            prev_t = ti
        if (not mi) and start is not None:
            intervals.append((start - pad, prev_t + pad))
            start = None
            prev_t = None

    if start is not None:
        intervals.append((start - pad, prev_t + pad))

    return intervals


# =========================================================
# Plots: overview figures
# =========================================================
def plot_f2_boxplot(streams, out_path):
    """
    Suggested caption:
    Boxplots show the median, interquartile range, and whiskers extending
    to 1.5×IQR; outliers are hidden.
    """
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


def plot_f4_alignment(stats_df, out_path, title):
    plot_df = stats_df.sort_values("start_offset_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(plot_df))))
    y = np.arange(len(plot_df))[::-1]

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.broken_barh([(row["start_offset_s"], max(0.01, row["_stream_duration_s"]))], (yi - 0.35, 0.7))

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["topic_name"])
    ax.set_xlabel("Time since selected segment start (s)")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_camera_frame_intervals(back_camera, scene_camera, out_fig):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False

    for ax, df, title in [
        (axes[0], back_camera, "Back camera"),
        (axes[1], scene_camera, "Scene camera"),
    ]:
        if df is not None and "t_unix_ns" in df.columns:
            ts = clean_time_ns(df["t_unix_ns"])
            dt_ms = compute_intervals_sec(ts) * 1e3
            dt_ms = dt_ms[np.isfinite(dt_ms)]
            if dt_ms.size > 0:
                ax.hist(dt_ms, bins=50)
                ax.set_xlabel("Frame interval Δt (ms)")
                ax.set_ylabel("Count")
                ax.set_title(title)
                ax.grid(True, linestyle="--", alpha=0.3)
                plotted = True
            else:
                ax.set_visible(False)
        else:
            ax.set_visible(False)

    if plotted:
        fig.suptitle("F3. Camera frame interval stability")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=300)
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


def plot_f15_gaze_stability(df_eye, out_fig):
    if df_eye is None or len(df_eye) == 0:
        return

    if "Sensor" in df_eye.columns:
        sub = df_eye[df_eye["Sensor"].astype(str).str.strip() == "Eye Tracker"].copy()
        if len(sub) > 0:
            df_eye = sub

    # raw Tobii-based yaw sanity check
    req = [
        "Gaze direction left X", "Gaze direction left Z",
        "Gaze direction right X", "Gaze direction right Z",
        "Validity left", "Validity right", "t_unix_ns"
    ]
    if any(c not in df_eye.columns for c in req):
        return

    lx = ensure_numeric(df_eye["Gaze direction left X"]).to_numpy()
    lz = ensure_numeric(df_eye["Gaze direction left Z"]).to_numpy()
    rx = ensure_numeric(df_eye["Gaze direction right X"]).to_numpy()
    rz = ensure_numeric(df_eye["Gaze direction right Z"]).to_numpy()
    vl = parse_validity_series(df_eye["Validity left"]).to_numpy()
    vr = parse_validity_series(df_eye["Validity right"]).to_numpy()
    t = ensure_numeric(df_eye["t_unix_ns"]).to_numpy()

    x = np.where((vl > 0) & (vr > 0), (lx + rx) / 2.0,
                 np.where(vl > 0, lx, np.where(vr > 0, rx, np.nan)))
    z = np.where((vl > 0) & (vr > 0), (lz + rz) / 2.0,
                 np.where(vl > 0, lz, np.where(vr > 0, rz, np.nan)))
    yaw = np.degrees(np.arctan2(x, z))

    t, yaw = clean_time_and_values(pd.Series(t), pd.Series(yaw))
    if t.size < 2:
        return

    dt_s = np.diff(t) / 1e9
    yaw_rate = np.diff(yaw) / np.where(dt_s > 1e-6, dt_s, np.nan)

    yaw = yaw[np.isfinite(yaw)]
    yaw_rate = yaw_rate[np.isfinite(yaw_rate)]
    if yaw.size == 0 or yaw_rate.size == 0:
        return

    yaw_lo, yaw_hi = np.percentile(yaw, [1, 99])
    yr_lo, yr_hi = np.percentile(yaw_rate, [1, 99])
    yaw_pad = 0.05 * max(1.0, yaw_hi - yaw_lo)
    yr_pad = 0.05 * max(1.0, yr_hi - yr_lo)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(yaw, bins=60, range=(yaw_lo - yaw_pad, yaw_hi + yaw_pad))
    axes[0].set_xlim(yaw_lo - yaw_pad, yaw_hi + yaw_pad)
    axes[0].set_xlabel("Gaze yaw (deg)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of gaze yaw")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    axes[1].hist(yaw_rate, bins=60, range=(yr_lo - yr_pad, yr_hi + yr_pad))
    axes[1].set_xlim(yr_lo - yr_pad, yr_hi + yr_pad)
    axes[1].set_xlabel("Gaze yaw rate (deg/s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of gaze angular velocity")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("F15. Eye-tracking / head-pose signal stability")
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

        rate = infer_rate_from_median(ts, fallback=10.0)
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


# =========================================================
# Analysis summaries
# =========================================================
def analyze_imu_vs_gnss_yawrate(df_gps, df_imu, out_csv, out_fig, title):
    t_gps, heading_deg, gnss_yawrate = get_gnss_heading_and_yawrate(df_gps)
    t_imu, imu_yaw, imu_col = get_imu_yawrate_series(df_imu)

    if t_gps.size < 2 or t_imu.size < 2 or gnss_yawrate.size == 0:
        return

    # remove NaN in GNSS-derived yawrate
    mask_gps = np.isfinite(gnss_yawrate)
    t_gps = t_gps[mask_gps]
    gnss_yawrate = gnss_yawrate[mask_gps]

    if t_gps.size < 2:
        return

    fs = 20.0
    g1, y_gnss = resample_series(t_gps, gnss_yawrate, fs)
    g2, y_imu = resample_series(t_imu, imu_yaw, fs)

    if g1.size == 0 or g2.size == 0:
        return

    t0 = max(g1[0], g2[0])
    t1e = min(g1[-1], g2[-1])
    if t1e <= t0:
        return

    common = np.arange(t0, t1e + 1, int(round(1e9 / fs)), dtype=np.int64)
    gnss_common = np.interp(common.astype(float), g1.astype(float), y_gnss.astype(float))
    imu_common = np.interp(common.astype(float), g2.astype(float), y_imu.astype(float))

    lag_df = sliding_window_lag(common, imu_common, gnss_common, fs_hz=fs, window_s=10.0, step_s=2.0, max_lag_s=2.0)

    if "lag_s" in lag_df.columns and len(lag_df) > 0:
        median_lag_s = float(lag_df["lag_s"].median())
        p95_abs_lag_s = float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95))
    else:
        median_lag_s = np.nan
        p95_abs_lag_s = np.nan

    if "corr_score" in lag_df.columns and len(lag_df) > 0:
        median_corr_score = float(lag_df["corr_score"].median())
    else:
        median_corr_score = np.nan

    err = imu_common - gnss_common
    summary = {
        "n_windows": len(lag_df),
        "median_lag_s": median_lag_s,
        "p95_abs_lag_s": p95_abs_lag_s,
        "median_corr_score": median_corr_score,
        "yawrate_rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "yawrate_mae": float(np.nanmean(np.abs(err))),
        "imu_signal": imu_col,
        "gnss_signal": "derived_from_head_mot",
    }
    pd.DataFrame([summary]).to_csv(out_csv, index=False)

    score = np.abs(np.gradient(imu_common) * fs) + np.abs(np.gradient(gnss_common) * fs)
    n_win = int(round(20.0 * fs))
    if len(score) > n_win:
        idx = int(np.argmax(np.convolve(score, np.ones(n_win), mode="valid")))
        mask = np.zeros_like(score, dtype=bool)
        mask[idx:idx + n_win] = True
    else:
        mask = np.ones_like(score, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    rel_t = common[mask] / 1e9 - common[mask][0] / 1e9

    axes[0].plot(rel_t, normalized(imu_common[mask]), label=f"IMU yaw rate ({imu_col})")
    axes[0].plot(rel_t, normalized(gnss_common[mask]), label="GNSS-derived yaw rate")
    axes[0].set_xlabel("Time within representative window (s)")
    axes[0].set_ylabel("Normalized signal")
    axes[0].set_title("Representative yaw-rate overlay")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    if "lag_s" in lag_df.columns and lag_df["lag_s"].dropna().shape[0] > 0:
        axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
        if np.isfinite(summary["median_lag_s"]):
            axes[1].axvline(summary["median_lag_s"], linestyle="--", label=f"median={summary['median_lag_s']:.3f}s")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid lag windows", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
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

    if "lag_s" in lag_df.columns and len(lag_df) > 0:
        median_lag_s = float(lag_df["lag_s"].median())
        p95_abs_lag_s = float(np.nanpercentile(np.abs(lag_df["lag_s"]), 95))
    else:
        median_lag_s = np.nan
        p95_abs_lag_s = np.nan

    if "corr_score" in lag_df.columns and len(lag_df) > 0:
        median_corr_score = float(lag_df["corr_score"].median())
    else:
        median_corr_score = np.nan

    summary = {
        "n_windows": len(lag_df),
        "median_lag_s": median_lag_s,
        "p95_abs_lag_s": p95_abs_lag_s,
        "median_corr_score": median_corr_score,
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

    if "lag_s" in lag_df.columns and lag_df["lag_s"].dropna().shape[0] > 0:
        axes[1].hist(lag_df["lag_s"].dropna(), bins=20)
        if np.isfinite(summary["median_lag_s"]):
            axes[1].axvline(
                summary["median_lag_s"],
                linestyle="--",
                label=f"median={summary['median_lag_s']:.3f}s"
            )
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid lag windows", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_xlabel("Estimated lag (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sliding-window lag distribution")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title)
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
        vl = parse_validity_series(df_eye["Validity left"])
        row["valid_left_ratio"] = float(vl.mean())

    if "Validity right" in df_eye.columns:
        vr = parse_validity_series(df_eye["Validity right"])
        row["valid_right_ratio"] = float(vr.mean())

    if "Validity left" in df_eye.columns and "Validity right" in df_eye.columns:
        vl = parse_validity_series(df_eye["Validity left"])
        vr = parse_validity_series(df_eye["Validity right"])
        row["both_valid_ratio"] = float((vl & vr).mean())

    if "Eye movement type" in df_eye.columns:
        eye_type = df_eye["Eye movement type"].astype(str).str.strip().str.lower()
        row["fixation_ratio"] = float((eye_type == "fixation").mean())
        row["eyes_not_found_ratio"] = float((eye_type == "eyesnotfound").mean())

    if "Eye movement event duration" in df_eye.columns:
        dur = ensure_numeric(df_eye["Eye movement event duration"])
        if dur.notna().any():
            row["eye_event_duration_median_ms"] = float(dur.median())
            row["eye_event_duration_p95_ms"] = float(np.nanpercentile(dur.dropna(), 95))

    pd.DataFrame([row]).to_csv(out_csv, index=False)


# =========================================================
# Plots: scenario-specific figures
# =========================================================
def plot_scenario_turning_control(df_gps, df_steer, df_imu, df_brake, out_fig, title, gps_speed_unit):
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

    # 1) GNSS speed
    if df_gps is not None and "g_speed" in df_gps.columns:
        t, spd = clean_time_and_values(df_gps["t_unix_ns"], df_gps["g_speed"])
        spd = convert_speed_to_mps(spd, gps_speed_unit)
        if t.size:
            axes[0].plot(t / 1e9 - t[0] / 1e9, spd, label="GNSS speed")
            axes[0].set_ylabel("Speed (m/s)")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No GNSS speed data", ha="center", va="center", transform=axes[0].transAxes)

    # 2) steering angle
    t_steer, steer, steer_col = get_best_steering_series(df_steer)
    if t_steer.size:
        axes[1].plot(t_steer / 1e9 - t_steer[0] / 1e9, steer, label=f"Steering ({steer_col})")
        axes[1].set_ylabel("Steer (deg)")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No steering data", ha="center", va="center", transform=axes[1].transAxes)

    # 3) IMU yaw-related
    t_imu, imu_yaw, imu_col = get_imu_yawrate_series(df_imu)
    if t_imu.size:
        axes[2].plot(t_imu / 1e9 - t_imu[0] / 1e9, imu_yaw, label=f"IMU yaw rate ({imu_col})")
        axes[2].set_ylabel("Yaw-related")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No IMU yaw data", ha="center", va="center", transform=axes[2].transAxes)

    # 4) GPS heading
    t_head, head_deg = get_gps_heading_deg(df_gps, unwrap=True)
    if t_head.size:
        axes[3].plot(t_head / 1e9 - t_head[0] / 1e9, head_deg, label="GPS heading (unwrapped)")
        axes[3].set_ylabel("Heading (deg)")
        axes[3].legend()
        axes[3].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, "No GPS heading data", ha="center", va="center", transform=axes[3].transAxes)

    # 5) brake
    if df_brake is not None and "force_total_n" in df_brake.columns:
        t, brake = clean_time_and_values(df_brake["t_unix_ns"], df_brake["force_total_n"])
        if t.size:
            axes[4].plot(t / 1e9 - t[0] / 1e9, brake, label="Brake force")
            axes[4].set_ylabel("Brake (N)")
            axes[4].set_ylim(*compute_brake_ylim(brake))
            axes[4].set_xlabel("Time since scenario start (s)")
            axes[4].legend()
            axes[4].grid(True, linestyle="--", alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, "No brake data", ha="center", va="center", transform=axes[4].transAxes)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

def plot_scenario_turn_consistency(df_gps, df_steer, df_imu, out_fig, title):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) steering angle
    t_steer, steer, steer_col = get_best_steering_series(df_steer)
    if t_steer.size:
        axes[0].plot(t_steer / 1e9 - t_steer[0] / 1e9, steer, label=f"Steering ({steer_col})")
        axes[0].set_ylabel("Steer (deg)")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No steering data", ha="center", va="center", transform=axes[0].transAxes)

    # 2) normalized yaw-rate comparison
    t_gps, heading_deg, gnss_yawrate = get_gnss_heading_and_yawrate(df_gps)
    t_imu, imu_yaw, imu_col = get_imu_yawrate_series(df_imu)

    have_panel2 = False
    if t_imu.size:
        axes[1].plot(t_imu / 1e9 - t_imu[0] / 1e9, normalized(imu_yaw), label=f"IMU yaw rate ({imu_col}, norm)")
        have_panel2 = True

    if t_gps.size and gnss_yawrate.size:
        mask = np.isfinite(gnss_yawrate)
        if np.sum(mask) >= 2:
            axes[1].plot(t_gps[mask] / 1e9 - t_gps[mask][0] / 1e9,
                         normalized(gnss_yawrate[mask]),
                         label="GNSS-derived yaw rate (norm)")
            have_panel2 = True

    if have_panel2:
        axes[1].set_ylabel("Normalized")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No yaw-rate data", ha="center", va="center", transform=axes[1].transAxes)

    # 3) GNSS heading
    if t_gps.size:
        axes[2].plot(t_gps / 1e9 - t_gps[0] / 1e9, heading_deg, label="GNSS heading (unwrapped)")
        axes[2].set_ylabel("Heading (deg)")
        axes[2].set_xlabel("Time since scenario start (s)")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No GNSS heading data", ha="center", va="center", transform=axes[2].transAxes)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

def plot_scenario_closed_loop_from_repr(
    df_gaze_repr,
    df_steer,
    df_brake,
    df_gps,
    df_power,
    df_imu,
    df_wheel,
    out_fig,
    title,
    gps_speed_unit,
    wheel_speed_unit,
    look_method="B",
):
    fig, axes = plt.subplots(7, 1, figsize=(12, 16), sharex=True)

    # 1) final gaze yaw in ego-bike frame
    gaze_df = compute_final_gaze_methods_yaw_from_repr(df_gaze_repr)

    if len(gaze_df) > 0:
        t0_abs = ensure_numeric(gaze_df["t_unix_ns"]).min() / 1e9

        plotted_any = False
        for method in ["A", "B", "C"]:
            col = f"yaw_{method}_deg"
            t, yaw = clean_time_and_values(gaze_df["t_unix_ns"], gaze_df[col])
            if t.size:
                axes[0].plot(t / 1e9 - t0_abs, yaw, color=ABC_COLORS[method], label=f"Gaze yaw {method}")
                plotted_any = True

        look_col = f"yaw_{str(look_method).upper()}_deg"
        t_look, yaw_look = clean_time_and_values(gaze_df["t_unix_ns"], gaze_df[look_col])
        if t_look.size:
            rel_t = t_look / 1e9 - t0_abs
            look_left, look_right = classify_glances_from_yaw(yaw_look, threshold_deg=25.0)
            axes[0].scatter(rel_t[look_left], np.full(np.sum(look_left), 35.0),
                            marker="|", color=ACADEMIC_GREEN, label=f"Look left ({look_method})")
            axes[0].scatter(rel_t[look_right], np.full(np.sum(look_right), -35.0),
                            marker="|", color=ACADEMIC_RED, label=f"Look right ({look_method})")

        # grey invalid intervals
        t_nf = ensure_numeric(gaze_df["t_unix_ns"]).to_numpy(dtype=float)
        enf = gaze_df["eyes_not_found"].to_numpy(dtype=bool)
        t_nf_rel = t_nf / 1e9 - t0_abs
        for a, b in extract_true_intervals(t_nf_rel, enf):
            axes[0].axvspan(a, b, color=ACADEMIC_GREY, alpha=0.25)

        if plotted_any:
            axes[0].set_ylabel("Gaze yaw (deg)")
            axes[0].legend(ncol=2, fontsize=8)
            axes[0].grid(True, linestyle="--", alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "Gaze file found but no valid A/B/C yaw could be plotted", ha="center", va="center", transform=axes[0].transAxes)
    else:
        axes[0].text(0.5, 0.5, "No final gaze file / data", ha="center", va="center", transform=axes[0].transAxes)

    # 2) steering
    t_steer, steer, steer_col = get_best_steering_series(df_steer)
    if t_steer.size:
        axes[1].plot(t_steer / 1e9 - t_steer[0] / 1e9, steer, label=f"Steering ({steer_col})")
        axes[1].set_ylabel("Steer (deg)")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No steering data", ha="center", va="center", transform=axes[1].transAxes)

    # 3) yaw rate
    t_imu, imu_yaw, imu_col = get_imu_yawrate_series(df_imu)
    t_gps_h, heading_deg, gnss_yawrate = get_gnss_heading_and_yawrate(df_gps)

    have_yaw_plot = False
    if t_imu.size:
        axes[2].plot(t_imu / 1e9 - t_imu[0] / 1e9, imu_yaw, label=f"IMU yaw rate ({imu_col})")
        have_yaw_plot = True
    if t_gps_h.size and gnss_yawrate.size:
        axes[2].plot(t_gps_h / 1e9 - t_gps_h[0] / 1e9, gnss_yawrate, label="GNSS-derived yaw rate")
        have_yaw_plot = True

    if have_yaw_plot:
        axes[2].set_ylabel("Yaw rate")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "No yaw-rate data", ha="center", va="center", transform=axes[2].transAxes)

    # 4) heading
    if t_gps_h.size:
        axes[3].plot(t_gps_h / 1e9 - t_gps_h[0] / 1e9, heading_deg, label="GNSS heading (unwrapped)")
        axes[3].set_ylabel("Heading (deg)")
        axes[3].legend()
        axes[3].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, "No heading data", ha="center", va="center", transform=axes[3].transAxes)

    # 5) brake
    if df_brake is not None and "force_total_n" in df_brake.columns:
        t, brake = clean_time_and_values(df_brake["t_unix_ns"], df_brake["force_total_n"])
        if t.size:
            axes[4].plot(t / 1e9 - t[0] / 1e9, brake, label="Brake force")
            axes[4].set_ylabel("Brake (N)")
            axes[4].set_ylim(*compute_brake_ylim(brake))
            axes[4].legend()
            axes[4].grid(True, linestyle="--", alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, "No brake data", ha="center", va="center", transform=axes[4].transAxes)
    else:
        axes[4].text(0.5, 0.5, "No brake data", ha="center", va="center", transform=axes[4].transAxes)

    # 6) power scatter only
    if df_power is not None and "p10_instantaneous_power_w" in df_power.columns:
        t, pwr = clean_time_and_values(df_power["t_unix_ns"], df_power["p10_instantaneous_power_w"])
        if t.size:
            axes[5].scatter(t / 1e9 - t[0] / 1e9, pwr, s=12, label="Pedal power events")
            axes[5].set_ylabel("Power (W)")
            axes[5].legend()
            axes[5].grid(True, linestyle="--", alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, "No power data", ha="center", va="center", transform=axes[5].transAxes)
    else:
        axes[5].text(0.5, 0.5, "No power data", ha="center", va="center", transform=axes[5].transAxes)

    # 7) speed: GNSS + wheel
    speed_plotted = False
    if df_gps is not None and "g_speed" in df_gps.columns:
        t, spd = clean_time_and_values(df_gps["t_unix_ns"], df_gps["g_speed"])
        spd = convert_speed_to_mps(spd, gps_speed_unit)
        if t.size:
            axes[6].plot(t / 1e9 - t[0] / 1e9, spd, label="GNSS speed")
            speed_plotted = True

    if df_wheel is not None and "speed_mps" in df_wheel.columns:
        t, spd_w = clean_time_and_values(df_wheel["t_unix_ns"], df_wheel["speed_mps"])
        spd_w = convert_speed_to_mps(spd_w, wheel_speed_unit)
        if t.size:
            axes[6].plot(t / 1e9 - t[0] / 1e9, spd_w, label="Wheel speed")
            speed_plotted = True

    if speed_plotted:
        axes[6].set_ylabel("Speed (m/s)")
        axes[6].set_xlabel("Time since scenario start (s)")
        axes[6].legend()
        axes[6].grid(True, linestyle="--", alpha=0.3)
    else:
        axes[6].text(0.5, 0.5, "No speed data", ha="center", va="center", transform=axes[6].transAxes)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close(fig)

# =========================================================
# Scenario parsing
# =========================================================
def parse_scenarios(csv_path: str) -> pd.DataFrame:
    scenarios = clean_columns(pd.read_csv(csv_path))

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

    if "initial_time" not in scenarios.columns or "start" not in scenarios.columns or "end" not in scenarios.columns:
        raise RuntimeError("critical_scenarios.csv must contain columns: initial time, start, end")

    scenarios["initial_time_s"] = pd.to_numeric(scenarios["initial_time"], errors="coerce")
    scenarios["start_offset_s"] = pd.to_numeric(scenarios["start"], errors="coerce")
    scenarios["end_offset_s"] = pd.to_numeric(scenarios["end"], errors="coerce")
    scenarios = scenarios.dropna(subset=["initial_time_s", "start_offset_s", "end_offset_s"]).copy()

    scenarios["start_abs_s"] = scenarios["initial_time_s"] + scenarios["start_offset_s"]
    scenarios["end_abs_s"] = scenarios["initial_time_s"] + scenarios["end_offset_s"]

    scenarios["start_ns"] = (scenarios["start_abs_s"] * 1e9).round().astype(np.int64)
    scenarios["end_ns"] = (scenarios["end_abs_s"] * 1e9).round().astype(np.int64)

    scenarios["scenario_id_str"] = (
        scenarios["scenario_id"].astype(str).str.strip().str.replace(".0", "", regex=False)
    )

    print("\n[DEBUG] parsed scenarios:")
    print(
        scenarios[
            ["scenario_id", "scenario_type", "initial_time_s", "start_offset_s", "end_offset_s", "start_abs_s", "end_abs_s"]
        ].to_string(index=False)
    )
    return scenarios


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Merged XLSX")
    ap.add_argument("--scenarios", required=True, help="critical_scenarios.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")

    ap.add_argument("--gps-speed-unit", default="mmps", choices=["mps", "mmps", "kmph", "mph"])
    ap.add_argument("--wheel-speed-unit", default="kmph", choices=["mps", "mmps", "kmph", "mph"])

    ap.add_argument("--scenario-ids", nargs="*", default=[], help="Optional list of scenario ids")
    ap.add_argument("--f4-scenario-id", default="8", help="Scenario id used for F4 alignment")
    ap.add_argument("--f6-scenario-id", default="8", help="Scenario id used for F6 speed consistency")
    ap.add_argument("--gaze-repr-root", default=None, help="Root folder containing scenario_{id}/gaze_repr.csv")
    ap.add_argument("--look-method", default="B", choices=["A", "B", "C"])

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(args.xlsx)

    gps = find_sheet(xl, ["gps"])
    steer = find_sheet(xl, ["potentiometer"])
    wheel = find_sheet(xl, ["wheel_speed", "wheel speed"])
    imu = find_sheet(xl, ["imu"])
    power = find_sheet(xl, ["powermeter"])
    # Use the raw Tobii export for sample-level quality analysis.  The
    # fixation export is a separate table; it is accepted as a fallback for
    # older workbooks that do not contain the raw table.
    eye = find_sheet(xl, ["eyetracker_raw", "eyetracker"])
    if eye is None:
        eye = find_sheet(xl, ["eyetracker_fixation"])
    brake = find_sheet(xl, ["break_sensor_right", "brake_sensor_right"])
    back_camera = find_sheet(xl, ["back_camera", "back_camera_timestamps"])
    scene_camera = find_sheet(xl, ["scene_camera", "scene_timestamps"])

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
        ("back_camera", back_camera, NOMINAL_RATES["back_camera"]),
        ("scene_camera", scene_camera, NOMINAL_RATES["scene_camera"]),
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
        nominal = infer_rate_from_median(ts, fallback=10.0)
        streams.append((name, ts, nominal))

    if not streams:
        raise RuntimeError("No valid streams found in XLSX.")

    # T1 / F2 full-session
    session_start_ns = min(ts[0] for _, ts, _ in streams)
    session_end_ns = max(ts[-1] for _, ts, _ in streams)

    t1 = pd.DataFrame(
        [compute_health_row(name, ts, nominal, session_start_ns, session_end_ns) for name, ts, nominal in streams]
    ).sort_values("topic_name").reset_index(drop=True)
    t1.to_csv(outdir / "T1_topic_health_summary.csv", index=False)

    plot_f2_boxplot(streams, outdir / "F2_sampling_interval_boxplot.png")
    plot_camera_frame_intervals(back_camera, scene_camera, outdir / "F3_camera_frame_interval_stability.png")

    scenarios = parse_scenarios(args.scenarios)

    # F4 based on one selected scenario
    target_id = str(args.f4_scenario_id).strip()
    f4_sub = scenarios[scenarios["scenario_id_str"] == target_id]

    if len(f4_sub) > 0:
        row = f4_sub.iloc[0]
        f4_start_ns = int(row["start_ns"])
        f4_end_ns = int(row["end_ns"])
        print(
            f"[DEBUG] Using F4 scenario {target_id}: "
            f"start_abs_s={row['start_abs_s']}, end_abs_s={row['end_abs_s']}, "
            f"duration_s={(f4_end_ns - f4_start_ns)/1e9:.3f}"
        )

        f4_rows = []
        for name, _, nominal in streams:
            src_df = None
            if name == "gps":
                src_df = gps
            elif name == "steering":
                src_df = steer
            elif name == "wheel_speed":
                src_df = wheel
            elif name == "imu":
                src_df = imu
            elif name == "powermeter":
                src_df = power
            elif name == "break_sensor_right":
                src_df = brake
            elif name == "back_camera":
                src_df = back_camera
            elif name == "scene_camera":
                src_df = scene_camera
            elif name.startswith("eyetracker_"):
                if eye is not None and "Sensor" in eye.columns:
                    sensor_label = name.replace("eyetracker_", "").replace("_", " ").lower()
                    wanted_map = {
                        "eye tracker": "Eye Tracker",
                        "gyroscope": "Gyroscope",
                        "accelerometer": "Accelerometer",
                        "magnetometer": "Magnetometer",
                        "scene camera": "Scene camera",
                    }
                    wanted_sensor = wanted_map.get(sensor_label, None)
                    if wanted_sensor is not None:
                        src_df = eye[eye["Sensor"].astype(str).str.strip() == wanted_sensor].copy()
            elif name.startswith("lidar_frame_"):
                src_df = lidar_frames.get(name, None)

            seg = crop_df_by_ns(src_df, f4_start_ns, f4_end_ns)
            ts = clean_time_ns(seg["t_unix_ns"]) if seg is not None and "t_unix_ns" in seg.columns else np.array([], dtype=float)
            f4_rows.append(compute_health_row(name, ts, nominal, f4_start_ns, f4_end_ns))

        t4 = pd.DataFrame(f4_rows).sort_values("topic_name").reset_index(drop=True)
        plot_f4_alignment(
            t4,
            outdir / "F4_relative_stream_alignment.png",
            f"F4. Relative start and end alignment within scenario {target_id}"
        )
    else:
        print(f"[DEBUG] F4 scenario {target_id} not found, fallback to full session")
        plot_f4_alignment(
            t1,
            outdir / "F4_relative_stream_alignment.png",
            "F4. Relative start and end alignment of sensor streams"
        )

    # F5 overall
    analyze_imu_vs_gnss_yawrate(
        gps, imu,
        outdir / "T3_imu_gnss_yawrate_summary.csv",
        outdir / "F5_imu_vs_gnss_yawrate.png",
        "F5. IMU yaw-rate vs GNSS-derived yaw-rate temporal consistency"
    )

    # F6 should use scenario 8 (or chosen scenario) only
    f6_id = str(args.f6_scenario_id).strip()
    f6_sub = scenarios[scenarios["scenario_id_str"] == f6_id]
    if len(f6_sub) > 0:
        row = f6_sub.iloc[0]
        f6_start_ns = int(row["start_ns"])
        f6_end_ns = int(row["end_ns"])

        wheel_f6 = crop_df_by_ns(wheel, f6_start_ns, f6_end_ns)
        gps_f6 = crop_df_by_ns(gps, f6_start_ns, f6_end_ns)

        analyze_speed_compare(
            wheel_f6, gps_f6,
            outdir / "T4_speed_gnss_residual_summary.csv",
            outdir / "F6_speed_compare_overall.png",
            f"F6. Wheel speed vs GNSS speed temporal consistency (scenario {f6_id})",
            args.wheel_speed_unit,
            args.gps_speed_unit,
        )
    else:
        print(f"[WARN] scenario {f6_id} not found for F6")

    plot_gps_accuracy(gps, outdir / "GPS_accuracy_overall.png", "GPS accuracy indicators from ubx_nav_pvt estimates")
    compute_t7_eye_quality(eye, outdir / "T7_eye_tracking_quality.csv")
    plot_f15_gaze_stability(eye, outdir / "F15_gaze_head_stability.png")
    plot_lidar_frame_health(
        {k: v for k, v in lidar_frames.items() if v is not None},
        outdir / "T6_lidar_frame_health_summary.csv",
        outdir / "F8_lidar_frame_health.png"
    )

    # Scenario-specific outputs
    if args.scenario_ids:
        wanted = set(str(x) for x in args.scenario_ids)
        scenarios_run = scenarios[scenarios["scenario_id_str"].isin(wanted)].copy()
    else:
        scenarios_run = scenarios.copy()

    gaze_repr_root = Path(args.gaze_repr_root) if args.gaze_repr_root else None

    for _, row in scenarios_run.iterrows():
        sid = str(row["scenario_id_str"])
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

        gaze_repr_seg = load_gaze_repr_for_scenario(gaze_repr_root, sid)
        if gaze_repr_seg is not None:
            gaze_repr_seg = crop_df_by_ns(gaze_repr_seg, start_ns, end_ns)

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
            f"Scenario {sid} ({sname}): speed, steering, IMU, GPS heading, brake",
            args.gps_speed_unit,
        )

        plot_scenario_turn_consistency(
            gps_seg, steer_seg, imu_seg,
            outdir / f"scenario_{sid}_turn_consistency.png",
            f"Scenario {sid} ({sname}): turning consistency across GNSS, potentiometer, and IMU",
        )

        plot_scenario_closed_loop_from_repr(
            gaze_repr_seg, steer_seg, brake_seg, gps_seg, power_seg, imu_seg, wheel_seg,
            outdir / f"scenario_{sid}_closed_loop.png",
            f"Scenario {sid} ({sname}): final gaze, control input, and dynamic response",
            args.gps_speed_unit,
            args.wheel_speed_unit,
            look_method=args.look_method,
        )
    print(f"Done. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
