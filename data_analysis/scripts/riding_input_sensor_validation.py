#!/usr/bin/env python3
"""Validate steering, brake and power-meter CSV streams for one session."""

import argparse
import hashlib
import json
import math
import platform
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


NS_PER_SECOND = 1_000_000_000

INPUT_FILES = {
    "brake": "brake_sensors_force_20260603_134654.csv",
    "power": "rally_payload_decoded_20260603_134654.csv",
    "steering": "steering_angle_20260603_134654.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value):
    path.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "t_unix_ns" not in frame.columns:
        raise ValueError(f"Missing t_unix_ns in {path}")
    frame = frame.copy()
    frame["t_unix_ns"] = numeric(frame["t_unix_ns"])
    frame = frame[frame["t_unix_ns"].notna() & (frame["t_unix_ns"] > 0)]
    frame = frame.sort_values("t_unix_ns").drop_duplicates("t_unix_ns", keep="first")
    return frame.reset_index(drop=True)


def timing_summary(ts_ns: np.ndarray, session_start_ns: int, session_end_ns: int) -> dict:
    ts_ns = np.asarray(ts_ns, dtype=np.float64)
    dt_s = np.diff(ts_ns) / NS_PER_SECOND if ts_ns.size > 1 else np.array([], dtype=float)
    positive_dt = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
    if positive_dt.size:
        median_dt = float(np.median(positive_dt))
        p95_dt = float(np.percentile(positive_dt, 95))
        max_gap = float(np.max(positive_dt))
        gap_mask = positive_dt > 2.0 * median_dt
        gap_count = int(np.sum(gap_mask))
        missing_count = int(
            np.sum(np.maximum(0.0, np.rint(positive_dt[gap_mask] / median_dt) - 1.0))
        )
        rate_hz = 1.0 / median_dt if median_dt > 0 else None
    else:
        median_dt = p95_dt = max_gap = None
        gap_count = missing_count = 0
        rate_hz = None

    session_duration_s = max(0.0, (session_end_ns - session_start_ns) / NS_PER_SECOND)
    stream_duration_s = (
        max(0.0, (ts_ns[-1] - ts_ns[0]) / NS_PER_SECOND) if ts_ns.size else 0.0
    )
    return {
        "n_messages": int(ts_ns.size),
        "start_offset_s": float((ts_ns[0] - session_start_ns) / NS_PER_SECOND) if ts_ns.size else None,
        "end_offset_s": float((ts_ns[-1] - session_start_ns) / NS_PER_SECOND) if ts_ns.size else None,
        "stream_duration_s": stream_duration_s,
        "session_coverage_fraction": stream_duration_s / session_duration_s if session_duration_s else None,
        "median_dt_ms": median_dt * 1000.0 if median_dt is not None else None,
        "p95_dt_ms": p95_dt * 1000.0 if p95_dt is not None else None,
        "max_gap_s": max_gap,
        "gaps_gt_2x_median": gap_count,
        "estimated_missing_messages": missing_count,
        "inferred_rate_hz": rate_hz,
    }


def finite_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(numeric(series).notna().mean())


def signal_statistics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = numeric(frame[column]).dropna().to_numpy(dtype=float)
        if not values.size:
            continue
        rows.append({
            "signal": column,
            "n_valid": int(values.size),
            "valid_fraction": float(values.size / len(frame)) if len(frame) else None,
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def build_summary(frames: dict[str, pd.DataFrame], session_start_ns: int, session_end_ns: int) -> pd.DataFrame:
    rows = []

    brake = frames["brake"]
    brake_row = {"sensor": "Brake force", "source_file": INPUT_FILES["brake"]}
    brake_row.update(timing_summary(brake["t_unix_ns"].to_numpy(), session_start_ns, session_end_ns))
    brake_row.update({
        "valid_fraction": float(
            (numeric(brake["ok_left"]).fillna(0).gt(0)
             & numeric(brake["ok_right"]).fillna(0).gt(0)).mean()
        ),
        "left_force_valid_fraction": finite_fraction(brake["left_force_n"]),
        "right_force_valid_fraction": finite_fraction(brake["right_force_n"]),
        "notes": "Decoded left/right force channels; valid when both ok flags are true.",
    })
    rows.append(brake_row)

    steering = frames["steering"]
    steering_row = {"sensor": "Steering angle", "source_file": INPUT_FILES["steering"]}
    steering_row.update(timing_summary(steering["t_unix_ns"].to_numpy(), session_start_ns, session_end_ns))
    steering_row.update({
        "valid_fraction": float(numeric(steering["ok"]).fillna(0).gt(0).mean()),
        "angle_valid_fraction": finite_fraction(steering["angle_deg"]),
        "saturation_fraction": finite_fraction(steering["angle_deg_clamped"]),
        "notes": "angle_deg is the decoded angle; angle_deg_clamped marks samples at the configured limits.",
    })
    rows.append(steering_row)

    power = frames["power"]
    power_row = {"sensor": "Power meter", "source_file": INPUT_FILES["power"]}
    power_row.update(timing_summary(power["t_unix_ns"].to_numpy(), session_start_ns, session_end_ns))
    power_row.update({
        "valid_fraction": finite_fraction(power["cadence_rpm"]),
        "standard_power_rows": int((power["page_name"] == "standard_power").sum()),
        "standard_torque_rows": int((power["page_name"] == "standard_torque").sum()),
        "instantaneous_power_valid_fraction": finite_fraction(power["p10_instantaneous_power_w"]),
        "cadence_valid_fraction": finite_fraction(power["cadence_rpm"]),
        "notes": "Power and cadence are decoded from Rally standard power/torque pages; page-specific fields are not valid on the other page.",
    })
    rows.append(power_row)
    return pd.DataFrame(rows)


def plot_riding_input(frames: dict[str, pd.DataFrame], session_start_ns: int, output_base: Path):
    brake = frames["brake"]
    power = frames["power"]
    steering = frames["steering"]
    t0_s = session_start_ns / NS_PER_SECOND

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True)

    t = numeric(steering["t_unix_ns"]).to_numpy(dtype=float) / NS_PER_SECOND - t0_s
    angle = numeric(steering["angle_deg"]).to_numpy(dtype=float)
    axes[0].plot(t, angle, color=COLORS["blue"], label="Steering angle")
    saturation = numeric(steering["angle_deg_clamped"]).notna().to_numpy()
    if saturation.any():
        axes[0].scatter(
            t[saturation],
            angle[saturation],
            s=4,
            color=COLORS["vermillion"],
            label="Configured limit",
            zorder=3,
        )
    axes[0].axhline(45.0, color=COLORS["grey"], linestyle="--", linewidth=0.8)
    axes[0].axhline(-45.0, color=COLORS["grey"], linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Steering angle")
    axes[0].legend(loc="upper right")

    t = numeric(brake["t_unix_ns"]).to_numpy(dtype=float) / NS_PER_SECOND - t0_s
    left = numeric(brake["left_force_n"]).to_numpy(dtype=float)
    right = numeric(brake["right_force_n"]).to_numpy(dtype=float)
    total = np.nan_to_num(left, nan=0.0) + np.nan_to_num(right, nan=0.0)
    axes[1].plot(t, left, color=COLORS["blue"], label="Left brake")
    axes[1].plot(t, right, color=COLORS["orange"], label="Right brake")
    axes[1].plot(t, total, color=COLORS["black"], linewidth=0.9, alpha=0.8, label="Total")
    axes[1].set_ylabel("Force (N)")
    axes[1].set_title("Brake force")
    axes[1].legend(loc="upper right")

    power_mask = power["page_name"].eq("standard_power")
    power_t = numeric(power.loc[power_mask, "t_unix_ns"]).to_numpy(dtype=float) / NS_PER_SECOND - t0_s
    power_values = numeric(power.loc[power_mask, "p10_instantaneous_power_w"]).to_numpy(dtype=float)
    cadence_t = numeric(power["t_unix_ns"]).to_numpy(dtype=float) / NS_PER_SECOND - t0_s
    cadence = numeric(power["cadence_rpm"]).to_numpy(dtype=float)
    finite_power = np.isfinite(power_values)
    axes[2].scatter(
        power_t[finite_power],
        power_values[finite_power],
        s=10,
        color=COLORS["vermillion"],
        label="Instantaneous power",
        zorder=3,
    )
    axes[2].set_ylabel("Power (W)")
    axes[2].set_title("Power-meter output and cadence")
    cadence_ax = axes[2].twinx()
    cadence_ax.plot(cadence_t, cadence, color=COLORS["purple"], linewidth=1.0, label="Cadence")
    cadence_ax.set_ylabel("Cadence (rpm)", color=COLORS["purple"])
    cadence_ax.tick_params(axis="y", colors=COLORS["purple"])
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = cadence_ax.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    for axis, label in zip(axes, ["(a)", "(b)", "(c)"]):
        panel_label(axis, label)
        axis.grid(True)
    axes[-1].set_xlabel("Elapsed time from first brake/steering sample (s)")
    fig.tight_layout()
    save_figure(fig, output_base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()

    session_dir = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if not session_dir.is_dir():
        raise SystemExit(f"Session directory does not exist: {session_dir}")
    if output.exists() or output.is_symlink():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")

    paths = {name: session_dir / filename for name, filename in INPUT_FILES.items()}
    frames = {name: read_input(path) for name, path in paths.items()}
    session_start_ns = min(int(frame["t_unix_ns"].min()) for frame in frames.values())
    session_end_ns = max(int(frame["t_unix_ns"].max()) for frame in frames.values())

    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True)
    figures.mkdir(parents=True)

    summary = build_summary(frames, session_start_ns, session_end_ns)
    summary.insert(0, "session_id", args.session_id or session_dir.name)
    summary.to_csv(tables / "riding_input_sensor_summary.csv", index=False)

    signal_tables = [
        signal_statistics(frames["steering"], ["angle_deg", "angle_deg_clamped"]),
        signal_statistics(frames["brake"], ["left_force_n", "right_force_n"]),
        signal_statistics(frames["power"], ["cadence_rpm", "p10_instantaneous_power_w"]),
    ]
    signal_stats = pd.concat(signal_tables, ignore_index=True)
    signal_stats.to_csv(tables / "riding_input_signal_statistics.csv", index=False)

    page_counts = (
        frames["power"].groupby(["page", "page_hex", "page_name"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    page_counts.to_csv(tables / "powermeter_page_counts.csv", index=False)

    plot_riding_input(frames, session_start_ns, figures / "riding_input_sensor_validation")

    caption = (
        "Riding-input sensor validation for the P9 session. (a) Decoded steering angle; "
        "red points mark samples at the configured angle limits. (b) Decoded left, right, "
        "and total brake force. (c) Rally instantaneous power events and cadence. "
        "The power meter starts later than the brake and steering streams, as reported in "
        "the accompanying timing table."
    )
    (output / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id or session_dir.name,
        "session_dir": str(session_dir),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "session_start_ns": session_start_ns,
        "session_end_ns": session_end_ns,
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
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "CHECKSUMS.sha256"):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Wrote riding-input validation to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
