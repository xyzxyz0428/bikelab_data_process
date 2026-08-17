#!/usr/bin/env python3
"""Generate GNSS/IMU ego-motion validation figures for one recording."""

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import gnss_imu_technical_validation as GNSS  # noqa: E402
from paper_style import apply_paper_style  # noqa: E402
import video_time_mapping as VIDEO_TIME  # noqa: E402

NS_PER_SECOND = 1_000_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="session")
    parser.add_argument(
        "--rosbag-time-source", choices=["record", "header"], default="record",
        help="Timestamp used for GNSS/video alignment; record is rosbag receive time",
    )
    parser.add_argument("--stationary-speed-threshold", type=float, default=0.15)
    parser.add_argument("--stationary-min-duration", type=float, default=10.0)
    parser.add_argument("--stationary-start-offset-s", type=float, default=6.0)
    parser.add_argument(
        "--stationary-reference-video-start-s", type=float, default=None,
        help="Use this video interval as the explicit stationary IMU reference",
    )
    parser.add_argument(
        "--stationary-reference-video-end-s", type=float, default=None,
        help="End of the explicit stationary IMU reference interval",
    )
    parser.add_argument("--course-min-speed", type=float, default=0.2)
    parser.add_argument("--course-max-accuracy", type=float, default=30.0)
    parser.add_argument("--max-lag", type=float, default=2.0)
    parser.add_argument("--lag-step", type=float, default=0.02)
    parser.add_argument("--turn-rate-threshold", type=float, default=0.015)
    parser.add_argument("--turn-duration", type=float, default=15.0)
    parser.add_argument("--video-start-s", type=float, default=None)
    parser.add_argument("--video-end-s", type=float, default=None)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow updating an existing derived output directory",
    )
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if (output.exists() or output.is_symlink()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    db_matches = sorted(session.glob("rosbag2_*/*.db3"))
    imu_matches = sorted(session.glob("imu_*.csv"))
    if len(db_matches) != 1 or len(imu_matches) != 1:
        raise SystemExit("Expected one rosbag db3 and one imu CSV")
    bag_file, imu_csv = db_matches[0], imu_matches[0]
    data = GNSS.read_bag_topics(bag_file, "sqlite3")
    GNSS.apply_rosbag_time_source(data, args.rosbag_time_source)
    # Keep the complete read-only topic frames so a stationary reference can
    # be outside the selected riding interval.
    data_full = {
        name: frame.copy() if isinstance(frame, pd.DataFrame) else frame.copy()
        if isinstance(frame, np.ndarray) else frame
        for name, frame in data.items()
    }
    bag_times = []
    for frame in data.values():
        if isinstance(frame, pd.DataFrame) and "t_ns" in frame.columns and len(frame):
            bag_times.extend([int(frame["t_ns"].min()), int(frame["t_ns"].max())])
    bag_start, bag_end = min(bag_times), max(bag_times)
    imu_streams_full, clock_audit = GNSS.load_imu(imu_csv, bag_start, bag_end)
    imu_streams = {dtype: frame.copy() for dtype, frame in imu_streams_full.items()}
    interval_start_ns, interval_end_ns = bag_start, bag_end
    camera_origin_ns = None
    camera_clock = None
    video_time_mapping = None
    if (args.video_start_s is None) != (args.video_end_s is None):
        raise SystemExit("Both --video-start-s and --video-end-s are required together")
    if (args.stationary_reference_video_start_s is None) != (args.stationary_reference_video_end_s is None):
        raise SystemExit(
            "Both --stationary-reference-video-start-s and --stationary-reference-video-end-s are required together"
        )
    if args.video_start_s is not None:
        timestamp_matches = sorted(session.glob("camera_*/timestamps.csv"))
        video_matches = sorted(session.glob("camera_*/video_mjpg.avi"))
        if len(timestamp_matches) != 1 or len(video_matches) != 1:
            raise SystemExit("Expected one camera video and timestamps.csv")
        camera_clock = VIDEO_TIME.load_video_clock(video_matches[0], timestamp_matches[0])
        camera_origin_ns = int(camera_clock["first_unix_ns"])
        interval_start_ns, interval_end_ns, video_time_mapping = (
            VIDEO_TIME.map_playback_interval(
                video_matches[0], timestamp_matches[0],
                args.video_start_s, args.video_end_s,
            )
        )
        for name, frame in list(data.items()):
            if isinstance(frame, pd.DataFrame) and "t_ns" in frame.columns:
                data[name] = frame[
                    frame["t_ns"].between(interval_start_ns, interval_end_ns, inclusive="both")
                ].copy().reset_index(drop=True)
        if "rawx_cno_t_ns" in data:
            cno_times = data["rawx_cno_t_ns"]
            keep = (cno_times >= interval_start_ns) & (cno_times <= interval_end_ns)
            data["rawx_cno"] = data["rawx_cno"][keep]
            data["rawx_cno_t_ns"] = cno_times[keep]
        imu_streams = {
            dtype: frame[
                frame["t_unix_ns"].between(interval_start_ns, interval_end_ns, inclusive="both")
            ].copy().reset_index(drop=True)
            for dtype, frame in imu_streams.items()
        }
        bag_start, bag_end = interval_start_ns, interval_end_ns
    streams = GNSS.build_stream_table(
        data,
        imu_streams,
        gnss_timestamp_field=(
            "rosbag record time" if args.rosbag_time_source == "record"
            else "header.stamp"
        ),
    )
    quality = GNSS.build_gnss_quality(data)
    east, north, up = GNSS.geodetic_to_enu(quality["latitude_deg"], quality["longitude_deg"], quality["height_m"])
    quality["east_m"], quality["north_m"], quality["up_m"] = east, north, up
    # Route panel: all valid PVT positions in the selected interval.  This is
    # deliberately broader than the quality-paired hAcc time series.
    route = data["pvt"].copy()
    # Include every PVT with a valid position, including no-carrier epochs;
    # the RTK state is still shown by colour in the route panel.
    route = route[~route["invalid_llh"]].copy()
    if len(route):
        route_east, route_north, route_up = GNSS.geodetic_to_enu(
            route["latitude_deg"], route["longitude_deg"], route["height_m"]
        )
        route["east_m"], route["north_m"], route["up_m"] = route_east, route_north, route_up
    route_coverage = {
        "source_topic": "/ubx_nav_pvt",
        "timestamp_source": args.rosbag_time_source,
        "valid_pvt_count": int(len(route)),
        "selected_interval_start_ns": int(interval_start_ns),
        "selected_interval_end_ns": int(interval_end_ns),
        "first_valid_pvt_ns": int(route["t_ns"].min()) if len(route) else None,
        "last_valid_pvt_ns": int(route["t_ns"].max()) if len(route) else None,
        "first_valid_pvt_header_ns": (
            int(route["header_ns"].min())
            if len(route) and "header_ns" in route.columns else None
        ),
        "last_valid_pvt_header_ns": (
            int(route["header_ns"].max())
            if len(route) and "header_ns" in route.columns else None
        ),
        "first_valid_pvt_record_ns": (
            int(route["record_ns"].min())
            if len(route) and "record_ns" in route.columns else None
        ),
        "last_valid_pvt_record_ns": (
            int(route["record_ns"].max())
            if len(route) and "record_ns" in route.columns else None
        ),
        "missing_before_first_valid_pvt_s": (
            float((int(route["t_ns"].min()) - interval_start_ns) / NS_PER_SECOND)
            if len(route) else None
        ),
        "missing_after_last_valid_pvt_s": (
            float((interval_end_ns - int(route["t_ns"].max())) / NS_PER_SECOND)
            if len(route) else None
        ),
        "camera_origin_ns": camera_origin_ns,
        "first_valid_pvt_video_s": (
            VIDEO_TIME.unix_ns_to_playback_seconds(
                camera_clock, int(route["t_ns"].min())
            ) if len(route) and camera_clock is not None else None
        ),
        "last_valid_pvt_video_s": (
            VIDEO_TIME.unix_ns_to_playback_seconds(
                camera_clock, int(route["t_ns"].max())
            ) if len(route) and camera_clock is not None else None
        ),
        "video_time_mapping": video_time_mapping,
    }
    gravity_window_end = int(quality["t_ns"].min()) + int(args.stationary_min_duration * NS_PER_SECOND)
    gravity_window = quality[quality["t_ns"] <= gravity_window_end]
    gravity_lat, gravity_lon, gravity_h = [float(gravity_window[col].median()) for col in ["latitude_deg", "longitude_deg", "height_m"]]
    gravity = GNSS.wgs84_normal_gravity(gravity_lat, gravity_h)

    quality_summary = GNSS.summarize_gnss(quality)
    rawx = data["rawx"]
    rawx_cno = data["rawx_cno"]
    raw_summary = {
        "session_id": args.session_id,
        "topic": "/ubx_rxm_rawx",
        "epochs": int(len(rawx)),
        "observations": int(rawx["decoded_measurements"].sum()) if len(rawx) else 0,
        "observations_per_epoch_median": float(rawx["decoded_measurements"].median()) if len(rawx) else None,
        "carrier_phase_valid_fraction": float(rawx["carrier_phase_valid"].sum() / rawx["decoded_measurements"].sum()) if len(rawx) and rawx["decoded_measurements"].sum() else None,
        "cno_median_dbhz": float(np.median(rawx_cno)) if len(rawx_cno) else None,
        "cno_p05_dbhz": float(np.quantile(rawx_cno, 0.05)) if len(rawx_cno) else None,
        "cno_p95_dbhz": float(np.quantile(rawx_cno, 0.95)) if len(rawx_cno) else None,
    }
    stationary_vel = data["vel"]
    stationary_imu = imu_streams[64]
    stationary_selection = "selected riding interval"
    stationary_reference_bounds = None
    stationary_reference_time_mapping = None
    stationary_reference_gnss = None
    if args.stationary_reference_video_start_s is not None:
        if camera_origin_ns is None:
            raise SystemExit(
                "An explicit stationary video reference requires --video-start-s and --video-end-s"
            )
        timestamp_matches = sorted(session.glob("camera_*/timestamps.csv"))
        video_matches = sorted(session.glob("camera_*/video_mjpg.avi"))
        ref_start_ns, ref_end_ns, stationary_reference_time_mapping = (
            VIDEO_TIME.map_playback_interval(
                video_matches[0], timestamp_matches[0],
                args.stationary_reference_video_start_s,
                args.stationary_reference_video_end_s,
            )
        )
        stationary_vel = data_full["vel"][
            data_full["vel"]["t_ns"].between(ref_start_ns, ref_end_ns, inclusive="both")
        ].copy()
        observed_reference_vel = stationary_vel.copy()
        # The normal IMU load is limited to the rosbag time span.  The camera
        # static reference can precede rosbag start, so load this explicit
        # CSV interval separately rather than silently dropping it.
        reference_imu_streams, _ = GNSS.load_imu(imu_csv, ref_start_ns, ref_end_ns)
        stationary_imu = reference_imu_streams[64].copy()
        stationary_selection = "explicit stationary video reference interval"
        stationary_reference_bounds = {
            "start_ns": int(ref_start_ns),
            "end_ns": int(ref_end_ns),
            "start_video_s": args.stationary_reference_video_start_s,
            "end_video_s": args.stationary_reference_video_end_s,
            "video_time_mapping": stationary_reference_time_mapping,
        }
        stationary_reference_gnss = {
            "sample_count": int(len(observed_reference_vel)),
            "maximum_ground_speed_mps": (
                float(observed_reference_vel["ground_speed_mps"].max())
                if len(observed_reference_vel) else None
            ),
            "samples_above_stationary_threshold": (
                int((observed_reference_vel["ground_speed_mps"] > args.stationary_speed_threshold).sum())
                if len(observed_reference_vel) else 0
            ),
            "role": "Consistency check only; the interval is explicitly labelled stationary from video",
        }
        # The camera can start before GNSS publishing. For an explicitly
        # labelled static video interval, use its complete raw-IMU span as the
        # candidate and validate it with gyro/acceleration stability gates.
        # Partial GNSS coverage is retained above as a consistency check.
        if len(stationary_imu):
            stationary_vel = pd.DataFrame({
                "t_ns": stationary_imu["t_unix_ns"].astype("int64").to_numpy(),
                "ground_speed_mps": np.zeros(len(stationary_imu), dtype=float),
            })
    gravity_metadata = {"location": "Dresden, Germany", "normal_gravity_mps2": gravity}
    stationary, imu_axes, stationary_samples = GNSS.stationary_quality(
        stationary_vel, stationary_imu, args.stationary_speed_threshold,
        args.stationary_min_duration, gravity, gravity_metadata,
        args.stationary_start_offset_s,
    )
    stationary["selection_scope"] = stationary_selection
    if stationary_reference_bounds is not None:
        stationary["reference_video_interval"] = stationary_reference_bounds
        stationary["reference_gnss_observation"] = stationary_reference_gnss
        stationary["selection"] = (
            "Explicit video interval labelled stationary; raw IMU samples were "
            "checked with gyro/acceleration stability gates and available GNSS "
            "speed was retained as a consistency check"
        )
        stationary["stationary_check_definition"] = (
            "Video-labelled static interval plus gyro-norm P95 <= 0.05 rad/s "
            "and acceleration-norm standard deviation <= 0.15 m/s^2; available "
            "GNSS speed is reported as a consistency check"
        )
    course = GNSS.build_course_rate(data["vel"], args.course_min_speed, args.course_max_accuracy)
    if len(course):
        curve, best_lag, best_corr = GNSS.lag_correlation(course, imu_streams[64], args.max_lag, args.lag_step, args.turn_rate_threshold)
        selected, turn_summary = GNSS.select_turn(course, args.turn_duration)
        course_turn, imu_turn = GNSS.build_turn_signals(selected, imu_streams[64], best_lag)
    else:
        curve = pd.DataFrame(); best_lag = best_corr = None; turn_summary = {}; course_turn = imu_turn = pd.DataFrame()

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    GNSS.plot_gnss_distributions(
        quality,
        figures / "gnss_quality_distributions",
        route=route,
        interval_start_ns=interval_start_ns,
        interval_end_ns=interval_end_ns,
    )
    if len(stationary_samples):
        GNSS.plot_imu_stationary(stationary_samples, imu_axes, gravity, figures / "imu_stationary_quality")
    if len(course):
        GNSS.plot_temporal_alignment(curve, course_turn, imu_turn, best_lag, best_corr, figures / "temporal_alignment_and_complementarity")

    streams.insert(0, "session_id", args.session_id)
    streams.to_csv(tables / "stream_timing.csv", index=False)
    GNSS.write_json(
        tables / "timestamp_alignment_audit.json",
        GNSS.build_timestamp_audit(data["pvt"], clock_audit),
    )
    quality.to_csv(tables / "gnss_quality_epochs.csv", index=False)
    route.to_csv(tables / "gnss_route_epochs.csv", index=False)
    GNSS.write_json(tables / "gnss_route_coverage.json", route_coverage)
    quality_summary.insert(0, "session_id", args.session_id)
    quality_summary.to_csv(tables / "gnss_solution_quality.csv", index=False)
    rawx.to_csv(tables / "raw_gnss_epoch_quality.csv", index=False)
    pd.DataFrame([raw_summary]).to_csv(tables / "raw_gnss_summary.csv", index=False)
    GNSS.write_json(tables / "gravity_reference.json", {"location": "Dresden, Germany", "normal_gravity_mps2": gravity, "latitude_deg": gravity_lat, "longitude_deg": gravity_lon, "ellipsoidal_height_m": gravity_h, "interpretation": "Model-derived normal gravity, not a gravimeter measurement"})
    GNSS.write_json(tables / "imu_stationary_summary.json", stationary)
    imu_axes.to_csv(tables / "imu_stationary_axis_statistics.csv", index=False)
    if len(stationary_samples):
        stationary_samples.to_csv(tables / "imu_stationary_figure_data.csv", index=False)
    if len(course):
        course.to_csv(tables / "gnss_course_rate.csv", index=False)
        curve.to_csv(tables / "temporal_lag_curve.csv", index=False)
        GNSS.write_json(tables / "representative_turn.json", {**turn_summary, "effective_lag_s": best_lag, "peak_normalized_correlation": best_corr, "lag_definition": "Effective signal alignment, not direct hardware clock offset"})

    (output / "figure_captions.txt").write_text(
        f"GNSS/IMU ego-motion validation for {args.session_id}. GNSS alignment uses {args.rosbag_time_source} timestamps. AVI player seconds are first converted to nominal-rate frame indices and then mapped through timestamps.csv. All time-series panels use the selected video interval start as t=0 and the selected interval end as the right boundary; missing GNSS samples remain blank. The four-panel GNSS figure uses valid /ubx_nav_pvt positions for panel (a); panels (b)--(d) use the quality-paired epochs. RTK status and hAcc are receiver outputs; hAcc is not measured position error. "
        "The IMU stationary figure uses model-derived normal gravity for Dresden. The acceleration norm is independent of sensor mounting tilt; axis-wise acceleration components are not compared with gravity. The temporal alignment figure reports an effective signal "
        "lag, not a hardware clock error.\n", encoding="utf-8"
    )
    interval_description = (
        f"video seconds {args.video_start_s:g}--{args.video_end_s:g}"
        if args.video_start_s is not None
        else "the complete rosbag interval"
    )
    (output / "README.txt").write_text(
        f"{args.session_id} GNSS/IMU ego-motion validation. The raw rosbag is opened read-only. GNSS alignment uses {args.rosbag_time_source} timestamps. AVI player seconds are converted through frame_idx and timestamps.csv rather than added directly to the first frame time. All GNSS time-series panels use the complete selected video interval {interval_description}; missing GNSS samples are not interpolated. The selected outputs are the GNSS-quality four-panel figure, the IMU stationary figure, and the temporal-alignment figure. Panel (a) uses all valid /ubx_nav_pvt positions; the other GNSS panels use the quality-paired epochs. "
        f"The reported interval is {interval_description}. "
        "The acceleration-norm comparison is valid for a tilted stationary bicycle because the vector norm does not depend on mounting orientation; axis-wise gravity components are not used as accuracy metrics. "
        "Camera height changes affect the gaze extrinsics only; GNSS/IMU calculations are unchanged.\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(), "session_id": args.session_id,
        "session_dir": str(session), "command": shlex.join([sys.executable, *sys.argv]),
        "inputs": [{"path": str(bag_file), "sha256": GNSS.sha256_path(bag_file)}, {"path": str(imu_csv), "sha256": GNSS.sha256_path(imu_csv)}],
        "parameters": vars(args), "gravity_reference": {"location": "Dresden, Germany", "normal_gravity_mps2": gravity},
        "timestamp_source": args.rosbag_time_source,
        "effective_lag_s": best_lag, "peak_correlation": best_corr,
        "analysis_interval": {"start_ns": interval_start_ns, "end_ns": interval_end_ns, "video_start_s": args.video_start_s, "video_end_s": args.video_end_s, "video_time_mapping": video_time_mapping},
        "stationary_reference": stationary_reference_bounds,
    }
    GNSS.write_json(output / "run_manifest.json", manifest)
    GNSS.write_checksums(output)
    print(f"Wrote GNSS/IMU ego-motion validation to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
