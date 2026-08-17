#!/usr/bin/env python3
"""Generate P8 F2, F6 and closed-loop workflow figures.

The F2 and F6 tables use the derived common interval.  The closed-loop panel
uses the requested video interval, converted with camera frame timestamps.
The ``--overwrite`` option updates a derived result directory only; source
files and previous result directories are not changed.
"""

import argparse
import importlib.util
import json
import math
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
import lidar_pcap_sampling as LIDAR_PCAP  # noqa: E402

P9_PATH = SCRIPT_DIR / "p9_speed_timing_closed_loop.py"
SPEC = importlib.util.spec_from_file_location("p9_workflow_functions", P9_PATH)
P9 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(P9)

NS_PER_SECOND = 1_000_000_000


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(session: Path, pattern: str) -> Path:
    matches = sorted(session.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {len(matches)}")
    return matches[0]


def crop(frame, start_ns, end_ns, column="t_unix_ns"):
    return P9.crop(frame, int(start_ns), int(end_ns), column=column)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--common-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="0616_1711_P8")
    parser.add_argument("--video-start-s", type=float, default=90.0)
    parser.add_argument("--video-end-s", type=float, default=120.0)
    parser.add_argument("--speed-pair-tolerance-s", type=float, default=0.2)
    parser.add_argument(
        "--f6-full-interval", action="store_true",
        help="Plot F6 over the complete common GNSS-speed interval instead of the selected 30 s window",
    )
    parser.add_argument(
        "--riding-input-dir", default="",
        help="Derived riding-input result containing the global neutral and brake bands",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow updating an existing derived output directory",
    )
    parser.add_argument("--extrinsics", default=str(SCRIPT_DIR / "bike_extrinsics_P8_20260616_camera_z1p105.json"))
    parser.add_argument(
        "--eyetracker-fixation",
        default="",
        help=(
            "Separate Tobii fixation-filter export. It only supplies fixation "
            "intervals; Methods A/B/C remain based on raw gazedata.gz."
        ),
    )
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    common_dir = Path(args.common_dir).resolve()
    output = Path(args.out).resolve()
    if (output.exists() or output.is_symlink()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    summary_path = common_dir.parent.parent / "tables" / "common_interval_and_steering_cleaning.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing common-interval summary: {summary_path}")
    interval = json.loads(summary_path.read_text(encoding="utf-8"))
    common_start_ns, common_end_ns = int(interval["start_ns"]), int(interval["end_ns"])

    # Use cropped streams for F2/F6.
    steering = P9.read_csv(find_one(common_dir, "steering_angle_*.csv"))
    brake = P9.read_csv(find_one(common_dir, "brake_sensors_force_*.csv"))
    imu = P9.read_csv(find_one(common_dir, "imu_*.csv"))
    wheel = P9.read_csv(find_one(common_dir, "speed_decoded_*.csv"))
    power = P9.read_csv(find_one(common_dir, "rally_payload_decoded_*.csv"))
    camera = pd.read_csv(find_one(common_dir, "timestamps.csv"), low_memory=False)
    gaze_cropped = pd.DataFrame()
    tobii_imu_cropped = pd.DataFrame()
    # The cropped files have the original recording timestamps already mapped
    # to Unix nanoseconds.  They are used for F2 only; the closed-loop window
    # below is read from the complete P8 streams to retain the requested video interval.
    recording = find_one(session, "**/recording.g3") if list(session.glob("**/recording.g3")) else None
    gaze_path = find_one(session, "**/gazedata.gz")
    tobii_imu_path = find_one(session, "**/imudata.gz")
    if recording is None:
        raise SystemExit("Missing recording.g3")
    metadata, gaze_full, tobii_imu_full = P9.read_tobii_recording(recording, gaze_path, tobii_imu_path)
    fixation_path = Path(args.eyetracker_fixation).resolve() if args.eyetracker_fixation else None
    if fixation_path is None:
        fixation_candidates = [
            path for path in sorted(session.rglob("eyetracker_fixation*"))
            if path.is_file()
            and path.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"}
        ]
        if len(fixation_candidates) == 1:
            fixation_path = fixation_candidates[0]
        elif len(fixation_candidates) > 1:
            raise SystemExit(
                "Multiple eyetracker_fixation exports were found; select one "
                "with --eyetracker-fixation"
            )
    fixation_intervals = pd.DataFrame()
    fixation_summary = None
    if fixation_path is not None:
        if not fixation_path.is_file():
            raise SystemExit(f"Fixation export does not exist: {fixation_path}")
        fixation_intervals, fixation_summary = P9.read_tobii_fixation_intervals(
            fixation_path, metadata["created"]
        )

    bag_file = find_one(session, "rosbag2_*/*.db3")
    bag_data = GNSS.read_bag_topics(bag_file, "sqlite3")
    # Camera timestamps are host/recording time, so use rosbag receive time
    # for the GNSS streams rather than the receiver/header clock.
    GNSS.apply_rosbag_time_source(bag_data, "record")
    velocity_full = bag_data["vel"].rename(columns={"t_ns": "t_unix_ns"}).copy()
    velocity = crop(velocity_full, common_start_ns, common_end_ns)
    bag_cropped = {}
    for name, frame in bag_data.items():
        if isinstance(frame, pd.DataFrame) and "t_ns" in frame.columns:
            bag_cropped[name] = crop(frame.rename(columns={"t_ns": "t_unix_ns"}), common_start_ns, common_end_ns)
            bag_cropped[name] = bag_cropped[name].rename(columns={"t_unix_ns": "t_ns"})
        else:
            bag_cropped[name] = frame

    wheel["wheel_speed_mps"] = pd.to_numeric(wheel["speed_mps"], errors="coerce") / 3.6
    wheel_time = pd.to_numeric(wheel["t_unix_ns"], errors="coerce")
    wheel_valid = wheel[
        np.isfinite(wheel["wheel_speed_mps"].to_numpy(dtype=float))
        & np.isfinite(wheel_time.to_numpy(dtype=float))
    ].copy()
    power_p10 = power[power["page_name"].eq("standard_power") & pd.to_numeric(power["p10_instantaneous_power_w"], errors="coerce").notna()].copy()
    raw_imu = imu[imu["dtype"].eq(64)].copy()
    for frame in [raw_imu, brake, steering]:
        for col in ["gyro_z", "left_force_n", "right_force_n", "angle_deg"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    pcap_matches = sorted(session.glob("*.pcap"))
    lidar_streams = None
    lidar_packet_counts = None
    if len(pcap_matches) == 1:
        lidar_streams, lidar_packet_counts = LIDAR_PCAP.extract_scan_timestamps(
            pcap_matches[0], common_start_ns, common_end_ns
        )
    sampling_summary, sampling_intervals = P9.build_sampling_streams(
        bag_cropped, imu, steering, brake, wheel, power, camera,
        crop(gaze_full, common_start_ns, common_end_ns),
        crop(tobii_imu_full, common_start_ns, common_end_ns),
        lidar_streams=lidar_streams,
    )
    selected, candidates = P9.select_representative_window(
        velocity, wheel_valid, steering, brake, power_p10, 30.0, 5.0
    )
    # Prefer a 30 s window without a long wheel-speed dropout.  The older
    # score maximises motion variation, but a single invalid block makes F6
    # harder to read.  Keep the original score as the tie-breaker.
    wheel_gaps = []
    for _, candidate in candidates.iterrows():
        wheel_part = crop(wheel_valid, int(candidate["start_ns"]), int(candidate["end_ns"]))
        dt = np.diff(pd.to_numeric(wheel_part["t_unix_ns"], errors="coerce").to_numpy(dtype=float)) / NS_PER_SECOND
        dt = dt[np.isfinite(dt) & (dt > 0)]
        wheel_gaps.append(float(np.max(dt)) if len(dt) else math.inf)
    candidates["wheel_max_gap_s"] = wheel_gaps
    continuous = candidates[candidates["wheel_max_gap_s"] <= 1.5]
    if len(continuous):
        best_idx = int(continuous["selection_score"].idxmax())
        candidates["selected"] = False
        candidates.loc[best_idx, "selected"] = True
        selected = candidates.loc[best_idx].to_dict()
    selected_start_ns, selected_end_ns = int(selected["start_ns"]), int(selected["end_ns"])
    if args.f6_full_interval:
        # F6 is a comparison of wheel speed and GNSS speed.  Limit the
        # plotted/evaluated window to the first and last valid wheel sample
        # inside the common interval; do not let leading/trailing periods
        # without wheel data appear as a sensor comparison.
        wheel_in_common = wheel_valid[
            wheel_valid["t_unix_ns"].between(common_start_ns, common_end_ns)
        ]
        if len(wheel_in_common):
            f6_start_ns = int(wheel_in_common["t_unix_ns"].min())
            f6_end_ns = int(wheel_in_common["t_unix_ns"].max())
        else:
            raise SystemExit(
                "F6 cannot be generated: no finite wheel-speed sample exists "
                "inside the common interval"
            )
    else:
        f6_start_ns, f6_end_ns = selected_start_ns, selected_end_ns
    gnss_speed, wheel_speed, paired, speed_summary = P9.prepare_speed_comparison(
        velocity, wheel_valid, f6_start_ns, f6_end_ns, args.speed_pair_tolerance_s
    )
    course_rate = P9.gnss_course_rate(velocity)

    # Camera timestamps establish the global origin for the requested video interval.
    raw_camera_path = find_one(session, "camera_*/timestamps.csv")
    raw_video_path = find_one(session, "camera_*/video_mjpg.avi")
    video_start_ns, video_end_ns, video_time_mapping = VIDEO_TIME.map_playback_interval(
        raw_video_path, raw_camera_path, args.video_start_s, args.video_end_s
    )
    camera_origin_ns = int(video_time_mapping["first_unix_ns"])
    # Raw streams are used for this exact video interval, which starts before
    # the first positive-speed sample in P8.  The source files are read-only.
    imu_full = P9.read_csv(find_one(session, "imu_*.csv"))
    steering_full = P9.read_csv(find_one(session, "steering_angle_*.csv"))
    brake_full = P9.read_csv(find_one(session, "brake_sensors_force_*.csv"))
    wheel_full = P9.read_csv(find_one(session, "speed_decoded_*.csv"))
    power_full = P9.read_csv(find_one(session, "rally_payload_decoded_*.csv"))
    wheel_full["wheel_speed_mps"] = pd.to_numeric(wheel_full["speed_mps"], errors="coerce") / 3.6
    raw_imu_full = imu_full[imu_full["dtype"].eq(64)].copy()
    power_full_p10 = power_full[power_full["page_name"].eq("standard_power")].copy()

    riding_dir = Path(args.riding_input_dir).resolve() if args.riding_input_dir else None
    steering_neutral = None
    brake_bands = None
    if riding_dir is not None:
        neutral_path = riding_dir / "tables" / "steering_neutral_reference.json"
        band_path = riding_dir / "tables" / "brake_zero_input_band.csv"
        if not neutral_path.is_file() or not band_path.is_file():
            raise SystemExit(
                f"Missing riding-input reference tables under {riding_dir}: "
                "steering_neutral_reference.json and brake_zero_input_band.csv are required"
            )
        steering_neutral = json.loads(neutral_path.read_text(encoding="utf-8"))
        band_table = pd.read_csv(band_path)
        brake_bands = {
            str(row["force_column"]): (
                float(row["zero_band_lower_n"]),
                float(row["zero_band_upper_n"]),
            )
            for _, row in band_table.iterrows()
        }

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    P9.plot_sampling_intervals(sampling_summary, sampling_intervals, figures / "F2_sampling_interval_boxplot")
    P9.plot_speed_comparison(gnss_speed, wheel_speed, paired, speed_summary, f6_start_ns, figures / "F6_speed_compare_overall")
    P9.plot_closed_loop(
        gaze_full, steering_full, brake_full, power_full_p10, raw_imu_full,
        P9.gnss_course_rate(velocity_full), velocity_full, wheel_full,
        video_start_ns, video_end_ns, figures / "P9_representative_closed_loop",
        Path(args.extrinsics),
        steering_neutral=steering_neutral,
        brake_bands=brake_bands,
        fixation_intervals=fixation_intervals,
    )

    sampling_summary.sort_values("median_interval_ms").to_csv(tables / "sampling_interval_summary.csv", index=False)
    if lidar_streams is not None:
        pd.DataFrame({
            name: pd.Series(timestamps, dtype="Int64")
            for name, timestamps in lidar_streams.items()
        }).to_csv(tables / "lidar_scan_timestamps.csv", index=False)
        P9.write_json(tables / "lidar_pcap_scan_extraction.json", {
            "pcap": str(pcap_matches[0]),
            "method": "MSOP first-block azimuth wrap (360 to 0 degrees)",
            "interval_start_ns": common_start_ns,
            "interval_end_ns": common_end_ns,
            "packet_counts": lidar_packet_counts,
            "scan_counts": {name: int(len(values)) for name, values in lidar_streams.items()},
        })
    candidates.to_csv(tables / "representative_window_candidates.csv", index=False)
    paired.to_csv(tables / "F6_speed_paired_samples.csv", index=False)
    pd.DataFrame([speed_summary]).to_csv(tables / "F6_speed_comparison_summary.csv", index=False)
    P9.write_json(tables / "selected_window.json", selected)
    gaze_window = P9.add_gaze_angle_methods(P9.smooth_gaze(crop(gaze_full, video_start_ns, video_end_ns)), Path(args.extrinsics))
    method_rows = []
    for column, label in [
        ("gaze_x_norm", "Tobii gaze2d horizontal coordinate"),
        ("gaze_x_relative_norm", "Raw 2-D relative horizontal position"),
        ("gaze_2d_relative_angle_deg", "Raw 2-D relative horizontal angle"),
        ("former_ego_angle_deg", "Former ego-angle calculation"),
        ("method_a_ego_angle_deg", "Method A: Tobii 3-D point"),
        ("method_b_ego_angle_deg", "Method B: binocular gaze direction"),
        ("method_c_ego_angle_deg", "Method C: 2-D back-projected ray"),
    ]:
        values = pd.to_numeric(gaze_window[column], errors="coerce").dropna()
        method_rows.append({"method": label, "n_valid": int(len(values)), "median": float(values.median()) if len(values) else None, "p95_abs": float(values.abs().quantile(0.95)) if len(values) else None, "minimum": float(values.min()) if len(values) else None, "maximum": float(values.max()) if len(values) else None, "over_100_deg": int((values.abs() > 100).sum()) if "angle" in column else None})
    pd.DataFrame(method_rows).to_csv(tables / "gaze_angle_method_summary.csv", index=False)
    validity_rows = []
    total_gaze_rows = int(len(gaze_window))
    for method in ("a", "b", "c"):
        fields = gaze_window[f"method_{method}_fields_valid"].fillna(False).astype(bool)
        forward = gaze_window[f"method_{method}_forward_valid"].fillna(False).astype(bool)
        valid = gaze_window[f"method_{method}_valid"].fillna(False).astype(bool)
        validity_rows.append({
            "method": f"Method {method.upper()}",
            "raw_gaze_records": total_gaze_rows,
            "required_fields_valid": int(fields.sum()),
            "image_domain_valid": (
                int(gaze_window["gaze_2d_image_valid"].sum())
                if method == "c" else None
            ),
            "forward_geometry_valid": int(forward.sum()),
            "output_valid": int(valid.sum()),
            "output_valid_percent": (
                100.0 * float(valid.mean()) if total_gaze_rows else None
            ),
        })
    pd.DataFrame(validity_rows).to_csv(
        tables / "gaze_method_validity_summary.csv", index=False
    )
    if fixation_summary is not None:
        fixation_window = fixation_intervals[
            (fixation_intervals["end_ns"] >= video_start_ns)
            & (fixation_intervals["start_ns"] <= video_end_ns)
        ].copy()
        fixation_window.to_csv(
            tables / "tobii_fixation_intervals.csv", index=False
        )
        fixation_summary = {
            **fixation_summary,
            "selected_window_event_count": int(len(fixation_window)),
            "selected_window_start_ns": int(video_start_ns),
            "selected_window_end_ns": int(video_end_ns),
        }
        P9.write_json(
            tables / "tobii_fixation_summary.json", fixation_summary
        )
    P9.write_json(tables / "video_interval_global_time.json", {
        "video_file": str(raw_video_path),
        "video_start_s": args.video_start_s, "video_end_s": args.video_end_s,
        "camera_origin_ns": camera_origin_ns, "start_ns": video_start_ns, "end_ns": video_end_ns,
        "start_utc": datetime.fromtimestamp(video_start_ns / NS_PER_SECOND, tz=timezone.utc).isoformat(),
        "end_utc": datetime.fromtimestamp(video_end_ns / NS_PER_SECOND, tz=timezone.utc).isoformat(),
        "time_mapping": video_time_mapping,
        "note": "AVI player seconds are mapped to nominal-rate frame indices and then to timestamps.csv unix_ns. Raw read-only streams are used for this panel."
    })

    captions = (
        f"F2. Sampling interval distributions for {args.session_id} sensor streams from one recording in an urban 30 km/h zone. "
        "LiDAR intervals are full-scan intervals detected from MSOP azimuth wrap, not UDP packet intervals. "
        "Boxes show the interquartile range, centre lines show medians, whiskers extend to 1.5 times the interquartile range, "
        "and points outside the whiskers are omitted. The horizontal axis is logarithmic. "
        f"F6. Comparison of wheel-sensor speed and receiver-provided GNSS ground speed for one recording in an urban 30 km/h zone, using {'the interval from the first to the last valid wheel-speed sample inside the common interval' if args.f6_full_interval else 'a 30 s dynamic window selected for non-empty sensor coverage'}; GNSS timestamps use rosbag record time; "
        "the wheel speed is converted from the legacy km/h field by division by 3.6. "
        f"{args.session_id} closed-loop workflow. Subfigures are ordered as (a) three static-frame "
        "gaze-angle methods, (b) steering, (c) raw gyroscope z-axis yaw-rate, (d) power, (e) left/right brake force, and (f) speed. "
        "Methods A--C are calculated from the raw Tobii gaze stream. Shaded intervals, when present, are Tobii-classified fixations read separately "
        "from the fixation export. Methods A--C use one static scene-camera "
        "optical-frame convention: the recording.g3 HUCS-to-camera rotation is applied to A/B and C uses the same "
        "camera ray; all angles are atan2(x_camera,z_camera). The former angle is retained only in the method-summary table, not plotted. "
        "When supplied, the steering panel uses the global neutral reference and the brake panel uses the full observed "
        "zero-input force ranges measured during the pre-ride video static interval."
    )
    (output / "figure_captions.txt").write_text(captions + "\n", encoding="utf-8")
    (output / "gaze_method_notes.txt").write_text(
        "Gaze-angle audit. Methods A--C use raw gazedata.gz, not the fixation "
        "export. Tobii gaze2d is a normalized image coordinate (0--1), not an "
        "ego-frame angle. Its relative horizontal position and calibrated angle "
        "are retained in the audit table but the raw position is not plotted. "
        "The P8 recording.g3 calibration supplies the "
        "rotation from Tobii HUCS to the scene-camera optical frame. Method A "
        "transforms the Tobii 3-D gaze point with that rotation; Method B applies "
        "it to the mean binocular direction; Method C back-projects the normalized "
        "2-D point with the same camera intrinsics. All three use atan2(x_camera, "
        "z_camera), where x is right and z is forward. Method A requires finite "
        "3-D gaze-point coordinates and a positive transformed forward component. "
        "Method B requires finite left- and right-eye direction vectors and a "
        "positive forward component of their mean. Method C requires finite "
        "normalized 2-D coordinates inside [0,1] x [0,1] and a forward "
        "back-projected ray. Thus, valid means numerical and geometric availability; "
        "it does not mean a Tobii fixation or independently verified gaze accuracy. "
        "This closed-loop script does not call the AprilTag head/back-pose "
        "estimation pipeline. Camera-frame availability, tag detections, and "
        "head/back-pose quality are therefore not part of the Method A--C validity "
        "counts. A--C are static camera-frame diagnostics and must not be described "
        "as dynamically head- or bicycle-frame-compensated gaze.\n\n"
        "The former calculation used atan2(raw Tobii y, raw Tobii x) as if the raw "
        "Tobii x axis were the forward bicycle axis. The P8 calibration shows that "
        "the forward component is z after conversion to the scene-camera frame, so "
        "the old formula could put a lateral component in the denominator and "
        "produce wrap-around values near +/-180 degrees. The former calculation is "
        "retained in the audit table only. After the explicit transform and forward "
        "gate, A, B and C have no samples above 100 degrees in the selected window. "
        "A and B use 3-D/eye-direction data; C uses the image ray and normally gives "
        "the most direct camera-based diagnostic. These are consistency checks, not "
        "an independent ground-truth measurement of gaze direction.\n",
        encoding="utf-8",
    )
    (output / "README.txt").write_text(
        f"{args.session_id} validation figures. F2/F6 use the derived common interval selected by GNSS ground speed >0.1 m/s. "
        f"The closed-loop figure uses AVI player seconds {args.video_start_s:g}--{args.video_end_s:g}, converted to frame indices at the nominal video rate and then mapped through timestamps.csv. "
        "The closed-loop signal interval is therefore kept separate from the common crop. The raw normalized gaze position "
        "is retained only in the audit table and is not plotted. Methods A/B/C "
        "are static-frame diagnostics; this script does not call the dynamic AprilTag head/back-pose pipeline. "
        "The steering and brake reference bands are taken from the riding-input validation result when supplied. "
        "Raw data and previous output directories are unchanged.\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "session_dir": str(session),
        "common_input_dir": str(common_dir),
        "command": shlex.join([sys.executable, *sys.argv]),
        "common_interval": interval,
        "selected_speed_window": selected,
        "video_interval": {"start_ns": video_start_ns, "end_ns": video_end_ns, "start_s": args.video_start_s, "end_s": args.video_end_s, "camera_origin_ns": camera_origin_ns, "time_mapping": video_time_mapping},
        "f6_interval": {
            "start_ns": f6_start_ns,
            "end_ns": f6_end_ns,
            "duration_s": (f6_end_ns - f6_start_ns) / NS_PER_SECOND,
            "selection": "first-to-last valid wheel-speed sample inside common interval" if args.f6_full_interval else "selected dynamic window",
        },
        "gnss_timestamp_source": "rosbag record time",
        "riding_input_reference": {
            "directory": str(riding_dir) if riding_dir is not None else None,
            "steering_neutral_used": steering_neutral is not None,
            "brake_zero_bands_used": brake_bands is not None,
        },
        "extrinsics": {"path": str(Path(args.extrinsics).resolve()), "sha256": sha256(Path(args.extrinsics).resolve())},
        "scripts": {
            "main": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "workflow_functions": {
                "path": str(P9_PATH),
                "sha256": sha256(P9_PATH),
            },
            "lidar_pcap_sampling": {
                "path": str((SCRIPT_DIR / "lidar_pcap_sampling.py").resolve()),
                "sha256": sha256(SCRIPT_DIR / "lidar_pcap_sampling.py"),
            },
        },
        "speed_comparison": speed_summary,
        "tobii_recording": {"created": metadata["created"], "duration_s": metadata["duration"]},
        "gaze_sources": {
            "method_a_b_c": str(gaze_path),
            "relative_normalized_2d": str(gaze_path),
            "fixation_intervals": (
                str(fixation_path) if fixation_path is not None else None
            ),
            "fixation_summary": fixation_summary,
        },
    }
    P9.write_json(output / "run_manifest.json", manifest)
    checksums = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "CHECKSUMS.sha256"):
        checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Wrote P8 F2/F6/closed-loop figures to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
