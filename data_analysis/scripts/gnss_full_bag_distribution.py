#!/usr/bin/env python3
"""Create GNSS quality figures from the complete rosbag recording.

This diagnostic deliberately uses the complete bag interval.  It is separate
from the video-cropped ego-motion result and does not modify source data or
previous result directories.
"""

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gnss_imu_technical_validation as GNSS  # noqa: E402
from paper_style import apply_paper_style  # noqa: E402
import video_time_mapping as VIDEO_TIME  # noqa: E402


def video_seconds_if_covered(camera_clock, unix_ns):
    """Return AVI playback time only when the record lies in camera coverage."""
    if camera_clock is None or unix_ns is None:
        return None
    unix_ns = int(unix_ns)
    if not (int(camera_clock["first_unix_ns"]) <= unix_ns
            <= int(camera_clock["last_unix_ns"])):
        return None
    return VIDEO_TIME.unix_ns_to_playback_seconds(camera_clock, unix_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--session-id", default="session_full_rosbag")
    parser.add_argument(
        "--rosbag-time-source", choices=["record", "header"], default="record",
        help="Timestamp used for the complete-bag timeline",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    session = Path(args.session_dir).resolve()
    output = Path(args.out).resolve()
    if (output.exists() or output.is_symlink()) and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    db_matches = sorted(session.glob("rosbag2_*/*.db3"))
    if len(db_matches) != 1:
        raise SystemExit("Expected one rosbag db3")
    bag_file = db_matches[0]

    data = GNSS.read_bag_topics(bag_file, "sqlite3")
    GNSS.apply_rosbag_time_source(data, args.rosbag_time_source)
    pvt = data["pvt"]
    if not len(pvt):
        raise SystemExit("No /ubx_nav_pvt records found")
    bag_start = min(int(frame["t_ns"].min()) for frame in data.values()
                    if isinstance(frame, pd.DataFrame) and len(frame) and "t_ns" in frame)
    bag_end = max(int(frame["t_ns"].max()) for frame in data.values()
                  if isinstance(frame, pd.DataFrame) and len(frame) and "t_ns" in frame)

    quality = GNSS.build_gnss_quality(data)
    q_east, q_north, q_up = GNSS.geodetic_to_enu(
        quality["latitude_deg"], quality["longitude_deg"], quality["height_m"]
    )
    quality["east_m"], quality["north_m"], quality["up_m"] = q_east, q_north, q_up
    route = pvt[~pvt["invalid_llh"]].copy()
    route_east, route_north, route_up = GNSS.geodetic_to_enu(
        route["latitude_deg"], route["longitude_deg"], route["height_m"]
    )
    route["east_m"], route["north_m"], route["up_m"] = route_east, route_north, route_up

    camera_origin_ns = None
    camera_end_ns = None
    camera_clock = None
    camera_matches = sorted(session.glob("camera_*/timestamps.csv"))
    video_matches = sorted(session.glob("camera_*/video_mjpg.avi"))
    if len(camera_matches) == 1 and len(video_matches) == 1:
        camera_clock = VIDEO_TIME.load_video_clock(video_matches[0], camera_matches[0])
        camera_origin_ns = int(camera_clock["first_unix_ns"])
        camera_end_ns = int(camera_clock["last_unix_ns"])

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    GNSS.plot_gnss_distributions(
        quality,
        figures / "gnss_quality_distributions_full_rosbag",
        route=route,
        interval_start_ns=bag_start,
        interval_end_ns=bag_end,
        route_title="Valid PVT route over the complete rosbag",
        quality_title="RTK state and hAcc over the complete rosbag",
    )
    GNSS.plot_gnss_route(quality, figures / "gnss_route_by_rtk_status_full_rosbag", route=route)

    quality.to_csv(tables / "gnss_quality_epochs_full_rosbag.csv", index=False)
    route.to_csv(tables / "gnss_route_epochs_full_rosbag.csv", index=False)
    first_route_ns = int(route["t_ns"].min()) if len(route) else None
    last_route_ns = int(route["t_ns"].max()) if len(route) else None
    first_route_video_s = video_seconds_if_covered(camera_clock, first_route_ns)
    last_route_video_s = video_seconds_if_covered(camera_clock, last_route_ns)
    GNSS.write_json(tables / "gnss_solution_quality_full_rosbag.json", {
        "session_id": args.session_id,
        "timestamp_source": args.rosbag_time_source,
        "bag_start_ns": bag_start,
        "bag_end_ns": bag_end,
        "duration_s": (bag_end - bag_start) / GNSS.NS_PER_SECOND,
        "pvt_rows": int(len(pvt)),
        "valid_route_rows": int(len(route)),
        "quality_rows": int(len(quality)),
        "first_route_ns": first_route_ns,
        "last_route_ns": last_route_ns,
        "camera_origin_ns": camera_origin_ns,
        "camera_end_ns": camera_end_ns,
        "camera_timestamp_duration_s": (
            (camera_end_ns - camera_origin_ns) / GNSS.NS_PER_SECOND
            if camera_origin_ns is not None and camera_end_ns is not None else None
        ),
        "camera_playback_duration_s": (
            camera_clock["playback_duration_s"]
            if camera_clock is not None else None
        ),
        "first_route_video_s": first_route_video_s,
        "last_route_video_s": last_route_video_s,
        "camera_end_after_last_route_record_time_s": (
            (camera_end_ns - last_route_ns) / GNSS.NS_PER_SECOND
            if last_route_ns is not None and camera_end_ns is not None else None
        ),
        "camera_end_after_last_route_playback_s": (
            camera_clock["playback_duration_s"] - last_route_video_s
            if camera_clock is not None and last_route_video_s is not None else None
        ),
        "video_time_mapping": ({
            key: value for key, value in camera_clock.items()
            if key not in {"frames", "unix_ns"}
        } if camera_clock is not None else None),
    })
    GNSS.write_json(tables / "timestamp_alignment_audit_full_rosbag.json",
                    GNSS.build_timestamp_audit(pvt, {}))
    (output / "README.txt").write_text(
        f"Complete-rosbag GNSS diagnostic for {args.session_id}. The rosbag is read-only. "
        f"GNSS analysis uses {args.rosbag_time_source} time; receiver UTC is retained only as a diagnostic field. "
        "The route panel includes every PVT epoch with valid latitude/longitude, including RTK float and no-carrier states. "
        "The hAcc, satellite and PDOP panels use quality-paired valid PVT/HPPOS epochs. "
        "This figure is not cropped to the camera riding interval. Camera/PVT coverage and any end gap are recorded in "
        "tables/gnss_solution_quality_full_rosbag.json. No position interpolation is applied.\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "session_dir": str(session),
        "command": shlex.join([sys.executable, *sys.argv]),
        "input": {"path": str(bag_file), "sha256": GNSS.sha256_path(bag_file)},
        "script": {"path": str(Path(__file__).resolve()), "sha256": GNSS.sha256_path(Path(__file__).resolve())},
        "timestamp_source": args.rosbag_time_source,
        "bag_interval": {"start_ns": bag_start, "end_ns": bag_end, "duration_s": (bag_end - bag_start) / GNSS.NS_PER_SECOND},
        "counts": {"pvt": int(len(pvt)), "valid_route": int(len(route)), "quality": int(len(quality))},
    }
    GNSS.write_json(output / "run_manifest.json", manifest)
    GNSS.write_checksums(output)
    print(f"Wrote complete-rosbag GNSS figures to {output}")


if __name__ == "__main__":
    apply_paper_style()
    main()
