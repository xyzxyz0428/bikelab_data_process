#!/usr/bin/env python3
"""Generate one reproducible EKF sweep configuration from the baseline YAML."""

import argparse
from pathlib import Path

import yaml


COMPARISON_NODES = (
    "compare_gps_course",
    "compare_gps_course_raw_gyro",
    "compare_gps_course_ahrs_rate",
)
STATE_SIZE = 15
YAW_INDEX = 5
YAW_RATE_INDEX = 11


def positive_or_none(value):
    """Parse a positive threshold, or the literal 'off'."""
    if value.lower() == "off":
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("threshold must be positive or 'off'")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yaw-process-noise", type=float, default=0.06)
    parser.add_argument("--yaw-rate-process-noise", type=float, default=0.02)
    parser.add_argument("--course-rejection", type=positive_or_none, default=None)
    parser.add_argument("--gyro-rejection", type=positive_or_none, default=None)
    parser.add_argument("--fuse-gnss-speed", action="store_true")
    args = parser.parse_args()

    if args.yaw_process_noise <= 0 or args.yaw_rate_process_noise <= 0:
        parser.error("process-noise values must be positive")

    base = Path(args.base).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out}")
    with base.open() as stream:
        config = yaml.safe_load(stream)

    for node_name in COMPARISON_NODES:
        params = config[node_name]["ros__parameters"]
        matrix = params["process_noise_covariance"]
        if len(matrix) != STATE_SIZE * STATE_SIZE:
            raise SystemExit(f"Unexpected process-noise size for {node_name}")
        matrix[YAW_INDEX * STATE_SIZE + YAW_INDEX] = args.yaw_process_noise
        matrix[YAW_RATE_INDEX * STATE_SIZE + YAW_RATE_INDEX] = (
            args.yaw_rate_process_noise
        )

        if args.course_rejection is None:
            params.pop("imu0_pose_rejection_threshold", None)
        else:
            params["imu0_pose_rejection_threshold"] = args.course_rejection

        if node_name != "compare_gps_course":
            if args.gyro_rejection is None:
                params.pop("imu1_twist_rejection_threshold", None)
            else:
                params["imu1_twist_rejection_threshold"] = args.gyro_rejection

        if args.fuse_gnss_speed:
            params.update({
                "twist0": "/compare_input/velocity",
                "twist0_config": [
                    False, False, False,
                    False, False, False,
                    True, False, False,
                    False, False, False,
                    False, False, False,
                ],
                "twist0_differential": False,
                "twist0_relative": False,
                "twist0_queue_size": 100,
            })
        else:
            for key in (
                "twist0", "twist0_config", "twist0_differential",
                "twist0_relative", "twist0_queue_size",
            ):
                params.pop(key, None)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, width=1000)
    print(out)


if __name__ == "__main__":
    main()
