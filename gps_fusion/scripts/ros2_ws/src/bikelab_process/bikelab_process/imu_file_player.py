#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import statistics
from bisect import bisect_left
from pathlib import Path

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu
from rosgraph_msgs.msg import Clock
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)


def ns_to_time_msg(ns: int) -> Time:
    msg = Time()
    msg.sec = int(ns // 1_000_000_000)
    msg.nanosec = int(ns % 1_000_000_000)
    return msg


def clean_key(key):
    return str(key).strip().lstrip("\ufeff")


def get_float(row, key, default=0.0):
    value = row.get(key, "")
    if value is None or value == "":
        return default
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def normalize_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def tilt_compensated_yaw_rate_native_frd(
    gyro_y,
    gyro_z,
    roll,
    pitch,
    yaw_sign=-1.0,
    minimum_abs_cos_pitch=0.1,
):
    """Convert native FRD body rates to a ZYX Euler yaw rate.

    ``roll`` and ``pitch`` use the native AHRS FRD/NED convention.  The
    default sign converts the native clockwise heading rate to ROS ENU yaw,
    for which ``yaw = pi/2 - heading + constant``.
    """
    values = (gyro_y, gyro_z, roll, pitch, yaw_sign)
    if not all(math.isfinite(value) for value in values):
        return None
    cos_pitch = math.cos(pitch)
    if abs(cos_pitch) < minimum_abs_cos_pitch:
        return None
    heading_rate = (
        gyro_y * math.sin(roll) + gyro_z * math.cos(roll)
    ) / cos_pitch
    return yaw_sign * heading_rate


def yaw_to_quaternion(yaw_rad):
    """
    Return quaternion in order qw, qx, qy, qz.
    ROS message assignment later uses x,y,z,w.
    """
    half = yaw_rad * 0.5
    return math.cos(half), 0.0, 0.0, math.sin(half)


def valid_quaternion(qw, qx, qy, qz):
    values = [qw, qx, qy, qz]
    if not all(math.isfinite(v) for v in values):
        return False
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return 0.5 < norm < 1.5


def read_table(path: str):
    path_obj = Path(path)

    if path_obj.suffix.lower() == ".csv":
        with open(path_obj, newline="") as fp:
            reader = csv.DictReader(fp)
            rows = []
            for row in reader:
                rows.append({clean_key(k): v for k, v in row.items()})
            return rows

    if path_obj.suffix.lower() in [".xlsx", ".xls"]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading Excel requires pandas and openpyxl. "
                "Install with: pip install pandas openpyxl"
            ) from exc

        df = pd.read_excel(path_obj)
        df.columns = [clean_key(c) for c in df.columns]
        return df.astype(object).where(df.notna(), "").to_dict(orient="records")

    raise RuntimeError(f"Unsupported IMU file format: {path_obj.suffix}")


class ImuFilePlayer(Node):
    def __init__(self):
        super().__init__("imu_file_player")

        self.declare_parameter("file_path", "")
        self.declare_parameter("topic", "/imu/data")
        self.declare_parameter("frame_id", "imu_link")

        # The test file stores heading in radians.
        self.declare_parameter("angle_unit", "rad")  # rad or deg

        # heading_north_to_enu:
        #   raw heading: 0 = North, clockwise positive
        #   ROS ENU yaw: 0 = East, counter-clockwise positive
        #   yaw_enu = pi/2 - heading
        #
        # csv_quat:
        #   directly publish CSV qw,qx,qy,qz
        #
        # heading_as_enu:
        #   use heading directly as ENU yaw
        self.declare_parameter("orientation_mode", "heading_north_to_enu")
        # Optional mounting yaw correction.
        self.declare_parameter("mount_yaw_offset_rad", 0.0)

        # dtype 64 = raw IMU gyro/acc
        # dtype 65 = AHRS orientation
        self.declare_parameter("dtype_imu", "64")
        self.declare_parameter("dtype_ahrs", "65")

        # Match nearest dtype64 row to each dtype65 row.
        self.declare_parameter("max_merge_dt_ms", 80.0)

        # Do not synthesize zero gyro/acceleration when raw IMU is missing.
        self.declare_parameter("publish_unmatched_ahrs", False)
        self.declare_parameter("drop_invalid_frames", True)

        self.declare_parameter("gyro_unit", "rad_s")   # rad_s or deg_s
        self.declare_parameter("acc_unit", "mps2")     # mps2 or g

        self.declare_parameter("time_offset_ns", 0)
        # Skip IMU rows older than the first bag clock to avoid a startup burst.
        self.declare_parameter("skip_rows_before_first_clock", True)

        # Use fixed covariances or estimate them from a stationary interval.
        self.declare_parameter("covariance_mode", "fixed")
        self.declare_parameter("static_start_s", 0.0)
        self.declare_parameter("static_duration_s", 10.0)
        self.declare_parameter("min_static_samples", 100)
        self.declare_parameter("require_static_window", True)
        self.declare_parameter("static_gyro_std_max_rad_s", 0.03)
        self.declare_parameter("static_acc_norm_std_max_mps2", 0.15)
        self.declare_parameter("covariance_report_path", "")

        # Values below are variances; floors bound the static estimates.
        self.declare_parameter("orientation_cov_roll", 0.05)
        self.declare_parameter("orientation_cov_pitch", 0.05)
        self.declare_parameter("orientation_cov_yaw", 0.10)
        self.declare_parameter("gyro_cov", 0.01)
        self.declare_parameter("gyro_covariance_scale", 1.0)
        self.declare_parameter("acc_cov", 0.20)
        self.declare_parameter("subtract_static_gyro_bias", False)

        # Floors cover errors not measured by static repeatability.
        self.declare_parameter("orientation_cov_floor_roll", 0.05)
        self.declare_parameter("orientation_cov_floor_pitch", 0.05)
        self.declare_parameter("orientation_cov_floor_yaw", 0.10)
        self.declare_parameter("gyro_cov_floor", 1.0e-6)
        self.declare_parameter("acc_cov_floor", 1.0e-4)
        self.declare_parameter("yaw_rate_source", "raw_gyro_z")
        self.declare_parameter("axis_conversion", "none")
        self.declare_parameter("tilt_min_abs_cos_pitch", 0.1)

        self.file_path = self.get_parameter("file_path").value
        self.topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.angle_unit = self.get_parameter("angle_unit").value
        self.orientation_mode = self.get_parameter("orientation_mode").value
        self.mount_yaw_offset_rad = float(
            self.get_parameter("mount_yaw_offset_rad").value
        )
        self.yaw_rate_source = str(
            self.get_parameter("yaw_rate_source").value
        ).strip()
        self.axis_conversion = str(
            self.get_parameter("axis_conversion").value
        ).strip()
        self.tilt_min_abs_cos_pitch = float(
            self.get_parameter("tilt_min_abs_cos_pitch").value
        )

        supported_rate_sources = {
            "raw_gyro_z",
            "ahrs_headingspeed",
            "tilt_compensated_raw_gyro",
        }
        if self.yaw_rate_source not in supported_rate_sources:
            raise RuntimeError(
                "yaw_rate_source must be one of "
                f"{sorted(supported_rate_sources)}, got: "
                f"{self.yaw_rate_source}"
            )
        if not 0.0 < self.tilt_min_abs_cos_pitch <= 1.0:
            raise RuntimeError("tilt_min_abs_cos_pitch must be in (0, 1]")
        if self.yaw_rate_source == "tilt_compensated_raw_gyro":
            if self.orientation_mode not in {
                "heading_north_to_enu", "heading_as_enu",
            }:
                raise RuntimeError(
                    "tilt_compensated_raw_gyro requires a heading-based "
                    "orientation_mode"
                )
            if self.axis_conversion != "frd_to_flu":
                raise RuntimeError(
                    "tilt_compensated_raw_gyro requires "
                    "axis_conversion=frd_to_flu"
                )
        self.dtype_imu = str(self.get_parameter("dtype_imu").value).strip()
        self.dtype_ahrs = str(self.get_parameter("dtype_ahrs").value).strip()
        self.max_merge_dt_ns = int(
            float(self.get_parameter("max_merge_dt_ms").value) * 1e6
        )
        self.publish_unmatched_ahrs = bool(
            self.get_parameter("publish_unmatched_ahrs").value
        )
        self.drop_invalid_frames = bool(
            self.get_parameter("drop_invalid_frames").value
        )

        self.gyro_unit = self.get_parameter("gyro_unit").value
        self.acc_unit = self.get_parameter("acc_unit").value
        self.time_offset_ns = int(self.get_parameter("time_offset_ns").value)
        self.skip_rows_before_first_clock = bool(
            self.get_parameter("skip_rows_before_first_clock").value
        )

        self.covariance_mode = str(
            self.get_parameter("covariance_mode").value
        ).strip().lower()
        self.static_start_s = float(self.get_parameter("static_start_s").value)
        self.static_duration_s = float(self.get_parameter("static_duration_s").value)
        self.min_static_samples = int(self.get_parameter("min_static_samples").value)
        self.require_static_window = bool(
            self.get_parameter("require_static_window").value
        )
        self.static_gyro_std_max_rad_s = float(
            self.get_parameter("static_gyro_std_max_rad_s").value
        )
        self.static_acc_norm_std_max_mps2 = float(
            self.get_parameter("static_acc_norm_std_max_mps2").value
        )
        self.covariance_report_path = str(
            self.get_parameter("covariance_report_path").value
        ).strip()

        self.orientation_cov = [
            float(self.get_parameter("orientation_cov_roll").value),
            float(self.get_parameter("orientation_cov_pitch").value),
            float(self.get_parameter("orientation_cov_yaw").value),
        ]
        fixed_gyro_cov = float(self.get_parameter("gyro_cov").value)
        self.gyro_covariance_scale = float(
            self.get_parameter("gyro_covariance_scale").value
        )
        if self.gyro_covariance_scale <= 0.0:
            raise RuntimeError("gyro_covariance_scale must be positive")
        self.subtract_static_gyro_bias = bool(
            self.get_parameter("subtract_static_gyro_bias").value
        )
        fixed_acc_cov = float(self.get_parameter("acc_cov").value)
        self.gyro_covariance = [fixed_gyro_cov] * 3
        self.gyro_bias = [0.0, 0.0, 0.0]
        self.acc_covariance = [fixed_acc_cov] * 3
        self.tilt_static_sample_count = 0
        self.tilt_published_frame_count = 0
        self.tilt_rejected_frame_count = 0

        self.orientation_cov_floor = [
            float(self.get_parameter("orientation_cov_floor_roll").value),
            float(self.get_parameter("orientation_cov_floor_pitch").value),
            float(self.get_parameter("orientation_cov_floor_yaw").value),
        ]
        self.gyro_cov_floor = float(self.get_parameter("gyro_cov_floor").value)
        self.acc_cov_floor = float(self.get_parameter("acc_cov_floor").value)

        if not self.file_path:
            raise RuntimeError(
                "No IMU file given. Use: "
                "ros2 run bikelab_process imu_file_player --ros-args "
                "-p file_path:=/path/to/imu.csv"
            )

        if not Path(self.file_path).exists():
            raise RuntimeError(f"IMU file not found: {self.file_path}")

        raw_rows = read_table(self.file_path)

        imu_rows = []
        ahrs_rows = []
        invalid_frames = 0

        for row in raw_rows:
            if "t_unix_ns" not in row or row["t_unix_ns"] in ["", None]:
                continue

            checks = [
                row.get(key) for key in ("crc8_ok", "crc16_ok", "end_ok")
                if row.get(key) not in (None, "")
            ]
            if (
                self.drop_invalid_frames
                and checks
                and not all(str(value).strip() == "1" for value in checks)
            ):
                invalid_frames += 1
                continue

            dtype = str(row.get("dtype", "")).strip()
            try:
                t_ns = int(str(row["t_unix_ns"]).strip())
            except (TypeError, ValueError):
                try:
                    t_ns = int(float(row["t_unix_ns"]))
                except (TypeError, ValueError):
                    continue
            t_ns += self.time_offset_ns

            row["_t_ns"] = t_ns

            if dtype == self.dtype_imu:
                imu_rows.append(row)
            elif dtype == self.dtype_ahrs:
                if self.row_has_valid_orientation(row):
                    ahrs_rows.append(row)

        imu_rows.sort(key=lambda r: r["_t_ns"])
        ahrs_rows.sort(key=lambda r: r["_t_ns"])

        if self.covariance_mode == "estimate_static":
            self.estimate_covariances_from_static_window(imu_rows, ahrs_rows)
        elif self.covariance_mode != "fixed":
            raise RuntimeError(
                "covariance_mode must be 'fixed' or 'estimate_static', got: "
                f"{self.covariance_mode}"
            )

        imu_times = [r["_t_ns"] for r in imu_rows]

        self.rows = []
        matched = 0
        unmatched = 0

        for ahrs in ahrs_rows:
            merged = dict(ahrs)
            nearest_imu = self.find_nearest_row(imu_rows, imu_times, ahrs["_t_ns"])

            if nearest_imu is not None:
                dt_ns = abs(nearest_imu["_t_ns"] - ahrs["_t_ns"])
                if dt_ns <= self.max_merge_dt_ns:
                    for key in [
                        "gyro_x", "gyro_y", "gyro_z",
                        "acc_x", "acc_y", "acc_z",
                        "mag_x_mG", "mag_y_mG", "mag_z_mG",
                        "imu_temp_C", "pressure_Pa", "pressure_temp_C",
                    ]:
                        if key in nearest_imu:
                            merged[key] = nearest_imu[key]

                    merged["raw_imu_t_unix_ns"] = nearest_imu["_t_ns"]
                    merged["raw_imu_dt_ns"] = nearest_imu["_t_ns"] - ahrs["_t_ns"]
                    matched += 1
                    self.rows.append(merged)
                else:
                    unmatched += 1
                    if self.publish_unmatched_ahrs:
                        merged["raw_imu_t_unix_ns"] = ""
                        merged["raw_imu_dt_ns"] = ""
                        self.rows.append(merged)
            else:
                unmatched += 1
                if self.publish_unmatched_ahrs:
                    merged["raw_imu_t_unix_ns"] = ""
                    merged["raw_imu_dt_ns"] = ""
                    self.rows.append(merged)

        self.rows.sort(key=lambda r: r["_t_ns"])
        self.row_times = [r["_t_ns"] for r in self.rows]
        self.idx = 0
        self.received_first_clock = False

        self.pub = self.create_publisher(Imu, self.topic, 200)

        clock_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self.on_clock,
            clock_qos,
        )

        self.get_logger().info(f"Loaded raw rows: {len(raw_rows)}")
        self.get_logger().info(f"Dropped invalid CRC/end frames: {invalid_frames}")
        self.get_logger().info(f"dtype {self.dtype_imu} IMU rows: {len(imu_rows)}")
        self.get_logger().info(f"dtype {self.dtype_ahrs} AHRS rows: {len(ahrs_rows)}")
        self.get_logger().info(f"Merged IMU+AHRS rows: {len(self.rows)}")
        self.get_logger().info(f"Matched AHRS with raw IMU: {matched}")
        self.get_logger().info(f"Unmatched AHRS: {unmatched}")
        self.get_logger().info(f"Publishing {self.topic}, frame_id={self.frame_id}")
        self.get_logger().info(f"orientation_mode={self.orientation_mode}, angle_unit={self.angle_unit}")
        self.get_logger().info(f"gyro_unit={self.gyro_unit}, acc_unit={self.acc_unit}")
        self.get_logger().info(f"covariance_mode={self.covariance_mode}")
        self.get_logger().info(
            "orientation covariance [roll,pitch,yaw] rad^2: "
            + ", ".join(f"{v:.8g}" for v in self.orientation_cov)
        )
        self.get_logger().info(
            "gyro covariance [x,y,z] (rad/s)^2: "
            + ", ".join(f"{v:.8g}" for v in self.gyro_covariance)
        )
        self.get_logger().info(
            "gyro covariance scale="
            f"{self.gyro_covariance_scale:.8g}, static bias subtraction="
            f"{self.subtract_static_gyro_bias}, bias [x,y,z] (rad/s)="
            + ", ".join(f"{v:.8g}" for v in self.gyro_bias)
        )
        self.get_logger().info(
            "acceleration covariance [x,y,z] (m/s^2)^2: "
            + ", ".join(f"{v:.8g}" for v in self.acc_covariance)
        )
        self.get_logger().info("Waiting for /clock from rosbag play --clock ...")
        self.get_logger().info(f"yaw_rate_source={self.yaw_rate_source}, axis_conversion={self.axis_conversion}")
        if self.yaw_rate_source == "tilt_compensated_raw_gyro":
            self.get_logger().info(
                "Tilt compensation uses dtype 64 gyro_y/gyro_z and dtype "
                "65 AHRS roll/pitch; minimum |cos(pitch)|="
                f"{self.tilt_min_abs_cos_pitch:.6g}"
            )

    @staticmethod
    def sample_variance(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if len(finite) < 2:
            return float("nan")
        mean = sum(finite) / len(finite)
        return sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)

    @staticmethod
    def unwrap_angles(values):
        out = []
        previous = None
        offset = 0.0

        for raw in values:
            value = float(raw)
            if not math.isfinite(value):
                continue

            if previous is not None:
                delta = value - previous
                if delta > math.pi:
                    offset -= 2.0 * math.pi
                elif delta < -math.pi:
                    offset += 2.0 * math.pi

            out.append(value + offset)
            previous = value

        return out

    def estimate_covariances_from_static_window(self, imu_rows, ahrs_rows):
        if not imu_rows or not ahrs_rows:
            raise RuntimeError(
                "Cannot estimate covariance: dtype 64 or dtype 65 rows are missing."
            )

        first_ns = min(imu_rows[0]["_t_ns"], ahrs_rows[0]["_t_ns"])
        start_ns = first_ns + int(self.static_start_s * 1e9)
        end_ns = start_ns + int(self.static_duration_s * 1e9)

        imu_static = [
            row for row in imu_rows
            if start_ns <= row["_t_ns"] <= end_ns
        ]
        ahrs_static = [
            row for row in ahrs_rows
            if start_ns <= row["_t_ns"] <= end_ns
        ]

        if (
            len(imu_static) < self.min_static_samples
            or len(ahrs_static) < self.min_static_samples
        ):
            raise RuntimeError(
                "Not enough samples in covariance estimation window: "
                f"IMU={len(imu_static)}, AHRS={len(ahrs_static)}, "
                f"minimum={self.min_static_samples}."
            )

        gyro_columns = ["gyro_x", "gyro_y", "gyro_z"]
        acc_columns = ["acc_x", "acc_y", "acc_z"]
        orientation_columns = ["roll", "pitch", "heading"]

        gyro_values = {
            key: [self.convert_gyro(get_float(row, key, float("nan")))
                  for row in imu_static]
            for key in gyro_columns
        }
        acc_values = {
            key: [self.convert_acc(get_float(row, key, float("nan")))
                  for row in imu_static]
            for key in acc_columns
        }

        orientation_values = {}
        for key in orientation_columns:
            vals = [
                self.to_rad(get_float(row, key, float("nan")))
                for row in ahrs_static
            ]
            orientation_values[key] = self.unwrap_angles(vals)

        gyro_var = [
            self.sample_variance(gyro_values[key])
            for key in gyro_columns
        ]
        gyro_mean = [
            statistics.mean(gyro_values[key])
            if gyro_values[key] else 0.0
            for key in gyro_columns
        ]
        if self.subtract_static_gyro_bias:
            self.gyro_bias = gyro_mean

        # Use AHRS headingspeed variance when it supplies angular_velocity.z.
        if self.yaw_rate_source == "ahrs_headingspeed":
            heading_speeds = [
                self.convert_gyro(
                    get_float(row, "headingspeed", float("nan"))
                )
                for row in ahrs_static
            ]
            gyro_var[2] = self.sample_variance(heading_speeds)
        elif self.yaw_rate_source == "tilt_compensated_raw_gyro":
            imu_times = [row["_t_ns"] for row in imu_static]
            tilt_rates = []
            for ahrs_row in ahrs_static:
                imu_row = self.find_nearest_row(
                    imu_static,
                    imu_times,
                    ahrs_row["_t_ns"],
                )
                if imu_row is None:
                    continue
                merge_dt_ns = abs(
                    imu_row["_t_ns"] - ahrs_row["_t_ns"]
                )
                if merge_dt_ns > self.max_merge_dt_ns:
                    continue
                gyro_y = self.convert_gyro(
                    get_float(imu_row, "gyro_y", float("nan"))
                )
                gyro_z = self.convert_gyro(
                    get_float(imu_row, "gyro_z", float("nan"))
                )
                if self.subtract_static_gyro_bias:
                    gyro_y -= self.gyro_bias[1]
                    gyro_z -= self.gyro_bias[2]
                rate = self.compute_tilt_compensated_yaw_rate(
                    gyro_y,
                    gyro_z,
                    ahrs_row,
                )
                if rate is not None:
                    tilt_rates.append(rate)
            self.tilt_static_sample_count = len(tilt_rates)
            if len(tilt_rates) < self.min_static_samples:
                raise RuntimeError(
                    "Not enough valid paired samples to estimate the "
                    "tilt-compensated yaw-rate covariance: "
                    f"{len(tilt_rates)} < {self.min_static_samples}"
                )
            gyro_var[2] = self.sample_variance(tilt_rates)
        acc_var = [
            self.sample_variance(acc_values[key])
            for key in acc_columns
        ]
        orientation_var = [
            self.sample_variance(orientation_values[key])
            for key in orientation_columns
        ]

        gyro_std_max = max(
            math.sqrt(max(v, 0.0))
            for v in gyro_var
            if math.isfinite(v)
        )

        acc_norm = []
        for row in imu_static:
            ax = self.convert_acc(get_float(row, "acc_x", float("nan")))
            ay = self.convert_acc(get_float(row, "acc_y", float("nan")))
            az = self.convert_acc(get_float(row, "acc_z", float("nan")))
            if all(math.isfinite(v) for v in [ax, ay, az]):
                acc_norm.append(math.sqrt(ax * ax + ay * ay + az * az))
        acc_norm_std = math.sqrt(max(self.sample_variance(acc_norm), 0.0))

        stationary = (
            gyro_std_max <= self.static_gyro_std_max_rad_s
            and acc_norm_std <= self.static_acc_norm_std_max_mps2
        )

        if not stationary:
            message = (
                "Selected covariance window may not be stationary: "
                f"max gyro std={gyro_std_max:.6g} rad/s "
                f"(limit {self.static_gyro_std_max_rad_s:.6g}), "
                f"acc-norm std={acc_norm_std:.6g} m/s^2 "
                f"(limit {self.static_acc_norm_std_max_mps2:.6g})."
            )
            if self.require_static_window:
                raise RuntimeError(message)
            self.get_logger().warning(message + " Keeping fixed covariances.")
            return

        if not all(math.isfinite(v) for v in orientation_var + gyro_var + acc_var):
            raise RuntimeError("Non-finite variance encountered during estimation.")

        self.orientation_cov = [
            max(orientation_var[i], self.orientation_cov_floor[i])
            for i in range(3)
        ]
        self.gyro_covariance = [
            max(v * self.gyro_covariance_scale, self.gyro_cov_floor)
            for v in gyro_var
        ]
        self.acc_covariance = [
            max(v, self.acc_cov_floor) for v in acc_var
        ]

        self.get_logger().info(
            "Estimated covariance from stationary interval "
            f"[{self.static_start_s:.3f}, "
            f"{self.static_start_s + self.static_duration_s:.3f}] s"
        )
        self.get_logger().info(
            "Raw orientation variance [roll,pitch,yaw] rad^2: "
            + ", ".join(f"{v:.8g}" for v in orientation_var)
        )
        self.get_logger().info(
            "Used orientation covariance [roll,pitch,yaw] rad^2: "
            + ", ".join(f"{v:.8g}" for v in self.orientation_cov)
        )
        self.get_logger().info(
            "Used gyro covariance [x,y,z] (rad/s)^2: "
            + ", ".join(f"{v:.8g}" for v in self.gyro_covariance)
        )
        self.get_logger().info(
            "Used acceleration covariance [x,y,z] (m/s^2)^2: "
            + ", ".join(f"{v:.8g}" for v in self.acc_covariance)
        )

        if self.covariance_report_path:
            report_path = Path(self.covariance_report_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "mode",
                "static_start_s",
                "static_duration_s",
                "imu_samples",
                "ahrs_samples",
                "gyro_std_max_rad_s",
                "acc_norm_std_mps2",
                "orientation_cov_roll_rad2",
                "orientation_cov_pitch_rad2",
                "orientation_cov_yaw_rad2",
                "gyro_cov_x_rad2_s2",
                "gyro_cov_y_rad2_s2",
                "gyro_cov_z_rad2_s2",
                "gyro_covariance_scale",
                "subtract_static_gyro_bias",
                "yaw_rate_source",
                "tilt_static_samples",
                "tilt_min_abs_cos_pitch",
                "gyro_bias_x_rad_s",
                "gyro_bias_y_rad_s",
                "gyro_bias_z_rad_s",
                "acc_cov_x_m2_s4",
                "acc_cov_y_m2_s4",
                "acc_cov_z_m2_s4",
            ]
            row = {
                "mode": self.covariance_mode,
                "static_start_s": self.static_start_s,
                "static_duration_s": self.static_duration_s,
                "imu_samples": len(imu_static),
                "ahrs_samples": len(ahrs_static),
                "gyro_std_max_rad_s": gyro_std_max,
                "acc_norm_std_mps2": acc_norm_std,
                "orientation_cov_roll_rad2": self.orientation_cov[0],
                "orientation_cov_pitch_rad2": self.orientation_cov[1],
                "orientation_cov_yaw_rad2": self.orientation_cov[2],
                "gyro_cov_x_rad2_s2": self.gyro_covariance[0],
                "gyro_cov_y_rad2_s2": self.gyro_covariance[1],
                "gyro_cov_z_rad2_s2": self.gyro_covariance[2],
                "gyro_covariance_scale": self.gyro_covariance_scale,
                "subtract_static_gyro_bias": self.subtract_static_gyro_bias,
                "yaw_rate_source": self.yaw_rate_source,
                "tilt_static_samples": self.tilt_static_sample_count,
                "tilt_min_abs_cos_pitch": self.tilt_min_abs_cos_pitch,
                "gyro_bias_x_rad_s": self.gyro_bias[0],
                "gyro_bias_y_rad_s": self.gyro_bias[1],
                "gyro_bias_z_rad_s": self.gyro_bias[2],
                "acc_cov_x_m2_s4": self.acc_covariance[0],
                "acc_cov_y_m2_s4": self.acc_covariance[1],
                "acc_cov_z_m2_s4": self.acc_covariance[2],
            }
            with open(report_path, "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            self.get_logger().info(
                f"Wrote covariance report: {report_path}"
            )

    @staticmethod
    def find_nearest_row(rows, times, t_ns):
        if not rows:
            return None

        i = bisect_left(times, t_ns)

        candidates = []
        if i < len(rows):
            candidates.append(rows[i])
        if i > 0:
            candidates.append(rows[i - 1])

        if not candidates:
            return None

        return min(candidates, key=lambda r: abs(r["_t_ns"] - t_ns))

    def to_rad(self, value):
        if self.angle_unit == "deg":
            return math.radians(value)
        return value

    def row_has_valid_orientation(self, row):
        if self.orientation_mode == "csv_quat":
            qw = get_float(row, "qw", float("nan"))
            qx = get_float(row, "qx", float("nan"))
            qy = get_float(row, "qy", float("nan"))
            qz = get_float(row, "qz", float("nan"))
            return valid_quaternion(qw, qx, qy, qz)

        heading = row.get("heading", "")
        if heading in ["", None]:
            return False

        try:
            h = float(heading)
        except Exception:
            return False

        return math.isfinite(h)

    def compute_orientation(self, row):
        if self.orientation_mode == "csv_quat":
            qw = get_float(row, "qw", 0.0)
            qx = get_float(row, "qx", 0.0)
            qy = get_float(row, "qy", 0.0)
            qz = get_float(row, "qz", 0.0)

            if not valid_quaternion(qw, qx, qy, qz):
                return None

            return qw, qx, qy, qz

        heading = self.to_rad(get_float(row, "heading", 0.0))

        if self.orientation_mode == "heading_north_to_enu":
            yaw = math.pi / 2.0 - heading + self.mount_yaw_offset_rad

        elif self.orientation_mode == "heading_as_enu":
            yaw = heading + self.mount_yaw_offset_rad

        else:
            raise RuntimeError(f"Unknown orientation_mode: {self.orientation_mode}")

        yaw = normalize_angle(yaw)
        return yaw_to_quaternion(yaw)

    def convert_gyro(self, value):
        if self.gyro_unit == "deg_s":
            return math.radians(value)
        return value

    def convert_acc(self, value):
        if self.acc_unit == "g":
            return value * 9.80665
        return value

    def compute_tilt_compensated_yaw_rate(self, gyro_y, gyro_z, row):
        """Return ENU yaw rate from native FRD gyro and AHRS tilt."""
        roll = self.to_rad(get_float(row, "roll", float("nan")))
        pitch = self.to_rad(get_float(row, "pitch", float("nan")))
        yaw_sign = (
            -1.0 if self.orientation_mode == "heading_north_to_enu" else 1.0
        )
        return tilt_compensated_yaw_rate_native_frd(
            gyro_y,
            gyro_z,
            roll,
            pitch,
            yaw_sign=yaw_sign,
            minimum_abs_cos_pitch=self.tilt_min_abs_cos_pitch,
        )

    def make_imu_msg(self, row):
        q = self.compute_orientation(row)
        if q is None:
            return None

        qw, qx, qy, qz = q

        msg = Imu()
        msg.header.stamp = ns_to_time_msg(row["_t_ns"])
        msg.header.frame_id = self.frame_id

        msg.orientation.w = qw
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz

        msg.orientation_covariance[0] = self.orientation_cov[0]
        msg.orientation_covariance[4] = self.orientation_cov[1]
        msg.orientation_covariance[8] = self.orientation_cov[2]

        native_gx = self.convert_gyro(get_float(row, "gyro_x", 0.0))
        native_gy = self.convert_gyro(get_float(row, "gyro_y", 0.0))
        native_gz = self.convert_gyro(get_float(row, "gyro_z", 0.0))

        if self.subtract_static_gyro_bias:
            native_gx -= self.gyro_bias[0]
            native_gy -= self.gyro_bias[1]
            native_gz -= self.gyro_bias[2]

        gx = native_gx
        gy = native_gy
        gz = native_gz

        ax = self.convert_acc(get_float(row, "acc_x", 0.0))
        ay = self.convert_acc(get_float(row, "acc_y", 0.0))
        az = self.convert_acc(get_float(row, "acc_z", 0.0))

        # Convert FRD sensor axes to ROS FLU by flipping y and z.
        if self.axis_conversion == "frd_to_flu":
            gy = -gy
            gz = -gz
            ay = -ay
            az = -az
        elif self.axis_conversion != "none":
            self.get_logger().warn(
                f"Unknown axis_conversion={self.axis_conversion}; using raw axes."
            )
        wz = gz

        # Optionally use AHRS headingspeed as the yaw rate.
        if self.yaw_rate_source == "ahrs_headingspeed":
            headingspeed = self.convert_gyro(
                get_float(row, "headingspeed", float("nan"))
            )

            if math.isfinite(headingspeed):
                # For yaw = pi/2 - heading + offset, yaw_rate = -headingspeed.
                wz = -headingspeed
            else:
                self.get_logger().warn(
                    "yaw_rate_source=ahrs_headingspeed but headingspeed is NaN; "
                    "falling back to converted raw gyro_z."
                )

        elif self.yaw_rate_source == "tilt_compensated_raw_gyro":
            compensated_rate = self.compute_tilt_compensated_yaw_rate(
                native_gy,
                native_gz,
                row,
            )
            if compensated_rate is None:
                self.tilt_rejected_frame_count += 1
                return None
            wz = compensated_rate
            self.tilt_published_frame_count += 1

        elif self.yaw_rate_source == "raw_gyro_z":
            wz = gz

        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = wz

        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az

        msg.angular_velocity_covariance[0] = self.gyro_covariance[0]
        msg.angular_velocity_covariance[4] = self.gyro_covariance[1]
        msg.angular_velocity_covariance[8] = self.gyro_covariance[2]

        msg.linear_acceleration_covariance[0] = self.acc_covariance[0]
        msg.linear_acceleration_covariance[4] = self.acc_covariance[1]
        msg.linear_acceleration_covariance[8] = self.acc_covariance[2]

        return msg

    def on_clock(self, clock_msg):
        now_ns = clock_msg.clock.sec * 1_000_000_000 + clock_msg.clock.nanosec

        if not self.received_first_clock:
            self.received_first_clock = True
            if self.skip_rows_before_first_clock:
                old_idx = self.idx
                self.idx = bisect_left(self.row_times, now_ns)
                skipped = self.idx - old_idx
                if skipped:
                    self.get_logger().info(
                        f"Skipped {skipped} IMU rows older than first /clock "
                        f"({now_ns})"
                    )

        while self.idx < len(self.rows) and self.rows[self.idx]["_t_ns"] <= now_ns:
            msg = self.make_imu_msg(self.rows[self.idx])
            if msg is not None:
                self.pub.publish(msg)
            self.idx += 1


def main():
    rclpy.init()
    node = ImuFilePlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
