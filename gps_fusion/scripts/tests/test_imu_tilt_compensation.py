#!/usr/bin/env python3
"""Unit tests for the native-FRD tilt compensation formula."""

import math
import sys
import unittest
from pathlib import Path


PACKAGE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws" / "src" / "bikelab_process"
)
sys.path.insert(0, str(PACKAGE_SOURCE))

from bikelab_process.imu_file_player import (  # noqa: E402
    tilt_compensated_yaw_rate_native_frd,
)


class TiltCompensationTests(unittest.TestCase):
    """Check signs, Euler-rate geometry, and invalid-input handling."""

    def test_level_frd_to_enu_sign(self):
        rate = tilt_compensated_yaw_rate_native_frd(
            gyro_y=0.0,
            gyro_z=-1.25,
            roll=0.0,
            pitch=0.0,
        )
        self.assertAlmostEqual(rate, 1.25)

    def test_zyx_euler_rate(self):
        roll = math.radians(30.0)
        pitch = math.radians(-10.0)
        gyro_y = 0.4
        gyro_z = -0.8
        expected = -(
            gyro_y * math.sin(roll) + gyro_z * math.cos(roll)
        ) / math.cos(pitch)
        rate = tilt_compensated_yaw_rate_native_frd(
            gyro_y,
            gyro_z,
            roll,
            pitch,
        )
        self.assertAlmostEqual(rate, expected)

    def test_heading_as_enu_sign(self):
        rate = tilt_compensated_yaw_rate_native_frd(
            gyro_y=0.0,
            gyro_z=0.75,
            roll=0.0,
            pitch=0.0,
            yaw_sign=1.0,
        )
        self.assertAlmostEqual(rate, 0.75)

    def test_rejects_nonfinite_tilt(self):
        rate = tilt_compensated_yaw_rate_native_frd(
            gyro_y=0.0,
            gyro_z=1.0,
            roll=float("nan"),
            pitch=0.0,
        )
        self.assertIsNone(rate)

    def test_rejects_near_gimbal_lock(self):
        rate = tilt_compensated_yaw_rate_native_frd(
            gyro_y=0.0,
            gyro_z=1.0,
            roll=0.0,
            pitch=math.radians(89.0),
            minimum_abs_cos_pitch=0.1,
        )
        self.assertIsNone(rate)


if __name__ == "__main__":
    unittest.main()
