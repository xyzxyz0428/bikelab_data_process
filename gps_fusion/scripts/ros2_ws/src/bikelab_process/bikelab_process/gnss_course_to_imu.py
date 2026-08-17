#!/usr/bin/env python3
"""Publish a yaw-only IMU orientation from u-blox GNSS course over ground."""

import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu

from ublox_ubx_msgs.msg import UBXNavVelNED


def heading_deg_to_enu_yaw(heading_deg):
    """Convert clockwise-from-North degrees to wrapped ROS ENU yaw."""
    yaw = math.pi / 2.0 - math.radians(float(heading_deg))
    return math.atan2(math.sin(yaw), math.cos(yaw))


class GnssCourseToImu(Node):
    """Convert valid NAV-VELNED course measurements to ROS ENU yaw."""

    HEADING_SCALE_DEG = 1.0e-5
    SPEED_SCALE_MPS = 0.01

    def __init__(self):
        """Initialize parameters and ROS interfaces."""
        super().__init__('gnss_course_to_imu')

        self.declare_parameter('input_topic', '/ubx_nav_vel_ned')
        self.declare_parameter('output_topic', '/gnss/course_imu')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('minimum_speed_mps', 2.0)
        self.declare_parameter('maximum_course_accuracy_deg', 30.0)
        self.declare_parameter('minimum_yaw_variance_rad2', 0.03)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.minimum_speed_mps = float(
            self.get_parameter('minimum_speed_mps').value
        )
        self.maximum_course_accuracy_deg = float(
            self.get_parameter('maximum_course_accuracy_deg').value
        )
        self.minimum_yaw_variance_rad2 = float(
            self.get_parameter('minimum_yaw_variance_rad2').value
        )

        if not math.isfinite(self.minimum_speed_mps):
            raise RuntimeError('minimum_speed_mps must be finite')
        if self.minimum_speed_mps < 0.0:
            raise RuntimeError('minimum_speed_mps must be non-negative')
        if (
            not math.isfinite(self.maximum_course_accuracy_deg)
            or self.maximum_course_accuracy_deg < 0.0
        ):
            raise RuntimeError(
                'maximum_course_accuracy_deg must be finite and non-negative'
            )
        if (
            not math.isfinite(self.minimum_yaw_variance_rad2)
            or self.minimum_yaw_variance_rad2 < 0.0
        ):
            raise RuntimeError(
                'minimum_yaw_variance_rad2 must be finite and non-negative'
            )

        self.publisher = self.create_publisher(Imu, self.output_topic, 50)
        self.subscription = self.create_subscription(
            UBXNavVelNED,
            self.input_topic,
            self.callback,
            100,
        )

        self.get_logger().info(
            f'Subscribing {self.input_topic}; publishing {self.output_topic} '
            f'in {self.frame_id}'
        )
        self.get_logger().info(
            'GNSS course gate: '
            f'speed >= {self.minimum_speed_mps:.3f} m/s, '
            'course accuracy <= '
            f'{self.maximum_course_accuracy_deg:.3f} deg'
        )

    def callback(self, msg):
        """Publish only course samples that pass speed and accuracy gates."""
        ground_speed_mps = float(msg.g_speed) * self.SPEED_SCALE_MPS
        course_accuracy_deg = (
            float(msg.c_acc) * self.HEADING_SCALE_DEG
        )

        if (
            not math.isfinite(ground_speed_mps)
            or ground_speed_mps < self.minimum_speed_mps
        ):
            return
        if (
            not math.isfinite(course_accuracy_deg)
            or course_accuracy_deg > self.maximum_course_accuracy_deg
        ):
            return

        # UBX heading is degrees clockwise from North. ROS ENU yaw is radians
        # counter-clockwise from East, hence yaw = pi/2 - heading.
        yaw = heading_deg_to_enu_yaw(
            float(msg.heading) * self.HEADING_SCALE_DEG
        )

        out = Imu()
        # Retain the acquisition timestamp from NAV-VELNED while expressing
        # the yaw measurement in the configured ROS body frame.
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id

        half_yaw = 0.5 * yaw
        out.orientation.x = 0.0
        out.orientation.y = 0.0
        out.orientation.z = math.sin(half_yaw)
        out.orientation.w = math.cos(half_yaw)

        course_accuracy_rad = math.radians(course_accuracy_deg)
        yaw_variance = max(
            course_accuracy_rad * course_accuracy_rad,
            self.minimum_yaw_variance_rad2,
        )
        # Roll and pitch are intentionally unusable; only yaw is measured.
        out.orientation_covariance[0] = 999.0
        out.orientation_covariance[4] = 999.0
        out.orientation_covariance[8] = yaw_variance

        # Per sensor_msgs/Imu convention, -1 in the first covariance element
        # marks angular velocity and linear acceleration as unavailable.
        out.angular_velocity_covariance[0] = -1.0
        out.linear_acceleration_covariance[0] = -1.0

        self.publisher.publish(out)


def main(args=None):
    """Run the GNSS course converter node."""
    rclpy.init(args=args)
    node = GnssCourseToImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
