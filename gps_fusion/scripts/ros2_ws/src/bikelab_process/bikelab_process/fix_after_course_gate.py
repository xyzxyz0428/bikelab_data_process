#!/usr/bin/env python3
"""Forward NavSatFix messages after the first valid GNSS course settles."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix


NANOSECONDS_PER_SECOND = 1_000_000_000


def stamp_to_nanoseconds(stamp):
    """Convert a builtin_interfaces/Time message to integer nanoseconds."""
    return (
        int(stamp.sec) * NANOSECONDS_PER_SECOND
        + int(stamp.nanosec)
    )


class FixAfterCourseGate(Node):
    """Gate fixes until a configurable delay after the first course sample."""

    def __init__(self):
        """Initialize parameters, state, and ROS interfaces."""
        super().__init__('fix_after_course_gate')

        self.declare_parameter('input_fix_topic', '/fix')
        self.declare_parameter('course_topic', '/gnss/course_imu')
        self.declare_parameter('output_fix_topic', '/fix/fusion')
        self.declare_parameter('settle_time_s', 0.5)

        self.input_fix_topic = str(
            self.get_parameter('input_fix_topic').value
        )
        self.course_topic = str(self.get_parameter('course_topic').value)
        self.output_fix_topic = str(
            self.get_parameter('output_fix_topic').value
        )
        self.settle_time_s = float(
            self.get_parameter('settle_time_s').value
        )

        if not math.isfinite(self.settle_time_s):
            raise RuntimeError('settle_time_s must be finite')
        if self.settle_time_s < 0.0:
            raise RuntimeError('settle_time_s must be non-negative')

        self.settle_time_ns = int(
            round(self.settle_time_s * NANOSECONDS_PER_SECOND)
        )
        self.unlock_after_ns = None
        self.unlocked = False

        self.publisher = self.create_publisher(
            NavSatFix,
            self.output_fix_topic,
            50,
        )
        self.fix_subscription = self.create_subscription(
            NavSatFix,
            self.input_fix_topic,
            self.fix_callback,
            qos_profile_sensor_data,
        )
        self.course_subscription = self.create_subscription(
            Imu,
            self.course_topic,
            self.course_callback,
            100,
        )

        self.get_logger().info(
            f'Gating {self.input_fix_topic} -> {self.output_fix_topic}; '
            f'waiting for {self.course_topic} plus '
            f'{self.settle_time_s:.3f} s'
        )

    def course_callback(self, msg):
        """Use only the first course timestamp to establish the gate."""
        if self.unlock_after_ns is not None:
            return

        course_stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        self.unlock_after_ns = course_stamp_ns + self.settle_time_ns

    def fix_callback(self, msg):
        """Open at the timestamp threshold, then forward later fixes."""
        if not self.unlocked:
            if self.unlock_after_ns is None:
                return

            fix_stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
            if fix_stamp_ns < self.unlock_after_ns:
                return

            self.unlocked = True
            self.get_logger().info(
                'GNSS fix gate opened at '
                f'{msg.header.stamp.sec}.'
                f'{msg.header.stamp.nanosec:09d}'
            )

        # Publish the original message without changing its header, frame, or
        # payload.
        self.publisher.publish(msg)


def main(args=None):
    """Run the NavSatFix course gate node."""
    rclpy.init(args=args)
    node = FixAfterCourseGate()
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
