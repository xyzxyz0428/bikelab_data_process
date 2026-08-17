#!/usr/bin/env python3
"""Start every comparison EKF from the same first common GPS odometry."""

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped


def stamp_nanoseconds(stamp):
    """Convert a ROS stamp to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ComparisonSensorGate(Node):
    """Discard yaw sensors until common GPS initializes all comparison EKFs."""

    def __init__(self):
        """Create gated publishers and subscribe to the common inputs."""
        super().__init__('comparison_sensor_gate')

        self.declare_parameter('gps_input_topic', '/odometry/gps_common')
        self.declare_parameter('gps_output_topic', '/compare_input/gps')
        self.declare_parameter('course_input_topic', '/gnss/course_imu')
        self.declare_parameter('course_output_topic', '/compare_input/course')
        self.declare_parameter('raw_rate_input_topic', '/imu/raw_gyro_rate')
        self.declare_parameter(
            'raw_rate_output_topic',
            '/compare_input/raw_gyro_rate',
        )
        self.declare_parameter(
            'ahrs_rate_input_topic',
            '/imu/ahrs_heading_rate',
        )
        self.declare_parameter(
            'ahrs_rate_output_topic',
            '/compare_input/ahrs_heading_rate',
        )
        self.declare_parameter('velocity_input_topic', '/gnss/vel_twist')
        self.declare_parameter(
            'velocity_output_topic', '/compare_input/velocity',
        )

        gps_input = str(self.get_parameter('gps_input_topic').value)
        gps_output = str(self.get_parameter('gps_output_topic').value)
        course_input = str(self.get_parameter('course_input_topic').value)
        course_output = str(self.get_parameter('course_output_topic').value)
        raw_input = str(self.get_parameter('raw_rate_input_topic').value)
        raw_output = str(self.get_parameter('raw_rate_output_topic').value)
        ahrs_input = str(self.get_parameter('ahrs_rate_input_topic').value)
        ahrs_output = str(self.get_parameter('ahrs_rate_output_topic').value)
        velocity_input = str(self.get_parameter('velocity_input_topic').value)
        velocity_output = str(
            self.get_parameter('velocity_output_topic').value
        )

        self.gps_publisher = self.create_publisher(Odometry, gps_output, 100)
        self.course_publisher = self.create_publisher(Imu, course_output, 100)
        self.raw_rate_publisher = self.create_publisher(
            Imu,
            raw_output,
            100,
        )
        self.ahrs_rate_publisher = self.create_publisher(
            Imu,
            ahrs_output,
            100,
        )
        self.velocity_publisher = self.create_publisher(
            TwistWithCovarianceStamped,
            velocity_output,
            100,
        )

        self.gps_subscription = self.create_subscription(
            Odometry,
            gps_input,
            self.gps_callback,
            100,
        )
        self.course_subscription = self.create_subscription(
            Imu,
            course_input,
            self.course_callback,
            100,
        )
        self.raw_rate_subscription = self.create_subscription(
            Imu,
            raw_input,
            self.raw_rate_callback,
            100,
        )
        self.ahrs_rate_subscription = self.create_subscription(
            Imu,
            ahrs_input,
            self.ahrs_rate_callback,
            100,
        )
        self.velocity_subscription = self.create_subscription(
            TwistWithCovarianceStamped,
            velocity_input,
            self.velocity_callback,
            100,
        )

        self.unlock_stamp_ns = None
        self.get_logger().info(
            'Comparison inputs are locked until the first common GPS '
            f'odometry on {gps_input}'
        )

    def gps_callback(self, message):
        """Publish common GPS first, then allow no-older yaw measurements."""
        stamp_ns = stamp_nanoseconds(message.header.stamp)
        self.gps_publisher.publish(message)
        if self.unlock_stamp_ns is None:
            self.unlock_stamp_ns = stamp_ns
            self.get_logger().info(
                'Comparison input gate unlocked at '
                f'{message.header.stamp.sec}.'
                f'{message.header.stamp.nanosec:09d}'
            )

    def publish_if_current(self, message, publisher):
        """Forward a yaw measurement only after common GPS initialization."""
        if self.unlock_stamp_ns is None:
            return
        if stamp_nanoseconds(message.header.stamp) < self.unlock_stamp_ns:
            return
        publisher.publish(message)

    def course_callback(self, message):
        """Gate GNSS course."""
        self.publish_if_current(message, self.course_publisher)

    def raw_rate_callback(self, message):
        """Gate raw gyro yaw-rate."""
        self.publish_if_current(message, self.raw_rate_publisher)

    def ahrs_rate_callback(self, message):
        """Gate AHRS headingspeed."""
        self.publish_if_current(message, self.ahrs_rate_publisher)

    def velocity_callback(self, message):
        """Gate GNSS horizontal speed."""
        self.publish_if_current(message, self.velocity_publisher)


def main(args=None):
    """Run the comparison sensor gate."""
    rclpy.init(args=args)
    node = ComparisonSensorGate()
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
