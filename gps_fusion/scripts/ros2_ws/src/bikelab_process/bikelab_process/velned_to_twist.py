#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistWithCovarianceStamped
from ublox_ubx_msgs.msg import UBXNavVelNED


def get_field(msg, *names, default=None):
    for name in names:
        if hasattr(msg, name):
            return getattr(msg, name)
    if default is not None:
        return default
    raise AttributeError(f"None of these fields exist in message: {names}")


class VelNedToTwist(Node):
    def __init__(self):
        super().__init__("velned_to_twist")

        self.declare_parameter("input_topic", "/ubx_nav_vel_ned")
        self.declare_parameter("output_topic", "/gnss/vel_twist")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("output_mode", "body_forward")
        self.declare_parameter("minimum_variance", 0.05)
        self.declare_parameter("lateral_variance", 0.05)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.output_mode = str(
            self.get_parameter("output_mode").value
        ).strip().lower()
        self.minimum_variance = float(
            self.get_parameter("minimum_variance").value
        )
        self.lateral_variance = float(
            self.get_parameter("lateral_variance").value
        )

        if self.output_mode not in ("body_forward", "enu_components"):
            raise RuntimeError(
                "output_mode must be 'body_forward' or 'enu_components'"
            )
        if self.output_mode == "enu_components" and self.frame_id == "base_link":
            self.get_logger().warning(
                "ENU velocity components must not be labelled base_link; "
                "use a navigation/world frame with a valid TF."
            )

        self.pub = self.create_publisher(
            TwistWithCovarianceStamped,
            self.output_topic,
            50,
        )

        self.sub = self.create_subscription(
            UBXNavVelNED,
            self.input_topic,
            self.callback,
            100,
        )

        self.get_logger().info(f"Subscribing {self.input_topic}")
        self.get_logger().info(f"Publishing {self.output_topic}")
        self.get_logger().info(
            f"output_mode={self.output_mode}, frame_id={self.frame_id}"
        )

    def callback(self, msg):
        out = TwistWithCovarianceStamped()

        if hasattr(msg, "header"):
            out.header = msg.header
        else:
            out.header.stamp = self.get_clock().now().to_msg()

        out.header.frame_id = self.frame_id

        # u-blox NAV-VELNED is usually cm/s.
        # NED:
        #   vel_n = North
        #   vel_e = East
        #   vel_d = Down
        #
        # ROS ENU approximation:
        #   x = East
        #   y = North
        #   z = Up
        vel_n = float(get_field(msg, "vel_n", "velN")) * 0.01
        vel_e = float(get_field(msg, "vel_e", "velE")) * 0.01
        vel_d = float(get_field(msg, "vel_d", "velD")) * 0.01

        if self.output_mode == "body_forward":
            # A bicycle is assumed to move forward with negligible lateral
            # slip.  NAV-VELNED components are earth-fixed and cannot be
            # labelled base_link directly; ground speed is frame invariant.
            ground_speed = float(
                get_field(
                    msg,
                    "g_speed",
                    "gSpeed",
                    default=(vel_n * vel_n + vel_e * vel_e) ** 0.5 * 100.0,
                )
            ) * 0.01
            out.twist.twist.linear.x = ground_speed
            out.twist.twist.linear.y = 0.0
            out.twist.twist.linear.z = 0.0
        else:
            out.twist.twist.linear.x = vel_e
            out.twist.twist.linear.y = vel_n
            out.twist.twist.linear.z = -vel_d

        # speed accuracy is usually cm/s
        s_acc = float(get_field(msg, "s_acc", "sAcc", default=50.0)) * 0.01
        var = max(s_acc * s_acc, self.minimum_variance)

        out.twist.covariance[0] = var
        out.twist.covariance[7] = (
            self.lateral_variance
            if self.output_mode == "body_forward"
            else var
        )
        out.twist.covariance[14] = var

        self.pub.publish(out)


def main():
    rclpy.init()
    node = VelNedToTwist()
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
