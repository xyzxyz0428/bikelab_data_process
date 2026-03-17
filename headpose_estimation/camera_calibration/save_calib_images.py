#!/usr/bin/env python3
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image


class CalibImageSaver(Node):
    def __init__(self):
        super().__init__("calib_image_saver")

        self.declare_parameter("topic", "/camera/image_raw")
        self.declare_parameter("output_dir", "./calib_images")
        self.declare_parameter("sec_per_frame", 1.0)
        self.declare_parameter("max_images", 30)

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.output_dir = Path(
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self.sec_per_frame = (
            self.get_parameter("sec_per_frame").get_parameter_value().double_value
        )
        self.max_images = (
            self.get_parameter("max_images").get_parameter_value().integer_value
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 明确指定 BEST_EFFORT
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos,
        )

        self.last_save_time = None
        self.saved_count = 0

        self.get_logger().info(f"Subscribing to {self.topic}")
        self.get_logger().info(f"Saving to {self.output_dir.resolve()}")
        self.get_logger().info(
            f"sec_per_frame={self.sec_per_frame}, max_images={self.max_images}"
        )

    def image_callback(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.last_save_time is not None:
            if now - self.last_save_time < self.sec_per_frame:
                return

        if self.saved_count >= self.max_images:
            self.get_logger().info("Reached max_images, shutting down.")
            rclpy.shutdown()
            return

        if msg.encoding.lower() != "bgr8":
            self.get_logger().warn(
                f"Unsupported encoding: {msg.encoding}. Expected bgr8."
            )
            return

        try:
            # 处理可能存在的 step padding
            data = np.frombuffer(msg.data, dtype=np.uint8)
            row_stride = msg.step
            expected = msg.width * 3

            img_2d = data.reshape((msg.height, row_stride))
            img = img_2d[:, :expected].reshape((msg.height, msg.width, 3))

            filename = self.output_dir / f"calib_{self.saved_count:04d}.png"
            ok = cv2.imwrite(str(filename), img)

            if ok:
                self.saved_count += 1
                self.last_save_time = now
                self.get_logger().info(
                    f"Saved {filename.name} "
                    f"(stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d})"
                )
            else:
                self.get_logger().warn(f"Failed to save {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to convert/save image: {e}")


def main():
    rclpy.init()
    node = CalibImageSaver()
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