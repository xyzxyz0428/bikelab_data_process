from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("bikelab_process"),
        "config",
        "bike_gps_imu_fusion.yaml",
    ])

    return LaunchDescription([
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="ekf_filter_node",
            output="screen",
            parameters=[config],
        ),

        Node(
            package="robot_localization",
            executable="navsat_transform_node",
            name="navsat_transform",
            output="screen",
            parameters=[config],
            arguments=["--ros-args", "--log-level", "navsat_transform:=debug"],
            remappings=[
                ("imu", "/imu/data"),
                ("gps/fix", "/fix"),
                ("odometry/filtered", "/odometry/filtered"),
                ("odometry/gps", "/odometry/gps"),
                ("gps/filtered", "/gps/filtered"),
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_gps_tf",
            arguments=[
                "--x", "0.135",
                "--y", "0.05",
                "--z", "1.46",
                "--yaw", "0",
                "--pitch", "0",
                "--roll", "0",
                "--frame-id", "base_link",
                "--child-frame-id", "ubx",
            ],
        ),

        # Node(
        #     package="tf2_ros",
        #     executable="static_transform_publisher",
        #     name="base_to_imu_tf",
        #     arguments=[
        #         "--x", "0.05",
        #         "--y", "0",
        #         "--z", "0.46",
        #         "--yaw", "0",
        #         "--pitch", "0",
        #         "--roll", "3.141592653589793",
        #         "--frame-id", "base_link",
        #         "--child-frame-id", "imu_link",
        #     ],
        # ),
    ])
