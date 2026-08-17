"""Replay the recorded GNSS bag and raw IMU CSV, then record fused topics."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def validate_launch_paths(context):
    """Reject missing inputs and an existing output before starting nodes."""
    bag = Path(LaunchConfiguration("bag").perform(context)).expanduser()
    imu_file = Path(
        LaunchConfiguration("imu_file").perform(context)
    ).expanduser()
    output_bag = Path(
        LaunchConfiguration("output_bag").perform(context)
    ).expanduser()
    if not bag.exists():
        raise RuntimeError(f"GNSS bag does not exist: {bag}")
    if not imu_file.is_file():
        raise RuntimeError(f"IMU file does not exist: {imu_file}")
    if output_bag.exists():
        raise RuntimeError(
            f"Output bag already exists; choose a new path: {output_bag}"
        )
    return []


def generate_launch_description():
    bag = LaunchConfiguration("bag")
    imu_file = LaunchConfiguration("imu_file")
    output_bag = LaunchConfiguration("output_bag")
    config_file = LaunchConfiguration("config_file")
    mount_yaw_offset = LaunchConfiguration("mount_yaw_offset_rad")
    imu_yaw_variance = LaunchConfiguration("imu_yaw_variance_rad2")
    yaw_rate_source = LaunchConfiguration("yaw_rate_source")
    fusion_fix_topic = LaunchConfiguration("fusion_fix_topic")
    playback_rate = LaunchConfiguration("playback_rate")
    start_offset = LaunchConfiguration("start_offset")
    ros_domain_id = LaunchConfiguration("ros_domain_id")

    default_config = PathJoinSubstitution([
        FindPackageShare("bikelab_process"),
        "config",
        "offline_gps_imu_fusion.yaml",
    ])

    imu_player = Node(
        package="bikelab_process",
        executable="imu_file_player",
        name="imu_file_player",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "file_path": imu_file,
            "topic": "/imu/data_clean",
            "frame_id": "base_link",
            "angle_unit": "rad",
            "orientation_mode": "heading_north_to_enu",
            # The default heading offset is specific to this test segment.
            "mount_yaw_offset_rad": mount_yaw_offset,
            "axis_conversion": "frd_to_flu",
            "yaw_rate_source": yaw_rate_source,
            "skip_rows_before_first_clock": True,
            "covariance_mode": "estimate_static",
            "static_start_s": 0.0,
            "static_duration_s": 10.0,
            "min_static_samples": 100,
            "require_static_window": True,
            "orientation_cov_floor_roll": 999.0,
            "orientation_cov_floor_pitch": 999.0,
            "orientation_cov_floor_yaw": imu_yaw_variance,
            "gyro_cov_floor": 1.0e-8,
            "acc_cov_floor": 1.0e-6,
        }],
    )

    velocity_converter = Node(
        package="bikelab_process",
        executable="velned_to_twist",
        name="velned_to_twist",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "input_topic": "/ubx_nav_vel_ned",
            "output_topic": "/gnss/vel_twist",
            "output_mode": "body_forward",
            "frame_id": "base_link",
        }],
    )

    course_converter = Node(
        package="bikelab_process",
        executable="gnss_course_to_imu",
        name="gnss_course_to_imu",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "input_topic": "/ubx_nav_vel_ned",
            "output_topic": "/gnss/course_imu",
            "frame_id": "base_link",
            "minimum_speed_mps": 2.0,
            "maximum_course_accuracy_deg": 30.0,
            "minimum_yaw_variance_rad2": 0.03,
        }],
    )

    fix_gate = Node(
        package="bikelab_process",
        executable="fix_after_course_gate",
        name="fix_after_course_gate",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "input_fix_topic": "/fix",
            "course_topic": "/gnss/course_imu",
            "output_fix_topic": "/fix/fusion",
            "settle_time_s": 0.5,
        }],
    )

    ekf_local = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_local_node",
        output="screen",
        parameters=[config_file],
        remappings=[("odometry/filtered", "/odometry/local")],
    )

    navsat = Node(
        package="robot_localization",
        executable="navsat_transform_node",
        name="navsat_transform",
        output="screen",
        parameters=[config_file],
        remappings=[
            ("imu", "/imu/data_clean"),
            ("gps/fix", fusion_fix_topic),
            # Use the global EKF output for the map-frame transform.
            ("odometry/filtered", "/odometry/filtered_global"),
            ("odometry/gps", "/odometry/gps"),
            ("gps/filtered", "/gps/filtered"),
        ],
    )

    ekf_global = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_global_node",
        output="screen",
        parameters=[config_file],
        remappings=[("odometry/filtered", "/odometry/filtered_global")],
    )

    gps_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_gps_tf",
        arguments=[
            "--x", "0.135", "--y", "0.05", "--z", "1.46",
            "--yaw", "0", "--pitch", "0", "--roll", "0",
            "--frame-id", "base_link", "--child-frame-id", "ubx",
        ],
    )

    recorder = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record", "--use-sim-time", "-o", output_bag,
            "/fix", "/fix/fusion", "/ubx_nav_vel_ned", "/imu/data_clean",
            "/ubx_nav_pvt", "/ubx_nav_hp_pos_llh",
            "/gnss/course_imu", "/gnss/vel_twist",
            "/odometry/local",
            "/odometry/gps", "/odometry/filtered_global", "/gps/filtered",
            "/diagnostics", "/tf", "/tf_static",
        ],
        output="screen",
    )

    # Let filters and recorder subscribe before simulated time starts.
    player = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play", bag,
            "--clock", "100.0",
            "--rate", playback_rate,
            "--start-offset", start_offset,
            "--disable-keyboard-controls",
            "--topics", "/fix", "/ubx_nav_vel_ned",
            "/ubx_nav_pvt", "/ubx_nav_hp_pos_llh",
        ],
        output="screen",
    )

    critical_exit_handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=action,
            on_exit=[EmitEvent(event=Shutdown(reason=f"{name} exited"))],
        ))
        for action, name in (
            (imu_player, "imu_file_player"),
            (velocity_converter, "velned_to_twist"),
            (course_converter, "gnss_course_to_imu"),
            (fix_gate, "fix_after_course_gate"),
            (ekf_local, "local EKF"),
            (navsat, "navsat_transform"),
            (ekf_global, "global EKF"),
            (recorder, "bag recorder"),
        )
    ]

    return LaunchDescription([
        DeclareLaunchArgument("bag"),
        DeclareLaunchArgument("imu_file"),
        DeclareLaunchArgument("output_bag", default_value="fused_gps_imu_result"),
        DeclareLaunchArgument("config_file", default_value=default_config),
        DeclareLaunchArgument(
            "mount_yaw_offset_rad",
            default_value="-3.141592653589793",
        ),
        DeclareLaunchArgument("imu_yaw_variance_rad2", default_value="0.7"),
        DeclareLaunchArgument(
            "yaw_rate_source",
            default_value="raw_gyro_z",
        ),
        DeclareLaunchArgument(
            "fusion_fix_topic",
            default_value="/fix/fusion",
        ),
        DeclareLaunchArgument("playback_rate", default_value="1.0"),
        DeclareLaunchArgument("start_offset", default_value="0.0"),
        DeclareLaunchArgument("ros_domain_id", default_value="73"),
        SetEnvironmentVariable("ROS_DOMAIN_ID", ros_domain_id),
        OpaqueFunction(function=validate_launch_paths),
        *critical_exit_handlers,
        RegisterEventHandler(OnProcessExit(
            target_action=player,
            on_exit=[TimerAction(
                period=1.0,
                actions=[EmitEvent(event=Shutdown(
                    reason="bag playback completed",
                ))],
            )],
        )),
        imu_player,
        velocity_converter,
        course_converter,
        fix_gate,
        ekf_local,
        navsat,
        ekf_global,
        gps_tf,
        recorder,
        TimerAction(period=3.0, actions=[player]),
    ])
