"""Run four GNSS/course/yaw-rate variants from one bag replay."""

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
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def validate_launch_paths(context):
    """Reject missing inputs and refuse to overwrite an existing result."""
    bag = Path(LaunchConfiguration('bag').perform(context)).expanduser()
    imu_file = Path(
        LaunchConfiguration('imu_file').perform(context)
    ).expanduser()
    output_bag = Path(
        LaunchConfiguration('output_bag').perform(context)
    ).expanduser()
    if not bag.exists():
        raise RuntimeError(f'GNSS bag does not exist: {bag}')
    if not imu_file.is_file():
        raise RuntimeError(f'IMU file does not exist: {imu_file}')
    if output_bag.exists():
        raise RuntimeError(
            f'Output bag already exists; choose a new path: {output_bag}'
        )
    return []


def imu_player(
    name,
    topic,
    yaw_rate_source,
    imu_file,
    time_offset_ns,
    gyro_covariance_scale,
    subtract_static_gyro_bias,
    static_start_s,
    static_duration_s,
    max_merge_dt_ms,
):
    """Create one rate-source-specific view of the same IMU CSV."""
    return Node(
        package='bikelab_process',
        executable='imu_file_player',
        name=name,
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'file_path': imu_file,
            'topic': topic,
            'frame_id': 'base_link',
            'angle_unit': 'rad',
            'orientation_mode': 'heading_north_to_enu',
            'mount_yaw_offset_rad': -3.141592653589793,
            'axis_conversion': 'frd_to_flu',
            'yaw_rate_source': yaw_rate_source,
            'time_offset_ns': ParameterValue(
                time_offset_ns, value_type=int,
            ),
            'gyro_covariance_scale': ParameterValue(
                gyro_covariance_scale, value_type=float,
            ),
            'subtract_static_gyro_bias': ParameterValue(
                subtract_static_gyro_bias, value_type=bool,
            ),
            'static_start_s': ParameterValue(
                static_start_s, value_type=float,
            ),
            'static_duration_s': ParameterValue(
                static_duration_s, value_type=float,
            ),
            'max_merge_dt_ms': ParameterValue(
                max_merge_dt_ms, value_type=float,
            ),
            'skip_rows_before_first_clock': True,
            'covariance_mode': 'estimate_static',
            'min_static_samples': 100,
            'require_static_window': True,
            # Comparison EKFs use angular_velocity.z; orientation is diagnostic.
            'orientation_cov_floor_roll': 999.0,
            'orientation_cov_floor_pitch': 999.0,
            'orientation_cov_floor_yaw': 0.7,
            'gyro_cov_floor': 1.0e-8,
            'acc_cov_floor': 1.0e-6,
        }],
    )


def ekf_node(name, output_topic, config_file, parameter_overrides=None):
    """Create an EKF instance with a unique output and shared config file."""
    parameters = [config_file]
    if parameter_overrides:
        parameters.append(parameter_overrides)
    return Node(
        package='robot_localization',
        executable='ekf_node',
        name=name,
        output='screen',
        parameters=parameters,
        remappings=[('odometry/filtered', output_topic)],
    )


def generate_launch_description():
    """Assemble the four-way comparison with shared inputs and map transform."""
    bag = LaunchConfiguration('bag')
    imu_file = LaunchConfiguration('imu_file')
    output_bag = LaunchConfiguration('output_bag')
    config_file = LaunchConfiguration('config_file')
    playback_rate = LaunchConfiguration('playback_rate')
    start_offset = LaunchConfiguration('start_offset')
    ros_domain_id = LaunchConfiguration('ros_domain_id')
    smooth_lagged_data = LaunchConfiguration('smooth_lagged_data')
    history_length = LaunchConfiguration('history_length')
    predict_to_current_time = LaunchConfiguration('predict_to_current_time')
    navsat_frequency = LaunchConfiguration('navsat_frequency')
    navsat_transform_timeout = LaunchConfiguration(
        'navsat_transform_timeout'
    )
    imu_time_offset_ns = LaunchConfiguration('imu_time_offset_ns')
    gyro_covariance_scale = LaunchConfiguration('gyro_covariance_scale')
    subtract_static_gyro_bias = LaunchConfiguration(
        'subtract_static_gyro_bias'
    )
    minimum_course_speed_mps = LaunchConfiguration(
        'minimum_course_speed_mps'
    )
    maximum_course_accuracy_deg = LaunchConfiguration(
        'maximum_course_accuracy_deg'
    )
    group3_yaw_rate_source = LaunchConfiguration(
        'group3_yaw_rate_source'
    )
    static_start_s = LaunchConfiguration('static_start_s')
    static_duration_s = LaunchConfiguration('static_duration_s')
    max_merge_dt_ms = LaunchConfiguration('max_merge_dt_ms')

    lag_aware_parameters = {
        'smooth_lagged_data': ParameterValue(
            smooth_lagged_data,
            value_type=bool,
        ),
        'history_length': ParameterValue(history_length, value_type=float),
        'predict_to_current_time': ParameterValue(
            predict_to_current_time,
            value_type=bool,
        ),
    }

    default_config = PathJoinSubstitution([
        FindPackageShare('bikelab_process'),
        'config',
        'offline_four_way_compare.yaml',
    ])

    group3_rate_player = imu_player(
        'imu_raw_rate_player',
        '/imu/raw_gyro_rate',
        group3_yaw_rate_source,
        imu_file,
        imu_time_offset_ns,
        gyro_covariance_scale,
        subtract_static_gyro_bias,
        static_start_s,
        static_duration_s,
        max_merge_dt_ms,
    )
    ahrs_rate_player = imu_player(
        'imu_ahrs_rate_player',
        '/imu/ahrs_heading_rate',
        'ahrs_headingspeed',
        imu_file,
        imu_time_offset_ns,
        gyro_covariance_scale,
        subtract_static_gyro_bias,
        static_start_s,
        static_duration_s,
        max_merge_dt_ms,
    )

    velocity_converter = Node(
        package='bikelab_process',
        executable='velned_to_twist',
        name='velned_to_twist',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/ubx_nav_vel_ned',
            'output_topic': '/gnss/vel_twist',
            'output_mode': 'body_forward',
            'frame_id': 'base_link',
        }],
    )

    course_converter = Node(
        package='bikelab_process',
        executable='gnss_course_to_imu',
        name='gnss_course_to_imu',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/ubx_nav_vel_ned',
            'output_topic': '/gnss/course_imu',
            'frame_id': 'base_link',
            'minimum_speed_mps': ParameterValue(
                minimum_course_speed_mps, value_type=float,
            ),
            'maximum_course_accuracy_deg': ParameterValue(
                maximum_course_accuracy_deg, value_type=float,
            ),
            'minimum_yaw_variance_rad2': 0.03,
        }],
    )

    fix_gate = Node(
        package='bikelab_process',
        executable='fix_after_course_gate',
        name='fix_after_course_gate',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_fix_topic': '/fix',
            'course_topic': '/gnss/course_imu',
            'output_fix_topic': '/fix/fusion',
            'settle_time_s': 0.5,
        }],
    )

    reference_local = ekf_node(
        'reference_local_ekf',
        '/reference/odometry/local',
        config_file,
    )
    reference_global = ekf_node(
        'reference_global_ekf',
        '/reference/odometry/global',
        config_file,
        lag_aware_parameters,
    )

    navsat = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_reference',
        output='screen',
        parameters=[config_file, {
            'frequency': ParameterValue(navsat_frequency, value_type=float),
            'transform_timeout': ParameterValue(
                navsat_transform_timeout,
                value_type=float,
            ),
        }],
        remappings=[
            ('imu', '/imu/ahrs_heading_rate'),
            ('gps/fix', '/fix/fusion'),
            ('odometry/filtered', '/reference/odometry/global'),
            ('odometry/gps', '/odometry/gps_common'),
            ('gps/filtered', '/reference/gps/filtered'),
        ],
    )

    comparison_gate = Node(
        package='bikelab_process',
        executable='comparison_sensor_gate',
        name='comparison_sensor_gate',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    comparison_nodes = [
        ekf_node(
            'compare_gps_course',
            '/compare/g02_gps_course',
            config_file,
            lag_aware_parameters,
        ),
        ekf_node(
            'compare_gps_course_raw_gyro',
            '/compare/g03_gps_course_raw_gyro',
            config_file,
            lag_aware_parameters,
        ),
        ekf_node(
            'compare_gps_course_ahrs_rate',
            '/compare/g04_gps_course_ahrs_rate',
            config_file,
            lag_aware_parameters,
        ),
    ]

    gps_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_gps_tf',
        arguments=[
            '--x', '0.135', '--y', '0.05', '--z', '1.46',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'base_link', '--child-frame-id', 'ubx',
        ],
    )

    recorder = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '--use-sim-time', '-o', output_bag,
            '/clock', '/fix', '/fix/fusion', '/ubx_nav_vel_ned',
            '/ubx_nav_pvt', '/ubx_nav_hp_pos_llh',
            '/gnss/course_imu', '/gnss/vel_twist',
            '/imu/raw_gyro_rate', '/imu/ahrs_heading_rate',
            '/odometry/gps_common',
            '/compare_input/gps', '/compare_input/course',
            '/compare_input/raw_gyro_rate',
            '/compare_input/ahrs_heading_rate',
            '/compare_input/velocity',
            '/reference/odometry/local', '/reference/odometry/global',
            '/reference/gps/filtered',
            '/compare/g02_gps_course',
            '/compare/g03_gps_course_raw_gyro',
            '/compare/g04_gps_course_ahrs_rate',
            '/diagnostics', '/tf', '/tf_static',
        ],
        output='screen',
    )

    player = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag,
            '--clock', '100.0',
            '--rate', playback_rate,
            '--start-offset', start_offset,
            '--disable-keyboard-controls',
            '--topics', '/fix', '/ubx_nav_vel_ned',
            '/ubx_nav_pvt', '/ubx_nav_hp_pos_llh',
        ],
        output='screen',
    )

    critical_actions = [
        group3_rate_player,
        ahrs_rate_player,
        velocity_converter,
        course_converter,
        fix_gate,
        reference_local,
        navsat,
        reference_global,
        comparison_gate,
        *comparison_nodes,
        recorder,
    ]
    critical_exit_handlers = [
        RegisterEventHandler(OnProcessExit(
            target_action=action,
            on_exit=[EmitEvent(event=Shutdown(
                reason=f'critical process {index} exited',
            ))],
        ))
        for index, action in enumerate(critical_actions, start=1)
    ]

    return LaunchDescription([
        DeclareLaunchArgument('bag'),
        DeclareLaunchArgument('imu_file'),
        DeclareLaunchArgument(
            'output_bag',
            default_value='four_way_compare_result',
        ),
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('playback_rate', default_value='1.0'),
        DeclareLaunchArgument('start_offset', default_value='300.0'),
        DeclareLaunchArgument('ros_domain_id', default_value='75'),
        DeclareLaunchArgument('smooth_lagged_data', default_value='false'),
        DeclareLaunchArgument('history_length', default_value='0.0'),
        DeclareLaunchArgument('predict_to_current_time', default_value='false'),
        DeclareLaunchArgument('navsat_frequency', default_value='10.0'),
        DeclareLaunchArgument('navsat_transform_timeout', default_value='0.2'),
        DeclareLaunchArgument('imu_time_offset_ns', default_value='0'),
        DeclareLaunchArgument('gyro_covariance_scale', default_value='1.0'),
        DeclareLaunchArgument(
            'minimum_course_speed_mps', default_value='2.0',
        ),
        DeclareLaunchArgument(
            'maximum_course_accuracy_deg', default_value='30.0',
        ),
        DeclareLaunchArgument(
            'subtract_static_gyro_bias', default_value='false',
        ),
        DeclareLaunchArgument(
            'group3_yaw_rate_source', default_value='raw_gyro_z',
        ),
        DeclareLaunchArgument('static_start_s', default_value='0.0'),
        DeclareLaunchArgument('static_duration_s', default_value='10.0'),
        DeclareLaunchArgument('max_merge_dt_ms', default_value='80.0'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', ros_domain_id),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
        OpaqueFunction(function=validate_launch_paths),
        *critical_exit_handlers,
        RegisterEventHandler(OnProcessExit(
            target_action=player,
            on_exit=[TimerAction(
                period=1.0,
                actions=[EmitEvent(event=Shutdown(
                    reason='bag playback completed',
                ))],
            )],
        )),
        group3_rate_player,
        ahrs_rate_player,
        velocity_converter,
        course_converter,
        fix_gate,
        reference_local,
        navsat,
        reference_global,
        comparison_gate,
        *comparison_nodes,
        gps_tf,
        recorder,
        TimerAction(period=3.0, actions=[player]),
    ])
