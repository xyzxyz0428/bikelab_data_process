#!/usr/bin/env python3
"""Check timing, frames, trajectory shape, and yaw consistency in a fused bag."""

import argparse
import bisect
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message


WGS84_SEMI_MAJOR_M = 6378137.0
WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def quaternion_yaw(quaternion):
    siny = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny, cosy)


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def percentile(values, fraction):
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def circular_mean(values):
    if not values:
        return None
    sine = sum(math.sin(value) for value in values)
    cosine = sum(math.cos(value) for value in values)
    return math.atan2(sine, cosine)


def correlation(left, right):
    pairs = [
        (x, y) for x, y in zip(left, right)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 3:
        return None
    mean_x = sum(item[0] for item in pairs) / len(pairs)
    mean_y = sum(item[1] for item in pairs) / len(pairs)
    covariance = sum(
        (x - mean_x) * (y - mean_y) for x, y in pairs
    )
    variance_x = sum((x - mean_x) ** 2 for x, _ in pairs)
    variance_y = sum((y - mean_y) ** 2 for _, y in pairs)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def open_bag(path, storage_id):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    return reader


def read_selected_topics(bag_path, storage_id, topics):
    reader = open_bag(bag_path, storage_id)
    type_map = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    selected = {
        topic: get_message(type_map[topic])
        for topic in topics if topic in type_map
    }
    rows = defaultdict(list)

    while reader.has_next():
        topic, data, record_ns = reader.read_next()
        if topic not in selected:
            continue
        message = deserialize_message(data, selected[topic])
        header_time = None
        if hasattr(message, 'header'):
            header_time = stamp_seconds(message.header.stamp)
        rows[topic].append((header_time, record_ns * 1.0e-9, message))
    return type_map, rows


def geodetic_to_ecef(latitude, longitude, altitude):
    """Convert WGS84 geodetic coordinates to Earth-centered coordinates."""
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)
    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    prime_vertical = WGS84_SEMI_MAJOR_M / math.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_latitude ** 2
    )
    x = (
        (prime_vertical + altitude)
        * cos_latitude
        * math.cos(longitude_rad)
    )
    y = (
        (prime_vertical + altitude)
        * cos_latitude
        * math.sin(longitude_rad)
    )
    z = (
        prime_vertical * (1.0 - WGS84_ECCENTRICITY_SQUARED)
        + altitude
    ) * sin_latitude
    return x, y, z


def local_enu(latitude, longitude, altitude, origin):
    """Project WGS84 coordinates into the origin's local tangent plane."""
    x, y, z = geodetic_to_ecef(latitude, longitude, altitude)
    delta_x = x - origin['ecef_x']
    delta_y = y - origin['ecef_y']
    delta_z = z - origin['ecef_z']
    latitude0 = math.radians(origin['latitude'])
    longitude0 = math.radians(origin['longitude'])
    east = (
        -math.sin(longitude0) * delta_x
        + math.cos(longitude0) * delta_y
    )
    north = (
        -math.sin(latitude0) * math.cos(longitude0) * delta_x
        - math.sin(latitude0) * math.sin(longitude0) * delta_y
        + math.cos(latitude0) * delta_z
    )
    return east, north


def topic_statistics(rows):
    times = [item[0] for item in rows if item[0] is not None]
    if len(times) < 2:
        return {'count': len(rows), 'duration_s': None, 'effective_rate_hz': None}
    duration = max(times) - min(times)
    return {
        'count': len(rows),
        'duration_s': duration,
        'effective_rate_hz': (
            (len(times) - 1) / duration if duration > 0.0 else None
        ),
    }


def nearest_index(times, value, tolerance):
    position = bisect.bisect_left(times, value)
    candidates = []
    if position < len(times):
        candidates.append(position)
    if position > 0:
        candidates.append(position - 1)
    if not candidates:
        return None
    best = min(candidates, key=lambda index: abs(times[index] - value))
    return best if abs(times[best] - value) <= tolerance else None


def match_trajectory(fix_points, odometry_points, tolerance):
    """Interpolate the GNSS trajectory at each odometry timestamp."""
    fix_times = [point['t'] for point in fix_points]
    matches = []
    for odometry in odometry_points:
        position = bisect.bisect_left(fix_times, odometry['t'])
        if position < len(fix_times) and fix_times[position] == odometry['t']:
            matches.append((fix_points[position], odometry))
            continue
        if position == 0 or position == len(fix_times):
            continue

        before = fix_points[position - 1]
        after = fix_points[position]
        before_gap = odometry['t'] - before['t']
        after_gap = after['t'] - odometry['t']
        interval = after['t'] - before['t']
        if (
            before_gap > tolerance
            or after_gap > tolerance
            or interval <= 0.0
        ):
            continue
        weight = before_gap / interval
        interpolated = {
            't': odometry['t'],
            'x': before['x'] + weight * (after['x'] - before['x']),
            'y': before['y'] + weight * (after['y'] - before['y']),
        }
        matches.append((interpolated, odometry))
    return matches


def error_summary(errors):
    return {
        'samples': len(errors),
        'median_m': percentile(errors, 0.5),
        'p95_m': percentile(errors, 0.95),
        'max_m': max(errors) if errors else None,
    }


def translation_alignment(matches, initial_samples=20):
    if not matches:
        return None
    subset = matches[:min(initial_samples, len(matches))]
    offset_x = statistics.median([
        odometry['x'] - fix['x'] for fix, odometry in subset
    ])
    offset_y = statistics.median([
        odometry['y'] - fix['y'] for fix, odometry in subset
    ])
    errors = [
        math.hypot(
            odometry['x'] - offset_x - fix['x'],
            odometry['y'] - offset_y - fix['y'],
        )
        for fix, odometry in matches
    ]
    result = error_summary(errors)
    result.update({'offset_x_m': offset_x, 'offset_y_m': offset_y})
    return result


def rigid_alignment(matches):
    if len(matches) < 2:
        return None
    source_x = [odometry['x'] for _, odometry in matches]
    source_y = [odometry['y'] for _, odometry in matches]
    target_x = [fix['x'] for fix, _ in matches]
    target_y = [fix['y'] for fix, _ in matches]
    mean_sx = sum(source_x) / len(source_x)
    mean_sy = sum(source_y) / len(source_y)
    mean_tx = sum(target_x) / len(target_x)
    mean_ty = sum(target_y) / len(target_y)

    cosine_term = 0.0
    sine_term = 0.0
    for sx, sy, tx, ty in zip(source_x, source_y, target_x, target_y):
        sx -= mean_sx
        sy -= mean_sy
        tx -= mean_tx
        ty -= mean_ty
        cosine_term += sx * tx + sy * ty
        sine_term += sx * ty - sy * tx
    angle = math.atan2(sine_term, cosine_term)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    errors = []
    for sx, sy, tx, ty in zip(source_x, source_y, target_x, target_y):
        aligned_x = (
            cosine * (sx - mean_sx) - sine * (sy - mean_sy) + mean_tx
        )
        aligned_y = (
            sine * (sx - mean_sx) + cosine * (sy - mean_sy) + mean_ty
        )
        errors.append(math.hypot(aligned_x - tx, aligned_y - ty))
    result = error_summary(errors)
    result.update({
        'rotation_deg_odometry_to_enu': math.degrees(angle),
        'source_centroid_x_m': mean_sx,
        'source_centroid_y_m': mean_sy,
        'target_centroid_x_m': mean_tx,
        'target_centroid_y_m': mean_ty,
    })
    return result


def build_courses(fix_points, half_window, minimum_speed):
    courses = []
    for index in range(half_window, len(fix_points) - half_window):
        before = fix_points[index - half_window]
        after = fix_points[index + half_window]
        delta_t = after['t'] - before['t']
        if delta_t <= 0.0:
            continue
        east = after['x'] - before['x']
        north = after['y'] - before['y']
        speed = math.hypot(east, north) / delta_t
        if speed < minimum_speed:
            continue
        courses.append({
            't': fix_points[index]['t'],
            'course': math.atan2(north, east),
            'speed': speed,
        })
    return courses


def angle_comparison(courses, angle_points, tolerance):
    times = [point['t'] for point in angle_points]
    errors = []
    for course in courses:
        index = nearest_index(times, course['t'], tolerance)
        if index is None:
            continue
        errors.append(wrap_pi(angle_points[index]['yaw'] - course['course']))
    absolute_degrees = [abs(math.degrees(error)) for error in errors]
    bias = circular_mean(errors)
    return {
        'samples': len(errors),
        'circular_bias_deg': math.degrees(bias) if bias is not None else None,
        'median_abs_error_deg': percentile(absolute_degrees, 0.5),
        'p95_abs_error_deg': percentile(absolute_degrees, 0.95),
    }


def imu_rate_consistency(imu_points):
    if len(imu_points) < 3:
        return None
    unwrapped = [imu_points[0]['yaw']]
    for point in imu_points[1:]:
        delta = wrap_pi(point['yaw'] - imu_points[len(unwrapped) - 1]['yaw'])
        unwrapped.append(unwrapped[-1] + delta)

    derivatives = []
    measured = []
    for index in range(1, len(imu_points) - 1):
        delta_t = imu_points[index + 1]['t'] - imu_points[index - 1]['t']
        if not 0.01 <= delta_t <= 0.5:
            continue
        derivatives.append(
            (unwrapped[index + 1] - unwrapped[index - 1]) / delta_t
        )
        measured.append(imu_points[index]['wz'])
    differences = [
        derivative - angular_rate
        for derivative, angular_rate in zip(derivatives, measured)
    ]
    return {
        'samples': len(derivatives),
        'correlation_dyaw_dt_vs_wz': correlation(derivatives, measured),
        'median_abs_difference_rad_s': percentile(
            [abs(value) for value in differences], 0.5
        ),
        'p95_abs_difference_rad_s': percentile(
            [abs(value) for value in differences], 0.95
        ),
    }


def evaluate(args):
    support_topics = ['/diagnostics', '/tf', '/tf_static']
    topics = list(dict.fromkeys(
        [args.fix_topic, '/fix', '/fix/fusion', args.imu_topic, args.course_topic]
        + args.odom_topics
        + support_topics
    ))
    topics = [topic for topic in topics if topic]
    type_map, raw = read_selected_topics(args.bag, args.storage_id, topics)
    warnings = []
    validation_errors = []

    fix_topic = args.fix_topic
    if not fix_topic:
        fix_topic = '/fix/fusion' if '/fix/fusion' in raw else '/fix'
    if fix_topic not in raw:
        raise RuntimeError(f'Missing reference topic: {fix_topic}')
    required_result_topics = [args.imu_topic] + args.odom_topics
    missing_result_topics = [
        topic for topic in required_result_topics if topic not in raw
    ]
    if missing_result_topics:
        raise RuntimeError(
            'Missing required fusion result topics: '
            + ', '.join(missing_result_topics)
        )
    if args.course_topic not in raw:
        warnings.append(
            f'Missing GNSS course topic: {args.course_topic}; '
            'yaw-course metrics are unavailable'
        )

    fix_messages = raw[fix_topic]
    first_fix = fix_messages[0][2]
    latitude0 = float(first_fix.latitude)
    longitude0 = float(first_fix.longitude)
    altitude0 = float(first_fix.altitude)
    if not math.isfinite(altitude0):
        altitude0 = 0.0
    ecef_x, ecef_y, ecef_z = geodetic_to_ecef(
        latitude0, longitude0, altitude0
    )
    origin = {
        'latitude': latitude0,
        'longitude': longitude0,
        'altitude': altitude0,
        'ecef_x': ecef_x,
        'ecef_y': ecef_y,
        'ecef_z': ecef_z,
    }
    fix_points = []
    for header_time, _, message in fix_messages:
        if header_time is None:
            continue
        latitude = float(message.latitude)
        longitude = float(message.longitude)
        altitude = float(message.altitude)
        if not all(math.isfinite(value) for value in (latitude, longitude)):
            continue
        if not math.isfinite(altitude):
            altitude = altitude0
        east, north = local_enu(
            latitude,
            longitude,
            altitude,
            origin,
        )
        fix_points.append({'t': header_time, 'x': east, 'y': north})
    fix_points.sort(key=lambda item: item['t'])
    courses = build_courses(
        fix_points, args.course_half_window, args.minimum_course_speed
    )

    topic_stats = {
        topic: topic_statistics(rows) for topic, rows in raw.items()
    }
    trajectory = {}
    odometry_angles = {}
    for topic in args.odom_topics:
        points = []
        frames = set()
        child_frames = set()
        invalid_values = 0
        for header_time, _, message in raw.get(topic, []):
            if header_time is None:
                continue
            frames.add(message.header.frame_id)
            child_frames.add(message.child_frame_id)
            values = (
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
                quaternion_yaw(message.pose.pose.orientation),
            )
            if not all(math.isfinite(value) for value in values):
                invalid_values += 1
                continue
            points.append({
                't': header_time,
                'x': values[0],
                'y': values[1],
                'yaw': values[2],
            })
        points.sort(key=lambda item: item['t'])
        if not points:
            continue
        matches = match_trajectory(fix_points, points, args.time_tolerance)
        trajectory[topic] = {
            'frame_ids': sorted(frames),
            'child_frame_ids': sorted(child_frames),
            'invalid_value_count': invalid_values,
            'matched_samples': len(matches),
            'translation_only': translation_alignment(matches),
            'rigid_se2': rigid_alignment(matches),
        }
        odometry_angles[topic] = angle_comparison(
            courses, points, args.time_tolerance
        )
        if 'odom' in frames and topic == '/odometry/filtered_global':
            warnings.append(
                '/odometry/filtered_global is in odom, not map; absolute ENU '
                'overlap is not expected'
            )
        expected_frame = None
        if topic == '/odometry/local':
            expected_frame = 'odom'
        elif topic in ('/odometry/gps', '/odometry/filtered_global'):
            expected_frame = 'map'
        if expected_frame and frames != {expected_frame}:
            validation_errors.append(
                f'{topic} frame_id is {sorted(frames)}, expected '
                f'{expected_frame}'
            )
        if (
            topic in ('/odometry/local', '/odometry/filtered_global')
            and child_frames != {'base_link'}
        ):
            validation_errors.append(
                f'{topic} child_frame_id is {sorted(child_frames)}, '
                'expected base_link'
            )
        if invalid_values:
            validation_errors.append(
                f'{topic} contains {invalid_values} non-finite poses'
            )

    imu_points = []
    for header_time, _, message in raw.get(args.imu_topic, []):
        if header_time is None:
            continue
        imu_points.append({
            't': header_time,
            'yaw': quaternion_yaw(message.orientation),
            'wz': float(message.angular_velocity.z),
        })
    imu_points.sort(key=lambda item: item['t'])

    course_points = []
    for header_time, _, message in raw.get(args.course_topic, []):
        if header_time is None:
            continue
        course_points.append({
            't': header_time,
            'yaw': quaternion_yaw(message.orientation),
        })
    course_points.sort(key=lambda item: item['t'])

    transform_pairs = set()
    for tf_topic in ('/tf', '/tf_static'):
        for _, _, message in raw.get(tf_topic, []):
            for transform in message.transforms:
                parent = transform.header.frame_id.lstrip('/')
                child = transform.child_frame_id.lstrip('/')
                transform_pairs.add((parent, child))
    for required_pair in (('map', 'odom'), ('odom', 'base_link')):
        if required_pair not in transform_pairs:
            validation_errors.append(
                'Missing required TF: '
                f'{required_pair[0]} -> {required_pair[1]}'
            )

    diagnostic_levels = defaultdict(int)
    diagnostic_error_messages = set()
    for _, _, message in raw.get('/diagnostics', []):
        for status in message.status:
            level = (
                status.level[0]
                if isinstance(status.level, (bytes, bytearray))
                else int(status.level)
            )
            diagnostic_levels[str(level)] += 1
            if level >= 2:
                diagnostic_error_messages.add(
                    f'{status.name}: {status.message}'
                )
    if diagnostic_error_messages:
        warnings.append(
            'Diagnostics contain error-level statuses; inspect '
            'diagnostic_error_messages'
        )

    result = {
        'ok': not validation_errors,
        'bag': str(Path(args.bag)),
        'reference_fix_topic': fix_topic,
        'topic_types': {
            topic: type_map.get(topic) for topic in topics
        },
        'topic_statistics': topic_stats,
        'reference_fix_origin': {
            'latitude': latitude0,
            'longitude': longitude0,
            'altitude': altitude0,
        },
        'moving_course_samples': len(courses),
        'trajectory_vs_fix_enu': trajectory,
        'odometry_yaw_vs_gnss_course': odometry_angles,
        'imu_yaw_vs_gnss_course': (
            angle_comparison(courses, imu_points, args.time_tolerance)
            if imu_points else None
        ),
        'gnss_course_yaw_vs_fix_course': (
            angle_comparison(courses, course_points, args.time_tolerance)
            if course_points else None
        ),
        'imu_yaw_rate_consistency': imu_rate_consistency(imu_points),
        'tf_pairs': [list(pair) for pair in sorted(transform_pairs)],
        'diagnostic_level_counts': dict(diagnostic_levels),
        'diagnostic_error_messages': sorted(diagnostic_error_messages),
        'validation_errors': validation_errors,
        'warnings': [
            f'{fix_topic} is also a fusion input, not independent ground '
            'truth; '
            'trajectory errors are diagnostic only.',
            *warnings,
        ],
    }
    return result


def format_number(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def print_result(result):
    print('=== GPS/IMU Fusion Result Check ===')
    print(f'Position comparison topic: {result["reference_fix_topic"]}')
    for topic, stats in result['topic_statistics'].items():
        print(
            f'{topic}: {stats["count"]} messages, '
            f'{format_number(stats["effective_rate_hz"])} Hz'
        )

    for topic, metrics in result['trajectory_vs_fix_enu'].items():
        translation = metrics['translation_only'] or {}
        rigid = metrics['rigid_se2'] or {}
        angle = (
            result.get('odometry_yaw_vs_gnss_course', {}).get(topic) or {}
        )
        print(f'\n{topic}, frame={metrics["frame_ids"]}:')
        print(
            '  Translation-only alignment: median=',
            format_number(translation.get('median_m')),
            'm, P95=', format_number(translation.get('p95_m')), 'm',
            sep='',
        )
        print(
            '  SE(2) rigid alignment: rotation=',
            format_number(rigid.get('rotation_deg_odometry_to_enu')),
            '°, median=', format_number(rigid.get('median_m')),
            'm, P95=', format_number(rigid.get('p95_m')), 'm',
            sep='',
        )
        print(
            '  Yaw vs. trajectory course: bias=',
            format_number(angle.get('circular_bias_deg')),
            '°, median abs=',
            format_number(angle.get('median_abs_error_deg')),
            '°, P95=',
            format_number(angle.get('p95_abs_error_deg')),
            '°',
            sep='',
        )

    consistency = result.get('imu_yaw_rate_consistency') or {}
    print(
        '\nIMU d(yaw)/dt vs. wz correlation: ',
        format_number(consistency.get('correlation_dyaw_dt_vs_wz')),
        sep='',
    )
    imu_course = result.get('imu_yaw_vs_gnss_course') or {}
    print(
        'IMU yaw vs. GNSS course: bias=',
        format_number(imu_course.get('circular_bias_deg')),
        '°, median abs=',
        format_number(imu_course.get('median_abs_error_deg')),
        '°',
        sep='',
    )
    gnss_course = result.get('gnss_course_yaw_vs_fix_course') or {}
    print(
        'GNSS course yaw vs. fix-derived course: bias=',
        format_number(gnss_course.get('circular_bias_deg')),
        '°, median abs=',
        format_number(gnss_course.get('median_abs_error_deg')),
        '°',
        sep='',
    )
    for warning in result['warnings']:
        print(f'[WARNING] {warning}')
    for error in result['validation_errors']:
        print(f'[ERROR] {error}')
    print('Structural validation:', 'PASS' if result['ok'] else 'FAIL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--storage-id', default='sqlite3')
    parser.add_argument(
        '--fix-topic',
        default='',
        help='default: /fix/fusion when present, otherwise /fix',
    )
    parser.add_argument('--imu-topic', default='/imu/data_clean')
    parser.add_argument('--course-topic', default='/gnss/course_imu')
    parser.add_argument(
        '--odom-topics', nargs='+',
        default=[
            '/odometry/gps',
            '/odometry/local',
            '/odometry/filtered_global',
        ],
    )
    parser.add_argument('--time-tolerance', type=float, default=0.25)
    parser.add_argument('--minimum-course-speed', type=float, default=2.0)
    parser.add_argument('--course-half-window', type=int, default=5)
    args = parser.parse_args()

    try:
        result = evaluate(args)
    except RuntimeError as error:
        print(f'[ERROR] {error}')
        raise SystemExit(2) from None
    print_result(result)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'JSON report: {output}')
    raise SystemExit(0 if result['ok'] else 2)


if __name__ == '__main__':
    main()
