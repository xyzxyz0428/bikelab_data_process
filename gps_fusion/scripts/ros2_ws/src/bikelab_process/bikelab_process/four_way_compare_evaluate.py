#!/usr/bin/env python3
"""Evaluate the single-replay four-way GNSS/yaw-rate comparison bag."""

import argparse
import bisect
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from bikelab_process.fusion_result_evaluate import (
    correlation,
    evaluate,
    percentile,
    quaternion_yaw,
    read_selected_topics,
    topic_statistics,
    wrap_pi,
)


GROUP_TOPICS = {
    # Group 1 uses common GPS odometry directly; without a yaw input, yaw is
    # unobservable and a position-only EKF is not a valid substitute.
    'gps_position': '/odometry/gps_common',
    'gps_course': '/compare/g02_gps_course',
    'gps_course_raw_gyro': '/compare/g03_gps_course_raw_gyro',
    'gps_course_ahrs_rate': '/compare/g04_gps_course_ahrs_rate',
}
RAW_RATE_TOPIC = '/imu/raw_gyro_rate'
AHRS_RATE_TOPIC = '/imu/ahrs_heading_rate'
GATED_GPS_TOPIC = '/compare_input/gps'
COURSE_TOPIC = '/gnss/course_imu'


def finite_summary(values, unit):
    """Summarize finite absolute differences."""
    finite = [abs(value) for value in values if math.isfinite(value)]
    return {
        'samples': len(finite),
        f'median_{unit}': percentile(finite, 0.5),
        f'p95_{unit}': percentile(finite, 0.95),
        f'max_{unit}': max(finite) if finite else None,
    }


def extract_odometry(rows):
    """Convert bag odometry rows to sorted planar samples."""
    points = []
    for header_time, _, message in rows:
        if header_time is None:
            continue
        points.append({
            't': header_time,
            'x': float(message.pose.pose.position.x),
            'y': float(message.pose.pose.position.y),
            'yaw': quaternion_yaw(message.pose.pose.orientation),
        })
    points.sort(key=lambda item: item['t'])
    return points


def pair_odometry(left, right, tolerance, compare_yaw=True):
    """Compare two odometry streams using nearest timestamps."""
    right_times = [point['t'] for point in right]
    position_differences = []
    yaw_differences = []
    time_differences = []
    for point in left:
        position = bisect.bisect_left(right_times, point['t'])
        candidates = []
        if position < len(right):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda index: abs(right[index]['t'] - point['t']),
        )
        delta_t = right[best]['t'] - point['t']
        if abs(delta_t) > tolerance:
            continue
        other = right[best]
        time_differences.append(delta_t)
        position_differences.append(
            math.hypot(point['x'] - other['x'], point['y'] - other['y'])
        )
        if compare_yaw:
            yaw_differences.append(
                math.degrees(wrap_pi(point['yaw'] - other['yaw']))
            )
    result = {
        'position': finite_summary(position_differences, 'm'),
        'yaw': (
            finite_summary(yaw_differences, 'deg')
            if compare_yaw else None
        ),
        'time': finite_summary(time_differences, 's'),
    }
    return result


def rate_comparison(raw_rows, ahrs_rows, tolerance):
    """Compare the two recorded ROS yaw-rate measurements."""
    ahrs_points = sorted(
        (
            header_time,
            float(message.angular_velocity.z),
            float(message.angular_velocity_covariance[8]),
        )
        for header_time, _, message in ahrs_rows
        if header_time is not None
    )
    ahrs_times = [point[0] for point in ahrs_points]
    raw_values = []
    ahrs_values = []
    time_differences = []
    raw_variances = []
    ahrs_variances = []
    for header_time, _, message in raw_rows:
        if header_time is None or not ahrs_points:
            continue
        position = bisect.bisect_left(ahrs_times, header_time)
        candidates = []
        if position < len(ahrs_points):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        best = min(
            candidates,
            key=lambda index: abs(ahrs_points[index][0] - header_time),
        )
        delta_t = ahrs_points[best][0] - header_time
        if abs(delta_t) > tolerance:
            continue
        raw_values.append(float(message.angular_velocity.z))
        ahrs_values.append(ahrs_points[best][1])
        time_differences.append(delta_t)
        raw_variances.append(float(message.angular_velocity_covariance[8]))
        ahrs_variances.append(ahrs_points[best][2])

    differences = [
        raw - ahrs for raw, ahrs in zip(raw_values, ahrs_values)
    ]
    return {
        'samples': len(differences),
        'correlation': correlation(raw_values, ahrs_values),
        'mean_raw_minus_ahrs_rad_s': (
            statistics.mean(differences) if differences else None
        ),
        'absolute_difference': finite_summary(differences, 'rad_s'),
        'time_difference': finite_summary(time_differences, 's'),
        'raw_variance_rad2_s2': (
            statistics.median(raw_variances) if raw_variances else None
        ),
        'ahrs_variance_rad2_s2': (
            statistics.median(ahrs_variances) if ahrs_variances else None
        ),
    }


def build_result(args):
    """Run the generic evaluation and add four-way pairwise diagnostics."""
    odom_topics = list(GROUP_TOPICS.values())
    generic_args = SimpleNamespace(
        bag=args.bag,
        storage_id=args.storage_id,
        fix_topic='/fix/fusion',
        imu_topic=AHRS_RATE_TOPIC,
        course_topic=COURSE_TOPIC,
        odom_topics=odom_topics,
        time_tolerance=args.time_tolerance,
        minimum_course_speed=args.minimum_course_speed,
        course_half_window=args.course_half_window,
    )
    result = evaluate(generic_args)
    # navsat_transform publishes a quaternion on its GPS odometry, but the
    # position-only baseline has no yaw observation. Do not expose that
    # placeholder orientation as a measured/evaluated yaw.
    result['odometry_yaw_vs_gnss_course'][
        GROUP_TOPICS['gps_position']
    ] = None

    extra_topics = [
        *odom_topics,
        RAW_RATE_TOPIC,
        AHRS_RATE_TOPIC,
        GATED_GPS_TOPIC,
        '/compare_input/course',
        '/compare_input/raw_gyro_rate',
        '/compare_input/ahrs_heading_rate',
    ]
    _, raw = read_selected_topics(args.bag, args.storage_id, extra_topics)
    points = {
        name: extract_odometry(raw.get(topic, []))
        for name, topic in GROUP_TOPICS.items()
    }
    group_statistics = {}
    for name, topic in GROUP_TOPICS.items():
        stats = topic_statistics(raw.get(topic, []))
        samples = points[name]
        stats.update({
            'first_stamp_s': samples[0]['t'] if samples else None,
            'last_stamp_s': samples[-1]['t'] if samples else None,
            'yaw_observable': name != 'gps_position',
        })
        group_statistics[name] = stats

    pairwise = {}
    names = list(GROUP_TOPICS)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            pairwise[f'{left_name}__vs__{right_name}'] = pair_odometry(
                points[left_name],
                points[right_name],
                args.pair_tolerance,
                compare_yaw=(
                    left_name != 'gps_position'
                    and right_name != 'gps_position'
                ),
            )

    gated_gps = raw.get(GATED_GPS_TOPIC, [])
    result['four_way'] = {
        'group_statistics': group_statistics,
        'first_common_gps_stamp_s': (
            gated_gps[0][0] if gated_gps else None
        ),
        'pairwise': pairwise,
        'raw_gyro_vs_ahrs_headingspeed': rate_comparison(
            raw.get(RAW_RATE_TOPIC, []),
            raw.get(AHRS_RATE_TOPIC, []),
            args.rate_tolerance,
        ),
    }
    return result


def format_number(value, digits=4):
    """Format a possibly absent numeric value."""
    return 'n/a' if value is None else f'{value:.{digits}f}'


def print_summary(result):
    """Print the comparison summary."""
    print('=== Four-Way Single-Replay Comparison ===')
    four_way = result['four_way']
    print('Group outputs:')
    for name, stats in four_way['group_statistics'].items():
        print(
            f'  {name}: {stats["count"]} messages, '
            f'{format_number(stats["effective_rate_hz"], 3)} Hz'
        )

    print('\nYaw vs. trajectory course by group:')
    angles = result['odometry_yaw_vs_gnss_course']
    for name, topic in GROUP_TOPICS.items():
        metrics = angles.get(topic) or {}
        if name == 'gps_position':
            print('  gps_position: N/A (position-only does not estimate yaw)')
            continue
        print(
            f'  {name}: median='
            f'{format_number(metrics.get("median_abs_error_deg"), 3)}°, '
            f'P95={format_number(metrics.get("p95_abs_error_deg"), 3)}°, '
            f'bias={format_number(metrics.get("circular_bias_deg"), 3)}°'
        )

    rate = four_way['raw_gyro_vs_ahrs_headingspeed']
    difference = rate['absolute_difference']
    print('\nRaw gyro_z vs. AHRS headingspeed:')
    print(
        f'  correlation={format_number(rate.get("correlation"), 5)}, '
        f'median abs diff='
        f'{format_number(difference.get("median_rad_s"), 6)} rad/s, '
        f'P95={format_number(difference.get("p95_rad_s"), 6)} rad/s'
    )

    key = 'gps_course_raw_gyro__vs__gps_course_ahrs_rate'
    pair = four_way['pairwise'].get(key) or {}
    position = pair.get('position') or {}
    yaw = pair.get('yaw') or {}
    print('\nDifference between the two yaw-rate fusion results:')
    print(
        f'  position median='
        f'{format_number(position.get("median_m"), 4)} m, '
        f'P95={format_number(position.get("p95_m"), 4)} m'
    )
    print(
        f'  yaw median={format_number(yaw.get("median_deg"), 4)}°, '
        f'P95={format_number(yaw.get("p95_deg"), 4)}°'
    )

    print(
        '\nNote: /fix and course are both fused inputs from the same GNSS '
        'receiver; these metrics are not independent absolute ground truth.'
    )


def main():
    """Parse arguments, evaluate the bag, and optionally save JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--storage-id', default='sqlite3')
    parser.add_argument('--time-tolerance', type=float, default=0.25)
    parser.add_argument('--pair-tolerance', type=float, default=0.03)
    parser.add_argument('--rate-tolerance', type=float, default=0.01)
    parser.add_argument('--minimum-course-speed', type=float, default=2.0)
    parser.add_argument('--course-half-window', type=int, default=5)
    args = parser.parse_args()

    try:
        result = build_result(args)
    except RuntimeError as error:
        print(f'[ERROR] {error}')
        raise SystemExit(2) from None
    print_summary(result)
    if args.out:
        output = Path(args.out)
        if output.exists():
            raise SystemExit(
                f'[ERROR] Refusing to overwrite existing report: {output}'
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'JSON report: {output}')
    raise SystemExit(0 if result['ok'] else 2)


if __name__ == '__main__':
    main()
