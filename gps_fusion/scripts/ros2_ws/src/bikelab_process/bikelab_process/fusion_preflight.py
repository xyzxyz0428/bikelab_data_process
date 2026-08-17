#!/usr/bin/env python3
"""Check a GNSS bag and an FDILink IMU CSV before replay."""

import argparse
import bisect
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sqlite3

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix


NS_PER_SECOND = 1_000_000_000


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def timestamp_ns(value):
    """Parse an integer nanosecond timestamp without losing precision."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        number = finite_float(value)
        return int(number) if number is not None else None


def resolve_db3(bag_path):
    path = Path(bag_path)
    if path.is_file() and path.suffix == '.db3':
        return path
    if not path.is_dir():
        raise RuntimeError(f'ROS bag path does not exist: {path}')
    files = sorted(path.glob('*.db3'))
    if len(files) != 1:
        raise RuntimeError(
            f'Expected exactly one .db3 file in {path}, found {len(files)}'
        )
    return files[0]


def inspect_bag(path):
    db_path = resolve_db3(path)
    connection = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        rows = connection.execute(
            'SELECT topics.name, topics.type, COUNT(messages.id), '
            'MIN(messages.timestamp), MAX(messages.timestamp) '
            'FROM topics LEFT JOIN messages '
            'ON messages.topic_id = topics.id '
            'GROUP BY topics.id ORDER BY topics.name'
        ).fetchall()

        fix_bounds = connection.execute(
            'SELECT messages.data, messages.timestamp '
            'FROM messages JOIN topics ON messages.topic_id = topics.id '
            'WHERE topics.name = ? ORDER BY messages.timestamp ASC LIMIT 1',
            ('/fix',),
        ).fetchone()
        fix_last = connection.execute(
            'SELECT messages.data, messages.timestamp '
            'FROM messages JOIN topics ON messages.topic_id = topics.id '
            'WHERE topics.name = ? ORDER BY messages.timestamp DESC LIMIT 1',
            ('/fix',),
        ).fetchone()
    finally:
        connection.close()

    topics = {}
    for name, message_type, count, first_ns, last_ns in rows:
        duration_s = None
        rate_hz = None
        if count and count > 1 and last_ns > first_ns:
            duration_s = (last_ns - first_ns) / NS_PER_SECOND
            rate_hz = (count - 1) / duration_s
        topics[name] = {
            'type': message_type,
            'count': count,
            'first_record_ns': first_ns,
            'last_record_ns': last_ns,
            'duration_s': duration_s,
            'effective_rate_hz': rate_hz,
        }
    fix_header_range = None
    fix_type = topics.get('/fix', {}).get('type')
    if (
        fix_bounds
        and fix_last
        and fix_type == 'sensor_msgs/msg/NavSatFix'
    ):
        first_message = deserialize_message(fix_bounds[0], NavSatFix)
        last_message = deserialize_message(fix_last[0], NavSatFix)
        first_header_ns = (
            first_message.header.stamp.sec * NS_PER_SECOND
            + first_message.header.stamp.nanosec
        )
        last_header_ns = (
            last_message.header.stamp.sec * NS_PER_SECOND
            + last_message.header.stamp.nanosec
        )
        fix_header_range = {
            'first_header_ns': first_header_ns,
            'last_header_ns': last_header_ns,
            'first_record_ns': fix_bounds[1],
            'last_record_ns': fix_last[1],
            'first_record_header_delta_s': (
                fix_bounds[1] - first_header_ns
            ) / NS_PER_SECOND,
            'last_record_header_delta_s': (
                fix_last[1] - last_header_ns
            ) / NS_PER_SECOND,
        }
    return {
        'db3': str(db_path),
        'topics': topics,
        'fix_header_range': fix_header_range,
    }


def inspect_imu(path):
    csv_path = Path(path)
    if not csv_path.is_file():
        raise RuntimeError(f'IMU CSV does not exist: {csv_path}')

    counts = Counter()
    valid_counts = Counter()
    first_ns = {}
    last_ns = {}
    previous_ns = {}
    intervals = {}
    timestamps = {}
    non_monotonic = Counter()
    invalid_numeric = Counter()
    invalid_timestamp_rows = 0
    total_rows = 0
    fields = None

    with csv_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {
            't_unix_ns', 'dtype',
            'heading', 'headingspeed',
            'roll', 'pitch',
            'gyro_x', 'gyro_y', 'gyro_z',
            'acc_x', 'acc_y', 'acc_z',
            'crc8_ok', 'crc16_ok', 'end_ok',
        }
        missing = required.difference(fields)
        if missing:
            raise RuntimeError(f'IMU CSV missing columns: {sorted(missing)}')

        for row in reader:
            total_rows += 1
            parsed_timestamp = timestamp_ns(row.get('t_unix_ns'))
            if parsed_timestamp is None:
                invalid_timestamp_rows += 1
                continue
            row_timestamp_ns = parsed_timestamp
            dtype = str(row.get('dtype', '')).strip()
            counts[dtype] += 1
            first_ns[dtype] = min(
                first_ns.get(dtype, row_timestamp_ns), row_timestamp_ns
            )
            last_ns[dtype] = max(
                last_ns.get(dtype, row_timestamp_ns), row_timestamp_ns
            )
            timestamps.setdefault(dtype, []).append(row_timestamp_ns)

            numeric_fields = ()
            if dtype == '64':
                numeric_fields = (
                    'gyro_x', 'gyro_y', 'gyro_z',
                    'acc_x', 'acc_y', 'acc_z',
                )
            elif dtype == '65':
                numeric_fields = (
                    'roll', 'pitch', 'heading', 'headingspeed',
                )
            if any(finite_float(row.get(key)) is None for key in numeric_fields):
                invalid_numeric[dtype] += 1

            checks = [row.get(key) for key in ('crc8_ok', 'crc16_ok', 'end_ok')]
            if all(str(value).strip() == '1' for value in checks):
                valid_counts[dtype] += 1

            if dtype in previous_ns:
                delta_ns = row_timestamp_ns - previous_ns[dtype]
                if 0 < delta_ns < 10 * NS_PER_SECOND:
                    intervals.setdefault(dtype, []).append(delta_ns)
                if delta_ns <= 0:
                    non_monotonic[dtype] += 1
            previous_ns[dtype] = row_timestamp_ns

    stats = {}
    for dtype, count in sorted(counts.items()):
        duration_s = (last_ns[dtype] - first_ns[dtype]) / NS_PER_SECOND
        deltas = sorted(intervals.get(dtype, []))
        median_delta_s = None
        if deltas:
            median_delta_s = deltas[len(deltas) // 2] / NS_PER_SECOND
        stats[dtype] = {
            'count': count,
            'valid_count': valid_counts[dtype],
            'valid_fraction': valid_counts[dtype] / count,
            'invalid_numeric_count': invalid_numeric[dtype],
            'non_monotonic_count': non_monotonic[dtype],
            'first_ns': first_ns[dtype],
            'last_ns': last_ns[dtype],
            'duration_s': duration_s,
            'effective_rate_hz': (
                (count - 1) / duration_s if count > 1 and duration_s > 0 else None
            ),
            'median_rate_hz': (
                1.0 / median_delta_s if median_delta_s else None
            ),
        }

    match = None
    imu_times = sorted(timestamps.get('64', []))
    ahrs_times = sorted(timestamps.get('65', []))
    if imu_times and ahrs_times:
        gaps = []
        matched = 0
        for ahrs_time in ahrs_times:
            position = bisect.bisect_left(imu_times, ahrs_time)
            candidates = imu_times[max(0, position - 1):position + 1]
            if not candidates:
                continue
            gap = min(abs(value - ahrs_time) for value in candidates)
            gaps.append(gap)
            if gap <= 80_000_000:
                matched += 1
        sorted_gaps = sorted(gaps)
        match = {
            'max_gap_ms': 80.0,
            'matched_count': matched,
            'ahrs_count': len(ahrs_times),
            'matched_fraction': matched / len(ahrs_times),
            'median_nearest_gap_ms': (
                sorted_gaps[len(sorted_gaps) // 2] / 1.0e6
            ),
        }

    return {
        'path': str(csv_path),
        'columns': fields,
        'total_rows': total_rows,
        'invalid_timestamp_rows': invalid_timestamp_rows,
        'dtypes': stats,
        'imu_ahrs_match': match,
    }


def overlap_interval(imu, bag, topic):
    topic_info = bag['topics'].get(topic)
    imu_types = [value for value in imu['dtypes'].values() if value['count']]
    if not topic_info or not topic_info['count'] or not imu_types:
        return None
    imu_start = min(item['first_ns'] for item in imu_types)
    imu_end = max(item['last_ns'] for item in imu_types)
    bag_start = topic_info['first_record_ns']
    bag_end = topic_info['last_record_ns']
    start = max(imu_start, bag_start)
    end = min(imu_end, bag_end)
    return {
        'start_ns': start,
        'end_ns': end,
        'duration_s': max(0.0, (end - start) / NS_PER_SECOND),
        'imu_lead_before_bag_s': (bag_start - imu_start) / NS_PER_SECOND,
        'imu_tail_after_bag_s': (imu_end - bag_end) / NS_PER_SECOND,
    }


def build_report(args):
    bag = inspect_bag(args.bag)
    imu = inspect_imu(args.imu_csv)
    errors = []
    warnings = []

    required_topics = {
        '/fix': 'sensor_msgs/msg/NavSatFix',
        '/ubx_nav_vel_ned': 'ublox_ubx_msgs/msg/UBXNavVelNED',
    }
    for topic, expected_type in required_topics.items():
        info = bag['topics'].get(topic)
        if not info or not info['count']:
            errors.append(f'Missing required bag topic: {topic}')
        elif info['type'] != expected_type:
            errors.append(
                f'{topic} has type {info["type"]}, expected {expected_type}'
            )

    fix_time = bag.get('fix_header_range')
    if fix_time:
        for label in ('first', 'last'):
            delta = fix_time[f'{label}_record_header_delta_s']
            if abs(delta) > 2.0:
                errors.append(
                    f'Bag {label} storage time and /fix header time differ '
                    f'by {delta:.3f} s; check --use-sim-time.'
                )

    if imu['invalid_timestamp_rows']:
        errors.append(
            f'IMU CSV has {imu["invalid_timestamp_rows"]} invalid timestamps'
        )

    for dtype, meaning in (('64', 'raw IMU'), ('65', 'AHRS')):
        info = imu['dtypes'].get(dtype)
        if not info or not info['count']:
            errors.append(f'Missing dtype {dtype} ({meaning}) rows in IMU CSV')
            continue
        if info['valid_fraction'] < 1.0:
            errors.append(
                f'dtype {dtype} valid-frame fraction is '
                f'{info["valid_fraction"]:.3%}; invalid frames must not be '
                'published'
            )
        if info['invalid_numeric_count']:
            errors.append(
                f'dtype {dtype} has {info["invalid_numeric_count"]} rows '
                'with missing or non-finite required values'
            )
        if info['non_monotonic_count']:
            errors.append(
                f'dtype {dtype} has {info["non_monotonic_count"]} '
                'non-monotonic timestamps'
            )

    match = imu.get('imu_ahrs_match')
    if match and match['matched_fraction'] < 0.99:
        errors.append(
            'Less than 99% of AHRS rows have a raw IMU row within 80 ms: '
            f'{match["matched_fraction"]:.3%}'
        )

    overlap = overlap_interval(imu, bag, '/fix')
    if overlap is None or overlap['duration_s'] <= 0:
        errors.append('IMU and /fix record time ranges do not overlap')
    else:
        if overlap['duration_s'] < 60.0:
            warnings.append(
                f'Only {overlap["duration_s"]:.1f} s of overlapping data'
            )
        if overlap['imu_lead_before_bag_s'] > 1.0:
            warnings.append(
                'IMU starts '
                f'{overlap["imu_lead_before_bag_s"]:.3f} s before the bag; '
                'keep skip_rows_before_first_clock enabled'
            )

    return {
        'ok': not errors,
        'bag': bag,
        'imu': imu,
        'overlap_with_fix': overlap,
        'errors': errors,
        'warnings': warnings,
    }


def print_report(report):
    print('=== GPS/IMU Input Check ===')
    print(f'Bag DB3: {report["bag"]["db3"]}')
    for topic in ('/fix', '/ubx_nav_vel_ned'):
        info = report['bag']['topics'].get(topic)
        if info:
            rate = info['effective_rate_hz']
            rate_text = 'n/a' if rate is None else f'{rate:.3f}'
            print(f'{topic}: {info["count"]} messages, {rate_text} Hz')

    for dtype, name in (('64', 'raw IMU'), ('65', 'AHRS')):
        info = report['imu']['dtypes'].get(dtype)
        if info:
            rate = info['effective_rate_hz']
            rate_text = 'n/a' if rate is None else f'{rate:.3f}'
            print(
                f'dtype {dtype} ({name}): {info["count"]} samples, '
                f'{rate_text} Hz, '
                f'valid {info["valid_fraction"]:.3%}'
            )

    match = report['imu'].get('imu_ahrs_match')
    if match:
        print(
            'IMU/AHRS matches within 80 ms: '
            f'{match["matched_fraction"]:.3%}, '
            f'median time gap {match["median_nearest_gap_ms"]:.3f} ms'
        )

    overlap = report.get('overlap_with_fix')
    if overlap:
        print(f'Overlap duration: {overlap["duration_s"]:.3f} s')
        print(f'IMU lead before bag: {overlap["imu_lead_before_bag_s"]:.3f} s')

    for message in report['warnings']:
        print(f'[WARNING] {message}')
    for message in report['errors']:
        print(f'[ERROR] {message}')
    print('Check result:', 'PASS' if report['ok'] else 'FAIL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True)
    parser.add_argument('--imu-csv', required=True)
    parser.add_argument('--json-out', default='')
    args = parser.parse_args()

    report = build_report(args)
    print_report(report)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        print(f'JSON report: {output}')
    raise SystemExit(0 if report['ok'] else 2)


if __name__ == '__main__':
    main()
