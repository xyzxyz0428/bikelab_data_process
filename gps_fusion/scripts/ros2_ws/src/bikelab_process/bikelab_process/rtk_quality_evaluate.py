#!/usr/bin/env python3
"""Evaluate RTK status and four-way consistency by quality stratum.

RTK status and ``/fix`` come from the same receiver, so this is not an
independent ground-truth evaluation.
"""

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bikelab_process.four_way_compare_evaluate import (
    GROUP_TOPICS,
    extract_odometry,
)
from bikelab_process.fusion_result_evaluate import (
    circular_mean,
    geodetic_to_ecef,
    local_enu,
    nearest_index,
    percentile,
    read_selected_topics,
    wrap_pi,
)


PVT_TOPIC = '/ubx_nav_pvt'
HPPOS_TOPIC = '/ubx_nav_hp_pos_llh'
STATUS_NAMES = {0: 'none', 1: 'float', 2: 'fixed'}
STRATUM_NAMES = ('all', 'fixed', 'fixed_hacc', 'stable_fixed')


def finite_summary(values, unit):
    """Return descriptive statistics for finite values."""
    finite = [float(value) for value in values if math.isfinite(value)]
    return {
        'samples': len(finite),
        f'median_{unit}': percentile(finite, 0.5),
        f'p95_{unit}': percentile(finite, 0.95),
        f'max_{unit}': max(finite) if finite else None,
        f'mean_{unit}': statistics.mean(finite) if finite else None,
    }


def percentage(part, whole):
    """Return a percentage, or None when the denominator is zero."""
    return 100.0 * part / whole if whole else None


def status_name(code):
    """Map a UBX carrier-solution status code to a stable label."""
    return STATUS_NAMES.get(int(code), f'unknown_{int(code)}')


def annotate_state_runs(epochs, maximum_gap_s):
    """
    Annotate status runs and continuous residence time in place.

    A run is split by either a carrier-status change or an acquisition gap
    larger than ``maximum_gap_s``. Duration is last sample time minus first
    sample time and therefore does not extrapolate beyond observed samples.
    """
    epochs.sort(key=lambda item: item['t_s'])
    runs = []
    current = None
    previous = None
    for epoch in epochs:
        split = (
            previous is None
            or epoch['carrier_status_code']
            != previous['carrier_status_code']
            or epoch['t_s'] - previous['t_s'] > maximum_gap_s
        )
        if split:
            current = {
                'run_id': len(runs),
                'carrier_status_code': epoch['carrier_status_code'],
                'carrier_status': epoch['carrier_status'],
                'start_s': epoch['t_s'],
                'end_s': epoch['t_s'],
                'duration_s': 0.0,
                'samples': 0,
            }
            runs.append(current)
        current['end_s'] = epoch['t_s']
        current['duration_s'] = current['end_s'] - current['start_s']
        current['samples'] += 1
        epoch['run_id'] = current['run_id']
        epoch['status_residence_s'] = epoch['t_s'] - current['start_s']
        previous = epoch
    return runs


def interval_weighted_state_time(epochs, maximum_gap_s):
    """Sum adjacent observed intervals, assigning each to its first sample."""
    durations = Counter()
    excluded_gap_s = 0.0
    excluded_gaps = 0
    for left, right in zip(epochs, epochs[1:]):
        delta = right['t_s'] - left['t_s']
        if delta <= 0.0:
            continue
        if delta > maximum_gap_s:
            excluded_gap_s += delta
            excluded_gaps += 1
            continue
        durations[left['carrier_status']] += delta
    return dict(durations), excluded_gaps, excluded_gap_s


def summarize_runs(runs):
    """Summarize run count and duration by carrier state."""
    by_state = {}
    states = sorted({run['carrier_status'] for run in runs})
    for state in states:
        selected = [run for run in runs if run['carrier_status'] == state]
        durations = [run['duration_s'] for run in selected]
        by_state[state] = {
            'runs': len(selected),
            'longest_duration_s': max(durations) if durations else None,
            'median_duration_s': percentile(durations, 0.5),
        }
    return by_state


def extract_pvt_epochs(rows, maximum_gap_s):
    """Extract carrier state and validity fields from UBX-NAV-PVT."""
    epochs = []
    for header_time, record_time, message in rows:
        time_s = header_time if header_time is not None else record_time
        epochs.append({
            't_s': float(time_s),
            'itow_ms': int(message.itow),
            'carrier_status_code': int(message.carr_soln.status),
            'carrier_status': status_name(message.carr_soln.status),
            'gnss_fix_ok': bool(message.gnss_fix_ok),
            'diff_soln': bool(message.diff_soln),
            'pvt_invalid_llh': bool(message.invalid_llh),
        })
    runs = annotate_state_runs(epochs, maximum_gap_s)
    return epochs, runs


def hp_horizontal_valid(message):
    """Return whether high-precision longitude and latitude are valid."""
    return not any((
        bool(message.invalid_lon),
        bool(message.invalid_lat),
        bool(message.invalid_lon_hp),
        bool(message.invalid_lat_hp),
    ))


def match_pvt_hp(pvt_epochs, hp_rows):
    """Join HPPOSLLH to PVT by exact iTOW, choosing the nearest duplicate."""
    by_itow = defaultdict(list)
    for epoch in pvt_epochs:
        by_itow[epoch['itow_ms']].append(epoch)

    timeline = []
    unmatched_hp = 0
    matched_pvt_ids = set()
    for header_time, record_time, message in hp_rows:
        time_s = header_time if header_time is not None else record_time
        candidates = by_itow.get(int(message.itow), [])
        if not candidates:
            unmatched_hp += 1
            continue
        pvt = min(candidates, key=lambda item: abs(item['t_s'] - time_s))
        matched_pvt_ids.add(id(pvt))
        timeline.append({
            't_s': float(time_s),
            'itow_ms': int(message.itow),
            'carrier_status_code': pvt['carrier_status_code'],
            'carrier_status': pvt['carrier_status'],
            # UBX-NAV-HPPOSLLH h_acc is in 0.1 mm.
            'h_acc_m': float(message.h_acc) * 1.0e-4,
            'hp_horizontal_valid': hp_horizontal_valid(message),
            'gnss_fix_ok': pvt['gnss_fix_ok'],
            'diff_soln': pvt['diff_soln'],
            'pvt_invalid_llh': pvt['pvt_invalid_llh'],
            'pvt_header_t_s': pvt['t_s'],
            'pvt_hp_delta_s': float(time_s) - pvt['t_s'],
            'run_id': pvt['run_id'],
            'fixed_residence_s': (
                pvt['status_residence_s']
                if pvt['carrier_status'] == 'fixed' else None
            ),
        })
    timeline.sort(key=lambda item: item['t_s'])
    matching = {
        'pvt_epochs': len(pvt_epochs),
        'hpposllh_epochs': len(hp_rows),
        'exact_itow_matches': len(timeline),
        'unmatched_hpposllh_epochs': unmatched_hp,
        'pvt_epochs_without_an_itow_match': (
            len(pvt_epochs) - len(matched_pvt_ids)
        ),
        'pvt_to_hppos_time_delta_s': finite_summary(
            [item['pvt_hp_delta_s'] for item in timeline], 's'
        ),
    }
    return timeline, matching


def quality_passes(epoch, stratum, h_acc_limit_m, stable_duration_s):
    """Apply one predeclared RTK-quality selector to one epoch."""
    if stratum == 'all':
        return True
    if epoch is None or epoch['carrier_status'] != 'fixed':
        return False
    if stratum == 'fixed':
        return True
    high_accuracy = (
        epoch['hp_horizontal_valid']
        and math.isfinite(epoch['h_acc_m'])
        and epoch['h_acc_m'] <= h_acc_limit_m
    )
    if stratum == 'fixed_hacc':
        return high_accuracy
    if stratum == 'stable_fixed':
        residence = epoch.get('fixed_residence_s')
        return (
            high_accuracy
            and residence is not None
            and residence >= stable_duration_s
        )
    raise ValueError(f'Unknown stratum: {stratum}')


def extract_fix_points(rows, quality_timeline, quality_tolerance_s):
    """Convert NavSatFix positions to ENU and attach nearest RTK quality."""
    valid_rows = []
    for header_time, _, message in rows:
        if header_time is None:
            continue
        latitude = float(message.latitude)
        longitude = float(message.longitude)
        altitude = float(message.altitude)
        if not math.isfinite(latitude) or not math.isfinite(longitude):
            continue
        valid_rows.append((header_time, message, altitude))
    valid_rows.sort(key=lambda item: item[0])
    if not valid_rows:
        raise RuntimeError('Reference fix topic has no finite stamped samples')

    first = valid_rows[0]
    altitude0 = first[2] if math.isfinite(first[2]) else 0.0
    ecef = geodetic_to_ecef(
        float(first[1].latitude), float(first[1].longitude), altitude0
    )
    origin = {
        'latitude': float(first[1].latitude),
        'longitude': float(first[1].longitude),
        'altitude': altitude0,
        'ecef_x': ecef[0],
        'ecef_y': ecef[1],
        'ecef_z': ecef[2],
    }
    quality_times = [item['t_s'] for item in quality_timeline]
    points = []
    for header_time, message, altitude in valid_rows:
        if not math.isfinite(altitude):
            altitude = altitude0
        east, north = local_enu(
            float(message.latitude), float(message.longitude), altitude, origin
        )
        index = nearest_index(
            quality_times, header_time, quality_tolerance_s
        )
        points.append({
            't': header_time,
            'x': east,
            'y': north,
            'quality': quality_timeline[index] if index is not None else None,
        })
    points.sort(key=lambda item: item['t'])
    return points, origin


def nearest_pairs(reference_points, odometry_points, tolerance_s):
    """Build a global nearest-time one-to-one assignment within tolerance."""
    times = [point['t'] for point in odometry_points]
    edges = []
    for reference_index, reference in enumerate(reference_points):
        first = bisect.bisect_left(
            times, reference['t'] - tolerance_s
        )
        last = bisect.bisect_right(
            times, reference['t'] + tolerance_s
        )
        for odometry_index in range(first, last):
            edges.append((
                abs(odometry_points[odometry_index]['t'] - reference['t']),
                reference_index,
                odometry_index,
            ))
    assigned_reference = set()
    assigned_odometry = set()
    assignments = []
    for _, reference_index, odometry_index in sorted(edges):
        if (
            reference_index in assigned_reference
            or odometry_index in assigned_odometry
        ):
            continue
        assigned_reference.add(reference_index)
        assigned_odometry.add(odometry_index)
        assignments.append((
            reference_points[reference_index],
            odometry_points[odometry_index],
            odometry_index,
        ))
    assignments.sort(key=lambda item: item[0]['t'])
    return assignments, len(assigned_odometry)


def estimate_translation(pairs, sample_count):
    """Estimate the map-to-ENU translation from initial paired epochs."""
    subset = pairs[:min(sample_count, len(pairs))]
    if not subset:
        return None
    return {
        'offset_x_m': statistics.median([
            odometry['x'] - reference['x']
            for reference, odometry, _ in subset
        ]),
        'offset_y_m': statistics.median([
            odometry['y'] - reference['y']
            for reference, odometry, _ in subset
        ]),
        'calibration_pairs': len(subset),
    }


def position_consistency(pairs, unique_odometry, translation):
    """Summarize translation-aligned planar consistency against /fix."""
    if not translation:
        return None
    east_errors = []
    north_errors = []
    planar_errors = []
    time_differences = []
    for reference, odometry, _ in pairs:
        east_error = (
            odometry['x'] - translation['offset_x_m'] - reference['x']
        )
        north_error = (
            odometry['y'] - translation['offset_y_m'] - reference['y']
        )
        east_errors.append(east_error)
        north_errors.append(north_error)
        planar_errors.append(math.hypot(east_error, north_error))
        time_differences.append(odometry['t'] - reference['t'])
    planar = finite_summary(planar_errors, 'm')
    planar['rmse_m'] = (
        math.sqrt(statistics.mean([value * value for value in planar_errors]))
        if planar_errors else None
    )
    return {
        'paired_reference_epochs': len(pairs),
        'unique_odometry_samples': unique_odometry,
        'pairing_time_difference_s': finite_summary(
            time_differences, 's'
        ),
        'planar_error': planar,
        'signed_east_error_m': finite_summary(east_errors, 'm'),
        'signed_north_error_m': finite_summary(north_errors, 'm'),
        'alignment': {
            **translation,
            'method': 'fixed translation from first all-stratum pairs',
            'rotation_applied': False,
        },
    }


def build_stratified_courses(
        fix_points, stratum, half_window, minimum_speed,
        h_acc_limit_m, stable_duration_s):
    """Build courses whose complete fix window passes a quality selector."""
    courses = []
    for index in range(half_window, len(fix_points) - half_window):
        window = fix_points[index - half_window:index + half_window + 1]
        if not all(
            quality_passes(
                point['quality'], stratum, h_acc_limit_m, stable_duration_s
            )
            for point in window
        ):
            continue
        before = window[0]
        center = fix_points[index]
        after = window[-1]
        delta_t = after['t'] - before['t']
        if delta_t <= 0.0:
            continue
        east = after['x'] - before['x']
        north = after['y'] - before['y']
        speed = math.hypot(east, north) / delta_t
        if speed < minimum_speed:
            continue
        courses.append({
            't': center['t'],
            'course': math.atan2(north, east),
            'speed': speed,
        })
    return courses


def yaw_consistency(courses, odometry_points, tolerance_s):
    """Compare odometry yaw with one-to-one nearest fix-derived course."""
    errors = []
    time_differences = []
    pairs, unique = nearest_pairs(courses, odometry_points, tolerance_s)
    for course, odometry, _ in pairs:
        time_differences.append(odometry['t'] - course['t'])
        errors.append(wrap_pi(
            odometry['yaw'] - course['course']
        ))
    absolute_degrees = [abs(math.degrees(error)) for error in errors]
    bias = circular_mean(errors)
    return {
        'paired_course_epochs': len(errors),
        'unique_odometry_samples': unique,
        'circular_bias_deg': math.degrees(bias) if bias is not None else None,
        'median_abs_error_deg': percentile(absolute_degrees, 0.5),
        'p95_abs_error_deg': percentile(absolute_degrees, 0.95),
        'max_abs_error_deg': (
            max(absolute_degrees) if absolute_degrees else None
        ),
        'pairing_time_difference_s': finite_summary(
            time_differences, 's'
        ),
    }


def hacc_by_state(timeline, state=None):
    """Summarize valid horizontal-accuracy estimates by state."""
    values = [
        item['h_acc_m'] for item in timeline
        if item['hp_horizontal_valid']
        and (state is None or item['carrier_status'] == state)
    ]
    return finite_summary(values, 'm')


def count_quality(timeline, h_acc_limit_m, stable_duration_s):
    """Count epochs in the four quality strata."""
    return {
        stratum: sum(
            quality_passes(
                epoch, stratum, h_acc_limit_m, stable_duration_s
            )
            for epoch in timeline
        )
        for stratum in STRATUM_NAMES
    }


def interval_weighted_quality_time(
        timeline, h_acc_limit_m, stable_duration_s, maximum_gap_s,
        start_s=None, end_s=None):
    """Measure quality-stratum time using clipped adjacent intervals."""
    durations = Counter()
    for left, right in zip(timeline, timeline[1:]):
        raw_delta = right['t_s'] - left['t_s']
        if raw_delta <= 0.0 or raw_delta > maximum_gap_s:
            continue
        interval_start = left['t_s']
        interval_end = right['t_s']
        if start_s is not None:
            interval_start = max(interval_start, start_s)
        if end_s is not None:
            interval_end = min(interval_end, end_s)
        delta = interval_end - interval_start
        if delta <= 0.0:
            continue
        for stratum in STRATUM_NAMES:
            if quality_passes(
                    left, stratum, h_acc_limit_m, stable_duration_s):
                durations[stratum] += delta
    return {
        stratum: float(durations[stratum]) for stratum in STRATUM_NAMES
    }


def build_result(args):
    """Read both bags and construct RTK and stratified diagnostics."""
    _, raw_gnss = read_selected_topics(
        args.gnss_bag, args.gnss_storage_id, [PVT_TOPIC, HPPOS_TOPIC]
    )
    missing_gnss = [
        topic for topic in (PVT_TOPIC, HPPOS_TOPIC)
        if not raw_gnss.get(topic)
    ]
    if missing_gnss:
        raise RuntimeError(
            'Missing required raw GNSS topics: ' + ', '.join(missing_gnss)
        )

    result_topics = [args.fix_topic, *GROUP_TOPICS.values()]
    _, raw_result = read_selected_topics(
        args.result_bag, args.result_storage_id, result_topics
    )
    missing_results = [
        topic for topic in result_topics if not raw_result.get(topic)
    ]
    if missing_results:
        raise RuntimeError(
            'Missing required four-way result topics: '
            + ', '.join(missing_results)
        )

    pvt_epochs, runs = extract_pvt_epochs(
        raw_gnss[PVT_TOPIC], args.maximum_state_gap
    )
    timeline, matching = match_pvt_hp(
        pvt_epochs, raw_gnss[HPPOS_TOPIC]
    )
    if not timeline:
        raise RuntimeError(
            'PVT and HPPOSLLH have no exact iTOW matches'
        )
    for epoch in timeline:
        epoch['stable_fixed_high_quality'] = quality_passes(
            epoch, 'stable_fixed', args.h_acc_limit,
            args.stable_fixed_duration,
        )

    fix_points, origin = extract_fix_points(
        raw_result[args.fix_topic], timeline, args.quality_time_tolerance
    )
    groups = {
        name: extract_odometry(raw_result[topic])
        for name, topic in GROUP_TOPICS.items()
    }
    empty_groups = [name for name, points in groups.items() if not points]
    if empty_groups:
        raise RuntimeError(
            'Four-way result topics have no finite stamped odometry: '
            + ', '.join(empty_groups)
        )
    first_common = max(
        [fix_points[0]['t']]
        + [points[0]['t'] for points in groups.values()]
    )
    last_common = min(
        [fix_points[-1]['t']]
        + [points[-1]['t'] for points in groups.values()]
    )
    if last_common <= first_common:
        raise RuntimeError('Four-way streams have no common result interval')
    common_fix = [
        point for point in fix_points
        if first_common <= point['t'] <= last_common
    ]
    for epoch in timeline:
        epoch['in_common_interval'] = (
            first_common <= epoch['t_s'] <= last_common
        )

    # Fix one translation per group using the all-quality common interval,
    # then reuse it in every stratum so quality filtering cannot recalibrate
    # away differences.
    translations = {}
    for name, points in groups.items():
        pairs, _ = nearest_pairs(
            common_fix, points, args.position_time_tolerance
        )
        translations[name] = estimate_translation(
            pairs, args.alignment_samples
        )

    strata = {}
    for stratum in STRATUM_NAMES:
        selected_fix = [
            point for point in common_fix
            if quality_passes(
                point['quality'], stratum, args.h_acc_limit,
                args.stable_fixed_duration,
            )
        ]
        courses = build_stratified_courses(
            fix_points, stratum, args.course_half_window,
            args.minimum_course_speed, args.h_acc_limit,
            args.stable_fixed_duration,
        )
        courses = [
            item for item in courses
            if first_common <= item['t'] <= last_common
        ]
        group_metrics = {}
        for name, points in groups.items():
            pairs, unique = nearest_pairs(
                selected_fix, points, args.position_time_tolerance
            )
            group_metrics[name] = {
                'topic': GROUP_TOPICS[name],
                'position_vs_fix_enu': position_consistency(
                    pairs, unique, translations[name]
                ),
                'yaw_vs_fix_derived_course': (
                    None if name == 'gps_position' else yaw_consistency(
                        courses, points, args.yaw_time_tolerance
                    )
                ),
            }
        strata[stratum] = {
            'selector': {
                'carrier_status': (
                    'any' if stratum == 'all' else 'fixed'
                ),
                'h_acc_max_m': (
                    args.h_acc_limit
                    if stratum in ('fixed_hacc', 'stable_fixed') else None
                ),
                'minimum_continuous_fixed_residence_s': (
                    args.stable_fixed_duration
                    if stratum == 'stable_fixed' else None
                ),
                'position_requires_center_epoch_quality': True,
                'course_requires_all_window_epochs_quality': True,
            },
            'reference_fix_samples': len(selected_fix),
            'fix_derived_course_samples': len(courses),
            'groups': group_metrics,
        }

    observed_state_counts = Counter(
        epoch['carrier_status'] for epoch in pvt_epochs
    )
    state_time, excluded_gaps, excluded_gap_s = (
        interval_weighted_state_time(pvt_epochs, args.maximum_state_gap)
    )
    total_state_time = sum(state_time.values())
    timeline_counts = count_quality(
        timeline, args.h_acc_limit, args.stable_fixed_duration
    )
    timeline_time = interval_weighted_quality_time(
        timeline, args.h_acc_limit, args.stable_fixed_duration,
        args.maximum_state_gap,
    )
    common_timeline = [
        epoch for epoch in timeline if epoch['in_common_interval']
    ]
    common_counts = count_quality(
        common_timeline, args.h_acc_limit, args.stable_fixed_duration
    )
    common_time = interval_weighted_quality_time(
        timeline, args.h_acc_limit, args.stable_fixed_duration,
        args.maximum_state_gap, first_common, last_common,
    )
    quality_match_count = sum(
        point['quality'] is not None for point in common_fix
    )

    known_states = list(STATUS_NAMES.values())
    other_states = sorted(set(observed_state_counts) - set(known_states))
    reported_states = known_states + other_states
    state_counts = {
        state: int(observed_state_counts[state]) for state in reported_states
    }
    reported_state_time = {
        state: float(state_time.get(state, 0.0))
        for state in reported_states
    }
    public_timeline = [{
        key: value for key, value in epoch.items()
        if key not in ('run_id', 'gnss_fix_ok', 'diff_soln',
                       'pvt_invalid_llh', 'pvt_header_t_s')
    } for epoch in timeline]
    return {
        'schema_version': '1.0',
        'reference_interpretation': {
            'role': 'Same-receiver RTK-fixed pseudo-reference',
            'independent_ground_truth': False,
            'reason': (
                '/fix and RTK status come from the same receiver; GNSS motion '
                'course is not body-yaw ground truth.'
            ),
        },
        'inputs': {
            'gnss_bag': str(Path(args.gnss_bag).resolve()),
            'four_way_result_bag': str(Path(args.result_bag).resolve()),
            'reference_fix_topic': args.fix_topic,
            'pvt_topic': PVT_TOPIC,
            'hpposllh_topic': HPPOS_TOPIC,
            'group_topics': GROUP_TOPICS,
        },
        'parameters': {
            'h_acc_limit_m': args.h_acc_limit,
            'stable_fixed_duration_s': args.stable_fixed_duration,
            'maximum_state_gap_s': args.maximum_state_gap,
            'quality_time_tolerance_s': args.quality_time_tolerance,
            'position_time_tolerance_s': args.position_time_tolerance,
            'yaw_time_tolerance_s': args.yaw_time_tolerance,
            'minimum_course_speed_mps': args.minimum_course_speed,
            'course_half_window_samples': args.course_half_window,
            'alignment_samples': args.alignment_samples,
        },
        'rtk_quality': {
            'pvt_epoch_count': len(pvt_epochs),
            'hpposllh_epoch_count': len(raw_gnss[HPPOS_TOPIC]),
            'pvt_hppos_matching': matching,
            'state_counts': state_counts,
            'state_percent_samples': {
                state: percentage(count, len(pvt_epochs))
                for state, count in state_counts.items()
            },
            'state_time_s': reported_state_time,
            'state_percent_observed_time': {
                state: percentage(duration, total_state_time)
                for state, duration in reported_state_time.items()
            },
            'time_accounting': {
                'method': 'adjacent interval assigned to first sample',
                'observed_time_s': total_state_time,
                'excluded_large_gaps': excluded_gaps,
                'excluded_large_gap_time_s': excluded_gap_s,
            },
            'runs': runs,
            'runs_by_state': summarize_runs(runs),
            'h_acc_m': {
                'all_valid_hppos': hacc_by_state(timeline),
                'fixed': hacc_by_state(timeline, 'fixed'),
                'float': hacc_by_state(timeline, 'float'),
            },
            'quality_epoch_counts': timeline_counts,
            'quality_time_s': timeline_time,
            'quality_percent_observed_time': {
                stratum: percentage(duration, timeline_time['all'])
                for stratum, duration in timeline_time.items()
            },
            'stable_fixed_high_quality_coverage_percent': percentage(
                timeline_counts['stable_fixed'], len(timeline)
            ),
            'stable_fixed_high_quality_time_coverage_percent': percentage(
                timeline_time['stable_fixed'], timeline_time['all']
            ),
        },
        'common_result_interval': {
            'start_s': first_common,
            'end_s': last_common,
            'duration_s': last_common - first_common,
            'reference_fix_samples': len(common_fix),
            'reference_fix_quality_matches': quality_match_count,
            'reference_fix_quality_match_percent': percentage(
                quality_match_count, len(common_fix)
            ),
            'quality_epoch_counts': common_counts,
            'quality_time_s': common_time,
            'quality_percent_observed_time': {
                stratum: percentage(duration, common_time['all'])
                for stratum, duration in common_time.items()
            },
            'stable_fixed_high_quality_coverage_percent': percentage(
                common_counts['stable_fixed'], len(common_timeline)
            ),
            'stable_fixed_high_quality_time_coverage_percent': percentage(
                common_time['stable_fixed'], common_time['all']
            ),
        },
        'reference_fix_origin': {
            key: origin[key] for key in ('latitude', 'longitude', 'altitude')
        },
        'strata': strata,
        'quality_timeline': public_timeline,
        'warnings': [
            'RTK fixed is a pseudo-reference, not independent ground truth.',
            'h_acc is a receiver-estimated accuracy, not measured error.',
            'Position metrics reuse the /fix input and measure internal '
            'consistency only.',
            'Yaw metrics compare body-yaw estimates with single-antenna '
            'direction of motion, not independent body-yaw truth.',
        ],
    }


def write_timeline_csv(path, timeline):
    """Write the quality timeline as CSV."""
    fields = [
        't_s', 'itow_ms', 'carrier_status_code', 'carrier_status',
        'h_acc_m', 'hp_horizontal_valid', 'fixed_residence_s',
        'stable_fixed_high_quality', 'in_common_interval',
        'pvt_hp_delta_s',
    ]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in timeline:
            writer.writerow({field: item.get(field) for field in fields})


def print_summary(result):
    """Print the RTK stratification summary."""
    quality = result['rtk_quality']
    counts = quality['state_counts']
    print('=== RTK-Stratified Test-Segment Evaluation ===')
    print(
        'PVT carrier states: '
        + ', '.join(f'{name}={count}' for name, count in counts.items())
    )
    fixed_hacc = quality['h_acc_m']['fixed']
    median_hacc = fixed_hacc['median_m']
    p95_hacc = fixed_hacc['p95_m']
    median_text = 'n/a' if median_hacc is None else f'{median_hacc:.4f} m'
    p95_text = 'n/a' if p95_hacc is None else f'{p95_hacc:.4f} m'
    print(
        f'RTK-fixed hAcc: median={median_text}, P95={p95_text}'
    )
    common = result['common_result_interval']
    print(
        f'Common interval: {common["duration_s"]:.3f} s, '
        f'{common["reference_fix_samples"]} reference fixes'
    )
    for name in STRATUM_NAMES:
        item = result['strata'][name]
        print(
            f'  {name}: fixes={item["reference_fix_samples"]}, '
            f'courses={item["fix_derived_course_samples"]}'
        )
    print(
        'WARNING: RTK fixed is a same-receiver pseudo-reference, not '
        'independent ground truth.'
    )


def output_paths_are_new(json_path, csv_path):
    """Reject existing output paths before reading the bags."""
    paths = [json_path] + ([csv_path] if csv_path else [])
    duplicates = len({path.resolve() for path in paths}) != len(paths)
    if duplicates:
        raise RuntimeError('JSON and CSV outputs must be different paths')
    existing = [path for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            'Refusing to overwrite existing output: '
            + ', '.join(str(path) for path in existing)
        )


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gnss-bag', required=True)
    parser.add_argument('--result-bag', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--csv', default='')
    parser.add_argument('--gnss-storage-id', default='sqlite3')
    parser.add_argument('--result-storage-id', default='sqlite3')
    parser.add_argument('--fix-topic', default='/fix/fusion')
    parser.add_argument('--h-acc-limit', type=float, default=0.1)
    parser.add_argument('--stable-fixed-duration', type=float, default=5.0)
    parser.add_argument('--maximum-state-gap', type=float, default=1.0)
    parser.add_argument('--quality-time-tolerance', type=float, default=0.05)
    parser.add_argument('--position-time-tolerance', type=float, default=0.25)
    parser.add_argument('--yaw-time-tolerance', type=float, default=0.25)
    parser.add_argument('--minimum-course-speed', type=float, default=2.0)
    parser.add_argument('--course-half-window', type=int, default=5)
    parser.add_argument('--alignment-samples', type=int, default=20)
    args = parser.parse_args()
    if args.h_acc_limit < 0.0:
        parser.error('--h-acc-limit must be non-negative')
    if args.stable_fixed_duration < 0.0:
        parser.error('--stable-fixed-duration must be non-negative')
    if args.maximum_state_gap <= 0.0:
        parser.error('--maximum-state-gap must be positive')
    if args.course_half_window < 1:
        parser.error('--course-half-window must be at least 1')
    if args.alignment_samples < 1:
        parser.error('--alignment-samples must be at least 1')
    return args


def main():
    """Evaluate bags and write new JSON and optional epoch CSV outputs."""
    args = parse_args()
    json_path = Path(args.out).expanduser()
    csv_path = Path(args.csv).expanduser() if args.csv else None
    try:
        output_paths_are_new(json_path, csv_path)
        result = build_result(args)
        # Recheck after bag reads to avoid overwrite races.
        output_paths_are_new(json_path, csv_path)
    except RuntimeError as error:
        raise SystemExit(f'[ERROR] {error}') from None

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_timeline_csv(csv_path, result['quality_timeline'])
    print_summary(result)
    print(f'JSON report: {json_path}')
    if csv_path:
        print(f'Quality timeline CSV: {csv_path}')


if __name__ == '__main__':
    main()
