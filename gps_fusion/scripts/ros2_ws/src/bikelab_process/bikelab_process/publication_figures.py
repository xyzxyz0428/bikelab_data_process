#!/usr/bin/env python3
"""Generate comparison figures from a four-way fusion result bag."""

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path

from bikelab_process.four_way_compare_evaluate import GROUP_TOPICS
from bikelab_process.fusion_result_evaluate import (
    build_courses,
    geodetic_to_ecef,
    local_enu,
    match_trajectory,
    nearest_index,
    percentile,
    quaternion_yaw,
    read_selected_topics,
    wrap_pi,
)

import cairo


FIX_TOPIC_DEFAULT = '/fix/fusion'
RAW_RATE_TOPIC = '/imu/raw_gyro_rate'
AHRS_RATE_TOPIC = '/imu/ahrs_heading_rate'
PVT_TOPIC = '/ubx_nav_pvt'
HP_TOPIC = '/ubx_nav_hp_pos_llh'
POSITION_TIME_TOLERANCE_S = 0.25
POSITION_ALIGNMENT_INITIAL_SAMPLES = 20

GROUP_LABELS = {
    'gps_position': 'Group 1: GNSS position only',
    'gps_course': 'Group 2: EKF with GNSS position + COG',
    'gps_course_raw_gyro': 'Group 3: Group 2 + raw-gyro yaw rate',
    'gps_course_ahrs_rate': 'Group 4: Group 2 + AHRS yaw rate',
}
GROUP_COLORS = {
    'gps_position': '#333333',
    'gps_course': '#0072B2',
    'gps_course_raw_gyro': '#D55E00',
    'gps_course_ahrs_rate': '#009E73',
}
LINE_STYLES = {
    'gps_position': [10.0, 6.0],
    'gps_course': [3.0, 5.0],
    'gps_course_raw_gyro': [],
    'gps_course_ahrs_rate': [12.0, 4.0, 2.0, 4.0],
}

WIDTH = 1600
HEIGHT = 1000
MARGIN_LEFT = 145
MARGIN_RIGHT = 75
MARGIN_TOP = 105
MARGIN_BOTTOM = 125


def finite(values):
    """Return finite float values only."""
    return [float(value) for value in values if math.isfinite(float(value))]


def stamp_seconds(stamp):
    """Convert a ROS stamp to floating-point seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def downsample(rows, maximum):
    """Uniformly reduce a sequence while retaining both endpoints."""
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    step = (len(rows) - 1) / float(maximum - 1)
    indices = sorted({round(index * step) for index in range(maximum)})
    return [rows[index] for index in indices]


def nice_step(span, target=6):
    """Return a human-friendly axis tick step."""
    if not math.isfinite(span) or span <= 0.0:
        return 1.0
    raw = span / max(target, 1)
    power = 10.0 ** math.floor(math.log10(raw))
    scaled = raw / power
    if scaled <= 1.0:
        factor = 1.0
    elif scaled <= 2.0:
        factor = 2.0
    elif scaled <= 5.0:
        factor = 5.0
    else:
        factor = 10.0
    return factor * power


def ticks(lower, upper, target=6):
    """Build inclusive, evenly spaced nice ticks."""
    step = nice_step(upper - lower, target)
    first = math.ceil(lower / step - 1.0e-12) * step
    values = []
    value = first
    while value <= upper + step * 1.0e-9 and len(values) < 100:
        values.append(value)
        value += step
    return values


def tick_text(value, span):
    """Format an axis value based on its displayed span."""
    if math.isclose(value, 0.0, abs_tol=max(abs(span), 1.0) * 1.0e-12):
        value = 0.0
    absolute = abs(value)
    if absolute >= 1.0e5 or (absolute and absolute < 1.0e-3):
        return f'{value:.2e}'
    if span < 1.0:
        return f'{value:.2f}'
    if span < 10.0:
        return f'{value:.1f}'
    return f'{value:.0f}'


def expanded_limits(values, fraction=0.05, fallback=(-1.0, 1.0)):
    """Return finite data limits with a proportional margin."""
    valid = finite(values)
    if not valid:
        return fallback
    lower = min(valid)
    upper = max(valid)
    span = upper - lower
    if span <= 1.0e-12:
        span = max(abs(lower), 1.0)
    return lower - fraction * span, upper + fraction * span


def equal_aspect_limits(xlim, ylim):
    """Expand limits so metres use the same scale in x and y."""
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    target_ratio = plot_width / plot_height
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]
    if xspan / yspan < target_ratio:
        wanted = yspan * target_ratio
        center = sum(xlim) / 2.0
        xlim = center - wanted / 2.0, center + wanted / 2.0
    else:
        wanted = xspan / target_ratio
        center = sum(ylim) / 2.0
        ylim = center - wanted / 2.0, center + wanted / 2.0
    return xlim, ylim


def set_source_hex(context, color, alpha=1.0):
    """Set a cairo source from a CSS-style hex color."""
    color = color.lstrip('#')
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    context.set_source_rgba(red, green, blue, alpha)


def text(context, x, y, value, size=25, anchor='left', color='#222222'):
    """Draw one line of text with simple horizontal anchoring."""
    context.save()
    context.select_font_face(
        'DejaVu Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
    )
    context.set_font_size(size)
    set_source_hex(context, color)
    extents = context.text_extents(str(value))
    if anchor == 'center':
        x -= extents.width / 2.0 + extents.x_bearing
    elif anchor == 'right':
        x -= extents.width + extents.x_bearing
    context.move_to(x, y)
    context.show_text(str(value))
    context.restore()


class Axes:
    """Cartesian axes drawn with Cairo."""

    def __init__(
        self, context, xlim, ylim, title, xlabel, ylabel,
        xtick_values=None, ytick_values=None,
        xtick_formatter=None, ytick_formatter=None,
    ):
        """Initialize and draw one set of Cartesian axes."""
        self.context = context
        self.xlim = xlim
        self.ylim = ylim
        self.left = MARGIN_LEFT
        self.right = WIDTH - MARGIN_RIGHT
        self.top = MARGIN_TOP
        self.bottom = HEIGHT - MARGIN_BOTTOM
        self.xtick_values = xtick_values
        self.ytick_values = ytick_values
        self.xtick_formatter = xtick_formatter
        self.ytick_formatter = ytick_formatter
        self._draw(title, xlabel, ylabel)

    def project(self, x, y):
        """Map data coordinates to surface coordinates."""
        px = self.left + (
            (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
            * (self.right - self.left)
        )
        py = self.bottom - (
            (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
            * (self.bottom - self.top)
        )
        return px, py

    def _draw(self, title_value, xlabel, ylabel):
        context = self.context
        context.set_line_width(1.0)
        xvalues = self.xtick_values or ticks(*self.xlim)
        for value in xvalues:
            x, _ = self.project(value, self.ylim[0])
            set_source_hex(context, '#E3E3E3')
            context.move_to(x, self.top)
            context.line_to(x, self.bottom)
            context.stroke()
            label = (
                self.xtick_formatter(value)
                if self.xtick_formatter else
                tick_text(value, self.xlim[1] - self.xlim[0])
            )
            text(
                context, x, self.bottom + 36,
                label,
                21, 'center', '#333333',
            )
        yvalues = self.ytick_values or ticks(*self.ylim)
        for value in yvalues:
            _, y = self.project(self.xlim[0], value)
            set_source_hex(context, '#E3E3E3')
            context.move_to(self.left, y)
            context.line_to(self.right, y)
            context.stroke()
            label = (
                self.ytick_formatter(value)
                if self.ytick_formatter else
                tick_text(value, self.ylim[1] - self.ylim[0])
            )
            text(
                context, self.left - 18, y + 7,
                label,
                21, 'right', '#333333',
            )
        set_source_hex(context, '#555555')
        context.set_line_width(1.7)
        context.rectangle(
            self.left, self.top,
            self.right - self.left, self.bottom - self.top,
        )
        context.stroke()
        text(context, WIDTH / 2.0, 58, title_value, 31, 'center')
        text(
            context, (self.left + self.right) / 2.0,
            HEIGHT - 35, xlabel, 25, 'center',
        )
        context.save()
        context.translate(38, (self.top + self.bottom) / 2.0)
        context.rotate(-math.pi / 2.0)
        text(context, 0, 0, ylabel, 25, 'center')
        context.restore()

    def line(self, points, color, width=3.0, dash=None, alpha=1.0):
        """Draw a clipped polyline."""
        valid = [
            (float(x), float(y)) for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if len(valid) < 2:
            return
        context = self.context
        context.save()
        context.rectangle(
            self.left, self.top,
            self.right - self.left, self.bottom - self.top,
        )
        context.clip()
        set_source_hex(context, color, alpha)
        context.set_line_width(width)
        context.set_line_join(cairo.LINE_JOIN_ROUND)
        context.set_line_cap(cairo.LINE_CAP_ROUND)
        context.set_dash(dash or [])
        first = True
        for x, y in valid:
            px, py = self.project(x, y)
            if first:
                context.move_to(px, py)
                first = False
            else:
                context.line_to(px, py)
        context.stroke()
        context.restore()

    def scatter(self, points, color, radius=3.0, alpha=0.45):
        """Draw clipped circular scatter points."""
        context = self.context
        context.save()
        context.rectangle(
            self.left, self.top,
            self.right - self.left, self.bottom - self.top,
        )
        context.clip()
        set_source_hex(context, color, alpha)
        for x, y in points:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            px, py = self.project(x, y)
            context.arc(px, py, radius, 0.0, 2.0 * math.pi)
            context.fill()
        context.restore()

    def horizontal(self, y, color='#777777', dash=None, width=1.5):
        """Draw a horizontal reference line."""
        self.line(
            [(self.xlim[0], y), (self.xlim[1], y)],
            color, width, dash or [7.0, 5.0],
        )

    def legend(self, entries, x=None, y=None):
        """Draw a white-backed line legend."""
        if not entries:
            return
        context = self.context
        x = x if x is not None else self.right - 470
        y = y if y is not None else self.top + 28
        height = 43 * len(entries) + 24
        context.save()
        set_source_hex(context, '#FFFFFF', 0.91)
        context.rectangle(x - 20, y - 29, 455, height)
        context.fill_preserve()
        set_source_hex(context, '#BBBBBB')
        context.set_line_width(1.0)
        context.stroke()
        for index, (label, color, dash) in enumerate(entries):
            line_y = y + index * 43
            set_source_hex(context, color)
            context.set_line_width(4.0)
            context.set_dash(dash or [])
            context.move_to(x, line_y)
            context.line_to(x + 65, line_y)
            context.stroke()
            text(context, x + 82, line_y + 8, label, 21)
        context.restore()


def render(base_path, draw):
    """Render one drawing callback to both SVG and PNG."""
    outputs = []
    svg_path = base_path.with_suffix('.svg')
    surface = cairo.SVGSurface(str(svg_path), WIDTH, HEIGHT)
    context = cairo.Context(surface)
    draw(context)
    surface.finish()
    outputs.append(svg_path)

    png_path = base_path.with_suffix('.png')
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    context = cairo.Context(surface)
    draw(context)
    surface.write_to_png(str(png_path))
    surface.finish()
    outputs.append(png_path)
    return outputs


def white_background(context):
    """Fill the plotting surface with white."""
    context.set_source_rgb(1.0, 1.0, 1.0)
    context.paint()


def extract_fix_points(rows):
    """Convert NavSatFix rows to sorted WGS84 local-ENU points."""
    valid = []
    for header_time, _, message in rows:
        latitude = float(message.latitude)
        longitude = float(message.longitude)
        altitude = float(message.altitude)
        if header_time is None or not all(
            math.isfinite(value) for value in (latitude, longitude)
        ):
            continue
        if not math.isfinite(altitude):
            altitude = 0.0
        valid.append((header_time, latitude, longitude, altitude))
    valid.sort(key=lambda item: item[0])
    if not valid:
        raise RuntimeError(
            'The selected reference fix topic has no valid data'
        )
    first = valid[0]
    ecef = geodetic_to_ecef(first[1], first[2], first[3])
    origin = {
        'latitude': first[1],
        'longitude': first[2],
        'altitude': first[3],
        'ecef_x': ecef[0],
        'ecef_y': ecef[1],
        'ecef_z': ecef[2],
    }
    points = []
    for time_value, latitude, longitude, altitude in valid:
        east, north = local_enu(latitude, longitude, altitude, origin)
        points.append({'t': time_value, 'x': east, 'y': north})
    return points


def extract_odometry(rows):
    """Convert Odometry bag rows to sorted planar samples."""
    points = []
    for header_time, _, message in rows:
        if header_time is None:
            continue
        values = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            quaternion_yaw(message.pose.pose.orientation),
        )
        if not all(math.isfinite(value) for value in values):
            continue
        points.append({
            't': header_time,
            'x': values[0],
            'y': values[1],
            'yaw': values[2],
        })
    return sorted(points, key=lambda point: point['t'])


def course_errors(courses, odometry, tolerance):
    """Return signed yaw-minus-course discrepancies with timestamps."""
    times = [point['t'] for point in odometry]
    errors = []
    for course in courses:
        index = nearest_index(times, course['t'], tolerance)
        if index is None:
            continue
        error = math.degrees(
            wrap_pi(odometry[index]['yaw'] - course['course'])
        )
        errors.append({
            't': course['t'],
            'error_deg': error,
            'abs_error_deg': abs(error),
            'speed_m_s': course['speed'],
        })
    return errors


def translation_only_errors(fix_points, odometry, tolerance):
    """Return planar distances after per-group x/y offset removal."""
    matches = match_trajectory(fix_points, odometry, tolerance)
    if not matches:
        return [], None, None
    subset = matches[:min(POSITION_ALIGNMENT_INITIAL_SAMPLES, len(matches))]
    offset_x = statistics.median([
        odometry_point['x'] - fix_point['x']
        for fix_point, odometry_point in subset
    ])
    offset_y = statistics.median([
        odometry_point['y'] - fix_point['y']
        for fix_point, odometry_point in subset
    ])
    rows = []
    for fix_point, odometry_point in matches:
        aligned_x = odometry_point['x'] - offset_x
        aligned_y = odometry_point['y'] - offset_y
        rows.append({
            't': odometry_point['t'],
            'gnss_input_east_m': fix_point['x'],
            'gnss_input_north_m': fix_point['y'],
            'aligned_output_east_m': aligned_x,
            'aligned_output_north_m': aligned_y,
            'distance_m': math.hypot(
                aligned_x - fix_point['x'],
                aligned_y - fix_point['y'],
            ),
        })
    return rows, offset_x, offset_y


def pair_rates(raw_rows, ahrs_rows, tolerance):
    """Pair raw gyro and AHRS heading-rate samples by nearest timestamp."""
    ahrs = sorted(
        (time_value, float(message.angular_velocity.z))
        for time_value, _, message in ahrs_rows
        if time_value is not None
        and math.isfinite(float(message.angular_velocity.z))
    )
    ahrs_times = [item[0] for item in ahrs]
    pairs = []
    for time_value, _, message in raw_rows:
        if time_value is None or not ahrs:
            continue
        raw_value = float(message.angular_velocity.z)
        if not math.isfinite(raw_value):
            continue
        position = bisect.bisect_left(ahrs_times, time_value)
        candidates = []
        if position < len(ahrs):
            candidates.append(position)
        if position > 0:
            candidates.append(position - 1)
        best = min(candidates, key=lambda i: abs(ahrs[i][0] - time_value))
        if abs(ahrs[best][0] - time_value) > tolerance:
            continue
        pairs.append({
            't': time_value,
            'raw_rad_s': raw_value,
            'ahrs_rad_s': ahrs[best][1],
            'difference_rad_s': raw_value - ahrs[best][1],
        })
    return pairs


def automatic_zoom(
    trajectories, size_m, tolerance=0.25,
    minimum_turn_angle_deg=20.0, turn_half_window_s=2.0,
):
    """Select the turn with the largest simultaneous inter-group spread."""
    available = {
        name: points for name, points in trajectories.items() if points
    }
    if not available:
        raise RuntimeError('No trajectories available for automatic zoom')
    base_name = max(available, key=lambda name: len(available[name]))
    base_full = available[base_name]
    base = downsample(base_full, 3000)
    base_times = [point['t'] for point in base_full]
    time_maps = {
        name: [point['t'] for point in points]
        for name, points in available.items()
    }
    best_score = -1.0
    best_center = None
    best_time = None
    best_turn_angle = None
    for point in base:
        before_position = bisect.bisect_left(
            base_times, point['t'] - turn_half_window_s
        )
        after_position = bisect.bisect_left(
            base_times, point['t'] + turn_half_window_s
        )
        if (
            before_position >= len(base_full)
            or after_position >= len(base_full)
        ):
            continue
        before = base_full[before_position]
        after = base_full[after_position]
        incoming_distance = math.hypot(
            point['x'] - before['x'], point['y'] - before['y']
        )
        outgoing_distance = math.hypot(
            after['x'] - point['x'], after['y'] - point['y']
        )
        if min(incoming_distance, outgoing_distance) < 2.0:
            continue
        incoming_heading = math.atan2(
            point['y'] - before['y'], point['x'] - before['x']
        )
        outgoing_heading = math.atan2(
            after['y'] - point['y'], after['x'] - point['x']
        )
        turn_angle = abs(math.degrees(wrap_pi(
            outgoing_heading - incoming_heading
        )))
        if turn_angle < minimum_turn_angle_deg:
            continue
        simultaneous = []
        for name, points in available.items():
            index = nearest_index(time_maps[name], point['t'], tolerance)
            if index is not None:
                simultaneous.append(points[index])
        if len(simultaneous) < 2:
            continue
        score = max(
            math.hypot(left['x'] - right['x'], left['y'] - right['y'])
            for left_index, left in enumerate(simultaneous)
            for right in simultaneous[left_index + 1:]
        )
        if score > best_score:
            best_score = score
            best_center = (
                statistics.mean(item['x'] for item in simultaneous),
                statistics.mean(item['y'] for item in simultaneous),
            )
            best_time = point['t']
            best_turn_angle = turn_angle
    if best_center is None:
        all_points = [
            point for points in available.values() for point in points
        ]
        best_center = (
            statistics.median(point['x'] for point in all_points),
            statistics.median(point['y'] for point in all_points),
        )
        best_score = None
        selection = 'median trajectory position; no qualifying turn found'
    else:
        selection = (
            'largest simultaneous pairwise group separation among turns'
        )
    half = size_m / 2.0
    limits = (
        best_center[0] - half,
        best_center[0] + half,
        best_center[1] - half,
        best_center[1] + half,
    )
    details = {
        'selection': selection,
        'base_group': base_name,
        'minimum_turn_angle_deg': minimum_turn_angle_deg,
        'turn_half_window_s': turn_half_window_s,
        'selected_timestamp_s': best_time,
        'selected_elapsed_s': (
            best_time - base_full[0]['t'] if best_time is not None else None
        ),
        'selected_turn_angle_deg': best_turn_angle,
        'center_east_m': best_center[0],
        'center_north_m': best_center[1],
    }
    return limits, best_score, details


def trajectory_figure(context, trajectories, title_value, limits=None):
    """Draw the four group trajectories with equal ENU scaling."""
    white_background(context)
    all_points = [
        point for points in trajectories.values() for point in points
    ]
    if limits is None:
        xlim = expanded_limits([point['x'] for point in all_points], 0.035)
        ylim = expanded_limits([point['y'] for point in all_points], 0.035)
    else:
        xlim = limits[0], limits[1]
        ylim = limits[2], limits[3]
    xlim, ylim = equal_aspect_limits(xlim, ylim)
    axes = Axes(
        context, xlim, ylim, title_value,
        'East [m]', 'North [m]',
    )
    entries = []
    for name in GROUP_TOPICS:
        points = trajectories.get(name, [])
        if not points:
            continue
        selected = downsample([(p['x'], p['y']) for p in points], 12000)
        axes.line(
            selected, GROUP_COLORS[name], 3.2,
            LINE_STYLES[name], 0.88,
        )
        entries.append((
            GROUP_LABELS[name], GROUP_COLORS[name], LINE_STYLES[name]
        ))
    axes.legend(entries)


def yaw_timeseries_figure(context, errors, time_origin):
    """Draw signed course consistency errors over elapsed time."""
    white_background(context)
    all_rows = [row for rows in errors.values() for row in rows]
    xvalues = [(row['t'] - time_origin) / 60.0 for row in all_rows]
    yvalues = [row['error_deg'] for row in all_rows]
    axes = Axes(
        context,
        expanded_limits(xvalues, 0.01),
        expanded_limits(yvalues, 0.05),
        'Yaw–course difference over time',
        'Elapsed time [min]', 'Yaw − position-derived course [deg]',
    )
    axes.horizontal(0.0)
    entries = []
    for name in ('gps_course', 'gps_course_raw_gyro',
                 'gps_course_ahrs_rate'):
        points = [
            ((row['t'] - time_origin) / 60.0, row['error_deg'])
            for row in errors.get(name, [])
        ]
        axes.line(
            downsample(points, 10000), GROUP_COLORS[name], 2.5,
            LINE_STYLES[name], 0.82,
        )
        entries.append((
            GROUP_LABELS[name], GROUP_COLORS[name], LINE_STYLES[name]
        ))
    axes.legend(entries)


def yaw_cdf_figure(context, errors):
    """Draw empirical CDFs of absolute course discrepancy."""
    white_background(context)
    all_values = [
        row['abs_error_deg'] for rows in errors.values() for row in rows
    ]
    maximum = max(all_values) if all_values else 1.0
    xstep = nice_step(maximum, 6)
    xupper = min(180.0, math.ceil(maximum / xstep) * xstep)
    xtick_values = []
    value = 0.0
    while value <= xupper + xstep * 1.0e-9:
        xtick_values.append(value)
        value += xstep
    if not math.isclose(xtick_values[-1], xupper):
        xtick_values.append(xupper)
    xlim = 0.0, xupper
    axes = Axes(
        context, xlim, (0.0, 1.0),
        'CDF of absolute yaw–course difference',
        'Absolute yaw–course difference [deg]',
        'Cumulative fraction of matched samples',
        xtick_values=xtick_values,
    )
    entries = []
    for name in ('gps_course', 'gps_course_raw_gyro',
                 'gps_course_ahrs_rate'):
        values = sorted(row['abs_error_deg'] for row in errors.get(name, []))
        if not values:
            continue
        points = [
            (value, (index + 1) / len(values))
            for index, value in enumerate(values)
        ]
        axes.line(
            downsample(points, 10000), GROUP_COLORS[name], 3.4,
            LINE_STYLES[name], 0.95,
        )
        entries.append((
            GROUP_LABELS[name], GROUP_COLORS[name], LINE_STYLES[name]
        ))
    axes.legend(entries, x=axes.right - 500, y=axes.top + 110)


def yaw_boxplot_figure(context, errors):
    """Draw boxplots of absolute course discrepancy."""
    white_background(context)
    names = [
        'gps_course', 'gps_course_raw_gyro', 'gps_course_ahrs_rate'
    ]
    values_by_name = {
        name: sorted(row['abs_error_deg'] for row in errors.get(name, []))
        for name in names
    }
    upper_whiskers = [
        percentile(values, 0.95)
        for values in values_by_name.values() if values
    ]
    ylim = 0.0, max(upper_whiskers) * 1.28
    group_ticks = {1: 'Group 2', 2: 'Group 3', 3: 'Group 4'}
    axes = Axes(
        context, (0.45, 3.55), ylim,
        'Absolute yaw–course difference by group',
        'Group', 'Absolute yaw–course difference [deg]',
        xtick_values=[1, 2, 3],
        xtick_formatter=lambda value: group_ticks[int(value)],
    )
    context.save()
    for index, name in enumerate(names, start=1):
        values = values_by_name[name]
        if not values:
            continue
        q1 = percentile(values, 0.25)
        median = percentile(values, 0.50)
        q3 = percentile(values, 0.75)
        lower = percentile(values, 0.05)
        upper = percentile(values, 0.95)
        x_left, y_q3 = axes.project(index - 0.22, q3)
        x_right, y_q1 = axes.project(index + 0.22, q1)
        _, y_median = axes.project(index, median)
        x_center, y_lower = axes.project(index, lower)
        _, y_upper = axes.project(index, upper)
        set_source_hex(context, GROUP_COLORS[name], 0.23)
        context.rectangle(x_left, y_q3, x_right - x_left, y_q1 - y_q3)
        context.fill_preserve()
        set_source_hex(context, GROUP_COLORS[name])
        context.set_line_width(3.0)
        context.stroke()
        context.move_to(x_left, y_median)
        context.line_to(x_right, y_median)
        context.stroke()
        context.move_to(x_center, y_q3)
        context.line_to(x_center, y_upper)
        context.move_to(x_center, y_q1)
        context.line_to(x_center, y_lower)
        context.stroke()
        context.move_to(x_center - 18, y_upper)
        context.line_to(x_center + 18, y_upper)
        context.move_to(x_center - 18, y_lower)
        context.line_to(x_center + 18, y_lower)
        context.stroke()
        text(
            context, x_center, y_median - 12,
            f'{median:.2f}°', 19, 'center', GROUP_COLORS[name],
        )
    context.restore()
    text(
        context, axes.left + 12, axes.top + 35,
        'Whiskers: P05–P95; box: Q1–Q3', 20, 'left', '#555555',
    )


def position_cdf_figure(context, errors):
    """Draw CDFs of planar distance to the shared GNSS input."""
    white_background(context)
    all_values = [
        row['distance_m'] for rows in errors.values() for row in rows
    ]
    maximum = max(all_values) if all_values else 1.0
    xstep = nice_step(maximum, 6)
    xupper = math.ceil(maximum / xstep) * xstep
    xtick_values = []
    value = 0.0
    while value <= xupper + xstep * 1.0e-9:
        xtick_values.append(value)
        value += xstep
    axes = Axes(
        context, (0.0, xupper), (0.0, 1.0),
        'CDF of translation-aligned distance to GNSS position input',
        'Planar distance to GNSS position input [m]',
        'Cumulative fraction of matched samples',
        xtick_values=xtick_values,
    )
    entries = []
    for name in GROUP_TOPICS:
        values = sorted(row['distance_m'] for row in errors.get(name, []))
        if not values:
            continue
        points = [
            (value, (index + 1) / len(values))
            for index, value in enumerate(values)
        ]
        axes.line(
            downsample(points, 12000), GROUP_COLORS[name], 3.4,
            LINE_STYLES[name], 0.95,
        )
        entries.append((
            GROUP_LABELS[name], GROUP_COLORS[name], LINE_STYLES[name]
        ))
    axes.legend(entries, x=axes.right - 500, y=axes.top + 95)


def position_boxplot_figure(context, errors):
    """Draw distributions of planar distance to the shared GNSS input."""
    white_background(context)
    names = list(GROUP_TOPICS)
    values_by_name = {
        name: sorted(row['distance_m'] for row in errors.get(name, []))
        for name in names
    }
    upper_whiskers = [
        percentile(values, 0.95)
        for values in values_by_name.values() if values
    ]
    ylim = 0.0, max(upper_whiskers) * 1.28
    group_ticks = {
        index: f'Group {index}' for index in range(1, len(names) + 1)
    }
    axes = Axes(
        context, (0.45, 4.55), ylim,
        'Translation-aligned distance to GNSS position input by group',
        'Group', 'Planar distance to GNSS position input [m]',
        xtick_values=[1, 2, 3, 4],
        xtick_formatter=lambda value: group_ticks[int(value)],
    )
    context.save()
    for index, name in enumerate(names, start=1):
        values = values_by_name[name]
        if not values:
            continue
        q1 = percentile(values, 0.25)
        median = percentile(values, 0.50)
        q3 = percentile(values, 0.75)
        lower = percentile(values, 0.05)
        upper = percentile(values, 0.95)
        x_left, y_q3 = axes.project(index - 0.22, q3)
        x_right, y_q1 = axes.project(index + 0.22, q1)
        _, y_median = axes.project(index, median)
        x_center, y_lower = axes.project(index, lower)
        _, y_upper = axes.project(index, upper)
        set_source_hex(context, GROUP_COLORS[name], 0.23)
        context.rectangle(x_left, y_q3, x_right - x_left, y_q1 - y_q3)
        context.fill_preserve()
        set_source_hex(context, GROUP_COLORS[name])
        context.set_line_width(3.0)
        context.stroke()
        context.move_to(x_left, y_median)
        context.line_to(x_right, y_median)
        context.stroke()
        context.move_to(x_center, y_q3)
        context.line_to(x_center, y_upper)
        context.move_to(x_center, y_q1)
        context.line_to(x_center, y_lower)
        context.stroke()
        context.move_to(x_center - 18, y_upper)
        context.line_to(x_center + 18, y_upper)
        context.move_to(x_center - 18, y_lower)
        context.line_to(x_center + 18, y_lower)
        context.stroke()
        label = f'{median:.2f} m'
        label_y = y_q3 - 13
        context.save()
        context.select_font_face(
            'DejaVu Sans', cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_NORMAL,
        )
        context.set_font_size(19)
        extents = context.text_extents(label)
        set_source_hex(context, '#FFFFFF', 0.90)
        context.rectangle(
            x_center - extents.width / 2.0 - 6,
            label_y + extents.y_bearing - 4,
            extents.width + 12,
            extents.height + 8,
        )
        context.fill()
        context.restore()
        text(
            context, x_center, label_y,
            label, 19, 'center', GROUP_COLORS[name],
        )
    context.restore()
    text(
        context, axes.left + 12, axes.top + 35,
        'x/y offset: median of first 20 matched samples',
        20, 'left', '#555555',
    )
    text(
        context, axes.left + 12, axes.top + 65,
        'Whiskers: P05–P95; box: Q1–Q3', 20, 'left', '#555555',
    )


def rate_timeseries_figure(context, pairs, time_origin):
    """Draw raw and AHRS yaw-rate time series."""
    white_background(context)
    xvalues = [(row['t'] - time_origin) / 60.0 for row in pairs]
    yvalues = [
        value for row in pairs
        for value in (row['raw_rad_s'], row['ahrs_rad_s'])
    ]
    axes = Axes(
        context, expanded_limits(xvalues, 0.01),
        expanded_limits(yvalues, 0.04),
        'Raw gyroscope and AHRS yaw rates',
        'Elapsed time [min]', 'Yaw rate [rad s⁻¹]',
    )
    axes.horizontal(0.0)
    raw = [
        ((row['t'] - time_origin) / 60.0, row['raw_rad_s'])
        for row in pairs
    ]
    ahrs = [
        ((row['t'] - time_origin) / 60.0, row['ahrs_rad_s'])
        for row in pairs
    ]
    axes.line(downsample(raw, 12000), '#0072B2', 2.3, [], 0.82)
    axes.line(downsample(ahrs, 12000), '#D55E00', 2.0, [9.0, 4.0], 0.82)
    axes.legend([
        ('Raw gyroscope', '#0072B2', []),
        ('AHRS yaw rate', '#D55E00', [9.0, 4.0]),
    ])


def rate_scatter_figure(context, pairs):
    """Draw paired raw-versus-AHRS yaw rates."""
    white_background(context)
    values = [
        value for row in pairs
        for value in (row['raw_rad_s'], row['ahrs_rad_s'])
    ]
    limits = expanded_limits(values, 0.05)
    axes = Axes(
        context, limits, limits,
        'Raw gyroscope versus AHRS yaw rate',
        'Raw gyroscope yaw rate [rad s⁻¹]',
        'AHRS yaw rate [rad s⁻¹]',
    )
    axes.line(
        [(limits[0], limits[0]), (limits[1], limits[1])],
        '#555555', 2.0, [8.0, 5.0],
    )
    selected = downsample(
        [(row['raw_rad_s'], row['ahrs_rad_s']) for row in pairs], 7000
    )
    axes.scatter(selected, '#0072B2', 3.0, 0.35)
    axes.legend([('1:1 line', '#555555', [8.0, 5.0])])


def rate_difference_figure(context, pairs, time_origin):
    """Draw raw-minus-AHRS yaw-rate differences over time."""
    white_background(context)
    points = [
        ((row['t'] - time_origin) / 60.0, row['difference_rad_s'])
        for row in pairs
    ]
    axes = Axes(
        context,
        expanded_limits([point[0] for point in points], 0.01),
        expanded_limits([point[1] for point in points], 0.05),
        'Raw gyroscope minus AHRS yaw rate',
        'Elapsed time [min]', 'Yaw-rate difference [rad s⁻¹]',
    )
    axes.horizontal(0.0)
    axes.line(downsample(points, 12000), '#6A3D9A', 2.3, [], 0.82)


def status_value(value):
    """Normalize common RTK state representations to 0/1/2."""
    if isinstance(value, str):
        normalized = value.strip().lower().replace('_', ' ')
        if 'fixed' in normalized:
            return 2
        if 'float' in normalized:
            return 1
        if normalized in ('none', 'no solution', 'invalid', 'single'):
            return 0
        try:
            return int(float(value))
        except ValueError:
            return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in (0, 1, 2) else None


def first_field(mapping, names):
    """Return the first present non-null mapping field."""
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def load_rtk_json(path):
    """Load a flexible epoch-oriented RTK JSON representation."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    candidates = document
    if isinstance(document, dict):
        for key in (
            'epochs', 'rtk_epochs', 'timeline', 'quality_timeline', 'samples'
        ):
            if isinstance(document.get(key), list):
                candidates = document[key]
                break
        else:
            nested = document.get('rtk_quality')
            if isinstance(nested, dict):
                for key in (
                    'epochs', 'timeline', 'quality_timeline', 'samples'
                ):
                    if isinstance(nested.get(key), list):
                        candidates = nested[key]
                        break
    if not isinstance(candidates, list):
        raise RuntimeError(
            'RTK JSON must contain a list under epochs, rtk_epochs, '
            'timeline, quality_timeline, samples, or rtk_quality'
        )
    rows = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        time_value = first_field(item, (
            't', 't_s', 'time_s', 'timestamp_s', 'header_time_s', 'ros_time_s',
            'stamp_s',
        ))
        state = first_field(item, (
            'carr_soln', 'carr_soln_status', 'carrier_status_code',
            'carrier_status', 'carrier_solution',
            'carrier_solution_status', 'rtk_status', 'status',
        ))
        accuracy = first_field(item, (
            'h_acc_m', 'hacc_m', 'horizontal_accuracy_m',
            'horizontal_accuracy',
        ))
        try:
            time_value = float(time_value)
        except (TypeError, ValueError):
            continue
        state = status_value(state)
        try:
            accuracy = float(accuracy) if accuracy is not None else None
        except (TypeError, ValueError):
            accuracy = None
        if state is None and accuracy is None:
            continue
        rows.append({'t': time_value, 'status': state, 'h_acc_m': accuracy})
    return sorted(rows, key=lambda row: row['t'])


def read_rtk_bag(path, storage_id):
    """Read carrier solution state and hAcc from a raw u-blox bag."""
    _, rows = read_selected_topics(
        str(path), storage_id, [PVT_TOPIC, HP_TOPIC]
    )
    hp_by_itow = {}
    for _, _, message in rows.get(HP_TOPIC, []):
        if any((
            bool(message.invalid_lon), bool(message.invalid_lat),
            bool(message.invalid_lon_hp), bool(message.invalid_lat_hp),
        )):
            continue
        hp_by_itow[int(message.itow)] = float(message.h_acc) * 1.0e-4
    output = []
    for header_time, _, message in rows.get(PVT_TOPIC, []):
        if header_time is None:
            continue
        state = status_value(message.carr_soln.status)
        accuracy = hp_by_itow.get(
            int(message.itow), float(message.h_acc) * 1.0e-3
        )
        output.append({
            't': header_time,
            'status': state,
            'h_acc_m': accuracy,
        })
    return sorted(output, key=lambda row: row['t'])


def rtk_figure(context, rows):
    """Draw carrier solution status and receiver hAcc on a common timeline."""
    white_background(context)
    time_origin = rows[0]['t']
    xvalues = [(row['t'] - time_origin) / 60.0 for row in rows]
    accuracies = [
        row['h_acc_m'] for row in rows
        if row['h_acc_m'] is not None and math.isfinite(row['h_acc_m'])
    ]
    hmax = max(percentile(accuracies, 0.99) or 0.1, 0.1)
    hmax *= 1.15
    axes = Axes(
        context, expanded_limits(xvalues, 0.01), (-0.2, 2.35),
        'RTK status and receiver-reported hAcc',
        'Elapsed time [min]', 'RTK status',
        ytick_values=[0, 1, 2],
        ytick_formatter=lambda value: {
            0: 'None', 1: 'Float', 2: 'Fixed'
        }[int(value)],
    )
    status_points = {
        0: [], 1: [], 2: [],
    }
    for row in rows:
        state = row['status']
        if state in status_points:
            status_points[state].append(
                ((row['t'] - time_origin) / 60.0, state)
            )
    for state, color in ((0, '#777777'), (1, '#E69F00'), (2, '#009E73')):
        axes.scatter(downsample(status_points[state], 9000), color, 3.2, 0.72)
    hpoints = []
    for row in rows:
        accuracy = row['h_acc_m']
        if accuracy is None or not math.isfinite(accuracy):
            continue
        normalized = min(max(accuracy / hmax, 0.0), 1.0)
        display_y = normalized * 2.25
        hpoints.append(((row['t'] - time_origin) / 60.0, display_y))
    axes.line(downsample(hpoints, 10000), '#CC79A7', 2.5, [], 0.80)

    threshold_m = 0.10
    threshold_y = min(threshold_m / hmax, 1.0) * 2.25
    axes.horizontal(
        threshold_y, '#A23B72', [10.0, 6.0], 2.2
    )
    _, threshold_surface_y = axes.project(0.0, threshold_y)
    text(
        context, axes.right - 15, threshold_surface_y - 10,
        'hAcc threshold: 0.10 m', 19, 'right', '#A23B72',
    )

    hacc_ticks = ticks(0.0, hmax, 5)
    if not hacc_ticks or not math.isclose(hacc_ticks[0], 0.0):
        hacc_ticks.insert(0, 0.0)
    if not math.isclose(hacc_ticks[-1], hmax):
        if len(hacc_ticks) > 1:
            final_step = hacc_ticks[-1] - hacc_ticks[-2]
        else:
            final_step = hmax
        if hmax - hacc_ticks[-1] < final_step * 0.5:
            hacc_ticks[-1] = hmax
        else:
            hacc_ticks.append(hmax)
    for accuracy in hacc_ticks:
        display_y = min(accuracy / hmax, 1.0) * 2.25
        _, surface_y = axes.project(0.0, display_y)
        set_source_hex(context, '#A23B72')
        context.set_line_width(1.4)
        context.move_to(axes.right, surface_y)
        context.line_to(axes.right + 9, surface_y)
        context.stroke()
        text(
            context, axes.right + 14, surface_y + 6,
            f'{accuracy:.2f}', 17, 'left', '#A23B72',
        )
    text(
        context, axes.right - 12, axes.top + 29,
        'Receiver-reported hAcc [m]', 18, 'right', '#A23B72',
    )
    axes.legend([
        ('No carrier-phase solution', '#777777', []),
        ('RTK float', '#E69F00', []),
        ('RTK fixed', '#009E73', []),
        ('Receiver-reported hAcc', '#CC79A7', []),
        ('hAcc threshold (0.10 m)', '#A23B72', [10.0, 6.0]),
    ], x=axes.left + 35, y=axes.top + 38)


def write_csv(path, fieldnames, rows):
    """Write a UTF-8 CSV with a fixed schema."""
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_summary(rows, section, group, metric, value, unit='', samples=''):
    """Append one summary row."""
    rows.append({
        'section': section,
        'group': group,
        'metric': metric,
        'value': '' if value is None else value,
        'unit': unit,
        'samples': samples,
    })


def correlation(left, right):
    """Compute Pearson correlation without NumPy."""
    if len(left) < 3 or len(left) != len(right):
        return None
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    numerator = sum(
        (x - mean_left) * (y - mean_right)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0.0 else None


def parse_zoom(values):
    """Validate one manually supplied ENU zoom rectangle."""
    xmin, xmax, ymin, ymax = map(float, values)
    if not xmin < xmax or not ymin < ymax:
        raise argparse.ArgumentTypeError(
            'zoom bounds must satisfy xmin < xmax and ymin < ymax'
        )
    return xmin, xmax, ymin, ymax


def validate_inputs(args):
    """Fail before creating output when input arguments are invalid."""
    bag = Path(args.bag).expanduser().resolve()
    evaluation = Path(args.evaluation_json).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if not bag.is_dir():
        raise RuntimeError(f'Result bag directory does not exist: {bag}')
    if not evaluation.is_file():
        raise RuntimeError(f'Evaluation JSON does not exist: {evaluation}')
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite existing path: {output}')
    if args.rtk_json and args.rtk_bag:
        raise RuntimeError('Use only one of --rtk-json and --rtk-bag')
    rtk_json = (
        Path(args.rtk_json).expanduser().resolve() if args.rtk_json else None
    )
    rtk_bag = (
        Path(args.rtk_bag).expanduser().resolve() if args.rtk_bag else None
    )
    if rtk_json and not rtk_json.is_file():
        raise RuntimeError(f'RTK JSON does not exist: {rtk_json}')
    if rtk_bag and not rtk_bag.is_dir():
        raise RuntimeError(f'RTK bag directory does not exist: {rtk_bag}')
    return bag, evaluation, output, rtk_json, rtk_bag


def generate(args):
    """Read all inputs, generate figures, and return a manifest document."""
    bag, evaluation_path, output, rtk_json, rtk_bag = validate_inputs(args)
    evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
    topics = [
        args.fix_topic, RAW_RATE_TOPIC, AHRS_RATE_TOPIC, *GROUP_TOPICS.values()
    ]
    type_map, bag_rows = read_selected_topics(
        str(bag), args.storage_id, topics
    )
    missing = [topic for topic in topics if topic not in type_map]
    if missing:
        raise RuntimeError(
            'Result bag is missing required topics: ' + ', '.join(missing)
        )

    fix_points = extract_fix_points(bag_rows[args.fix_topic])
    trajectories = {
        name: extract_odometry(bag_rows[topic])
        for name, topic in GROUP_TOPICS.items()
    }
    if any(not points for points in trajectories.values()):
        empty = [name for name, points in trajectories.items() if not points]
        raise RuntimeError('Empty group trajectories: ' + ', '.join(empty))
    courses = build_courses(
        fix_points, args.course_half_window, args.minimum_course_speed
    )
    yaw_errors = {
        name: course_errors(courses, trajectories[name], args.time_tolerance)
        for name in (
            'gps_course', 'gps_course_raw_gyro', 'gps_course_ahrs_rate'
        )
    }
    position_errors = {}
    position_offsets = {}
    for name in GROUP_TOPICS:
        rows, offset_x, offset_y = translation_only_errors(
            fix_points, trajectories[name], POSITION_TIME_TOLERANCE_S
        )
        position_errors[name] = rows
        position_offsets[name] = (offset_x, offset_y)
    rate_pairs = pair_rates(
        bag_rows[RAW_RATE_TOPIC], bag_rows[AHRS_RATE_TOPIC],
        args.rate_tolerance,
    )
    if not courses or any(not rows for rows in yaw_errors.values()):
        raise RuntimeError('No matched yaw/course samples were produced')
    if any(not rows for rows in position_errors.values()):
        raise RuntimeError(
            'No translation-only position matches for one or more groups'
        )
    if not rate_pairs:
        raise RuntimeError('No paired raw/AHRS yaw-rate samples were produced')

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    generated = []
    warnings = []
    try:
        figures = output / 'figures'
        tables = output / 'tables'
        figures.mkdir()
        tables.mkdir()

        generated += render(
            figures / 'trajectory_all',
            lambda context: trajectory_figure(
                context, trajectories, 'Trajectory comparison'
            ),
        )
        auto_limits, auto_spread, auto_details = automatic_zoom(
            trajectories, args.auto_zoom_size_m
        )
        generated += render(
            figures / 'trajectory_zoom_auto',
            lambda context: trajectory_figure(
                context, trajectories,
                'Local trajectory comparison (turn A)',
                auto_limits,
            ),
        )
        manual_zooms = []
        for index, values in enumerate(args.zoom or [], start=1):
            limits = parse_zoom(values)
            turn_label = chr(ord('A') + index)
            manual_zooms.append({
                'file_index': index,
                'turn': turn_label,
                'xmin': limits[0],
                'xmax': limits[1],
                'ymin': limits[2],
                'ymax': limits[3],
            })
            generated += render(
                figures / f'trajectory_zoom_manual_{index:02d}',
                lambda context, limits=limits, turn_label=turn_label: (
                    trajectory_figure(
                        context, trajectories,
                        f'Local trajectory comparison (turn {turn_label})',
                        limits,
                    )
                ),
            )

        time_origin = min(point['t'] for point in fix_points)
        generated += render(
            figures / 'yaw_course_error_timeseries',
            lambda context: yaw_timeseries_figure(
                context, yaw_errors, time_origin
            ),
        )
        generated += render(
            figures / 'yaw_course_error_cdf',
            lambda context: yaw_cdf_figure(context, yaw_errors),
        )
        generated += render(
            figures / 'yaw_course_error_boxplot',
            lambda context: yaw_boxplot_figure(context, yaw_errors),
        )
        generated += render(
            figures / 'position_discrepancy_cdf',
            lambda context: position_cdf_figure(context, position_errors),
        )
        generated += render(
            figures / 'position_discrepancy_boxplot',
            lambda context: position_boxplot_figure(
                context, position_errors
            ),
        )
        generated += render(
            figures / 'yaw_rate_timeseries',
            lambda context: rate_timeseries_figure(
                context, rate_pairs, time_origin
            ),
        )
        generated += render(
            figures / 'yaw_rate_scatter',
            lambda context: rate_scatter_figure(context, rate_pairs),
        )
        generated += render(
            figures / 'yaw_rate_difference',
            lambda context: rate_difference_figure(
                context, rate_pairs, time_origin
            ),
        )

        rtk_rows = []
        if rtk_json:
            rtk_rows = load_rtk_json(rtk_json)
        elif rtk_bag:
            rtk_rows = read_rtk_bag(rtk_bag, args.rtk_storage_id)
        if (rtk_json or rtk_bag) and not rtk_rows:
            warnings.append(
                'RTK input supplied, but no plottable epochs found'
            )
        if rtk_rows:
            generated += render(
                figures / 'rtk_status_hacc_timeseries',
                lambda context: rtk_figure(context, rtk_rows),
            )

        yaw_table = []
        for name, rows in yaw_errors.items():
            for row in rows:
                yaw_table.append({
                    'group': name,
                    'label': GROUP_LABELS[name],
                    'timestamp_s': f'{row["t"]:.9f}',
                    'elapsed_s': f'{row["t"] - time_origin:.9f}',
                    'signed_error_deg': f'{row["error_deg"]:.12g}',
                    'absolute_error_deg': f'{row["abs_error_deg"]:.12g}',
                    'reference_speed_m_s': f'{row["speed_m_s"]:.12g}',
                })
        write_csv(
            tables / 'yaw_course_errors.csv',
            [
                'group', 'label', 'timestamp_s', 'elapsed_s',
                'signed_error_deg', 'absolute_error_deg',
                'reference_speed_m_s',
            ],
            yaw_table,
        )
        generated.append(tables / 'yaw_course_errors.csv')

        position_table = []
        for name, rows in position_errors.items():
            offset_x, offset_y = position_offsets[name]
            for row in rows:
                position_table.append({
                    'group': name,
                    'label': GROUP_LABELS[name],
                    'timestamp_s': f'{row["t"]:.9f}',
                    'elapsed_s': f'{row["t"] - time_origin:.9f}',
                    'interpolated_gnss_east_m': (
                        f'{row["gnss_input_east_m"]:.12g}'
                    ),
                    'interpolated_gnss_north_m': (
                        f'{row["gnss_input_north_m"]:.12g}'
                    ),
                    'aligned_output_east_m': (
                        f'{row["aligned_output_east_m"]:.12g}'
                    ),
                    'aligned_output_north_m': (
                        f'{row["aligned_output_north_m"]:.12g}'
                    ),
                    'planar_distance_to_gnss_input_m': (
                        f'{row["distance_m"]:.12g}'
                    ),
                    'output_minus_gnss_east_offset_m': f'{offset_x:.12g}',
                    'output_minus_gnss_north_offset_m': f'{offset_y:.12g}',
                })
        write_csv(
            tables / 'translation_only_position_discrepancy.csv',
            [
                'group', 'label', 'timestamp_s', 'elapsed_s',
                'interpolated_gnss_east_m',
                'interpolated_gnss_north_m',
                'aligned_output_east_m', 'aligned_output_north_m',
                'planar_distance_to_gnss_input_m',
                'output_minus_gnss_east_offset_m',
                'output_minus_gnss_north_offset_m',
            ],
            position_table,
        )
        generated.append(
            tables / 'translation_only_position_discrepancy.csv'
        )

        rate_table = [{
            'timestamp_s': f'{row["t"]:.9f}',
            'elapsed_s': f'{row["t"] - time_origin:.9f}',
            'raw_gyro_z_rad_s': f'{row["raw_rad_s"]:.12g}',
            'ahrs_headingspeed_rad_s': f'{row["ahrs_rad_s"]:.12g}',
            'raw_minus_ahrs_rad_s': f'{row["difference_rad_s"]:.12g}',
        } for row in rate_pairs]
        write_csv(
            tables / 'yaw_rate_pairs.csv',
            [
                'timestamp_s', 'elapsed_s', 'raw_gyro_z_rad_s',
                'ahrs_headingspeed_rad_s', 'raw_minus_ahrs_rad_s',
            ],
            rate_table,
        )
        generated.append(tables / 'yaw_rate_pairs.csv')

        if rtk_rows:
            rtk_origin = rtk_rows[0]['t']
            write_csv(
                tables / 'rtk_timeline.csv',
                [
                    'timestamp_s', 'elapsed_s', 'carrier_solution_status',
                    'horizontal_accuracy_m',
                ],
                [{
                    'timestamp_s': f'{row["t"]:.9f}',
                    'elapsed_s': f'{row["t"] - rtk_origin:.9f}',
                    'carrier_solution_status': (
                        '' if row['status'] is None else row['status']
                    ),
                    'horizontal_accuracy_m': (
                        '' if row['h_acc_m'] is None
                        else f'{row["h_acc_m"]:.12g}'
                    ),
                } for row in rtk_rows],
            )
            generated.append(tables / 'rtk_timeline.csv')

        summary = []
        group_stats = evaluation.get('four_way', {}).get(
            'group_statistics', {}
        )
        for name in GROUP_TOPICS:
            stats = group_stats.get(name, {})
            add_summary(
                summary, 'output', name, 'message_count',
                stats.get('count', len(trajectories[name])),
                'messages', stats.get('count', len(trajectories[name])),
            )
            add_summary(
                summary, 'output', name, 'effective_rate',
                stats.get('effective_rate_hz'), 'Hz', stats.get('count', ''),
            )
        for name, rows in position_errors.items():
            values = [row['distance_m'] for row in rows]
            offset_x, offset_y = position_offsets[name]
            add_summary(
                summary, 'position_consistency', name,
                'matched_samples', len(rows), 'samples', len(rows),
            )
            add_summary(
                summary, 'position_consistency', name,
                'output_minus_gnss_east_offset', offset_x, 'm', len(rows),
            )
            add_summary(
                summary, 'position_consistency', name,
                'output_minus_gnss_north_offset', offset_y, 'm', len(rows),
            )
            add_summary(
                summary, 'position_consistency', name,
                'median_planar_distance_to_gnss_input',
                percentile(values, 0.5),
                'm', len(rows),
            )
            add_summary(
                summary, 'position_consistency', name,
                'p95_planar_distance_to_gnss_input',
                percentile(values, 0.95),
                'm', len(rows),
            )
            add_summary(
                summary, 'position_consistency', name,
                'maximum_planar_distance_to_gnss_input', max(values),
                'm', len(rows),
            )
            expected = (
                evaluation.get('trajectory_vs_fix_enu', {})
                .get(GROUP_TOPICS[name], {})
                .get('translation_only', {})
            )
            expected_median = expected.get('median_m')
            actual_median = percentile(values, 0.5)
            if (
                expected_median is not None
                and abs(expected_median - actual_median) > 1.0e-3
            ):
                warnings.append(
                    f'{name}: plotted median distance to GNSS input '
                    f'({actual_median:.6f} m) differs from evaluation JSON '
                    f'({expected_median:.6f} m); check the input topics'
                )
        for name, rows in yaw_errors.items():
            absolute = [row['abs_error_deg'] for row in rows]
            signed = [math.radians(row['error_deg']) for row in rows]
            sine = sum(math.sin(value) for value in signed)
            cosine = sum(math.cos(value) for value in signed)
            bias = math.degrees(math.atan2(sine, cosine))
            add_summary(
                summary, 'course_consistency', name,
                'matched_samples', len(rows), 'samples', len(rows),
            )
            add_summary(
                summary, 'course_consistency', name,
                'median_absolute_error', percentile(absolute, 0.5),
                'deg', len(rows),
            )
            add_summary(
                summary, 'course_consistency', name,
                'p95_absolute_error', percentile(absolute, 0.95),
                'deg', len(rows),
            )
            add_summary(
                summary, 'course_consistency', name,
                'circular_bias', bias, 'deg', len(rows),
            )
            expected = evaluation.get(
                'odometry_yaw_vs_gnss_course', {}
            ).get(GROUP_TOPICS[name]) or {}
            expected_median = expected.get('median_abs_error_deg')
            actual_median = percentile(absolute, 0.5)
            if (
                expected_median is not None
                and abs(expected_median - actual_median) > 1.0e-3
            ):
                warnings.append(
                    f'{name}: plotted median ({actual_median:.6f} deg) '
                    f'differs from evaluation JSON ({expected_median:.6f} '
                    'deg); check metric CLI settings'
                )

        differences = [row['difference_rad_s'] for row in rate_pairs]
        raw_values = [row['raw_rad_s'] for row in rate_pairs]
        ahrs_values = [row['ahrs_rad_s'] for row in rate_pairs]
        add_summary(
            summary, 'yaw_rate', 'raw_vs_ahrs', 'matched_samples',
            len(rate_pairs), 'samples', len(rate_pairs),
        )
        add_summary(
            summary, 'yaw_rate', 'raw_vs_ahrs', 'correlation',
            correlation(raw_values, ahrs_values), '', len(rate_pairs),
        )
        add_summary(
            summary, 'yaw_rate', 'raw_vs_ahrs',
            'median_absolute_difference',
            percentile([abs(value) for value in differences], 0.5),
            'rad/s', len(rate_pairs),
        )
        add_summary(
            summary, 'yaw_rate', 'raw_vs_ahrs',
            'p95_absolute_difference',
            percentile([abs(value) for value in differences], 0.95),
            'rad/s', len(rate_pairs),
        )
        add_summary(
            summary, 'automatic_zoom', 'all', 'maximum_group_spread',
            auto_spread, 'm', '',
        )
        add_summary(
            summary, 'automatic_zoom', 'all', 'selected_elapsed_time',
            auto_details['selected_elapsed_s'], 's', '',
        )
        add_summary(
            summary, 'automatic_zoom', 'all', 'selected_turn_angle',
            auto_details['selected_turn_angle_deg'], 'deg', '',
        )
        for key, value in zip(
            ('xmin', 'xmax', 'ymin', 'ymax'), auto_limits
        ):
            add_summary(summary, 'automatic_zoom', 'all', key, value, 'm')

        if rtk_rows:
            states = [
                row['status'] for row in rtk_rows
                if row['status'] is not None
            ]
            accuracies = [
                row['h_acc_m'] for row in rtk_rows
                if row['h_acc_m'] is not None and math.isfinite(row['h_acc_m'])
            ]
            for state, label in ((0, 'none'), (1, 'float'), (2, 'fixed')):
                count = sum(value == state for value in states)
                add_summary(
                    summary, 'rtk_quality', label, 'sample_count',
                    count, 'samples', len(states),
                )
                add_summary(
                    summary, 'rtk_quality', label, 'sample_fraction',
                    count / len(states) if states else None,
                    'fraction', len(states),
                )
            add_summary(
                summary, 'rtk_quality', 'all', 'hacc_median',
                percentile(accuracies, 0.5), 'm', len(accuracies),
            )
            add_summary(
                summary, 'rtk_quality', 'all', 'hacc_p95',
                percentile(accuracies, 0.95), 'm', len(accuracies),
            )

        write_csv(
            output / 'publication_summary.csv',
            ['section', 'group', 'metric', 'value', 'unit', 'samples'],
            summary,
        )
        generated.append(output / 'publication_summary.csv')

        captions = [
            'trajectory_all: Groups 1-4 trajectories in the common map frame.',
            'trajectory_zoom_auto: Turn A, selected as the qualifying turn '
            'with the largest simultaneous pairwise group separation.',
        ]
        for zoom in manual_zooms:
            captions.append(
                f'trajectory_zoom_manual_{zoom["file_index"]:02d}: Turn '
                f'{zoom["turn"]}; requested ENU bounds: East '
                f'[{zoom["xmin"]:.3f}, {zoom["xmax"]:.3f}] m, North '
                f'[{zoom["ymin"]:.3f}, {zoom["ymax"]:.3f}] m.'
            )
        captions += [
            'position_discrepancy_cdf: CDF of planar distance from each '
            'group output to the interpolated shared GNSS position input '
            '(/fix/fusion), after removing the median x/y offset from the '
            'first 20 matches. The GNSS position is an input, not ground '
            'truth.',
            'position_discrepancy_boxplot: Distribution of the same '
            'translation-aligned distance. Boxes show Q1-Q3, center lines '
            'and labels show the median, and whiskers show P05-P95.',
            'yaw_course_error_timeseries: Signed wrapped difference between '
            'odometry yaw and course derived from /fix/fusion positions. The '
            'course is not independent heading ground truth.',
            'yaw_course_error_cdf: Empirical CDF of the absolute wrapped '
            'yaw-course difference.',
            'yaw_course_error_boxplot: Absolute wrapped yaw-course '
            'difference by group. Boxes show Q1-Q3, center lines and labels '
            'show the median, and whiskers show P05-P95.',
            'yaw_rate_timeseries: ROS-FLU yaw rates from the raw gyroscope '
            'z-axis and the device AHRS headingspeed field.',
            'yaw_rate_scatter: Timestamp-matched raw-gyroscope and AHRS yaw '
            'rates; the dashed line denotes equality.',
            'yaw_rate_difference: Raw-gyroscope yaw rate minus AHRS yaw rate '
            'for timestamp-matched samples.',
        ]
        if rtk_rows:
            captions.append(
                'rtk_status_hacc_timeseries: u-blox carrier-solution status '
                'and receiver-reported hAcc over the full GNSS input bag. '
                'hAcc is not a measured position error.'
            )
        (output / 'figure_captions.txt').write_text(
            '\n'.join(captions) + '\n', encoding='utf-8'
        )
        generated.append(output / 'figure_captions.txt')

        manifest = {
            'schema_version': 1,
            'input_result_bag': str(bag),
            'input_evaluation_json': str(evaluation_path),
            'input_rtk_json': str(rtk_json) if rtk_json else None,
            'input_rtk_bag': str(rtk_bag) if rtk_bag else None,
            'reference_fix_topic': args.fix_topic,
            'course_metric': {
                'half_window_samples': args.course_half_window,
                'minimum_speed_m_s': args.minimum_course_speed,
                'nearest_time_tolerance_s': args.time_tolerance,
                'definition': 'wrap(odometry_yaw - fix_derived_course)',
            },
            'translation_only_position_metric': {
                'reference_topic': args.fix_topic,
                'reference_frame': 'local ENU from first reference fix',
                'interpolation_time_tolerance_s': (
                    POSITION_TIME_TOLERANCE_S
                ),
                'alignment_initial_matched_samples': (
                    POSITION_ALIGNMENT_INITIAL_SAMPLES
                ),
                'alignment': (
                    'subtract the median output-minus-fix east/north offset '
                    'from the initial matched samples; no rotation or scale'
                ),
                'distance_definition': (
                    'hypot(aligned_output_east - interpolated_fix_east, '
                    'aligned_output_north - interpolated_fix_north)'
                ),
                'reference_role': (
                    'shared GNSS position input; not independent ground truth'
                ),
                'sampling': 'one value per matched output sample',
            },
            'rate_pair_tolerance_s': args.rate_tolerance,
            'automatic_zoom_bounds_m': {
                'xmin': auto_limits[0], 'xmax': auto_limits[1],
                'ymin': auto_limits[2], 'ymax': auto_limits[3],
            },
            'automatic_zoom_maximum_group_spread_m': auto_spread,
            'automatic_zoom_selection': auto_details,
            'manual_zoom_bounds_m': manual_zooms,
            'warnings': warnings,
            'generated_files': sorted(
                str(path.relative_to(output)) for path in generated
            ),
        }
        manifest_path = output / 'generation_manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        return manifest
    except (RuntimeError, OSError, ValueError):
        # Keep the newly created directory for diagnosis.
        raise


def build_parser():
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, help='four-way result bag')
    parser.add_argument(
        '--evaluation-json', required=True,
        help='four_way_compare_evaluate JSON for the same result bag',
    )
    parser.add_argument(
        '--out', required=True,
        help='new output directory; existing paths are rejected',
    )
    parser.add_argument('--storage-id', default='sqlite3')
    parser.add_argument('--fix-topic', default=FIX_TOPIC_DEFAULT)
    parser.add_argument('--course-half-window', type=int, default=5)
    parser.add_argument('--minimum-course-speed', type=float, default=2.0)
    parser.add_argument('--time-tolerance', type=float, default=0.25)
    parser.add_argument('--rate-tolerance', type=float, default=0.01)
    parser.add_argument(
        '--auto-zoom-size-m', type=float, default=20.0,
        help='nominal width and height of the automatic turn view',
    )
    parser.add_argument(
        '--zoom', action='append', nargs=4, metavar=(
            'XMIN', 'XMAX', 'YMIN', 'YMAX'
        ), help='explicit ENU local view; may be repeated',
    )
    parser.add_argument('--rtk-json', default='')
    parser.add_argument('--rtk-bag', default='')
    parser.add_argument('--rtk-storage-id', default='sqlite3')
    return parser


def main():
    """CLI entry point."""
    args = build_parser().parse_args()
    if args.course_half_window < 1:
        raise SystemExit('[ERROR] --course-half-window must be at least 1')
    if args.minimum_course_speed < 0.0:
        raise SystemExit('[ERROR] --minimum-course-speed cannot be negative')
    if args.time_tolerance <= 0.0 or args.rate_tolerance <= 0.0:
        raise SystemExit('[ERROR] time tolerances must be positive')
    if args.auto_zoom_size_m <= 0.0:
        raise SystemExit('[ERROR] --auto-zoom-size-m must be positive')
    try:
        manifest = generate(args)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f'[ERROR] {error}') from None
    print(f'[OK] Comparison figures: {Path(args.out).resolve()}')
    print(f'[OK] Generated files: {len(manifest["generated_files"])}')
    for warning in manifest['warnings']:
        print(f'[WARNING] {warning}')


if __name__ == '__main__':
    main()
