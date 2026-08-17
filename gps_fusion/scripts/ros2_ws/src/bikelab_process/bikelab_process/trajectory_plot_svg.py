#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path


def read_xy_csv(path, x_key, y_key, max_points=None, align_start=False):
    rows = []
    path = Path(path)

    if not path.exists():
        return rows

    with open(path, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                x = float(row[x_key])
                y = float(row[y_key])
            except Exception:
                continue
            rows.append((x, y))

    if align_start and rows:
        x0, y0 = rows[0]
        rows = [(x - x0, y - y0) for x, y in rows]

    if max_points and len(rows) > max_points:
        step = max(1, len(rows) // max_points)
        rows = rows[::step]

    return rows


def bounds(all_series):
    xs = []
    ys = []
    for _, _, _, _, points in all_series:
        for x, y in points:
            xs.append(x)
            ys.append(y)

    if not xs:
        return -1, 1, -1, 1

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = xmax - xmin
    dy = ymax - ymin

    if dx < 1e-9:
        dx = 1.0
    if dy < 1e-9:
        dy = 1.0

    margin_x = dx * 0.08
    margin_y = dy * 0.08

    return xmin - margin_x, xmax + margin_x, ymin - margin_y, ymax + margin_y


def map_point(x, y, xmin, xmax, ymin, ymax, width, height, pad):
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    sx = pad + (x - xmin) / (xmax - xmin) * plot_w
    sy = height - pad - (y - ymin) / (ymax - ymin) * plot_h
    return sx, sy


def make_polyline(points, xmin, xmax, ymin, ymax, width, height, pad):
    coords = []
    for x, y in points:
        sx, sy = map_point(x, y, xmin, xmax, ymin, ymax, width, height, pad)
        coords.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(coords)


def write_svg(out_path, series, width=1200, height=900, xlim=None, ylim=None, title="Trajectory comparison"):
    pad = 80

    if xlim is None or ylim is None:
        xmin, xmax, ymin, ymax = bounds(series)
        if xlim is not None:
            xmin, xmax = xlim
        if ylim is not None:
            ymin, ymax = ylim
    else:
        xmin, xmax = xlim
        ymin, ymax = ylim

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="white"/>')

    # axes box
    svg.append(f'<rect x="{pad}" y="{pad}" width="{width - 2*pad}" height="{height - 2*pad}" fill="none" stroke="gray" stroke-width="1"/>')

    # title
    svg.append(f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-family="Arial">{title}</text>')

    # axis labels
    svg.append(f'<text x="{width/2}" y="{height - 25}" text-anchor="middle" font-size="18" font-family="Arial">x / East [m]</text>')
    svg.append(f'<text x="25" y="{height/2}" text-anchor="middle" font-size="18" font-family="Arial" transform="rotate(-90 25 {height/2})">y / North [m]</text>')

    # grid
    for i in range(6):
        tx = pad + i / 5 * (width - 2 * pad)
        xval = xmin + i / 5 * (xmax - xmin)
        svg.append(f'<line x1="{tx:.2f}" y1="{pad}" x2="{tx:.2f}" y2="{height-pad}" stroke="#eeeeee" stroke-width="1"/>')
        svg.append(f'<text x="{tx:.2f}" y="{height-pad+25}" text-anchor="middle" font-size="13" font-family="Arial">{xval:.2f}</text>')

        ty = height - pad - i / 5 * (height - 2 * pad)
        yval = ymin + i / 5 * (ymax - ymin)
        svg.append(f'<line x1="{pad}" y1="{ty:.2f}" x2="{width-pad}" y2="{ty:.2f}" stroke="#eeeeee" stroke-width="1"/>')
        svg.append(f'<text x="{pad-10}" y="{ty+5:.2f}" text-anchor="end" font-size="13" font-family="Arial">{yval:.2f}</text>')

    # plot
    legend_y = 70
    for label, color, style, stroke_width, points in series:
        if not points:
            continue

        polyline = make_polyline(points, xmin, xmax, ymin, ymax, width, height, pad)

        dash = ""
        if style == "dashed":
            dash = ' stroke-dasharray="10,6"'
        elif style == "dotted":
            dash = ' stroke-dasharray="2,6"'

        svg.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash} stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>'
        )

        # Start point: circle
        sx, sy = map_point(points[0][0], points[0][1], xmin, xmax, ymin, ymax, width, height, pad)
        svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="5" fill="{color}"/>')

        # End point: square
        ex, ey = map_point(points[-1][0], points[-1][1], xmin, xmax, ymin, ymax, width, height, pad)
        svg.append(f'<rect x="{ex-4:.2f}" y="{ey-4:.2f}" width="8" height="8" fill="{color}"/>')

        # legend
        legend_dash = ""
        if style == "dashed":
            legend_dash = ' stroke-dasharray="10,6"'
        elif style == "dotted":
            legend_dash = ' stroke-dasharray="2,6"'

        svg.append(f'<line x1="{width-pad-320}" y1="{legend_y}" x2="{width-pad-270}" y2="{legend_y}" stroke="{color}" stroke-width="{stroke_width}"{legend_dash}/>')
        svg.append(f'<text x="{width-pad-255}" y="{legend_y+5}" font-size="15" font-family="Arial">{label} ({len(points)} pts)</text>')
        legend_y += 24

    svg.append("</svg>")

    out_path = Path(out_path)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Plot exported trajectory CSV files as SVG."
    )
    parser.add_argument("--dir", required=True, help="Directory containing exported CSV files")
    parser.add_argument("--out", default="trajectory_xy.svg", help="Output SVG filename")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--align-start", action="store_true", help="Shift each trajectory so its first point becomes (0,0)")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, help="Manual x-axis limits: xmin xmax")
    parser.add_argument("--ylim", nargs=2, type=float, default=None, help="Manual y-axis limits: ymin ymax")
    args = parser.parse_args()

    d = Path(args.dir)

    series = [
        (
            "/fix local ENU",
            "black",
            "solid",
            2.5,
            read_xy_csv(d / "fix_enu.csv", "x_east_m", "y_north_m", args.max_points, args.align_start),
        ),
        (
            "/odometry/gps",
            "red",
            "dashed",
            2.2,
            read_xy_csv(d / "odometry_gps.csv", "x", "y", args.max_points, args.align_start),
        ),

        (
            "/odometry/filtered_global",
            "green",
            "solid",
            3.0,
            read_xy_csv(d / "odometry_filtered_global.csv", "x", "y", args.max_points, args.align_start),
        ),
    ]

    series = [(label, color, style, width, pts) for label, color, style, width, pts in series if pts]

    if not series:
        raise RuntimeError(f"No valid trajectory CSV found in {d}")

    title = "Trajectory comparison"
    if args.align_start:
        title += " (start-aligned)"
    else:
        title += " (absolute coordinates)"

    out_path = d / args.out
    write_svg(out_path, series, xlim=args.xlim, ylim=args.ylim, title=title)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
