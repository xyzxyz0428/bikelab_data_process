#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare AHRS heading conventions with GNSS course.

Use vel_n/vel_e when available; otherwise derive course from ENU positions.
"""

import argparse
import bisect
import csv
import math
from pathlib import Path


def to_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def percentile(vals, p):
    vals = sorted(v for v in vals if v is not None and math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def circular_mean(angles):
    s = 0.0
    c = 0.0
    n = 0
    for a in angles:
        if a is not None and math.isfinite(a):
            s += math.sin(a)
            c += math.cos(a)
            n += 1
    if n == 0:
        return float("nan")
    return math.atan2(s / n, c / n)


def corrcoef(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 10:
        return float("nan")
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-24 or vy <= 1e-24:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def canonical_cols(fieldnames):
    return {c.strip().lower(): c for c in fieldnames}


def find_exact(fieldnames, names):
    cmap = canonical_cols(fieldnames)
    for name in names:
        if name.lower() in cmap:
            return cmap[name.lower()]
    return None


def find_position_cols(fieldnames):
    x = find_exact(fieldnames, ["x_east_m", "east_m", "enu_x", "x"])
    y = find_exact(fieldnames, ["y_north_m", "north_m", "enu_y", "y"])
    return x, y


def find_velocity_cols(fieldnames):
    # Match exact velocity names so ENU position columns are not misclassified.
    vn = find_exact(fieldnames, [
        "vel_n", "vel_north", "v_n", "vn", "north_velocity",
        "velocity_north", "veln"
    ])
    ve = find_exact(fieldnames, [
        "vel_e", "vel_east", "v_e", "ve", "east_velocity",
        "velocity_east", "vele"
    ])
    return vn, ve


def time_value_to_ns(v):
    if v is None:
        return None
    av = abs(v)
    if av > 1e17:
        return int(round(v))
    if av > 1e14:
        return int(round(v * 1e3))
    if av > 1e11:
        return int(round(v * 1e6))
    if av > 1e8:
        return int(round(v * 1e9))
    return int(round(v * 1e9))


def get_time_from_row(row, fieldnames):
    # Prefer sensor / message time, put bag_time last.
    for c in [
        "t_unix_ns", "t", "header_stamp", "stamp",
        "timestamp_ns", "time_ns", "time", "unix_ns",
        "bag_time",
    ]:
        real = find_exact(fieldnames, [c])
        if real is not None:
            return time_value_to_ns(to_float(row.get(real)))

    sec_col = find_exact(fieldnames, [
        "header.stamp.sec", "header_stamp_sec", "stamp.sec", "stamp_sec",
        "sec", "secs",
    ])
    nsec_col = find_exact(fieldnames, [
        "header.stamp.nanosec", "header_stamp_nanosec",
        "stamp.nanosec", "stamp_nanosec", "nanosec", "nsec", "nanosecs",
    ])
    if sec_col and nsec_col:
        sec = to_float(row.get(sec_col))
        nsec = to_float(row.get(nsec_col))
        if sec is not None and nsec is not None:
            return int(round(sec * 1e9 + nsec))

    return None


def read_imu_ahrs(path, angle_unit):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []

        if find_exact(fieldnames, ["heading"]) is None:
            raise RuntimeError(f"Cannot find heading column in IMU CSV. columns={fieldnames}")

        for row in r:
            dtype_col = find_exact(fieldnames, ["dtype"])
            if dtype_col is not None:
                dtype = to_float(row.get(dtype_col))
                if dtype != 65:
                    continue

            ok = True
            for c in ["crc8_ok", "crc16_ok", "end_ok"]:
                real = find_exact(fieldnames, [c])
                if real is not None:
                    v = to_float(row.get(real))
                    if v != 1:
                        ok = False
                        break
            if not ok:
                continue

            t_ns = get_time_from_row(row, fieldnames)
            h = to_float(row.get(find_exact(fieldnames, ["heading"])))
            hs_col = find_exact(fieldnames, ["headingspeed"])
            hs = to_float(row.get(hs_col)) if hs_col else None

            if t_ns is None or h is None:
                continue

            if angle_unit == "deg":
                h = math.radians(h)

            rows.append({
                "t_ns": t_ns,
                "heading": h,
                "headingspeed": hs if hs is not None else float("nan"),
            })

    rows.sort(key=lambda x: x["t_ns"])
    return rows


def read_gnss_course(path, speed_unit):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        vel_n_col, vel_e_col = find_velocity_cols(fieldnames)
        x_col, y_col = find_position_cols(fieldnames)

        raw = []
        for row in r:
            t_ns = get_time_from_row(row, fieldnames)
            if t_ns is None:
                continue

            if vel_n_col is not None and vel_e_col is not None:
                vn = to_float(row.get(vel_n_col))
                ve = to_float(row.get(vel_e_col))
                if vn is None or ve is None:
                    continue
                raw.append(("velocity", t_ns, vn, ve))
            elif x_col is not None and y_col is not None:
                x = to_float(row.get(x_col))
                y = to_float(row.get(y_col))
                if x is None or y is None:
                    continue
                raw.append(("position", t_ns, x, y))
            else:
                raise RuntimeError(
                    "Expected vel_n/vel_e or x_east_m/y_north_m; "
                    f"columns={fieldnames}"
                )

    raw.sort(key=lambda x: x[1])
    if not raw:
        raise RuntimeError("No usable GNSS rows.")

    mode = raw[0][0]
    out = []

    if mode == "velocity":
        speeds_raw = [math.hypot(v1, v2) for _, _, v1, v2 in raw]
        med_speed_raw = percentile(speeds_raw, 50)
        use_cmps = False
        if speed_unit == "cmps":
            use_cmps = True
        elif speed_unit == "auto" and med_speed_raw > 30.0:
            use_cmps = True

        factor = 0.01 if use_cmps else 1.0
        for _, t_ns, vn_raw, ve_raw in raw:
            vn = vn_raw * factor
            ve = ve_raw * factor
            speed = math.hypot(vn, ve)
            if speed <= 1e-9:
                continue
            course = math.atan2(vn, ve)
            out.append({
                "t_ns": t_ns,
                "course": course,
                "speed": speed,
                "source": f"velocity vel_n/vel_e, unit={'cmps' if use_cmps else 'mps'}",
            })
        return out

    # position-derived course: central difference
    n = len(raw)
    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1

        _, t0, x0, y0 = raw[j0]
        _, t1, x1, y1 = raw[j1]
        dt = (t1 - t0) / 1e9
        if dt <= 1e-6:
            continue

        ve = (x1 - x0) / dt
        vn = (y1 - y0) / dt
        speed = math.hypot(vn, ve)
        if speed <= 1e-9:
            continue
        course = math.atan2(vn, ve)
        out.append({
            "t_ns": raw[i][1],
            "course": course,
            "speed": speed,
            "source": "position-derived from x_east_m/y_north_m",
        })

    return out


def candidate_yaw(name, h):
    if name == "heading_as_enu":
        return h, +1.0, "heading already equals ROS ENU yaw"
    if name == "north_zero_clockwise__pi_over_2_minus_heading":
        return math.pi / 2.0 - h, -1.0, "heading 0=North, clockwise positive"
    if name == "north_zero_counterclockwise__pi_over_2_plus_heading":
        return math.pi / 2.0 + h, +1.0, "heading 0=North, counter-clockwise positive"
    if name == "west_zero_clockwise__pi_minus_heading":
        return math.pi - h, -1.0, "diagnostic"
    if name == "east_zero_clockwise__minus_heading":
        return -h, -1.0, "heading 0=East, clockwise positive"
    if name == "south_zero_clockwise__minus_pi_over_2_minus_heading":
        return -math.pi / 2.0 - h, -1.0, "diagnostic: pi/2-heading with extra 180 deg"
    raise KeyError(name)


CANDIDATES = [
    "heading_as_enu",
    "north_zero_clockwise__pi_over_2_minus_heading",
    "north_zero_counterclockwise__pi_over_2_plus_heading",
    "west_zero_clockwise__pi_minus_heading",
    "east_zero_clockwise__minus_heading",
    "south_zero_clockwise__minus_pi_over_2_minus_heading",
]


def unwrap_sequence(vals):
    out = []
    prev = None
    offset = 0.0
    for v in vals:
        if prev is not None:
            d = v - prev
            if d > math.pi:
                offset -= 2 * math.pi
            elif d < -math.pi:
                offset += 2 * math.pi
        out.append(v + offset)
        prev = v
    return out


def gradient(vals, ts):
    n = len(vals)
    out = [float("nan")] * n
    for i in range(n):
        if n < 2:
            break
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = ts[j1] - ts[j0]
        if dt > 1e-9:
            out[i] = (vals[j1] - vals[j0]) / dt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imu-csv", required=True)
    ap.add_argument("--gnss-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--angle-unit", choices=["rad", "deg"], default="rad")
    ap.add_argument("--speed-unit", choices=["auto", "mps", "cmps"], default="auto")
    ap.add_argument("--speed-threshold", type=float, default=2.0)
    ap.add_argument("--time-tolerance-ms", type=float, default=150.0)
    ap.add_argument("--min-samples", type=int, default=100)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imu = read_imu_ahrs(args.imu_csv, args.angle_unit)
    gnss_all = read_gnss_course(args.gnss_csv, args.speed_unit)
    gnss = [g for g in gnss_all if g["speed"] >= args.speed_threshold]

    if not gnss:
        raise RuntimeError("No GNSS samples after speed filter.")

    gt = [g["t_ns"] for g in gnss]
    matched = []
    tol_ns = int(args.time_tolerance_ms * 1e6)

    for row in imu:
        i = bisect.bisect_left(gt, row["t_ns"])
        candidates = []
        if i < len(gt):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(gt[k] - row["t_ns"]))
        dt_ns = abs(gt[j] - row["t_ns"])
        if dt_ns <= tol_ns:
            g = gnss[j]
            matched.append({
                **row,
                "course": g["course"],
                "speed": g["speed"],
                "dt_ms": dt_ns / 1e6,
            })

    if len(matched) < args.min_samples:
        raise RuntimeError(
            f"Too few matched samples: {len(matched)}. "
            "Try lower speed threshold or larger time tolerance."
        )

    source = gnss[0]["source"]

    summary_rows = []
    ts = [(m["t_ns"] - matched[0]["t_ns"]) / 1e9 for m in matched]
    headings = [m["heading"] for m in matched]
    courses = [m["course"] for m in matched]
    headingspeeds = [m["headingspeed"] for m in matched]

    for cand in CANDIDATES:
        yaws = []
        rate_sign = 1.0
        meaning = ""
        for h in headings:
            y, rate_sign, meaning = candidate_yaw(cand, h)
            yaws.append(wrap_pi(y))

        errs = [wrap_pi(y - c) for y, c in zip(yaws, courses)]
        abs_err_deg = [abs(math.degrees(e)) for e in errs]
        bias = circular_mean(errs)
        debiased_abs_deg = [abs(math.degrees(wrap_pi(e - bias))) for e in errs]

        yaws_unwrapped = unwrap_sequence(yaws)
        dyaw_dt = gradient(yaws_unwrapped, ts)
        wz_from_hs = [
            rate_sign * hs if math.isfinite(hs) else float("nan")
            for hs in headingspeeds
        ]
        rate_corr = corrcoef(dyaw_dt, wz_from_hs)

        summary_rows.append({
            "candidate": cand,
            "meaning": meaning,
            "yaw_rate_from_headingspeed": "+headingspeed" if rate_sign > 0 else "-headingspeed",
            "n_matched": len(matched),
            "median_abs_error_deg": percentile(abs_err_deg, 50),
            "mean_abs_error_deg": mean(abs_err_deg),
            "p95_abs_error_deg": percentile(abs_err_deg, 95),
            "max_abs_error_deg": max(abs_err_deg),
            "circular_bias_deg": math.degrees(bias),
            "debiased_median_abs_error_deg": percentile(debiased_abs_deg, 50),
            "debiased_p95_abs_error_deg": percentile(debiased_abs_deg, 95),
            "corr_dyaw_dt_vs_candidate_headingspeed": rate_corr,
        })

    summary_rows.sort(key=lambda r: r["median_abs_error_deg"])

    summary_path = outdir / "heading_convention_summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = list(summary_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    matched_path = outdir / "heading_convention_matched_samples.csv"
    with open(matched_path, "w", newline="") as f:
        fieldnames = [
            "t_ns", "heading", "headingspeed", "course", "speed", "dt_ms"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in matched:
            w.writerow({k: m[k] for k in fieldnames})

    print("\n=== Input summary ===")
    print(f"IMU AHRS samples: {len(imu)}")
    print(f"GNSS samples before speed filter: {len(gnss_all)}")
    print(f"GNSS samples after speed filter: {len(gnss)}")
    print(f"Matched samples: {len(matched)}")
    print(f"GNSS course source: {source}")
    print(f"Speed threshold: {args.speed_threshold:.3f} m/s")
    print(f"Time tolerance: {args.time_tolerance_ms:.1f} ms")

    print("\n=== Candidate ranking by median absolute error ===")
    cols = [
        "candidate",
        "yaw_rate_from_headingspeed",
        "median_abs_error_deg",
        "mean_abs_error_deg",
        "p95_abs_error_deg",
        "circular_bias_deg",
        "debiased_median_abs_error_deg",
        "corr_dyaw_dt_vs_candidate_headingspeed",
    ]
    print(",".join(cols))
    for r in summary_rows:
        print(",".join(str(r[c]) for c in cols))

    best = summary_rows[0]
    print("\n=== Ranking result ===")
    print(f"Lowest-error candidate for this input: {best['candidate']}")
    print(f"Meaning: {best['meaning']}")
    print(f"AHRS yaw-rate conversion: {best['yaw_rate_from_headingspeed']}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote matched samples: {matched_path}")


if __name__ == "__main__":
    main()
