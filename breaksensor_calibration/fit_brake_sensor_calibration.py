#!/usr/bin/env python3
"""
Fit linear calibration relations for the two brake force sensors from an ODS file.

Expected calibration sheet layout, based on Sheet1:
    A: force_g_left (with base)
    B: adc_mean_left
    D: force_g_right (with base)
    E: adc_mean_right

Output relation:
    force_g = slope * adc + intercept

Example:
    python3 fit_brake_sensor_calibration.py \
        --calibration-file 'Calibration_Right_&_Left_12May.ods' \
        --sheet Sheet1 \
        --output brake_sensor_calibration.json
"""

import argparse
import json
import math
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

ODS_NS = {
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}

Point = Tuple[float, float]  # (adc, force_g)


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def read_ods_sheet(path: str, sheet_name: str) -> List[List[str]]:
    """Read a sheet from an .ods file using only Python stdlib."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("content.xml"))

    for sheet in root.findall(".//table:table", ODS_NS):
        name = sheet.attrib.get(f"{{{ODS_NS['table']}}}name")
        if name != sheet_name:
            continue

        rows: List[List[str]] = []
        for row_node in sheet.findall("table:table-row", ODS_NS):
            row_repeat = int(row_node.attrib.get(f"{{{ODS_NS['table']}}}number-rows-repeated", "1"))
            row_values: List[str] = []

            for cell in row_node.findall("table:table-cell", ODS_NS):
                col_repeat = int(cell.attrib.get(f"{{{ODS_NS['table']}}}number-columns-repeated", "1"))
                value = cell.attrib.get(f"{{{ODS_NS['office']}}}value")
                if value is None:
                    texts = ["".join(p.itertext()) for p in cell.findall(".//text:p", ODS_NS)]
                    value = "\n".join(t for t in texts if t).strip()

                # Avoid accidentally expanding thousands of empty repeated columns.
                for _ in range(min(col_repeat, 200)):
                    row_values.append(value)

            # Avoid accidentally expanding thousands of empty repeated rows.
            for _ in range(min(row_repeat, 200)):
                rows.append(row_values)

        return rows

    raise ValueError(f"Sheet '{sheet_name}' not found in {path}")


def extract_points(rows: List[List[str]]) -> Dict[str, List[Point]]:
    """Extract left/right (adc_mean, force_g) pairs from Sheet1 layout."""
    left: List[Point] = []
    right: List[Point] = []

    for row in rows[1:]:  # skip header
        # left: force in column A, adc in column B
        if len(row) >= 2:
            force_g = _to_float(row[0])
            adc = _to_float(row[1])
            if force_g is not None and adc is not None:
                left.append((adc, force_g))

        # right: force in column D, adc in column E
        if len(row) >= 5:
            force_g = _to_float(row[3])
            adc = _to_float(row[4])
            if force_g is not None and adc is not None:
                right.append((adc, force_g))

    if len(left) < 2:
        raise ValueError("Not enough left sensor calibration points found.")
    if len(right) < 2:
        raise ValueError("Not enough right sensor calibration points found.")

    return {"left": left, "right": right}


def linear_fit(points: List[Point]) -> Dict[str, float]:
    """Least-squares fit: force_g = slope * adc + intercept."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(points)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("All ADC values are identical; cannot fit a line.")
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))

    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    preds = [slope * x + intercept for x in xs]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, preds))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    rmse_g = math.sqrt(ss_res / n)

    return {
        "slope_g_per_adc": slope,
        "intercept_g": intercept,
        "r2": r2,
        "rmse_g": rmse_g,
        "n_points": n,
        "adc_min": min(xs),
        "adc_max": max(xs),
        "force_g_min": min(ys),
        "force_g_max": max(ys),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit brake sensor ADC-to-force calibration.")
    parser.add_argument("--calibration-file", default="Calibration_Right_&_Left_12May.ods")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--output", default="brake_sensor_calibration.json")
    args = parser.parse_args()

    rows = read_ods_sheet(args.calibration_file, args.sheet)
    points = extract_points(rows)

    result = {
        "model": "force_g = slope_g_per_adc * adc + intercept_g",
        "source_file": os.path.abspath(args.calibration_file),
        "source_sheet": args.sheet,
        "sensors": {
            "left": linear_fit(points["left"]),
            "right": linear_fit(points["right"]),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[INFO] Saved calibration to {args.output}")
    for side in ("left", "right"):
        fit = result["sensors"][side]
        print(
            f"{side:>5}: force_g = {fit['slope_g_per_adc']:.10f} * adc "
            f"+ {fit['intercept_g']:.10f}; "
            f"R2={fit['r2']:.6f}, RMSE={fit['rmse_g']:.2f} g, n={fit['n_points']}"
        )


if __name__ == "__main__":
    main()
