#!/usr/bin/env python3
"""Create row-filtered steering CSV files without changing source files.

The outlier classification reuses ``classify_steering`` from the historical
P9 cleaning workflow.  Only rows assigned to an explicit jump/outlier class
are removed.  Plausible measurements at the configured steering boundary are
kept, and no interpolation or value replacement is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCRIPT_DIR = REPOSITORY_ROOT / "data_analysis" / "scripts"
if str(ANALYSIS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_SCRIPT_DIR))

from p9_clean_and_crop import classify_steering  # noqa: E402


JUMP_CLASSES = {
    "abnormal_clamped_limit",
    "abnormal_transition_to_clamped_run",
    "abnormal_transition_from_clamped_run",
    "isolated_rate_spike",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def filtered_destination(source: Path, suffix: str) -> Path:
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")


def write_exact_row_subset(
    source: Path,
    destination: Path,
    remove_indices: set[int],
    expected_rows: int,
) -> None:
    """Copy the original header and retained physical CSV rows byte-for-byte."""

    with source.open("rb") as stream:
        lines = stream.readlines()
    if len(lines) != expected_rows + 1:
        raise RuntimeError(
            "Cannot preserve physical rows safely: expected "
            f"{expected_rows + 1} lines, found {len(lines)}"
        )

    try:
        with destination.open("xb") as stream:
            stream.write(lines[0])
            for row_index, line in enumerate(lines[1:]):
                if row_index not in remove_indices:
                    stream.write(line)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def verify_exact_row_subset(
    source: Path,
    destination: Path,
    remove_indices: set[int],
    expected_rows: int,
) -> None:
    """Verify the derived file without interpreting retained raw field bytes."""

    with source.open("rb") as stream:
        source_lines = stream.readlines()
    if len(source_lines) != expected_rows + 1:
        raise RuntimeError(
            "Cannot verify physical rows safely: expected "
            f"{expected_rows + 1} lines, found {len(source_lines)}"
        )
    expected_lines = [source_lines[0]] + [
        line
        for row_index, line in enumerate(source_lines[1:])
        if row_index not in remove_indices
    ]
    with destination.open("rb") as stream:
        output_lines = stream.readlines()
    if output_lines != expected_lines:
        raise RuntimeError(
            f"Derived file is not an exact retained-row subset: {destination}"
        )


def scan_file(path: Path, args: argparse.Namespace) -> tuple[dict, pd.DataFrame | None]:
    source_hash = sha256_file(path)
    row = {
        "source_file": str(path),
        "relative_file": str(path.relative_to(args.root)),
        "folder": str(path.parent),
        "source_sha256": source_hash,
        "output_file": "",
        "output_sha256": "",
        "status": "",
        "error": "",
    }
    try:
        frame = pd.read_csv(path, low_memory=False)
        row["rows_before"] = int(len(frame))
        _, audit, calibration = classify_steering(
            frame,
            limit_deg=args.steering_limit_deg,
            maximum_extrapolation_deg=args.maximum_extrapolation_deg,
            maximum_transition_rate_deg_s=args.maximum_transition_rate_deg_s,
            maximum_contiguous_gap_s=args.maximum_contiguous_gap_s,
        )
    except Exception as exc:
        row.update(
            {
                "rows_before": int(row.get("rows_before", 0)),
                "jump_rows_removed": 0,
                "rows_after": int(row.get("rows_before", 0)),
                "removed_fraction": 0.0,
                "classification_counts": "",
                "calibration_r_squared": None,
                "status": "unassessable",
                "error": str(exc),
            }
        )
        return row, None

    classifications = audit["quality_classification"].astype(str)
    jump_mask = classifications.isin(JUMP_CLASSES).to_numpy()
    jump_count = int(np.count_nonzero(jump_mask))
    counts = Counter(classifications)
    row.update(
        {
            "jump_rows_removed": jump_count,
            "rows_after": int(len(frame) - jump_count),
            "removed_fraction": jump_count / len(frame) if len(frame) else 0.0,
            "classification_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
            "calibration_r_squared": calibration["r_squared"],
        }
    )
    if jump_count == 0:
        row["status"] = "no_jump_detected"
        return row, audit

    destination = filtered_destination(path, args.suffix)
    row["output_file"] = str(destination)
    if not args.write:
        row["status"] = "would_create"
        return row, audit
    remove_indices = set(np.flatnonzero(jump_mask).tolist())
    if destination.exists() or destination.is_symlink():
        try:
            verify_exact_row_subset(path, destination, remove_indices, len(frame))
        except Exception as exc:
            row["status"] = "output_exists_mismatch"
            row["error"] = str(exc)
            return row, audit
        row["output_sha256"] = sha256_file(destination)
        row["status"] = "existing_verified"
        return row, audit

    write_exact_row_subset(path, destination, remove_indices, len(frame))
    if sha256_file(path) != source_hash:
        raise RuntimeError(f"Source file changed while processing: {path}")

    try:
        verify_exact_row_subset(path, destination, remove_indices, len(frame))
    except Exception:
        destination.unlink()
        raise

    row["output_sha256"] = sha256_file(destination)
    row["status"] = "created"
    return row, audit


def write_reports(report_dir: Path, rows: list[dict], args: argparse.Namespace) -> None:
    report_dir.mkdir(parents=True, exist_ok=False)
    columns = [
        "source_file",
        "relative_file",
        "folder",
        "rows_before",
        "jump_rows_removed",
        "rows_after",
        "removed_fraction",
        "classification_counts",
        "calibration_r_squared",
        "status",
        "output_file",
        "source_sha256",
        "output_sha256",
        "error",
    ]
    with (report_dir / "steering_outlier_removal_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "root": str(args.root),
        "write_enabled": args.write,
        "source_pattern": "steering*.csv excluding generated suffix",
        "suffix": args.suffix,
        "parameters": {
            "steering_limit_deg": args.steering_limit_deg,
            "maximum_extrapolation_deg": args.maximum_extrapolation_deg,
            "maximum_transition_rate_deg_s": args.maximum_transition_rate_deg_s,
            "maximum_contiguous_gap_s": args.maximum_contiguous_gap_s,
            "removed_classes": sorted(JUMP_CLASSES),
        },
        "files_scanned": len(rows),
        "status_counts": dict(status_counts),
        "files_with_detected_jumps": sum(
            int(row.get("jump_rows_removed", 0)) > 0 for row in rows
        ),
        "total_jump_rows_removed": sum(
            int(row.get("jump_rows_removed", 0))
            for row in rows
            if row.get("status") in {"created", "existing_verified"}
        ),
        "created_files": [
            row["output_file"]
            for row in rows
            if row.get("status") in {"created", "existing_verified"}
        ],
        "unassessable_files": [
            {"file": row["source_file"], "reason": row["error"]}
            for row in rows
            if row.get("status") == "unassessable"
        ],
    }
    (report_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (report_dir / "created_folders.txt").open("w", encoding="utf-8") as stream:
        folders = sorted(
            {
                row["folder"]
                for row in rows
                if row.get("status") in {"created", "existing_verified"}
            }
        )
        stream.write("\n".join(folders))
        if folders:
            stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--suffix", default="_outlierremove")
    parser.add_argument("--steering-limit-deg", type=float, default=45.0)
    parser.add_argument("--maximum-extrapolation-deg", type=float, default=10.0)
    parser.add_argument("--maximum-transition-rate-deg-s", type=float, default=250.0)
    parser.add_argument("--maximum-contiguous-gap-s", type=float, default=0.25)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not args.root.is_dir():
        parser.error(f"Root directory does not exist: {args.root}")
    if args.write and args.report_dir is None:
        parser.error("--report-dir is required with --write")
    if args.report_dir is not None:
        args.report_dir = args.report_dir.resolve()
        if args.report_dir.exists() or args.report_dir.is_symlink():
            parser.error(f"Report directory already exists: {args.report_dir}")
    return args


def main() -> int:
    args = parse_args()
    files = sorted(
        path
        for path in args.root.rglob("steering*.csv")
        if args.suffix not in path.stem
    )
    rows = []
    for path in files:
        row, _ = scan_file(path, args)
        rows.append(row)
        if row["status"] in {
            "created",
            "existing_verified",
            "would_create",
            "unassessable",
            "output_exists_mismatch",
        }:
            print(
                f"{row['status']}: {row['relative_file']} "
                f"({row.get('jump_rows_removed', 0)} jump rows)"
            )
    if args.report_dir is not None:
        write_reports(args.report_dir, rows, args)
    status_counts = Counter(row["status"] for row in rows)
    print(f"Scanned {len(rows)} files: {dict(status_counts)}")
    return 0 if not status_counts.get("output_exists_mismatch") else 2


if __name__ == "__main__":
    raise SystemExit(main())
