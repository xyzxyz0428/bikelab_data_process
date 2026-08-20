#!/usr/bin/env python3
"""Create steering CSV derivatives containing only hardware-jump removal.

Continuous measurements clipped at the configured steering-angle limits are
retained. A row is removed only when temporal continuity provides evidence of
a hardware jump. No source file is overwritten, and no retained value is
interpolated or changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


NS_PER_SECOND = 1_000_000_000.0
HARDWARE_JUMP_CLASSES = {
    "hardware_jump_clamped_limit",
    "hardware_jump_transition_to_clamped_run",
    "hardware_jump_transition_from_clamped_run",
    "isolated_rate_spike",
}


def classify_hardware_jumps(
    frame: pd.DataFrame,
    *,
    limit_deg: float,
    maximum_transition_rate_deg_s: float,
    maximum_contiguous_gap_s: float,
    near_limit_fringe_deg: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Classify discontinuities while retaining continuous limit saturation.

    Reaching +/- ``limit_deg`` is not evidence of a fault by itself. A limit
    run is rejected only when an available entry or exit transition changes
    side or exceeds the angular-rate threshold. Missing transition evidence at
    a file boundary or beside an invalid row is treated as uncertain and is
    retained rather than deleted.
    """

    required = {"t_unix_ns", "ok", "adc_raw", "angle_deg"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Missing required steering columns: {missing}")

    result = frame.copy().reset_index(drop=True)
    time = pd.to_numeric(result["t_unix_ns"], errors="coerce").to_numpy(float)
    ok = pd.to_numeric(result["ok"], errors="coerce").eq(1).to_numpy()
    adc = pd.to_numeric(result["adc_raw"], errors="coerce").to_numpy(float)
    angle = pd.to_numeric(result["angle_deg"], errors="coerce").to_numpy(float)
    finite = np.isfinite(time) & np.isfinite(adc) & np.isfinite(angle)
    # Invalid decoder rows are not evidence of a steering hardware jump.  Keep
    # them outside the limit-run classifier and preserve them in the derivative.
    limit = finite & ok & np.isclose(np.abs(angle), limit_deg, atol=1.0e-9)

    classification = np.full(
        len(result), "invalid_decode_or_nonfinite", dtype=object
    )
    classification[finite & ok & ~limit] = "regular_measurement"
    classification[limit] = "continuous_limit_saturation"
    hardware_jump = np.zeros(len(result), dtype=bool)
    run_id = np.full(len(result), -1, dtype=int)
    run_duration = np.full(len(result), np.nan)
    entry_rate = np.full(len(result), np.nan)
    exit_rate = np.full(len(result), np.nan)
    entry_discontinuous = np.zeros(len(result), dtype=bool)
    exit_discontinuous = np.zeros(len(result), dtype=bool)
    abnormal_runs: list[tuple[int, int, float]] = []

    maximum_gap_ns = maximum_contiguous_gap_s * NS_PER_SECOND

    def transition_is_bad(
        neighbor_index: int,
        limit_index: int,
        sign: float,
    ) -> tuple[bool, float]:
        if not (0 <= neighbor_index < len(result)):
            return False, math.nan
        if not (finite[neighbor_index] and finite[limit_index]):
            return False, math.nan
        if neighbor_index < limit_index:
            dt_s = (time[limit_index] - time[neighbor_index]) / NS_PER_SECOND
        else:
            dt_s = (time[neighbor_index] - time[limit_index]) / NS_PER_SECOND
        if dt_s <= 0 or dt_s > maximum_contiguous_gap_s:
            return False, math.nan
        if limit[neighbor_index]:
            return bool(np.sign(angle[neighbor_index]) != sign), math.nan
        if not ok[neighbor_index]:
            return False, math.nan
        rate = abs(sign * limit_deg - angle[neighbor_index]) / dt_s
        # Zero is not the opposite steering side.  A zero-to-limit transition
        # is rejected only when its rate exceeds the physical-rate threshold.
        wrong_side = angle[neighbor_index] * sign < 0
        return bool(wrong_side or rate > maximum_transition_rate_deg_s), rate

    run_counter = 0
    index = 0
    while index < len(result):
        if not limit[index]:
            index += 1
            continue
        first = index
        sign = 1.0 if angle[first] > 0 else -1.0
        last = first
        while last + 1 < len(result) and limit[last + 1]:
            step_ns = time[last + 1] - time[last]
            if np.sign(angle[last + 1]) != sign:
                break
            if not (0 < step_ns <= maximum_gap_ns):
                break
            last += 1

        indices = np.arange(first, last + 1)
        run_id[indices] = run_counter
        run_duration[indices] = (time[last] - time[first]) / NS_PER_SECOND
        bad_entry, entry_value = transition_is_bad(first - 1, first, sign)
        bad_exit, exit_value = transition_is_bad(last + 1, last, sign)
        entry_rate[indices] = entry_value
        exit_rate[indices] = exit_value
        entry_discontinuous[indices] = bad_entry
        exit_discontinuous[indices] = bad_exit
        if bad_entry or bad_exit:
            hardware_jump[indices] = True
            classification[indices] = "hardware_jump_clamped_limit"
            abnormal_runs.append((first, last, sign))

        run_counter += 1
        index = last + 1

    # Remove same-side near-limit bridge samples only when they connect an
    # otherwise regular trace to a run with proven discontinuity.
    fringe_threshold_deg = limit_deg - near_limit_fringe_deg
    regular = finite & ok & ~limit & ~hardware_jump
    for first, last, sign in abnormal_runs:
        left_candidates: list[int] = []
        cursor = first - 1
        while (
            cursor >= 0
            and regular[cursor]
            and angle[cursor] * sign > 0
            and abs(angle[cursor]) >= fringe_threshold_deg
            and 0 < time[cursor + 1] - time[cursor] <= maximum_gap_ns
        ):
            left_candidates.append(cursor)
            cursor -= 1
        if left_candidates and cursor >= 0 and regular[cursor]:
            edge = left_candidates[-1]
            dt_s = (time[edge] - time[cursor]) / NS_PER_SECOND
            rate = abs(angle[edge] - angle[cursor]) / dt_s if dt_s > 0 else math.inf
            if (
                0 < dt_s <= maximum_contiguous_gap_s
                and rate > maximum_transition_rate_deg_s
            ):
                hardware_jump[left_candidates] = True
                regular[left_candidates] = False
                classification[left_candidates] = (
                    "hardware_jump_transition_to_clamped_run"
                )

        right_candidates: list[int] = []
        cursor = last + 1
        while (
            cursor < len(result)
            and regular[cursor]
            and angle[cursor] * sign > 0
            and abs(angle[cursor]) >= fringe_threshold_deg
            and 0 < time[cursor] - time[cursor - 1] <= maximum_gap_ns
        ):
            right_candidates.append(cursor)
            cursor += 1
        if right_candidates and cursor < len(result) and regular[cursor]:
            edge = right_candidates[-1]
            dt_s = (time[cursor] - time[edge]) / NS_PER_SECOND
            rate = abs(angle[cursor] - angle[edge]) / dt_s if dt_s > 0 else math.inf
            if (
                0 < dt_s <= maximum_contiguous_gap_s
                and rate > maximum_transition_rate_deg_s
            ):
                hardware_jump[right_candidates] = True
                regular[right_candidates] = False
                classification[right_candidates] = (
                    "hardware_jump_transition_from_clamped_run"
                )

    # Reject an isolated non-limit spike only when its two transitions are
    # implausible but the samples on either side remain mutually continuous.
    for index in range(1, len(result) - 1):
        if not regular[index] or not regular[index - 1] or not regular[index + 1]:
            continue
        dt_before = (time[index] - time[index - 1]) / NS_PER_SECOND
        dt_after = (time[index + 1] - time[index]) / NS_PER_SECOND
        dt_across = (time[index + 1] - time[index - 1]) / NS_PER_SECOND
        if min(dt_before, dt_after, dt_across) <= 0:
            continue
        if max(dt_before, dt_after) > maximum_contiguous_gap_s:
            continue
        incoming = abs(angle[index] - angle[index - 1]) / dt_before
        outgoing = abs(angle[index + 1] - angle[index]) / dt_after
        across = abs(angle[index + 1] - angle[index - 1]) / dt_across
        if (
            incoming > maximum_transition_rate_deg_s
            and outgoing > maximum_transition_rate_deg_s
            and across <= maximum_transition_rate_deg_s
        ):
            hardware_jump[index] = True
            regular[index] = False
            classification[index] = "isolated_rate_spike"

    audit = result.copy()
    audit.insert(0, "source_row_index", np.arange(len(result), dtype=int))
    audit["is_limit_value"] = limit
    audit["limit_run_id"] = run_id
    audit["limit_run_duration_s"] = run_duration
    audit["entry_rate_deg_s"] = entry_rate
    audit["exit_rate_deg_s"] = exit_rate
    audit["entry_discontinuous"] = entry_discontinuous
    audit["exit_discontinuous"] = exit_discontinuous
    audit["quality_classification"] = classification
    audit["is_hardware_jump"] = hardware_jump
    return hardware_jump, audit


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


def exact_row_subset_lines(
    source: Path,
    remove_indices: set[int],
    expected_rows: int,
) -> list[bytes]:
    """Return the original header and retained physical CSV rows."""

    with source.open("rb") as stream:
        lines = stream.readlines()
    if len(lines) != expected_rows + 1:
        raise RuntimeError(
            "Cannot preserve physical rows safely: expected "
            f"{expected_rows + 1} lines, found {len(lines)}"
        )
    return [lines[0]] + [
        line
        for row_index, line in enumerate(lines[1:])
        if row_index not in remove_indices
    ]


def write_exact_row_subset(
    source: Path,
    destination: Path,
    remove_indices: set[int],
    expected_rows: int,
    *,
    replace: bool,
) -> None:
    """Write retained rows byte-for-byte, optionally replacing atomically."""

    lines = exact_row_subset_lines(source, remove_indices, expected_rows)

    if replace:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.writelines(lines)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary_path, source.stat().st_mode & 0o777)
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            raise
        return

    try:
        with destination.open("xb") as stream:
            stream.writelines(lines)
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

    expected_lines = exact_row_subset_lines(source, remove_indices, expected_rows)
    with destination.open("rb") as stream:
        output_lines = stream.readlines()
    if output_lines != expected_lines:
        raise RuntimeError(
            f"Derived file is not an exact retained-row subset: {destination}"
        )


def preflight_reconciliation(
    root: Path,
    manifest_path: Path,
    suffix: str,
) -> dict[str, dict[str, str]]:
    """Verify old source/output hashes before any delete or replacement."""

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        previous_rows = list(csv.DictReader(stream))
    if not previous_rows:
        raise RuntimeError(f"Previous manifest contains no rows: {manifest_path}")

    previous_by_source: dict[str, dict[str, str]] = {}
    allowed_outputs: set[Path] = set()
    errors: list[str] = []
    for row in previous_rows:
        source = Path(row["source_file"]).resolve()
        if str(source) in previous_by_source:
            errors.append(f"Duplicate source entry in previous manifest: {source}")
            continue
        previous_by_source[str(source)] = row
        if not source.is_file():
            errors.append(f"Previous source is missing: {source}")
            continue
        expected_source_hash = row.get("source_sha256", "")
        if not expected_source_hash or sha256_file(source) != expected_source_hash:
            errors.append(f"Source hash differs from previous manifest: {source}")

        output_text = row.get("output_file", "")
        if not output_text:
            continue
        output = Path(output_text).resolve()
        allowed_outputs.add(output)
        if not output.is_file():
            errors.append(f"Previous derived output is missing: {output}")
            continue
        expected_output_hash = row.get("output_sha256", "")
        if not expected_output_hash or sha256_file(output) != expected_output_hash:
            errors.append(f"Derived output hash differs from previous manifest: {output}")

    actual_outputs = {
        path.resolve()
        for path in root.rglob(f"steering*{suffix}.csv")
        if path.is_file()
    }
    unexpected_outputs = sorted(actual_outputs.difference(allowed_outputs))
    if unexpected_outputs:
        errors.extend(
            f"Derived output is outside the previous manifest: {path}"
            for path in unexpected_outputs
        )
    if errors:
        preview = "\n".join(f"- {message}" for message in errors[:20])
        raise RuntimeError(
            "Reconciliation preflight failed; no files were changed:\n" + preview
        )
    return previous_by_source


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
        jump_mask, audit = classify_hardware_jumps(
            frame,
            limit_deg=args.steering_limit_deg,
            maximum_transition_rate_deg_s=args.maximum_transition_rate_deg_s,
            maximum_contiguous_gap_s=args.maximum_contiguous_gap_s,
            near_limit_fringe_deg=args.near_limit_fringe_deg,
        )
    except Exception as exc:
        row.update(
            {
                "rows_before": int(row.get("rows_before", 0)),
                "jump_rows_removed": 0,
                "rows_after": int(row.get("rows_before", 0)),
                "removed_fraction": 0.0,
                "classification_counts": "",
                "status": "unassessable",
                "error": str(exc),
            }
        )
        return row, None

    classifications = audit["quality_classification"].astype(str)
    jump_count = int(np.count_nonzero(jump_mask))
    counts = Counter(classifications)
    row.update(
        {
            "jump_rows_removed": jump_count,
            "rows_after": int(len(frame) - jump_count),
            "removed_fraction": jump_count / len(frame) if len(frame) else 0.0,
            "classification_counts": json.dumps(dict(sorted(counts.items())), sort_keys=True),
        }
    )
    destination = filtered_destination(path, args.suffix)
    previous = args.previous_by_source.get(str(path.resolve()))
    previous_output = bool(previous and previous.get("output_file"))
    if previous_output:
        row["previous_output_sha256"] = previous.get("output_sha256", "")
        row["output_file"] = str(destination)

    if jump_count == 0:
        if not args.write:
            row["status"] = (
                "would_delete_obsolete_output"
                if previous_output
                else "no_jump_detected"
            )
            return row, audit
        if previous_output:
            if not destination.is_file():
                raise RuntimeError(f"Expected derived file is missing: {destination}")
            destination.unlink()
            row["status"] = "deleted_obsolete_output"
        else:
            row["status"] = "no_jump_detected"
        return row, audit

    row["output_file"] = str(destination)
    if not args.write:
        row["status"] = "would_replace" if previous_output else "would_create"
        return row, audit
    remove_indices = set(np.flatnonzero(jump_mask).tolist())
    replace = destination.exists() or destination.is_symlink()
    if replace and not previous_output:
        row["status"] = "output_exists_mismatch"
        row["error"] = "Refusing to replace an output outside the previous manifest"
        return row, audit

    write_exact_row_subset(
        path,
        destination,
        remove_indices,
        len(frame),
        replace=replace,
    )
    if sha256_file(path) != source_hash:
        raise RuntimeError(f"Source file changed while processing: {path}")

    try:
        verify_exact_row_subset(path, destination, remove_indices, len(frame))
    except Exception:
        destination.unlink()
        raise

    row["output_sha256"] = sha256_file(destination)
    row["status"] = "replaced" if replace else "created"
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
        "status",
        "output_file",
        "source_sha256",
        "previous_output_sha256",
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
            "rule": "hardware_jump_only",
            "steering_limit_deg": args.steering_limit_deg,
            "near_limit_fringe_deg": args.near_limit_fringe_deg,
            "maximum_transition_rate_deg_s": args.maximum_transition_rate_deg_s,
            "maximum_contiguous_gap_s": args.maximum_contiguous_gap_s,
            "removed_classes": sorted(HARDWARE_JUMP_CLASSES),
        },
        "reconciled_from_manifest": (
            str(args.reconcile_manifest) if args.reconcile_manifest else None
        ),
        "files_scanned": len(rows),
        "status_counts": dict(status_counts),
        "files_with_detected_jumps": sum(
            int(row.get("jump_rows_removed", 0)) > 0 for row in rows
        ),
        "total_jump_rows_removed": sum(
            int(row.get("jump_rows_removed", 0))
            for row in rows
            if row.get("status") in {"created", "replaced"}
        ),
        "active_derived_files": [
            row["output_file"]
            for row in rows
            if row.get("status") in {"created", "replaced"}
        ],
        "deleted_obsolete_files": [
            row["output_file"]
            for row in rows
            if row.get("status") == "deleted_obsolete_output"
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
    with (report_dir / "active_output_folders.txt").open("w", encoding="utf-8") as stream:
        folders = sorted(
            {
                row["folder"]
                for row in rows
                if row.get("status") in {"created", "replaced"}
            }
        )
        stream.write("\n".join(folders))
        if folders:
            stream.write("\n")
    with (report_dir / "deleted_output_folders.txt").open(
        "w", encoding="utf-8"
    ) as stream:
        folders = sorted(
            {
                row["folder"]
                for row in rows
                if row.get("status") == "deleted_obsolete_output"
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
    parser.add_argument(
        "--reconcile-manifest",
        type=Path,
        help="Previous manifest authorising hash-verified delete/replace actions",
    )
    parser.add_argument("--steering-limit-deg", type=float, default=45.0)
    parser.add_argument("--near-limit-fringe-deg", type=float, default=10.0)
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
    if args.reconcile_manifest is not None:
        args.reconcile_manifest = args.reconcile_manifest.resolve()
        if not args.reconcile_manifest.is_file():
            parser.error(
                f"Reconciliation manifest does not exist: {args.reconcile_manifest}"
            )
    return args


def main() -> int:
    args = parse_args()
    args.previous_by_source = (
        preflight_reconciliation(args.root, args.reconcile_manifest, args.suffix)
        if args.reconcile_manifest
        else {}
    )
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
            "replaced",
            "deleted_obsolete_output",
            "would_create",
            "would_replace",
            "would_delete_obsolete_output",
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
