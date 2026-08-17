#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cut_bikelab_streams.py

Cut Bikelab GPS / RTK / sensor CSV or XLSX files to the same valid time window.
This script does NOT merge files into one XLSX anymore.

It reads files recursively from --input-dir, finds target streams by filename prefix,
adds/standardizes t_unix_ns if needed, filters by time window, and writes one
trimmed CSV/XLSX per stream.

Typical usage with manual time window:
--------------------------------------
python cut_bikelab_streams.py \
  --input-dir /raw_data_process/source \
  --output-dir /raw_data_process/source/trimmed_session_001 \
  --start-unix-ns 1773159006000000000 \
  --end-unix-ns   1773159123000000000

Usage with automatic common overlap:
------------------------------------
python cut_bikelab_streams.py \
  --input-dir /raw_data_process/source \
  --output-dir /raw_data_process/source/trimmed_session_001 \
  --auto-overlap
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd


STREAMS = [
    # GPS / RTK topics exported from ros2 unbag
    {"name": "fix", "prefixes": ["fix"], "time_mode": "ros_header"},
    {"name": "ubx_nav_pvt", "prefixes": ["ubx_nav_pvt"], "time_mode": "ros_header"},
    {"name": "ubx_nav_hp_pos_llh", "prefixes": ["ubx_nav_hp_pos_llh"], "time_mode": "ros_header"},
    {"name": "ubx_nav_vel_ned", "prefixes": ["ubx_nav_vel_ned"], "time_mode": "ros_header"},
    {"name": "ubx_nav_cov", "prefixes": ["ubx_nav_cov"], "time_mode": "ros_header"},
    {"name": "ubx_nav_status", "prefixes": ["ubx_nav_status"], "time_mode": "ros_header"},
    {"name": "ubx_nav_sat", "prefixes": ["ubx_nav_sat"], "time_mode": "ros_header"},
    {"name": "ubx_nav_sig", "prefixes": ["ubx_nav_sig"], "time_mode": "ros_header"},
    {"name": "ubx_nav_dop", "prefixes": ["ubx_nav_dop"], "time_mode": "ros_header"},
    {"name": "ubx_rxm_rtcm", "prefixes": ["ubx_rxm_rtcm"], "time_mode": "ros_header"},

    # Bike interface CSVs
    {"name": "steering_angle", "prefixes": ["steering_angle"], "time_mode": "t_unix_ns"},
    {"name": "wheel_speed", "prefixes": ["speed_decoded"], "time_mode": "t_unix_ns"},
    {"name": "powermeter", "prefixes": ["rally_payload_decoded"], "time_mode": "t_unix_ns"},
    {"name": "imu", "prefixes": ["imu"], "time_mode": "t_unix_ns"},
    {"name": "brake_sensors_force", "prefixes": ["brake_sensors_force", "fsr"], "time_mode": "t_unix_ns"},
]


SUPPORTED_SUFFIXES = [".csv", ".txt", ".xlsx", ".xlsm", ".xls"]


def read_table_robust(path: Path) -> pd.DataFrame:
    """Read CSV/TXT/XLSX robustly."""
    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        return pd.read_excel(path)

    attempts = ["utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16-le", "utf-16-be"]
    last_err = None

    for enc in attempts:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e

    # Last attempt: maybe a mislabeled Excel file
    try:
        return pd.read_excel(path)
    except Exception:
        pass

    raise RuntimeError(f"Could not read file {path}: {last_err}")


def drop_duplicate_columns(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        dup_cols = df.columns[dup_mask].tolist()
        print(f"[WARN] {file_name} has duplicate columns; keeping first occurrence only: {dup_cols}")
        df = df.loc[:, ~dup_mask].copy()
    return df


def find_matching_file(input_dir: Path, prefixes: List[str]) -> Optional[Path]:
    candidates = []

    for pfx in prefixes:
        matches = sorted(input_dir.rglob(f"{pfx}*"))
        matches = [
            m for m in matches
            if m.is_file() and m.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        candidates.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for m in candidates:
        if m not in seen:
            unique.append(m)
            seen.add(m)

    if not unique:
        return None

    if len(unique) > 1:
        print(f"[WARN] Multiple files found for prefixes {prefixes}; using: {unique[0]}")

    return unique[0]


def get_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sanitize_time_column_ns(t_ns: pd.Series, file_name: str) -> pd.Series:
    """
    Remove obvious invalid timestamps such as 0 / tiny values mixed with valid Unix epoch ns.
    """
    s = pd.to_numeric(t_ns, errors="coerce")
    valid = s.dropna()

    if valid.empty:
        return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    valid_pos = valid[valid > 0]
    if valid_pos.empty:
        return pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    med = valid_pos.median()
    mask = s > 0

    # Unix epoch ns is around 1e18.
    # Reject obvious startup garbage if the stream is Unix-time based.
    if med > 1e17:
        lower = med * 0.1
        upper = med * 10.0
        mask &= (s >= lower) & (s <= upper)

    dropped = int((~mask & s.notna()).sum())
    if dropped > 0:
        print(f"[INFO] {file_name}: dropped {dropped} invalid timestamp rows.")

    s = s.where(mask, pd.NA)
    return s.astype("Int64")


def add_unified_time_column(df: pd.DataFrame, time_mode: str, file_name: str) -> pd.DataFrame:
    """
    Ensure df has a clean t_unix_ns column.
    """
    df = drop_duplicate_columns(df, file_name)

    if time_mode == "ros_header":
        sec_col = get_column(df, [
            "header.stamp.sec",
            "header_stamp_sec",
            "header.stamp.sec.",
            "stamp.sec",
            "sec",
        ])

        nsec_col = get_column(df, [
            "header.stamp.nanosec",
            "header_stamp_nanosec",
            "header.stamp.nsec",
            "stamp.nanosec",
            "nanosec",
            "nsec",
        ])

        if sec_col is None or nsec_col is None:
            # Some exported files may already have t_unix_ns
            t_col = get_column(df, ["t_unix_ns", "unix_ns", "timestamp_ns", "ts_ns"])
            if t_col is None:
                raise ValueError(
                    f"{file_name}: missing ROS header stamp columns and no t_unix_ns.\n"
                    f"Available columns: {list(df.columns)}"
                )
            df["t_unix_ns"] = pd.to_numeric(df[t_col], errors="coerce").astype("Int64")
        else:
            sec = pd.to_numeric(df[sec_col], errors="coerce")
            nsec = pd.to_numeric(df[nsec_col], errors="coerce")
            df["t_unix_ns"] = (sec * 1_000_000_000 + nsec).astype("Int64")

    elif time_mode == "t_unix_ns":
        t_col = get_column(df, [
            "t_unix_ns",
            "unix_ns",
            "timestamp_ns",
            "ts_ns",
            "timestamp",
        ])

        if t_col is None:
            raise ValueError(
                f"{file_name}: missing usable timestamp column.\n"
                f"Expected one of: t_unix_ns, unix_ns, timestamp_ns, ts_ns, timestamp\n"
                f"Available columns: {list(df.columns)}"
            )

        df["t_unix_ns"] = pd.to_numeric(df[t_col], errors="coerce").astype("Int64")

    else:
        raise ValueError(f"{file_name}: unknown time_mode '{time_mode}'")

    df["t_unix_ns"] = sanitize_time_column_ns(df["t_unix_ns"], file_name)
    return df


def load_streams(input_dir: Path, strict: bool) -> Dict[str, Tuple[Path, pd.DataFrame]]:
    loaded = {}

    for stream in STREAMS:
        name = stream["name"]
        prefixes = stream["prefixes"]
        time_mode = stream["time_mode"]

        path = find_matching_file(input_dir, prefixes)

        if path is None:
            msg = f"No file found for stream '{name}' with prefixes {prefixes}"
            if strict:
                raise FileNotFoundError(msg)
            print(f"[WARN] {msg}; skipped.")
            continue

        print(f"[READ] {name}: {path}")
        df = read_table_robust(path)
        df = add_unified_time_column(df, time_mode, path.name)

        loaded[name] = (path, df)

    return loaded


def get_common_overlap(loaded: Dict[str, Tuple[Path, pd.DataFrame]]) -> Tuple[int, int]:
    ranges = []

    for name, (path, df) in loaded.items():
        valid_t = df["t_unix_ns"].dropna()

        if valid_t.empty:
            print(f"[WARN] {name}: no valid timestamps; excluded from overlap calculation.")
            continue

        t_min = int(valid_t.min())
        t_max = int(valid_t.max())
        ranges.append((name, t_min, t_max))

    if not ranges:
        raise RuntimeError("No stream has valid timestamps. Cannot determine common overlap.")

    start = max(r[1] for r in ranges)
    end = min(r[2] for r in ranges)

    print("\n[INFO] Stream time ranges:")
    for name, t_min, t_max in ranges:
        print(f"  {name:24s} [{t_min}, {t_max}]")

    print(f"\n[INFO] Common overlap: [{start}, {end}]")

    if start > end:
        print("\n[WARN] No common overlap across all loaded streams.")
        print("[WARN] This can happen if one file belongs to another session, e.g. brake file from another date.")

    return start, end


def cut_and_save(
    loaded: Dict[str, Tuple[Path, pd.DataFrame]],
    output_dir: Path,
    start_unix_ns: int,
    end_unix_ns: int,
    output_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []

    for name, (src_path, df) in loaded.items():
        t = df["t_unix_ns"]
        cut = df[t.notna() & (t >= start_unix_ns) & (t <= end_unix_ns)].copy()
        cut = cut.sort_values("t_unix_ns").reset_index(drop=True)

        if output_format == "csv":
            out_path = output_dir / f"{name}_cut.csv"
            cut.to_csv(out_path, index=False)
        elif output_format == "xlsx":
            out_path = output_dir / f"{name}_cut.xlsx"
            cut.to_excel(out_path, index=False)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

        if len(cut) > 0:
            t_min = int(cut["t_unix_ns"].min())
            t_max = int(cut["t_unix_ns"].max())
        else:
            t_min = pd.NA
            t_max = pd.NA

        report_rows.append({
            "stream": name,
            "source_file": str(src_path),
            "output_file": str(out_path),
            "rows_original": len(df),
            "rows_cut": len(cut),
            "t_min_cut": t_min,
            "t_max_cut": t_max,
        })

        print(f"[OK] {name:24s} rows: {len(df):8d} -> {len(cut):8d}  saved: {out_path}")

    report = pd.DataFrame(report_rows)
    report_path = output_dir / "cut_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n[OK] Wrote report: {report_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Folder containing CSV/XLSX files. Search is recursive."
    )

    ap.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Folder for trimmed output files."
    )

    ap.add_argument(
        "--start-unix-ns",
        type=int,
        default=None,
        help="Start timestamp in Unix ns."
    )

    ap.add_argument(
        "--end-unix-ns",
        type=int,
        default=None,
        help="End timestamp in Unix ns."
    )

    ap.add_argument(
        "--auto-overlap",
        action="store_true",
        help="Use common valid time overlap across all loaded streams."
    )

    ap.add_argument(
        "--output-format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Output each stream as separate CSV or XLSX. Default: csv."
    )

    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any configured stream is missing."
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    loaded = load_streams(input_dir, strict=args.strict)

    if not loaded:
        raise RuntimeError("No stream loaded. Check input directory and file prefixes.")

    if args.auto_overlap:
        start_unix_ns, end_unix_ns = get_common_overlap(loaded)
    else:
        if args.start_unix_ns is None or args.end_unix_ns is None:
            raise ValueError(
                "Please provide --start-unix-ns and --end-unix-ns, "
                "or use --auto-overlap."
            )
        start_unix_ns = args.start_unix_ns
        end_unix_ns = args.end_unix_ns

    if start_unix_ns > end_unix_ns:
        raise ValueError(
            f"Invalid time window: start {start_unix_ns} > end {end_unix_ns}. "
            "Maybe one stream belongs to a different session."
        )

    print(f"\n[INFO] Cutting all streams to:")
    print(f"  start_unix_ns = {start_unix_ns}")
    print(f"  end_unix_ns   = {end_unix_ns}\n")

    cut_and_save(
        loaded=loaded,
        output_dir=output_dir,
        start_unix_ns=start_unix_ns,
        end_unix_ns=end_unix_ns,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()