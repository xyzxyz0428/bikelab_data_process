#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict


def to_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def read_csv_dicts(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-timestamps-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument(
        "--max-frame-gap",
        type=int,
        default=2,
        help="maximum allowed gap in frame_idx to still be considered continuous"
    )
    ap.add_argument(
        "--max-time-gap-ms",
        type=float,
        default=100.0,
        help="maximum allowed gap in time (ms) to still be considered continuous"
    )
    ap.add_argument(
        "--min-window-rows",
        type=int,
        default=3,
        help="discard windows shorter than this many rows"
    )
    args = ap.parse_args()

    rows = read_csv_dicts(args.scene_timestamps_csv)

    # group by tag_id
    grouped = defaultdict(list)
    for r in rows:
        tag_id = to_int(r.get("tag_id"))
        frame_idx = to_int(r.get("frame_idx"))
        unix_ns = to_int(r.get("unix_ns"))
        if tag_id is None or frame_idx is None or unix_ns is None:
            continue
        grouped[tag_id].append({
            "tag_id": tag_id,
            "frame_idx": frame_idx,
            "unix_ns": unix_ns,
            "filename": r.get("filename", "")
        })

    windows = []

    for tag_id, items in grouped.items():
        items.sort(key=lambda x: x["frame_idx"])

        current = [items[0]]
        for prev, cur in zip(items[:-1], items[1:]):
            frame_gap = cur["frame_idx"] - prev["frame_idx"]
            time_gap_ms = (cur["unix_ns"] - prev["unix_ns"]) / 1e6

            continuous = (
                frame_gap <= args.max_frame_gap and
                time_gap_ms <= args.max_time_gap_ms
            )

            if continuous:
                current.append(cur)
            else:
                if len(current) >= args.min_window_rows:
                    windows.append(current)
                current = [cur]

        if len(current) >= args.min_window_rows:
            windows.append(current)

    out_rows = []
    window_id = 0
    for w in windows:
        window_id += 1
        tag_id = w[0]["tag_id"]
        start_frame = w[0]["frame_idx"]
        end_frame = w[-1]["frame_idx"]
        start_ts = w[0]["unix_ns"]
        end_ts = w[-1]["unix_ns"]

        out_rows.append({
            "window_id": window_id,
            "tag_id": tag_id,
            "start_frame_idx": start_frame,
            "end_frame_idx": end_frame,
            "start_unix_ns": start_ts,
            "end_unix_ns": end_ts,
            "num_rows": len(w),
            "duration_ms": (end_ts - start_ts) / 1e6
        })

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_id",
                "tag_id",
                "start_frame_idx",
                "end_frame_idx",
                "start_unix_ns",
                "end_unix_ns",
                "num_rows",
                "duration_ms",
            ]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"saved {len(out_rows)} windows to {args.output_csv}")


if __name__ == "__main__":
    main()