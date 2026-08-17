#!/usr/bin/env python3
"""Record RoboSense Web UI WebSocket frames with host receive timestamps.

RoboSense WebSocket paths and request messages vary by model and firmware.
This logger therefore takes the exact URL and optional request frames from the
command line.  It preserves every received frame and extracts candidate
PTP/time-synchronization fields without assuming a fixed vendor schema.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import re
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = (
    "host_receive_unix_ns",
    "host_receive_utc",
    "logger_hostname",
    "sensor_label",
    "websocket_url",
    "frame_type",
    "normalized_sync_status",
    "sync_candidates_json",
    "time_candidates_json",
    "parse_error",
)

POSITIVE_STATUS = {"lock", "locked", "fixed", "synchronized", "synchronised", "true", "1"}
NEGATIVE_STATUS = {
    "unlock",
    "unlocked",
    "unfixed",
    "not fixed",
    "not synchronized",
    "not synchronised",
    "false",
    "0",
}


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid name: {value!r}")
    return cleaned


def utc_iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()


def flatten_json(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{path}[{index}]")
    else:
        yield path, value


def candidate_fields(parsed: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sync_candidates: dict[str, Any] = {}
    time_candidates: dict[str, Any] = {}
    for path, value in flatten_json(parsed):
        lower = path.lower().replace("_", "-")
        if any(token in lower for token in ("ptp", "sync", "lock", "fixed")):
            sync_candidates[path] = value
        if any(
            token in lower
            for token in ("sync-data", "device-time", "lidar-time", "clock-time", "timestamp")
        ):
            time_candidates[path] = value
    return sync_candidates, time_candidates


def normalize_status(sync_candidates: dict[str, Any], text: str) -> str:
    preferred: list[str] = []
    for key, value in sync_candidates.items():
        key_lower = key.lower()
        if any(token in key_lower for token in ("status", "state", "fixed", "lock")):
            preferred.append(str(value).strip().lower())
    for value in preferred:
        if value in NEGATIVE_STATUS or value.startswith("un") or "not sync" in value:
            return "unlocked"
        if value in POSITIVE_STATUS:
            return "locked"
    lowered = text.lower()
    negative_match = re.search(
        r"(?:time\s*sync|ptp)[^\n]{0,80}(unlock(?:ed)?|unfixed|not\s+sync(?:hronized|hronised)?)",
        lowered,
    )
    if negative_match:
        return "unlocked"
    positive_match = re.search(
        r"(?:time\s*sync|ptp)[^\n]{0,80}(lock(?:ed)?|fixed|sync(?:hronized|hronised))",
        lowered,
    )
    return "locked" if positive_match else "unknown"


def parse_sensor(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=ws://host/path.")
    label, url = value.split("=", 1)
    if not label.strip() or not re.match(r"^wss?://", url.strip(), flags=re.IGNORECASE):
        raise argparse.ArgumentTypeError("Use LABEL=ws://host/path.")
    return safe_name(label), url.strip()


def read_send_messages(args: argparse.Namespace) -> list[str]:
    messages = list(args.send_text)
    for path in args.send_file:
        messages.append(path.expanduser().read_text(encoding="utf-8").rstrip("\r\n"))
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record raw RoboSense Web UI WebSocket status frames."
    )
    parser.add_argument(
        "--sensor",
        action="append",
        required=True,
        type=parse_sensor,
        metavar="LABEL=URL",
        help="Repeat for each LiDAR.",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("time_sync_logs"))
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds; 0 means until Ctrl+C.")
    parser.add_argument("--send-text", action="append", default=[], help="Text frame sent once after connection.")
    parser.add_argument("--send-file", action="append", default=[], type=Path)
    parser.add_argument("--poll-text", help="Optional text frame sent repeatedly.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--origin", help="Optional HTTP Origin required by the device firmware.")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification for wss:// URLs.")
    parser.add_argument("--open-timeout", type=float, default=5.0)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--max-message-bytes", type=int, default=2_000_000)
    return parser.parse_args()


async def poll_sender(websocket: Any, text: str, interval_s: float, deadline: float | None) -> None:
    while deadline is None or time.monotonic() < deadline:
        await websocket.send(text)
        await asyncio.sleep(interval_s)


async def record_sensor(
    label: str,
    url: str,
    args: argparse.Namespace,
    connect: Any,
    raw_file: Any,
    status_writer: csv.DictWriter,
    status_file: Any,
    deadline: float | None,
    send_messages: list[str],
    logger_hostname: str,
) -> dict[str, Any]:
    ssl_context = None
    if url.lower().startswith("wss://") and args.insecure:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    frame_count = 0
    connection_count = 0
    connection_error_count = 0
    while deadline is None or time.monotonic() < deadline:
        connect_kwargs: dict[str, Any] = {
            "open_timeout": args.open_timeout,
            "max_size": args.max_message_bytes,
        }
        if args.origin:
            connect_kwargs["origin"] = args.origin
        if ssl_context is not None:
            connect_kwargs["ssl"] = ssl_context
        try:
            async with connect(url, **connect_kwargs) as websocket:
                connection_count += 1
                event_ns = time.time_ns()
                raw_file.write(
                    json.dumps(
                        {
                            "event": "connected",
                            "host_unix_ns": event_ns,
                            "host_utc": utc_iso_from_ns(event_ns),
                            "sensor_label": label,
                            "url": url,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                raw_file.flush()
                for message in send_messages:
                    await websocket.send(message)

                poll_task = None
                if args.poll_text:
                    poll_task = asyncio.create_task(
                        poll_sender(websocket, args.poll_text, args.poll_interval, deadline)
                    )
                try:
                    while deadline is None or time.monotonic() < deadline:
                        timeout = 1.0 if deadline is None else max(
                            0.05, min(1.0, deadline - time.monotonic())
                        )
                        try:
                            frame = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                        except asyncio.TimeoutError:
                            continue
                        receive_ns = time.time_ns()
                        frame_type = "binary" if isinstance(frame, bytes) else "text"
                        if isinstance(frame, bytes):
                            raw_payload: Any = {
                                "encoding": "base64",
                                "data": base64.b64encode(frame).decode("ascii"),
                            }
                            text = ""
                            parsed = None
                            parse_error = "binary frame"
                        else:
                            raw_payload = frame
                            text = frame
                            try:
                                parsed = json.loads(frame)
                                parse_error = ""
                            except json.JSONDecodeError as exc:
                                parsed = None
                                parse_error = f"JSONDecodeError: {exc}"

                        if parsed is not None:
                            sync_candidates, time_candidates = candidate_fields(parsed)
                        else:
                            sync_candidates, time_candidates = {}, {}
                        normalized = normalize_status(sync_candidates, text)
                        record = {
                            "event": "frame",
                            "host_receive_unix_ns": receive_ns,
                            "host_receive_utc": utc_iso_from_ns(receive_ns),
                            "logger_hostname": logger_hostname,
                            "sensor_label": label,
                            "websocket_url": url,
                            "frame_type": frame_type,
                            "normalized_sync_status": normalized,
                            "sync_candidates": sync_candidates,
                            "time_candidates": time_candidates,
                            "parse_error": parse_error,
                            "raw_payload": raw_payload,
                        }
                        raw_file.write(json.dumps(record, default=str, sort_keys=True) + "\n")
                        status_writer.writerow(
                            {
                                "host_receive_unix_ns": receive_ns,
                                "host_receive_utc": record["host_receive_utc"],
                                "logger_hostname": logger_hostname,
                                "sensor_label": label,
                                "websocket_url": url,
                                "frame_type": frame_type,
                                "normalized_sync_status": normalized,
                                "sync_candidates_json": json.dumps(sync_candidates, sort_keys=True),
                                "time_candidates_json": json.dumps(time_candidates, sort_keys=True),
                                "parse_error": parse_error,
                            }
                        )
                        raw_file.flush()
                        status_file.flush()
                        frame_count += 1
                finally:
                    if poll_task is not None:
                        poll_task.cancel()
                        await asyncio.gather(poll_task, return_exceptions=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            connection_error_count += 1
            event_ns = time.time_ns()
            raw_file.write(
                json.dumps(
                    {
                        "event": "connection_error",
                        "host_unix_ns": event_ns,
                        "host_utc": utc_iso_from_ns(event_ns),
                        "sensor_label": label,
                        "url": url,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            raw_file.flush()
            if deadline is not None and time.monotonic() >= deadline:
                break
            await asyncio.sleep(args.reconnect_delay)
    return {
        "sensor_label": label,
        "websocket_url": url,
        "frame_count": frame_count,
        "connection_count": connection_count,
        "connection_error_count": connection_error_count,
    }


async def run(args: argparse.Namespace, connect: Any) -> int:
    logger_hostname = socket.gethostname()
    output_dir = (
        args.output_root.expanduser().resolve()
        / safe_name(args.session_id)
        / safe_name(f"robosense_websocket_{logger_hostname}")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    send_messages = read_send_messages(args)
    start_ns = time.time_ns()
    metadata = {
        "schema_version": 1,
        "session_id": args.session_id,
        "logger_hostname": logger_hostname,
        "sensors": [{"label": label, "url": url} for label, url in args.sensor],
        "start_unix_ns": start_ns,
        "start_utc": utc_iso_from_ns(start_ns),
        "requested_duration_s": args.duration,
        "poll_interval_s": args.poll_interval,
        "on_connect_messages": send_messages,
        "poll_message": args.poll_text,
        "interpretation_note": (
            "Lock/Fixed is a device synchronization state. It is not a measured "
            "clock offset. Raw frames are authoritative; normalized fields are "
            "best-effort extraction across firmware-specific schemas."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    raw_path = output_dir / "robosense_websocket_frames.jsonl"
    status_path = output_dir / "robosense_websocket_status.csv"
    start_monotonic = time.monotonic()
    deadline = None if args.duration == 0 else start_monotonic + args.duration
    interrupted = False

    with raw_path.open("x", encoding="utf-8", buffering=1) as raw_file, status_path.open(
        "x", newline="", encoding="utf-8", buffering=1
    ) as status_file:
        status_writer = csv.DictWriter(status_file, fieldnames=CSV_FIELDS)
        status_writer.writeheader()
        tasks = [
            asyncio.create_task(
                record_sensor(
                    label,
                    url,
                    args,
                    connect,
                    raw_file,
                    status_writer,
                    status_file,
                    deadline,
                    send_messages,
                    logger_hostname,
                )
            )
            for label, url in args.sensor
        ]
        sensor_summaries: list[dict[str, Any]] = []
        try:
            sensor_summaries = list(await asyncio.gather(*tasks))
        except KeyboardInterrupt:
            interrupted = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    summary = {
        "schema_version": 1,
        "session_id": args.session_id,
        "interrupted": interrupted,
        "start_unix_ns": start_ns,
        "end_unix_ns": time.time_ns(),
        "elapsed_monotonic_s": time.monotonic() - start_monotonic,
        "sensors": sensor_summaries,
        "total_frame_count": sum(item["frame_count"] for item in sensor_summaries),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0 if summary["total_frame_count"] else 2


def main() -> int:
    args = parse_args()
    if args.duration < 0 or args.poll_interval <= 0 or args.reconnect_delay <= 0:
        raise SystemExit("Duration must be non-negative and intervals must be positive.")
    try:
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect
    except ImportError as exc:
        raise SystemExit(
            "The 'websockets' package is required. Install it in the acquisition "
            "environment with: python3 -m pip install websockets"
        ) from exc
    return asyncio.run(run(args, connect))


if __name__ == "__main__":
    raise SystemExit(main())
