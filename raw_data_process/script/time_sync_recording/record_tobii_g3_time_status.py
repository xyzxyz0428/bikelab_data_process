#!/usr/bin/env python3
"""Record Tobii Pro Glasses 3 NTP state and device/host time samples.

This logger is read-only: it does not enable NTP and does not set device time.
For each device-time request, host timestamps are taken immediately before and
after the API call.  The reported device-minus-host value uses the host request
midpoint and is therefore an API round-trip estimate, not an NTP offset report.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable


CSV_FIELDS = (
    "sample_index",
    "host_request_start_unix_ns",
    "host_request_end_unix_ns",
    "host_midpoint_unix_ns",
    "host_midpoint_utc",
    "device_time_reported",
    "device_time_unix_ns",
    "device_minus_host_midpoint_ms",
    "device_time_request_rtt_ms",
    "ntp_is_enabled",
    "ntp_is_synchronized",
    "ntp_enabled_request_rtt_ms",
    "ntp_synchronized_request_rtt_ms",
    "hostname",
    "recording_unit",
    "error",
)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid name: {value!r}")
    return cleaned


def datetime_to_unix_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(round(value.timestamp() * 1_000_000_000))


def utc_iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def bounded_call(awaitable: Awaitable[Any], timeout_s: float) -> dict[str, Any]:
    start_ns = time.time_ns()
    try:
        value = await asyncio.wait_for(awaitable, timeout=timeout_s)
        error = ""
        ok = True
    except Exception as exc:
        value = None
        error = f"{type(exc).__name__}: {exc}"
        ok = False
    end_ns = time.time_ns()
    return {
        "ok": ok,
        "value": value,
        "error": error,
        "start_unix_ns": start_ns,
        "end_unix_ns": end_ns,
        "rtt_ms": (end_ns - start_ns) / 1e6,
    }


class DirectSystem:
    """Minimal read-only implementation of the required Glasses 3 API calls."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.next_id = 0

    async def _get(self, property_name: str) -> Any:
        self.next_id += 1
        request_id = self.next_id
        await self.websocket.send(
            json.dumps(
                {
                    "path": f"/system.{property_name}",
                    "id": request_id,
                    "method": "GET",
                }
            )
        )
        while True:
            response = json.loads(await self.websocket.recv())
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise RuntimeError(str(response.get("message") or response["error"]))
            return response.get("body")

    async def get_name(self) -> str:
        return str(await self._get("name"))

    async def get_head_unit_serial(self) -> str:
        return str(await self._get("head-unit-serial"))

    async def get_recording_unit_serial(self) -> str:
        return str(await self._get("recording-unit-serial"))

    async def get_version(self) -> str:
        return str(await self._get("version"))

    async def get_timezone(self) -> str:
        return str(await self._get("timezone"))

    async def get_time(self) -> datetime:
        value = str(await self._get("time"))
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    async def get_ntp_is_enabled(self) -> bool:
        return bool(await self._get("ntp-is-enabled"))

    async def get_ntp_is_synchronized(self) -> bool:
        return bool(await self._get("ntp-is-synchronized"))


class DirectG3:
    def __init__(self, websocket: Any) -> None:
        self.system = DirectSystem(websocket)


class DirectConnection:
    def __init__(self, hostname: str, connect: Any, timeout_s: float) -> None:
        self.hostname = hostname
        self.connect = connect
        self.timeout_s = timeout_s
        self.websocket: Any = None

    async def __aenter__(self) -> DirectG3:
        self.websocket = await self.connect(
            f"ws://{self.hostname}/websocket",
            subprotocols=["g3api"],
            open_timeout=self.timeout_s,
            max_size=2_000_000,
        )
        return DirectG3(self.websocket)

    async def __aexit__(self, *_: Any) -> None:
        if self.websocket is not None:
            await self.websocket.close()


class DirectConnector:
    def __init__(self, connect: Any, timeout_s: float) -> None:
        self.connect = connect
        self.timeout_s = timeout_s

    def with_hostname(self, hostname: str) -> DirectConnection:
        return DirectConnection(hostname, self.connect, self.timeout_s)


async def read_metadata(g3: Any, timeout_s: float) -> dict[str, Any]:
    calls = {
        "name": g3.system.get_name(),
        "head_unit_serial": g3.system.get_head_unit_serial(),
        "recording_unit_serial": g3.system.get_recording_unit_serial(),
        "version": g3.system.get_version(),
        "timezone": g3.system.get_timezone(),
    }
    results: dict[str, Any] = {}
    for name, awaitable in calls.items():
        result = await bounded_call(awaitable, timeout_s)
        results[name] = result["value"] if result["ok"] else None
        if result["error"]:
            results[f"{name}_error"] = result["error"]
    return results


async def collect_sample(g3: Any, sample_index: int, timeout_s: float, hostname: str) -> dict[str, Any]:
    time_result = await bounded_call(g3.system.get_time(), timeout_s)
    enabled_result = await bounded_call(g3.system.get_ntp_is_enabled(), timeout_s)
    synchronized_result = await bounded_call(
        g3.system.get_ntp_is_synchronized(), timeout_s
    )
    errors = [
        result["error"]
        for result in (time_result, enabled_result, synchronized_result)
        if result["error"]
    ]

    device_time = time_result["value"] if time_result["ok"] else None
    device_time_ns = datetime_to_unix_ns(device_time) if isinstance(device_time, datetime) else None
    host_midpoint_ns = (
        time_result["start_unix_ns"] + time_result["end_unix_ns"]
    ) // 2
    offset_ms = (
        (device_time_ns - host_midpoint_ns) / 1e6
        if device_time_ns is not None
        else None
    )
    return {
        "sample_index": sample_index,
        "host_request_start_unix_ns": time_result["start_unix_ns"],
        "host_request_end_unix_ns": time_result["end_unix_ns"],
        "host_midpoint_unix_ns": host_midpoint_ns,
        "host_midpoint_utc": utc_iso_from_ns(host_midpoint_ns),
        "device_time_reported": device_time.isoformat() if isinstance(device_time, datetime) else "",
        "device_time_unix_ns": device_time_ns if device_time_ns is not None else "",
        "device_minus_host_midpoint_ms": f"{offset_ms:.6f}" if offset_ms is not None else "",
        "device_time_request_rtt_ms": f"{time_result['rtt_ms']:.6f}",
        "ntp_is_enabled": enabled_result["value"] if enabled_result["ok"] else "",
        "ntp_is_synchronized": (
            synchronized_result["value"] if synchronized_result["ok"] else ""
        ),
        "ntp_enabled_request_rtt_ms": f"{enabled_result['rtt_ms']:.6f}",
        "ntp_synchronized_request_rtt_ms": f"{synchronized_result['rtt_ms']:.6f}",
        "hostname": hostname,
        "recording_unit": "",
        "error": " | ".join(errors),
        "raw_calls": {
            "device_time": time_result,
            "ntp_is_enabled": enabled_result,
            "ntp_is_synchronized": synchronized_result,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record Tobii Glasses 3 NTP state and bounded device/host time samples."
    )
    parser.add_argument("--hostname", default=os.getenv("G3_HOSTNAME"), help="Glasses hostname or IP.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("time_sync_logs"))
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds; 0 means until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    parser.add_argument(
        "--g3-pylib-src",
        type=Path,
        default=Path("/home/bikelab_ws/ros2_ws/src/bikelab_interfaces2/glasses3-pylib/src"),
        help="Path containing the g3pylib package when it is not installed.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace, connect_to_glasses: Any, api_backend: str = "g3pylib") -> int:
    local_hostname = socket.gethostname()
    output_dir = (
        args.output_root.expanduser().resolve()
        / safe_name(args.session_id)
        / safe_name(f"tobii_{local_hostname}")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "session_id": args.session_id,
        "logger_hostname": local_hostname,
        "glasses_hostname": args.hostname,
        "start_unix_ns": time.time_ns(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "interval_s": args.interval,
        "requested_duration_s": args.duration,
        "api_backend": api_backend,
        "measurement_note": (
            "device_minus_host_midpoint_ms uses host timestamps bracketing the "
            "Glasses API get_time call. It is an API round-trip estimate, not "
            "an NTP protocol offset or an independent accuracy measurement."
        ),
        "read_only": True,
        "time_reference_note": (
            "Computer 2 is the intentionally offline local time reference. "
            "The target is agreement with that local timeline, not UTC traceability."
        ),
    }

    samples_path = output_dir / "tobii_time_sync_samples.csv"
    raw_path = output_dir / "tobii_time_sync_raw.jsonl"
    sample_index = 0
    interrupted = False
    start_monotonic = time.monotonic()
    next_sample = start_monotonic

    try:
        async with connect_to_glasses.with_hostname(args.hostname) as g3:
            device_metadata = await read_metadata(g3, args.request_timeout)
            metadata["device"] = device_metadata
            recording_unit = str(device_metadata.get("recording_unit_serial") or "")
            write_json(output_dir / "metadata.json", metadata)

            with samples_path.open("x", newline="", encoding="utf-8", buffering=1) as csv_file, raw_path.open(
                "x", encoding="utf-8", buffering=1
            ) as raw_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                while args.duration == 0 or time.monotonic() - start_monotonic < args.duration:
                    now = time.monotonic()
                    if now < next_sample:
                        await asyncio.sleep(min(next_sample - now, 0.2))
                        continue
                    sample = await collect_sample(
                        g3, sample_index, args.request_timeout, local_hostname
                    )
                    sample["recording_unit"] = recording_unit
                    writer.writerow({field: sample.get(field, "") for field in CSV_FIELDS})
                    raw_file.write(json.dumps(sample, default=str, sort_keys=True) + "\n")
                    csv_file.flush()
                    raw_file.flush()
                    sample_index += 1
                    next_sample = time.monotonic() + args.interval
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        metadata["connection_error"] = f"{type(exc).__name__}: {exc}"
        if not (output_dir / "metadata.json").exists():
            write_json(output_dir / "metadata.json", metadata)

    summary = {
        "schema_version": 1,
        "session_id": args.session_id,
        "sample_count": sample_index,
        "interrupted": interrupted,
        "start_unix_ns": metadata["start_unix_ns"],
        "end_unix_ns": time.time_ns(),
        "elapsed_monotonic_s": time.monotonic() - start_monotonic,
        "connection_error": metadata.get("connection_error", ""),
    }
    write_json(output_dir / "summary.json", summary)
    print(output_dir)
    return 0 if sample_index else 2


def main() -> int:
    args = parse_args()
    if not args.hostname:
        raise SystemExit("Pass --hostname or set G3_HOSTNAME.")
    if args.interval <= 0 or args.duration < 0 or args.request_timeout <= 0:
        raise SystemExit("Intervals/timeouts must be positive and duration non-negative.")
    if args.g3_pylib_src.exists():
        sys.path.insert(0, str(args.g3_pylib_src.resolve()))
    g3_import_error = ""
    try:
        from g3pylib import connect_to_glasses

        connector = connect_to_glasses
        api_backend = "g3pylib"
    except ImportError as exc:
        g3_import_error = str(exc)
        try:
            try:
                from websockets.asyncio.client import connect
            except ImportError:
                from websockets import connect
        except ImportError as websocket_exc:
            raise SystemExit(
                "Neither g3pylib nor the lightweight WebSocket fallback is available. "
                "Install the fallback with 'python3 -m pip install websockets~=10.3'. "
                f"g3pylib error: {g3_import_error}; websockets error: {websocket_exc}"
            ) from websocket_exc
        connector = DirectConnector(connect, args.request_timeout)
        api_backend = "direct_read_only_websocket"
    return asyncio.run(run(args, connector, api_backend))


if __name__ == "__main__":
    raise SystemExit(main())
