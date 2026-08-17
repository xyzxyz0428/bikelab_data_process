#!/usr/bin/env python3
"""Record host NTP/chrony state and optional local PTP evidence.

The same script is intended for Computer 1 and Computer 2.  It never changes
the system clock or synchronization configuration.  Each run creates a new
session directory and stores both parsed CSV samples and unmodified command
outputs for later audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


TRACKING_FIELDS = (
    "reference_id",
    "reference_name",
    "stratum",
    "reference_time_utc",
    "system_time_offset_s",
    "last_offset_s",
    "rms_offset_s",
    "frequency_ppm",
    "residual_frequency_ppm",
    "skew_ppm",
    "root_delay_s",
    "root_dispersion_s",
    "update_interval_s",
    "leap_status",
)

CSV_FIELDS = (
    "sample_index",
    "host_unix_ns_before",
    "host_unix_ns_after",
    "host_midpoint_unix_ns",
    "host_utc_midpoint",
    "sample_duration_ms",
    "monotonic_ns",
    "hostname",
    "role",
    "peer",
    "chrony_available",
    "reference_id",
    "reference_name",
    "stratum",
    "reference_time_utc",
    "system_time_offset_s",
    "last_offset_s",
    "rms_offset_s",
    "frequency_ppm",
    "residual_frequency_ppm",
    "skew_ppm",
    "root_delay_s",
    "root_dispersion_s",
    "update_interval_s",
    "leap_status",
    "selected_source",
    "selected_source_raw",
    "ntp_synchronized",
    "ping_rtt_ms",
    "pmc_ok",
    "error_summary",
)


def utc_iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid name: {value!r}")
    return cleaned


def run_command(command: Sequence[str], timeout_s: float = 3.0) -> dict[str, Any]:
    started_ns = time.time_ns()
    try:
        completed = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "started_unix_ns": started_ns,
            "finished_unix_ns": time.time_ns(),
        }
    except FileNotFoundError as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "started_unix_ns": started_ns,
            "finished_unix_ns": time.time_ns(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "command": list(command),
            "returncode": None,
            "stdout": stdout,
            "stderr": f"timeout after {timeout_s:.3f} s; {stderr}",
            "started_unix_ns": started_ns,
            "finished_unix_ns": time.time_ns(),
        }


def first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def parse_tracking(text: str) -> dict[str, str]:
    line = first_nonempty_line(text)
    if not line:
        return {}
    values = next(csv.reader([line]))
    return {
        name: values[index].strip()
        for index, name in enumerate(TRACKING_FIELDS)
        if index < len(values)
    }


def parse_selected_source(text: str) -> tuple[str, str]:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        values = next(csv.reader([stripped]))
        marker_index = next(
            (index for index, value in enumerate(values[:3]) if "*" in value),
            None,
        )
        if marker_index is not None:
            # chronyc versions commonly emit either `^,*,address,...` or
            # `^*,address,...`; support both layouts.
            if marker_index == 1 and len(values) > 2:
                source = values[2].strip()
            elif marker_index == 0 and len(values) > 1:
                source = values[1].strip()
            else:
                source = ""
            return source, stripped
    return "", ""


def parse_timedatectl(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("NTPSynchronized="):
            return line.split("=", 1)[1].strip()
    return ""


def parse_ping_rtt_ms(text: str) -> str:
    match = re.search(r"time[=<]([0-9.]+)\s*ms", text)
    return match.group(1) if match else ""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_snapshot(interface: str | None) -> dict[str, Any]:
    commands: dict[str, Sequence[str]] = {
        "date_utc": ("date", "--iso-8601=ns", "--utc"),
        "timedatectl": (
            "timedatectl",
            "show",
            "-p",
            "Timezone",
            "-p",
            "NTP",
            "-p",
            "NTPSynchronized",
        ),
        "ip_address": ("ip", "-j", "address", "show"),
        "ip_route": ("ip", "route", "show"),
        "chrony_version": ("chronyc", "--version"),
        "ptp4l_version": ("ptp4l", "-v"),
        "pmc_version": ("pmc", "-v"),
    }
    if interface:
        commands["interface_timestamp_capabilities"] = ("ethtool", "-T", interface)
        commands["interface_details"] = ("ip", "-j", "address", "show", "dev", interface)
    return {name: run_command(command, timeout_s=5.0) for name, command in commands.items()}


def start_ptp_capture(interface: str, output_path: Path) -> tuple[subprocess.Popen[str] | None, str]:
    if shutil.which("tcpdump") is None:
        return None, "tcpdump was not found"
    command = [
        "tcpdump",
        "-i",
        interface,
        "-s",
        "256",
        "-U",
        "-w",
        str(output_path),
        "udp",
        "port",
        "319",
        "or",
        "udp",
        "port",
        "320",
        "or",
        "ether",
        "proto",
        "0x88f7",
    ]
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.25)
        if process.poll() is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            return None, stderr or f"tcpdump exited with code {process.returncode}"
        return process, ""
    except OSError as exc:
        return None, str(exc)


def stop_ptp_capture(process: subprocess.Popen[str] | None) -> dict[str, Any]:
    if process is None:
        return {"started": False}
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        _, stderr = process.communicate(timeout=8.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        _, stderr = process.communicate(timeout=3.0)
    return {
        "started": True,
        "returncode": process.returncode,
        "stderr": (stderr or "").strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record chrony/NTP status on Computer 1 or Computer 2."
    )
    parser.add_argument("--role", required=True, choices=("computer1", "computer2"))
    parser.add_argument("--peer", required=True, help="Other computer IP address.")
    parser.add_argument("--session-id", required=True, help="Use the same ID on both computers.")
    parser.add_argument("--output-root", type=Path, default=Path("time_sync_logs"))
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds; 0 means until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--command-timeout", type=float, default=3.0)
    parser.add_argument("--include-ptp", action="store_true", help="Query the local ptp4l daemon with pmc.")
    parser.add_argument("--ptp-query-interval", type=float, default=5.0)
    parser.add_argument("--capture-ptp", action="store_true", help="Capture PTP packets with tcpdump.")
    parser.add_argument("--interface", help="Network interface used for PTP capture and ethtool metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.duration < 0 or args.ptp_query_interval <= 0:
        raise SystemExit("Intervals must be positive and duration must be non-negative.")
    if args.capture_ptp and not args.interface:
        raise SystemExit("--capture-ptp requires --interface.")

    hostname = socket.gethostname()
    session_id = safe_name(args.session_id)
    run_name = safe_name(f"{args.role}_{hostname}")
    output_dir = args.output_root.expanduser().resolve() / session_id / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": 2,
        "session_id": session_id,
        "role": args.role,
        "hostname": hostname,
        "peer": args.peer,
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "start_unix_ns": time.time_ns(),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "interval_s": args.interval,
        "requested_duration_s": args.duration,
        "include_ptp": args.include_ptp,
        "capture_ptp": args.capture_ptp,
        "interface": args.interface,
        "startup_commands": command_snapshot(args.interface),
    }
    write_json(output_dir / "metadata.json", metadata)

    capture_process: subprocess.Popen[str] | None = None
    capture_error = ""
    if args.capture_ptp:
        capture_process, capture_error = start_ptp_capture(
            args.interface, output_dir / "ptp_control_packets.pcap"
        )

    samples_path = output_dir / "host_sync_samples.csv"
    raw_path = output_dir / "command_outputs.jsonl"
    start_monotonic = time.monotonic()
    next_sample = start_monotonic
    next_ptp_query = start_monotonic
    sample_index = 0
    interrupted = False

    with samples_path.open("x", newline="", encoding="utf-8", buffering=1) as csv_file, raw_path.open(
        "x", encoding="utf-8", buffering=1
    ) as raw_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            while args.duration == 0 or time.monotonic() - start_monotonic < args.duration:
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(min(next_sample - now, 0.2))
                    continue

                host_before_ns = time.time_ns()
                commands = {
                    "chrony_tracking": run_command(
                        ("chronyc", "-c", "tracking"), args.command_timeout
                    ),
                    "chrony_sources": run_command(
                        ("chronyc", "-c", "-n", "sources"), args.command_timeout
                    ),
                    "chrony_sourcestats": run_command(
                        ("chronyc", "-c", "-n", "sourcestats"), args.command_timeout
                    ),
                    "timedatectl": run_command(
                        ("timedatectl", "show", "-p", "NTPSynchronized"),
                        args.command_timeout,
                    ),
                    "peer_ping": run_command(
                        ("ping", "-n", "-c", "1", "-W", "1", args.peer),
                        max(args.command_timeout, 1.5),
                    ),
                }
                pmc_queried = False
                if args.include_ptp and time.monotonic() >= next_ptp_query:
                    pmc_queried = True
                    commands["pmc_current_data_set"] = run_command(
                        ("pmc", "-u", "-b", "0", "GET CURRENT_DATA_SET"),
                        args.command_timeout,
                    )
                    commands["pmc_port_data_set"] = run_command(
                        ("pmc", "-u", "-b", "0", "GET PORT_DATA_SET"),
                        args.command_timeout,
                    )
                    commands["pmc_time_status_np"] = run_command(
                        ("pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"),
                        args.command_timeout,
                    )
                    next_ptp_query = time.monotonic() + args.ptp_query_interval

                host_after_ns = time.time_ns()
                midpoint_ns = (host_before_ns + host_after_ns) // 2
                tracking = parse_tracking(commands["chrony_tracking"]["stdout"])
                selected_source, selected_source_raw = parse_selected_source(
                    commands["chrony_sources"]["stdout"]
                )
                errors = [
                    f"{name}: {result['stderr'].strip()}"
                    for name, result in commands.items()
                    if result["returncode"] != 0 and result["stderr"].strip()
                ]
                pmc_results = [result for name, result in commands.items() if name.startswith("pmc_")]
                row = {
                    "sample_index": sample_index,
                    "host_unix_ns_before": host_before_ns,
                    "host_unix_ns_after": host_after_ns,
                    "host_midpoint_unix_ns": midpoint_ns,
                    "host_utc_midpoint": utc_iso_from_ns(midpoint_ns),
                    "sample_duration_ms": f"{(host_after_ns - host_before_ns) / 1e6:.6f}",
                    "monotonic_ns": time.monotonic_ns(),
                    "hostname": hostname,
                    "role": args.role,
                    "peer": args.peer,
                    "chrony_available": commands["chrony_tracking"]["returncode"] == 0,
                    **tracking,
                    "selected_source": selected_source,
                    "selected_source_raw": selected_source_raw,
                    "ntp_synchronized": parse_timedatectl(commands["timedatectl"]["stdout"]),
                    "ping_rtt_ms": parse_ping_rtt_ms(commands["peer_ping"]["stdout"]),
                    "pmc_ok": (
                        all(result["returncode"] == 0 for result in pmc_results)
                        if pmc_queried
                        else ""
                    ),
                    "error_summary": " | ".join(errors),
                }
                writer.writerow(row)
                raw_file.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "host_midpoint_unix_ns": midpoint_ns,
                            "commands": commands,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                csv_file.flush()
                raw_file.flush()
                sample_index += 1
                next_sample = time.monotonic() + args.interval
        except KeyboardInterrupt:
            interrupted = True

    capture_result = stop_ptp_capture(capture_process)
    if capture_error:
        capture_result["start_error"] = capture_error
    summary = {
        "schema_version": 2,
        "session_id": session_id,
        "role": args.role,
        "hostname": hostname,
        "sample_count": sample_index,
        "interrupted": interrupted,
        "start_unix_ns": metadata["start_unix_ns"],
        "end_unix_ns": time.time_ns(),
        "elapsed_monotonic_s": time.monotonic() - start_monotonic,
        "ptp_capture": capture_result,
    }
    write_json(output_dir / "summary.json", summary)
    print(output_dir)
    if sample_index == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
