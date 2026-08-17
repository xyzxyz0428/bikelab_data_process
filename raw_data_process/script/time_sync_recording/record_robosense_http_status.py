#!/usr/bin/env python3
"""Record RoboSense diagnostic-page PTP status without changing the device.

RS-Helios firmware used by the front and rear LiDARs exposes its PTP state on
``/cgi-bin/diagnostic_info.cgi`` rather than through a WebSocket.  This logger
performs GET requests only, preserves each complete HTML response, and writes
the parsed input-field values to CSV.  The unit of the displayed
``PTP Master Offset`` is deliberately kept unspecified unless documented for
the exact firmware.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


CSV_FIELDS = (
    "sample_index",
    "sensor_label",
    "sensor_ip",
    "request_url",
    "host_request_start_unix_ns",
    "host_request_end_unix_ns",
    "host_midpoint_unix_ns",
    "host_midpoint_utc",
    "request_rtt_ms",
    "http_status",
    "http_date_header",
    "response_bytes",
    "ptp_status",
    "ptp_master_offset_raw",
    "ptp_master_offset_unit",
    "phase_lock_status",
    "time_sync_mode_reported",
    "time_sync_source",
    "all_input_fields_json",
    "error",
)


class InputFieldParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: dict[str, str] = {}
        self.current_select_name: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value for key, value in attrs}
        if tag == "select":
            self.current_select_name = attributes.get("name")
            return
        if tag == "option" and self.current_select_name and "selected" in attributes:
            self.fields[self.current_select_name] = (attributes.get("value") or "").strip()
            return
        if tag != "input":
            return
        name = attributes.get("name")
        if name:
            self.fields[name] = (attributes.get("value") or "").strip()

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "select":
            self.current_select_name = None


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid name: {value!r}")
    return cleaned


def utc_iso_from_ns(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc).isoformat()


def parse_sensor(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=IP_OR_HOSTNAME.")
    label, host = value.split("=", 1)
    label = safe_name(label)
    host = host.strip()
    if not host or "/" in host or "://" in host:
        raise argparse.ArgumentTypeError("Use LABEL=IP_OR_HOSTNAME without a URL path.")
    return label, host


def field_case_insensitive(fields: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in fields.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return ""


def fetch_page(url: str, timeout_s: float) -> dict[str, Any]:
    start_ns = time.time_ns()
    body = b""
    status: int | str = ""
    headers: dict[str, str] = {}
    error = ""
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BikeLab-TimeSyncLogger/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = response.status
            headers = dict(response.headers.items())
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        headers = dict(exc.headers.items()) if exc.headers else {}
        body = exc.read()
        error = f"HTTPError: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end_ns = time.time_ns()
    return {
        "start_unix_ns": start_ns,
        "end_unix_ns": end_ns,
        "http_status": status,
        "headers": headers,
        "body": body,
        "error": error,
    }


def decode_and_parse(body: bytes) -> tuple[str, dict[str, str], str]:
    if not body:
        return "", {}, ""
    text = body.decode("gb18030", errors="replace")
    parser = InputFieldParser()
    try:
        parser.feed(text)
        parse_error = ""
    except Exception as exc:
        parse_error = f"{type(exc).__name__}: {exc}"
    return text, parser.fields, parse_error


def collect_startup_pages(
    sensors: list[tuple[str, str]], timeout_s: float
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    requests = [
        (label, host, page_name, f"http://{host}{path}")
        for label, host in sensors
        for page_name, path in (
            ("device_info", "/cgi-bin/device_info.cgi"),
            ("parameter_settings", "/cgi-bin/param_setting.cgi"),
        )
    ]
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(fetch_page, url, timeout_s) for _, _, _, url in requests]
        results = [future.result() for future in futures]
    records: list[dict[str, Any]] = []
    time_sources: dict[str, str] = {}
    for (label, host, page_name, url), result in zip(requests, results):
        text, fields, parse_error = decode_and_parse(result["body"])
        if page_name == "parameter_settings":
            time_sources[host] = field_case_insensitive(
                fields, "TimeSyncSrc", "time_sync_src"
            )
        records.append(
            {
                "sensor_label": label,
                "sensor_ip": host,
                "page": page_name,
                "request_url": url,
                "host_request_start_unix_ns": result["start_unix_ns"],
                "host_request_end_unix_ns": result["end_unix_ns"],
                "http_status": result["http_status"],
                "response_headers": result["headers"],
                "fields": fields,
                "error": " | ".join(value for value in (result["error"], parse_error) if value),
                "raw_html": text,
            }
        )
    return records, time_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record read-only RoboSense CGI diagnostic PTP status."
    )
    parser.add_argument(
        "--sensor",
        action="append",
        required=True,
        type=parse_sensor,
        metavar="LABEL=IP",
        help="Repeat for each LiDAR, for example front=192.168.1.201.",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("time_sync_logs"))
    parser.add_argument("--duration", type=float, default=300.0, help="Seconds; 0 means until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=2.0)
    parser.add_argument(
        "--path",
        default="/cgi-bin/diagnostic_info.cgi",
        help="Read-only diagnostic path used by the installed firmware.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration < 0 or args.interval <= 0 or args.request_timeout <= 0:
        raise SystemExit("Duration must be non-negative and intervals/timeouts positive.")
    if not args.path.startswith("/"):
        raise SystemExit("--path must begin with '/'.")

    logger_hostname = socket.gethostname()
    output_dir = (
        args.output_root.expanduser().resolve()
        / safe_name(args.session_id)
        / safe_name(f"robosense_http_{logger_hostname}")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    start_ns = time.time_ns()
    metadata = {
        "schema_version": 1,
        "read_only": True,
        "session_id": args.session_id,
        "logger_hostname": logger_hostname,
        "sensors": [{"label": label, "host": host} for label, host in args.sensor],
        "diagnostic_path": args.path,
        "start_unix_ns": start_ns,
        "start_utc": utc_iso_from_ns(start_ns),
        "requested_duration_s": args.duration,
        "interval_s": args.interval,
        "ptp_master_offset_unit": "not declared by the installed Web UI",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    startup_pages, time_sources = collect_startup_pages(args.sensor, args.request_timeout)
    (output_dir / "robosense_startup_pages.json").write_text(
        json.dumps(startup_pages, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "robosense_http_ptp_status.csv"
    raw_path = output_dir / "robosense_http_responses.jsonl"
    start_monotonic = time.monotonic()
    sample_index = 0
    response_count = 0
    interrupted = False

    with csv_path.open("x", newline="", encoding="utf-8", buffering=1) as csv_file, raw_path.open(
        "x", encoding="utf-8", buffering=1
    ) as raw_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        try:
            while args.duration == 0 or time.monotonic() - start_monotonic < args.duration:
                cycle_started = time.monotonic()
                sensor_requests = [
                    (label, host, f"http://{host}{args.path}")
                    for label, host in args.sensor
                ]
                with ThreadPoolExecutor(max_workers=len(sensor_requests)) as executor:
                    futures = [
                        executor.submit(fetch_page, url, args.request_timeout)
                        for _, _, url in sensor_requests
                    ]
                    cycle_results = [future.result() for future in futures]
                for (label, host, url), result in zip(sensor_requests, cycle_results):
                    text, fields, parse_error = decode_and_parse(result["body"])
                    midpoint_ns = (result["start_unix_ns"] + result["end_unix_ns"]) // 2
                    error = " | ".join(value for value in (result["error"], parse_error) if value)
                    if result["http_status"] == 200 and result["body"] and not result["error"]:
                        response_count += 1
                    row = {
                        "sample_index": sample_index,
                        "sensor_label": label,
                        "sensor_ip": host,
                        "request_url": url,
                        "host_request_start_unix_ns": result["start_unix_ns"],
                        "host_request_end_unix_ns": result["end_unix_ns"],
                        "host_midpoint_unix_ns": midpoint_ns,
                        "host_midpoint_utc": utc_iso_from_ns(midpoint_ns),
                        "request_rtt_ms": f"{(result['end_unix_ns'] - result['start_unix_ns']) / 1e6:.6f}",
                        "http_status": result["http_status"],
                        "http_date_header": result["headers"].get("Date", ""),
                        "response_bytes": len(result["body"]),
                        "ptp_status": field_case_insensitive(
                            fields,
                            "ptp_remote_sync_status_text",
                            "ptp_status",
                            "time_sync_status_text",
                            "time_sync_status",
                            "sync_status",
                        ),
                        "ptp_master_offset_raw": field_case_insensitive(
                            fields,
                            "PTP_Master_Offset_text",
                            "ptp_master_offset",
                            "time_sync_data_text",
                            "time_sync_data",
                        ),
                        "ptp_master_offset_unit": "Web UI not specified",
                        "phase_lock_status": field_case_insensitive(fields, "phase_lock_text"),
                        "time_sync_mode_reported": field_case_insensitive(
                            fields, "time_sync_mode", "time_sync_mode_text"
                        ),
                        "time_sync_source": time_sources.get(host, ""),
                        "all_input_fields_json": json.dumps(fields, sort_keys=True),
                        "error": error,
                    }
                    writer.writerow(row)
                    raw_file.write(
                        json.dumps(
                            {
                                **{key: row[key] for key in CSV_FIELDS if key != "all_input_fields_json"},
                                "all_input_fields": fields,
                                "response_headers": result["headers"],
                                "raw_html": text,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    csv_file.flush()
                    raw_file.flush()
                sample_index += 1
                sleep_s = args.interval - (time.monotonic() - cycle_started)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            interrupted = True

    summary = {
        "schema_version": 1,
        "session_id": args.session_id,
        "sample_cycle_count": sample_index,
        "successful_http_response_count": response_count,
        "interrupted": interrupted,
        "start_unix_ns": start_ns,
        "end_unix_ns": time.time_ns(),
        "elapsed_monotonic_s": time.monotonic() - start_monotonic,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return 0 if response_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
