#!/usr/bin/env python3
"""Extract LiDAR scan timestamps from a three-sensor RoboSense PCAP.

The PCAP is streamed sequentially.  A scan boundary is detected when the
azimuth of the first MSOP measurement block wraps from the end of a rotation
to its beginning.  UDP packet timestamps are not reported as LiDAR frame
timestamps.
"""

import socket
import struct
from pathlib import Path

import numpy as np


LIDAR_IPS = {
    "192.168.1.200": "Near-range LiDAR",
    "192.168.1.201": "Front LiDAR",
    "192.168.1.202": "Rear LiDAR",
}
MSOP_PORTS = {
    "192.168.1.200": 2000,
    "192.168.1.201": 2010,
    "192.168.1.202": 2020,
}


def _pcap_format(header):
    magic = header[:4]
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000),
        b"\xa1\xb2\xc3\xd4": (">", 1_000),
        b"M<\xb2\xa1": ("<", 1),
        b"\xa1\xb2<M": (">", 1),
    }
    if magic not in formats:
        raise ValueError("Unsupported PCAP magic number")
    return formats[magic]


def extract_scan_timestamps(pcap_path, start_ns=None, end_ns=None):
    """Return scan-boundary capture timestamps for IPs .200, .201 and .202."""
    pcap_path = Path(pcap_path).resolve()
    scans = {name: [] for name in LIDAR_IPS.values()}
    previous_azimuth = {ip: None for ip in LIDAR_IPS}
    packet_counts = {ip: 0 for ip in LIDAR_IPS}
    start_ns = int(start_ns) if start_ns is not None else None
    end_ns = int(end_ns) if end_ns is not None else None

    with pcap_path.open("rb", buffering=8 * 1024 * 1024) as stream:
        global_header = stream.read(24)
        if len(global_header) != 24:
            raise ValueError(f"Invalid PCAP header: {pcap_path}")
        endian, fraction_to_ns = _pcap_format(global_header)
        unpack_packet = struct.Struct(endian + "IIII").unpack

        while True:
            packet_header = stream.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                raise ValueError("Truncated PCAP packet header")
            seconds, fraction, captured_length, _ = unpack_packet(packet_header)
            timestamp_ns = seconds * 1_000_000_000 + fraction * fraction_to_ns
            if end_ns is not None and timestamp_ns > end_ns:
                break
            if start_ns is not None and timestamp_ns < start_ns:
                stream.seek(captured_length, 1)
                continue
            # The Ethernet/IP/UDP headers and the first MSOP measurement
            # block fit in the first 96 captured bytes.  Seek over the point
            # payload instead of reading it; this is important for multi-GB
            # captures on an external data volume.
            prefix_length = min(captured_length, 96)
            packet = stream.read(prefix_length)
            stream.seek(captured_length - prefix_length, 1)
            if len(packet) != prefix_length:
                raise ValueError("Truncated PCAP packet")
            if len(packet) < 42 or packet[12:14] != b"\x08\x00":
                continue
            ip_offset = 14
            if packet[ip_offset + 9] != 17:  # UDP
                continue
            header_length = (packet[ip_offset] & 0x0F) * 4
            source_ip = socket.inet_ntoa(packet[ip_offset + 12:ip_offset + 16])
            if source_ip not in LIDAR_IPS:
                continue
            udp_offset = ip_offset + header_length
            if len(packet) < udp_offset + 8:
                continue
            source_port = struct.unpack("!H", packet[udp_offset:udp_offset + 2])[0]
            if source_port != MSOP_PORTS[source_ip]:
                continue
            payload = packet[udp_offset + 8:]
            block = payload.find(b"\xff\xee", 40)
            if block < 0 or block + 4 > len(payload):
                continue
            azimuth = struct.unpack("!H", payload[block + 2:block + 4])[0]
            if azimuth > 35999:
                continue
            previous = previous_azimuth[source_ip]
            packet_counts[source_ip] += 1
            # Normal packet-to-packet changes are a few degrees.  A decrease
            # greater than 180 degrees denotes the 360 -> 0 degree wrap.
            if previous is not None and previous - azimuth > 18_000:
                scans[LIDAR_IPS[source_ip]].append(timestamp_ns)
            previous_azimuth[source_ip] = azimuth

    return {
        name: np.asarray(timestamps, dtype=np.int64)
        for name, timestamps in scans.items()
    }, packet_counts
