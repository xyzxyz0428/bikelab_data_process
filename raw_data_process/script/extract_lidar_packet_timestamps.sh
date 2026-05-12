#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash extract_lidar_packet_timestamps_v2.sh your_capture.pcapng out_dir \
#       192.168.1.200 192.168.1.201 192.168.1.202
#
# Output:
#   One CSV per LiDAR IP with packet-level timestamps and packet metadata.
#   Fields include:
#     t_unix_ns, frame.number, frame.time_epoch, frame.time_relative,
#     ip.src, ip.dst, udp.srcport, udp.dstport, udp.length, frame.len, frame.cap_len

if [ "$#" -lt 5 ]; then
  echo "Usage: $0 <pcap_or_pcapng> <out_dir> <ip1> <ip2> <ip3>"
  exit 1
fi

PCAP="$1"
OUTDIR="$2"
shift 2
IPS=("$@")

mkdir -p "$OUTDIR"

for IP in "${IPS[@]}"; do
  SAFE_IP=${IP//./_}
  RAWCSV="$OUTDIR/lidar_packets_${SAFE_IP}_raw.csv"
  OUTCSV="$OUTDIR/lidar_packets_${SAFE_IP}.csv"

  echo "Extracting packets for $IP -> $OUTCSV"

  tshark -r "$PCAP" \
    -Y "ip.src == $IP && udp" \
    -T fields \
    -E header=y -E separator=, -E quote=d \
    -e frame.number \
    -e frame.time_epoch \
    -e frame.time_relative \
    -e ip.src \
    -e ip.dst \
    -e udp.srcport \
    -e udp.dstport \
    -e udp.length \
    -e frame.len \
    -e frame.cap_len \
    > "$RAWCSV"

  python3 - <<PY
import pandas as pd
from pathlib import Path

raw_path = Path(r"$RAWCSV")
out_path = Path(r"$OUTCSV")

df = pd.read_csv(raw_path)

# frame.time_epoch is in seconds -> convert to unix nanoseconds
df["frame.time_epoch"] = pd.to_numeric(df["frame.time_epoch"], errors="coerce")
df["t_unix_ns"] = (df["frame.time_epoch"] * 1_000_000_000).round().astype("Int64")

# Reorder columns
front_cols = ["t_unix_ns"]
other_cols = [c for c in df.columns if c not in front_cols]
df = df[front_cols + other_cols]

df.to_csv(out_path, index=False)
raw_path.unlink(missing_ok=True)
print(f"Wrote: {out_path}")
PY

done

echo "Done. CSV files are in: $OUTDIR"