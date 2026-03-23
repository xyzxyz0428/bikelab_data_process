#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash extract_lidar_packet_timestamps.sh your_capture.pcapng out_dir \
#       192.168.1.200 192.168.1.201 192.168.1.202
#
# Output: one CSV per LiDAR IP with packet-level timestamps extracted by tshark.

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
    -e frame.len \
    > "$OUTCSV"
done

echo "Done. CSV files are in: $OUTDIR"
