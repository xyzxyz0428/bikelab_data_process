#!/usr/bin/env python3
import os
import csv
import time
import math
import statistics
from typing import List

import smbus

# -------- config --------
I2C_BUS = 1
I2C_ADDR = 0x24
REG = 0x17

SAMPLE_PERIOD_S = 0.05   # 20 Hz
SAMPLES_PER_STEP = 100   # each calibration point: 100 samples ~ 5 s
OUT_DIR = os.path.expanduser("~/bikelab_interface_logs")


class FSRReader:
    def __init__(self):
        self.bus = smbus.SMBus(I2C_BUS)

    def read_adc(self) -> int:
        self.bus.write_byte(I2C_ADDR, REG)
        return int(self.bus.read_word_data(I2C_ADDR, REG))

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass


def collect_samples(reader: FSRReader, n: int, period_s: float) -> List[int]:
    vals = []
    next_t = time.time()
    for _ in range(n):
        vals.append(reader.read_adc())
        next_t += period_s
        sleep_s = next_t - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_t = time.time()
    return vals


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_csv = os.path.join(OUT_DIR, f"fsr_calibration_points_{ts}.csv")

    reader = FSRReader()

    try:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "label",
                "force_g",
                "force_n",
                "adc_mean",
                "adc_std",
                "adc_min",
                "adc_max",
                "num_samples",
            ])

            print(f"[INFO] Output: {out_csv}")
            print("[INFO] Enter known load in grams. Example: 0, 100, 200, 500")
            print("[INFO] Press Enter with empty input to finish.\n")

            while True:
                s = input("Known load [g]: ").strip()
                if s == "":
                    break

                force_g = float(s)
                force_n = force_g * 0.00980665
                label = f"{force_g:.1f}g"

                input(f"[INFO] Apply {label}, keep stable, then press Enter to start sampling...")

                vals = collect_samples(reader, SAMPLES_PER_STEP, SAMPLE_PERIOD_S)

                adc_mean = statistics.mean(vals)
                adc_std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                adc_min = min(vals)
                adc_max = max(vals)

                print(
                    f"[INFO] {label}: mean={adc_mean:.2f}, std={adc_std:.2f}, "
                    f"min={adc_min}, max={adc_max}"
                )

                w.writerow([
                    label,
                    force_g,
                    force_n,
                    adc_mean,
                    adc_std,
                    adc_min,
                    adc_max,
                    len(vals),
                ])
                f.flush()

    finally:
        reader.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()