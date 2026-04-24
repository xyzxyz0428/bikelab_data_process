#!/usr/bin/env python3
import os
import csv
import time
import math
from typing import List

import smbus

# -------- config --------
I2C_BUS = 1
I2C_ADDR = 0x24
REG = 0x17

SAMPLE_PERIOD_S = 0.1     # 10 Hz
FLUSH_ROWS = 500

# baseline from plate + support
BASELINE_G = 127.0

# second calibration process
# (adc_mean, extra_force_g) sorted by adc_mean
CAL_POINTS = [
    (0.48,   0.0),
    (73.34,  151.0),
    (113.52, 191.0),
    (167.17, 323.0),
    (239.23, 583.0),
    (296.07, 808.0),
    (363.59, 997.0),
    (402.77, 1581.0),
    (479.74, 2172.0),
    (520.39, 2598.0),
]

out_dir = os.path.expanduser("~/bikelab_interface_logs")
ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
OUT_CSV = os.path.join(out_dir, f"fsr_force_interp_{ts}.csv")


def now_unix_ns() -> int:
    return time.time_ns()


def g_to_newton(force_g: float) -> float:
    return force_g * 0.00980665


def adc_to_extra_force_g(adc: float, points=CAL_POINTS) -> float:
    """
    Piecewise linear interpolation.
    Input: adc raw value
    Output: extra force in grams (excluding baseline)
    """
    adc = float(adc)

    # clamp below range
    if adc <= points[0][0]:
        return points[0][1]

    # clamp above range
    if adc >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= adc <= x1:
            return y0 + (y1 - y0) * (adc - x0) / (x1 - x0)

    return float("nan")


def adc_to_total_force_g(adc: float, baseline_g: float = BASELINE_G) -> float:
    return adc_to_extra_force_g(adc) + baseline_g


class FSRForceInterpLogger:
    def __init__(self, out_csv: str, flush_rows: int = 500):
        self.out_csv = out_csv
        self.flush_rows = int(flush_rows)

        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

        self.bus = smbus.SMBus(I2C_BUS)
        self.rows: List[list] = []

        self._open_csv()

    def _open_csv(self):
        new_file = not os.path.exists(self.out_csv) or os.path.getsize(self.out_csv) == 0
        self.f = open(self.out_csv, "a", newline="")
        self.w = csv.writer(self.f)
        if new_file:
            self.w.writerow([
                "t_unix_ns",
                "ok",
                "adc_raw",
                "force_extra_g",
                "force_extra_n",
                "force_total_g",
                "force_total_n",
                "error",
            ])
            self.f.flush()

    def read_adc(self) -> int:
        """
        Keep the same read style as your existing script.
        """
        self.bus.write_byte(I2C_ADDR, REG)
        return int(self.bus.read_word_data(I2C_ADDR, REG))

    def log_once(self):
        t = now_unix_ns()
        try:
            adc = self.read_adc()

            force_extra_g = adc_to_extra_force_g(adc)
            force_extra_n = g_to_newton(force_extra_g)

            force_total_g = force_extra_g + BASELINE_G
            force_total_n = g_to_newton(force_total_g)

            row = [
                t,
                1,
                adc,
                force_extra_g,
                force_extra_n,
                force_total_g,
                force_total_n,
                "",
            ]
        except Exception as e:
            row = [
                t,
                0,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                str(e),
            ]

        self.rows.append(row)

        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        self.w.writerows(self.rows)
        self.f.flush()
        self.rows.clear()

    def close(self):
        try:
            self.flush()
        finally:
            try:
                self.f.close()
            except Exception:
                pass
            try:
                self.bus.close()
            except Exception:
                pass

    def run(self, period_s: float = SAMPLE_PERIOD_S):
        print(f"[INFO] Logging FSR interpolated force to {self.out_csv}")
        print(f"[INFO] Sampling period: {period_s}s, flush every {self.flush_rows} rows")

        next_t = time.time()
        try:
            while True:
                self.log_once()

                next_t += period_s
                sleep_s = next_t - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.time()
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl-C received. Flushing and exiting...")
        finally:
            self.close()
            print("[INFO] Done.")


if __name__ == "__main__":
    logger = FSRForceInterpLogger(out_csv=OUT_CSV, flush_rows=FLUSH_ROWS)
    logger.run(period_s=SAMPLE_PERIOD_S)