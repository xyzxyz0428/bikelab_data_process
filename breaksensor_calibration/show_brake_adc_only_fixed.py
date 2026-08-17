#!/usr/bin/env python3
"""
Show raw ADC values of both brake sensors only.
No force conversion. No CSV saving.

Left brake sensor register:  0x17
Right brake sensor register: 0x14
Default I2C address: 0x24
"""

import argparse
import time

import smbus

I2C_BUS = 1
I2C_ADDR = 0x24
LEFT_REG = 0x17
RIGHT_REG = 0x14
DEFAULT_PERIOD_S = 0.1  # 10 Hz


def read_adc(bus: smbus.SMBus, i2c_addr: int, reg: int) -> int:
    """Read raw ADC value from one register."""
    bus.write_byte(i2c_addr, reg)
    raw = bus.read_word_data(i2c_addr, reg)
    return int(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw ADC values from left and right brake sensors."
    )
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PERIOD_S,
        help="Sampling period in seconds. Default: 0.1 s = 10 Hz.",
    )
    parser.add_argument(
        "--i2c-bus",
        type=int,
        default=I2C_BUS,
        help="I2C bus number. Default: 1.",
    )
    parser.add_argument(
        "--i2c-addr",
        type=lambda x: int(x, 0),
        default=I2C_ADDR,
        help="I2C device address. Default: 0x24. You can pass decimal or hex, e.g. 0x24.",
    )
    args = parser.parse_args()

    bus = smbus.SMBus(args.i2c_bus)

    print("[INFO] Showing raw ADC values only. No file will be saved.")
    print(f"[INFO] I2C bus: {args.i2c_bus}, I2C addr: 0x{args.i2c_addr:02x}")
    print(f"[INFO] Left reg: 0x{LEFT_REG:02x}, right reg: 0x{RIGHT_REG:02x}")
    print("[INFO] Press Ctrl-C to stop.\n")
    print("t_unix_ns,left_adc_raw,right_adc_raw,left_error,right_error")

    next_t = time.time()
    try:
        while True:
            t_ns = time.time_ns()

            left_adc = ""
            right_adc = ""
            left_error = ""
            right_error = ""

            try:
                left_adc = read_adc(bus, args.i2c_addr, LEFT_REG)
            except Exception as e:
                left_error = str(e)

            try:
                right_adc = read_adc(bus, args.i2c_addr, RIGHT_REG)
            except Exception as e:
                right_error = str(e)

            print(f"{t_ns},{left_adc},{right_adc},{left_error},{right_error}", flush=True)

            next_t += args.period
            sleep_s = next_t - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        try:
            bus.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
