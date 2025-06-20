"""Synthetic sensor data generator for GranaryPredict.

Produces a CSV containing at minimum the columns required by the
pipeline:
    detection_time, detection_cycle, grid_x, grid_y, grid_z,
    temperature_grain, temperature_inside, temperature_outside,
    humidity_warehouse, humidity_outside, grain_type

Additional metadata columns are included to mimic real exports.

Usage (from project root):
    python scripts/generate_fake_sensor_data.py \
        --days 30 \
        --grid 4 5 3 \
        --output data/raw/synthetic_sensor_data.csv
"""
from __future__ import annotations

import itertools
import math
import random
from argparse import ArgumentParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

GRAIN_TYPES = [
    "Mid-to-late indica rice",
    "Early indica rice",
    "Japonica rice",
    "Yellow corn",
    "Soybeans",
    "Wheat",
]


def parse_args() -> tuple[int, Tuple[int, int, int], Path]:
    parser = ArgumentParser(description="Generate synthetic sensor CSV for GranaryPredict")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--grid", nargs=3, type=int, metavar=("X", "Y", "Z"), default=(4, 5, 3), help="Grid dimensions X Y Z")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/synthetic_sensor_data.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()
    return args.days, tuple(args.grid), args.output


def seasonal_temperature(base_day: int) -> float:
    """Return an outside temperature with mild seasonal pattern."""
    # peak around day 180 (~mid-year), amplitude 8°C, base 20°C
    return 20 + 8 * math.sin(2 * math.pi * base_day / 365)


def generate_dataframe(days: int, grid: Tuple[int, int, int]) -> pd.DataFrame:
    start = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    timestamps: List[datetime] = [start + timedelta(days=i) for i in range(days)]

    rows = []
    detection_cycle = 0

    gx, gy, gz = grid
    grain_type = random.choice(GRAIN_TYPES)

    for day_index, ts in enumerate(timestamps):
        outside_temp = seasonal_temperature((datetime.now() - ts).days) + np.random.normal(0, 1.5)
        outside_humidity = np.clip(85 - 0.5 * (outside_temp - 28) + np.random.normal(0, 3), 60, 95)

        # simple inside temp model: outside + bias - ventilation effect
        ventilation_on = int(np.random.rand() < 0.3)
        inside_temp = outside_temp - (1.5 if ventilation_on else 0) + 2.0  # warehouses warmer than outside
        humidity_inside = outside_humidity - np.random.uniform(10, 20)

        for x, y, z in itertools.product(range(gx), range(gy), range(gz)):
            detection_cycle += 1

            # Position influences:
            edge = int(x in (0, gx - 1) or y in (0, gy - 1))
            height_factor = z / max(gz - 1, 1)

            grain_temp = inside_temp - 6 + edge * 1.5 + height_factor * 3 + np.random.normal(0, 0.3)
            grain_temp = np.clip(grain_temp, 12, 30)

            rows.append(
                {
                    "detection_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "detection_cycle": detection_cycle,
                    "grid_x": x,
                    "grid_y": y,
                    "grid_z": z,
                    "temperature_grain": round(grain_temp, 2),
                    "temperature_inside": round(inside_temp, 2),
                    "temperature_outside": round(outside_temp, 2),
                    "humidity_warehouse": round(humidity_inside, 1),
                    "humidity_outside": round(outside_humidity, 1),
                    "grain_type": grain_type,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    days, grid, output = parse_args()
    df = generate_dataframe(days, grid)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8")
    resolved = output.resolve()
    try:
        rel_path = resolved.relative_to(Path.cwd())
    except ValueError:
        rel_path = resolved
    print(f"✅ Generated {len(df)} rows → {rel_path}")
    print("Columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main() 