#!/usr/bin/env python3
"""Remove leading commas from each row of a CSV and prepend the standard header.

Usage:
    python scripts/fix_training_csv.py \
        --input data/preloaded/TrainingData.csv \
        --output data/processed/TrainingData_fixed.csv

If you omit --input/--output, the defaults above are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

def clean_csv(input_path: Path, output_path: Path) -> None:
    """Copy `input_path` to `output_path` while removing a single leading comma on each line."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Ensure the parent directory for output exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8", errors="ignore") as fin, \
            output_path.open("w", encoding="utf-8", newline="") as fout:
        for raw_line in fin:
            # Preserve empty lines as-is
            if raw_line == "\n":
                fout.write(raw_line)
                continue

            # Remove UTF-8 BOM if present then a single leading comma
            line = raw_line.lstrip("\ufeff")
            if line.startswith(","):
                line = line[1:]

            fout.write(line)


def parse_args() -> argparse.Namespace:  # noqa: D401 (simple function)
    parser = argparse.ArgumentParser(description="Fix training data CSV format")
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path("data/preloaded/TrainingData.csv"),
        help="Path to the raw CSV with leading commas"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("data/processed/TrainingData_fixed.csv"),
        help="Destination for the cleaned CSV"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    clean_csv(args.input, args.output)
    print(f"✔ Cleaned CSV written to: {args.output}") 