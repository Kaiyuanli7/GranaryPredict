from __future__ import annotations

"""Utilities to reorganise a single *mixed* CSV into per-silo daily slices.

Usage (CLI):
    python -m granarypredict.data_organizer input.csv  # writes to data/raw/by_silo

The dashboard imports and triggers `organize_mixed_csv` automatically when it
sees more than one (granary_id, heap_id) combination in the uploaded file.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .data_utils import comprehensive_sort
from .ingestion import standardize_granary_csv

logger = logging.getLogger(__name__)

__all__ = ["organize_mixed_csv"]


def _check_cols(df: pd.DataFrame, cols: Tuple[str, ...]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def organize_mixed_csv(csv_path: str | Path, *, out_dir: str | Path = "data/raw/by_silo") -> int:
    """Split *csv_path* into individual CSVs per (granary, heap, date).

    Parameters
    ----------
    csv_path : str | Path
        Path to the mixed CSV.
    out_dir : str | Path, default ``data/raw/by_silo``
        Root directory where the organised files are written.

    Returns
    -------
    int
        Number of files written.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    df = pd.read_csv(csv_path, encoding="utf-8")
    df = standardize_granary_csv(df)
    df = comprehensive_sort(df)

    _check_cols(df, ("granary_id", "heap_id", "detection_time"))

    df["_date"] = pd.to_datetime(df["detection_time"], errors="coerce").dt.date

    files_written = 0
    for (granary_id, heap_id, date), part in df.groupby(
        ["granary_id", "heap_id", "_date"], sort=False
    ):
        target_dir = out_dir / str(granary_id) / str(heap_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / f"{date}.csv"
        part.drop(columns=["_date"]).to_csv(file_path, index=False)
        files_written += 1

    logger.info("Organized '%s' into %d slice files under %s", csv_path.name, files_written, out_dir)
    return files_written


# ---------------------------------------------------------------------------
# CLI entry-point: python -m granarypredict.data_organizer <csv> [out_dir]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Organize a mixed CSV into per-silo daily files")
    parser.add_argument("csv", help="Path to the mixed CSV file")
    parser.add_argument("--out", default="data/raw/by_silo", help="Output folder root")
    args = parser.parse_args()

    n = organize_mixed_csv(args.csv, out_dir=args.out)
    print(f"Wrote {n} files under {args.out}") 