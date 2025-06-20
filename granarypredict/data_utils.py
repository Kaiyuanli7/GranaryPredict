from __future__ import annotations

"""Utility helpers for data ordering & grouping (May-2025)."""

from typing import List

import pandas as pd

__all__ = [
    "comprehensive_sort",
    "assign_group_id",
]

# Default sort hierarchy – columns will be used only if present.
_SORT_HIERARCHY: List[str] = [
    "granary_id",  # warehouse identifier
    "heap_id",     # silo / heap identifier
    "grid_x",
    "grid_y",
    "grid_z",
    "detection_time",  # timestamp last to keep chronological order within sensor
]


def comprehensive_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* sorted according to the canonical hierarchy.

    1. granary_id ➜ 2. heap_id ➜ 3. grid_x ➜ 4. grid_y ➜ 5. grid_z ➜ 6. detection_time

    Columns missing in *df* are simply ignored.
    """
    sort_cols = [c for c in _SORT_HIERARCHY if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def assign_group_id(df: pd.DataFrame, *, col_name: str = "_group_id") -> pd.DataFrame:
    """Add a column (*col_name*) that uniquely identifies a physical silo.

    Priority: granary_id+heap_id → granary_id → heap_id → constant "all".
    """
    if {"granary_id", "heap_id"}.issubset(df.columns):
        df[col_name] = df["granary_id"].astype(str) + "_" + df["heap_id"].astype(str)
    elif "granary_id" in df.columns:
        df[col_name] = df["granary_id"].astype(str)
    elif "heap_id" in df.columns:
        df[col_name] = df["heap_id"].astype(str)
    else:
        df[col_name] = "all"
    return df 