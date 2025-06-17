from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add cyclical and calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame.
    timestamp_col : str, default "detection_time"
        Name of the column that stores the timestamp.  If that column is missing
        the function will try common alternatives (``timestamp``, ``datetime``,
        ``date``, ``time``) before raising a ``KeyError``.
    """

    df = df.copy()

    # Auto-detect timestamp column if the default is missing
    if timestamp_col not in df.columns:
        common_alts = ["timestamp", "datetime", "date", "time"]
        found_alt = next((c for c in common_alts if c in df.columns), None)
        if found_alt is not None:
            logger.info("create_time_features: using '%s' as timestamp column (auto-detected)", found_alt)
            timestamp_col = found_alt
        else:
            raise KeyError(
                f"Timestamp column '{timestamp_col}' not found in dataframe. "
                "Add the column or pass the correct name via the 'timestamp_col' argument."
            )

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df["year"] = df[timestamp_col].dt.year
    df["month"] = df[timestamp_col].dt.month
    df["day"] = df[timestamp_col].dt.day
    df["hour"] = df[timestamp_col].dt.hour

    # Cyclical encodings
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def create_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine grid_x/y/z into useful spatial indices."""
    if {"grid_x", "grid_y", "grid_z"}.issubset(df.columns):
        df = df.copy()
        df["grid_index"] = (
            df["grid_x"].astype(int).astype(str)
            + "_"
            + df["grid_y"].astype(int).astype(str)
            + "_"
            + df["grid_z"].astype(int).astype(str)
        )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object/category dtypes to integer codes (label encoding)."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes
    return df


def select_feature_target(
    df: pd.DataFrame,
    target_col: str = "temperature_grain",
    drop_cols: Tuple[str, ...] = (
        "temperature_grain",
        "detection_time",
        "granary_id",
        "heap_id",
        "forecast_day",
        # Redundant or constant columns – duplicates of grid_x/y
        "line_no",
        "layer_no",
        "line",
        "layer",
    ),
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X, y frames."""
    X = df.drop(columns=list(drop_cols), errors="ignore")
    X = encode_categoricals(X)
    y = df[target_col]
    return X, y


__all__ = [
    "create_time_features",
    "create_spatial_features",
    "encode_categoricals",
    "select_feature_target",
] 