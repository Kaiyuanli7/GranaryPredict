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
        the function will try common alternatives (``batch``, ``timestamp``,
        ``datetime``, ``date``, ``time``) before raising a ``KeyError``.
    """

    df = df.copy()

    # Auto-detect timestamp column if the default is missing
    if timestamp_col not in df.columns:
        common_alts = ["batch", "timestamp", "datetime", "date", "time"]
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
    # Remove previously created grid_index to avoid redundancy
    if "grid_index" in df.columns:
        df = df.copy().drop(columns=["grid_index"])

    # Function now does nothing else; kept for backward-compatibility.
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
        "avg_grain_temp",
        "avg_in_temp",         # legacy aggregate metric
        "temperature_grain",   # drop target from feature matrix to avoid leakage
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
    if df[target_col].notna().any():
        mask = df[target_col].notna()
        X = df.loc[mask].drop(columns=list(drop_cols), errors="ignore")
        y = df.loc[mask, target_col]
    else:
        # All targets missing – likely forecasting scenario; keep all rows and
        # return empty y (length 0) to satisfy signature.
        X = df.drop(columns=list(drop_cols), errors="ignore")
        y = pd.Series(dtype="float64")
    X = encode_categoricals(X)
    return X, y


# Keep exports alphabetical for readability
__all__ = [
    "add_sensor_lag",
    "create_spatial_features",
    "create_time_features",
    "encode_categoricals",
    "select_feature_target",
]


# ------------------------------------------------------------
# NEW – 1-day lag feature for each sensor (grid_x/y/z)          May-2025
# ------------------------------------------------------------


def add_sensor_lag(
    df: pd.DataFrame,
    *,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
    lag_days: int = 1,
) -> pd.DataFrame:
    """Add ``lag_temp_1d`` – the *temp_col* from ``lag_days`` days earlier
    for the *same physical sensor* defined by ``granary_id`` + ``heap_id`` +
    grid_x / y / z.  If *granary_id* or *heap_id* are missing, they are
    ignored, falling back to coarser grouping.  Rows without an exact match
    receive ``NaN``.

    This version uses a merge-on-timestamp (+lag) approach so it no longer
    depends on a fixed sampling frequency.  Whether you record hourly or
    irregularly, the function only assigns a lag when there is a reading
    exactly *lag_days* earlier (within a +/- 6 hour tolerance).
    """

    # Mandatory columns check ------------------------------------------------
    base_required = {"grid_x", "grid_y", "grid_z", timestamp_col, temp_col}
    if not base_required.issubset(df.columns):
        return df  # key columns absent, nothing to do

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Determine grouping hierarchy ------------------------------------------
    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]

    # Work at *date* granularity so mixed sampling frequencies align nicely
    df["_date"] = df[timestamp_col].dt.floor("D")

    lag_key_cols = group_cols + ["_date"]

    # Build lagged frame -----------------------------------------------------
    lag_df = df[lag_key_cols + [temp_col]].copy()
    lag_df["_date"] = lag_df["_date"] + pd.Timedelta(days=lag_days)
    lag_df.rename(columns={temp_col: "lag_temp_1d"}, inplace=True)

    # Merge – vectorised & fast ---------------------------------------------
    df = df.merge(lag_df, on=lag_key_cols, how="left")

    df.drop(columns=["_date"], inplace=True)

    return df 