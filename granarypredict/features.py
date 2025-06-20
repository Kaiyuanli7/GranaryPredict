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
    "add_multi_lag",
    "add_rolling_stats",
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


# ---------------------------------------------------------------------------
# NEW – Extra temperature features (May-2025)
# ---------------------------------------------------------------------------


def _add_single_lag(df: pd.DataFrame, lag_days: int, *,
                    temp_col: str = "temperature_grain",
                    timestamp_col: str = "detection_time") -> pd.DataFrame:
    """Return *df* with a new column ``lag_temp_<lag>d`` computed exactly like
    ``add_sensor_lag`` but for an arbitrary day offset."""

    if lag_days == 1:
        # Already handled by original add_sensor_lag; skip duplication.
        return df

    if {"grid_x", "grid_y", "grid_z", temp_col, timestamp_col}.issubset(df.columns):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
        df["_date"] = df[timestamp_col].dt.floor("D")

        lag_df = df[group_cols + ["_date", temp_col]].copy()
        lag_df["_date"] = lag_df["_date"] + pd.Timedelta(days=lag_days)
        lag_df.rename(columns={temp_col: f"lag_temp_{lag_days}d"}, inplace=True)

        df = df.merge(lag_df, on=group_cols + ["_date"], how="left")
        df.drop(columns=["_date"], inplace=True)
    return df


def add_multi_lag(
    df: pd.DataFrame,
    *,
    lags: tuple[int, ...] = (1, 7, 30),
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add multiple lag columns (e.g. 1-, 7-, 30-day) and ΔT feature.

    Relies on ``add_sensor_lag`` for the 1-day lag and `_add_single_lag` for
    longer offsets.
    """

    df = add_sensor_lag(df, temp_col=temp_col, timestamp_col=timestamp_col)
    for d in lags:
        if d == 1:
            continue
        df = _add_single_lag(df, d, temp_col=temp_col, timestamp_col=timestamp_col)

    # ΔT = current − 1-day lag
    if temp_col in df.columns and "lag_temp_1d" in df.columns:
        df["delta_temp_1d"] = df[temp_col] - df["lag_temp_1d"]
    return df


def add_rolling_stats(
    df: pd.DataFrame,
    *,
    window_days: int = 7,
    temp_col: str = "temperature_grain",
    timestamp_col: str = "detection_time",
) -> pd.DataFrame:
    """Add rolling mean and std of *temp_col* over *window_days* for each sensor."""

    if not {temp_col, timestamp_col}.issubset(df.columns):
        return df

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df.sort_values(timestamp_col, inplace=True)

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    roll_mean_col = f"roll_mean_{window_days}d"
    roll_std_col = f"roll_std_{window_days}d"

    if group_cols:
        df[roll_mean_col] = (
            df.groupby(group_cols)[temp_col]
            .transform(lambda s: s.rolling(window_days, min_periods=1).mean())
        )
        df[roll_std_col] = (
            df.groupby(group_cols)[temp_col]
            .transform(lambda s: s.rolling(window_days, min_periods=1).std())
        )
    else:
        df[roll_mean_col] = df[temp_col].rolling(window_days, min_periods=1).mean()
        df[roll_std_col] = df[temp_col].rolling(window_days, min_periods=1).std()

    return df 