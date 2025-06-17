from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Simple cleaning pipeline:
    - Strip column names
    - Drop fully empty columns
    - Handle duplicated rows
    - Replace obvious missing value markers (-999, "-", "NA") with NaN
    """
    logger.info("Running basic cleaning on df shape=%s", df.shape)

    df = df.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Replace common missing markers
    df.replace({"-": pd.NA, "NA": pd.NA, "N/A": pd.NA, -999: pd.NA}, inplace=True)

    # Drop empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # Remove duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.debug("Dropped %d duplicate rows", before - len(df))

    return df


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "ffill",
    numeric_strategy: str = "mean",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fill missing values.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        For non-numeric columns. Options: 'ffill', 'bfill', 'mode'.
    numeric_strategy : str
        For numeric columns. Options: 'mean', 'median', 'interpolate'.
    limit : int, optional
        Max number of consecutive NaNs to forward/backward fill.
    """
    df = df.copy()

    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.columns.difference(num_cols)

    if numeric_strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif numeric_strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif numeric_strategy == "interpolate":
        df[num_cols] = df[num_cols].interpolate(limit=limit)
    else:
        raise ValueError("Invalid numeric_strategy")

    # Object / category columns
    if strategy == "ffill":
        df[obj_cols] = df[obj_cols].ffill(limit=limit)
    elif strategy == "bfill":
        df[obj_cols] = df[obj_cols].bfill(limit=limit)
    elif strategy == "mode":
        modes = df[obj_cols].mode().iloc[0]
        df[obj_cols] = df[obj_cols].fillna(modes)
    else:
        raise ValueError("Invalid strategy")

    return df


__all__ = [
    "basic_clean",
    "fill_missing",
] 