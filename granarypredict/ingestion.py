from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import requests

from .config import RAW_DATA_DIR, METEOROLOGY_API_BASE, COMPANY_API_BASE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_granary_csv(
    file_path: str | Path,
    *,
    encoding: str = "utf-8",
    dtype: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Generic CSV loader that handles common encodings.

    Parameters
    ----------
    file_path : str | Path
        Path to csv file.
    encoding : str
        File encoding, defaults to utf-8 but can be gbk for Chinese files.
    dtype : dict[str, str], optional
        Explicit dtype mapping when pandas cannot infer.
    """
    file_path = Path(file_path)
    logger.info("Loading CSV %s", file_path)
    df = pd.read_csv(file_path, encoding=encoding, dtype=dtype)
    return df


def fetch_meteorology(
    location: str,
    start: str,
    end: str,
    *,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Placeholder REST client that fetches meteorological data.

    This function currently mocks API responses because the real endpoint
    is not publicly available. Replace the body with actual request logic.
    """
    logger.info("Fetching weather for %s from %s to %s", location, start, end)
    # Example query – replace with real parameters
    params = {
        "location": location,
        "start": start,
        "end": end,
        "key": api_key or "demo-key",
    }
    try:
        response = requests.get(f"{METEOROLOGY_API_BASE}/historical", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as exc:
        logger.warning("Weather API unavailable, returning empty frame: %s", exc)
        return pd.DataFrame()


def fetch_company_data(
    endpoint: Literal[
        "granaries",
        "heaps",
        "sensors",
        "operations",
    ],
    params: Optional[dict[str, str]] = None,
    *,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Generic GET request to company's data service."""
    url = f"{COMPANY_API_BASE}/{endpoint}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception as exc:
        logger.warning("Company API unavailable (%s), returning empty frame.", exc)
        return pd.DataFrame()


def standardize_result147(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Result_147.csv format to standard columns used in pipeline."""
    # Column mapping for multiple export formats (Result_147 and the new StorePoint format)
    mapping = {
        # ----- core timestamp / spatial / target -----
        "batch": "detection_time",        # timestamp
        "temp": "temperature_grain",     # target variable
        "x": "grid_x",
        "y": "grid_y",
        "z": "grid_z",

        # ----- environmental variables -----
        "indoor_temp": "temperature_inside",
        "outdoor_temp": "temperature_outside",
        "indoor_humidity": "humidity_warehouse",
        "outdoor_humidity": "humidity_outside",

        # ----- warehouse / grain metadata -----
        "storeType": "warehouse_type",

        # ----- new StorePoint header additions -----
        "storepoint_id": "storepoint_id",
        "storepointName": "granary_id",
        "storeName": "heap_id",
        "kdjd": "longitude",            # 经度 (longitude)
        "kdwd": "latitude",             # 纬度 (latitude)
        "kqdz": "address_cn",           # 库区地址 (Chinese)
        "storeId": "warehouse_id",
        "locatType": "location_type",
        "line_no": "line_no",
        "layer_no": "layer_no",
        "avg_in_temp": "avg_in_temp",
        "max_temp": "max_temp",
        "min_temp": "min_temp",
    }
    df = df.rename(columns=mapping)
    # ensure correct dtypes
    df["detection_time"] = pd.to_datetime(df["detection_time"])
    nums = [
        "temperature_grain",
        "temperature_inside",
        "temperature_outside",
        "humidity_warehouse",
        "humidity_outside",
        "avg_in_temp",
        "max_temp",
        "min_temp",
        "longitude",
        "latitude",
    ]
    for col in nums:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


__all__ = [
    "read_granary_csv",
    "fetch_meteorology",
    "fetch_company_data",
    "standardize_result147",
    "standardize_granary_csv",
]


# ------------------------------------------------------------
# NEW – Canonical schema standardisation helper  (May-2025)
# ------------------------------------------------------------

# Desired external CSV header (user-facing):
#   Granary_ID,longitude,latitude,address,Heap_ID,batch,temp,x,y,z,
#   avg_grain_temp,max_temp,min_temp,indoor_temp,indoor_humidity,
#   outdoor_temp,outdoor_humidity,storeType

# Internal pipeline keeps its original snake-case column names so that the
# rest of the codebase (features/, dashboard, etc.) remains unchanged.

_CANONICAL_MAP: dict[str, str] = {
    # ----- ID / location -----
    "Granary_ID": "granary_id",
    "Heap_ID": "heap_id",
    "longitude": "longitude",
    "latitude": "latitude",
    "address": "address_cn",  # keep same field semantics – Chinese address OK

    # ----- timestamp & spatial coords -----
    "batch": "detection_time",
    "x": "grid_x",
    "y": "grid_y",
    "z": "grid_z",

    # ----- sensor & temperature measurements -----
    "temp": "temperature_grain",      # individual probe reading
    "avg_grain_temp": "avg_grain_temp",  # pile-wide daily average
    "max_temp": "max_temp",
    "min_temp": "min_temp",

    # ----- environmental -----
    "indoor_temp": "temperature_inside",
    "indoor_humidity": "humidity_warehouse",
    "outdoor_temp": "temperature_outside",
    "outdoor_humidity": "humidity_outside",

    # ----- misc -----
    "storeType": "warehouse_type",

    # ----- Backward-compatibility aliases (old StorePoint / Result_147) -----
    "storepointName": "granary_id",
    "storeName": "heap_id",
    "kdjd": "longitude",   # longitude
    "kdwd": "latitude",    # latitude
    "kqdz": "address_cn",
    "avg_in_temp": "avg_grain_temp",  # old name for average grain temperature
}


def _rename_and_select(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns via ``_CANONICAL_MAP`` and keep only recognised ones."""
    rename_map = {c: _CANONICAL_MAP[c] for c in df.columns if c in _CANONICAL_MAP}
    df = df.rename(columns=rename_map)
    keep_cols = set(_CANONICAL_MAP.values())
    df = df[[c for c in df.columns if c in keep_cols]].copy()
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast datatypes for numeric columns and parse datetimes."""
    # Parse timestamp
    if "detection_time" in df.columns:
        df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")

    # Numeric coercion list – everything except IDs / address / types
    numeric_like = [
        "longitude",
        "latitude",
        "grid_x",
        "grid_y",
        "grid_z",
        "temperature_grain",
        "avg_grain_temp",
        "max_temp",
        "min_temp",
        "temperature_inside",
        "temperature_outside",
        "humidity_warehouse",
        "humidity_outside",
    ]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_granary_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any recognised CSV variant to the canonical internal schema.

    Steps:
    1. Rename columns via ``_CANONICAL_MAP`` (case-sensitive).
    2. Drop columns that are not part of the canonical internal schema.
    3. Parse *detection_time* to datetime and coerce numeric columns.
    """

    logger.info("Standardising CSV to canonical schema. Incoming cols=%s", list(df.columns))
    df = _rename_and_select(df)
    df = _coerce_types(df)
    logger.info("After standardisation cols=%s", list(df.columns))
    return df 