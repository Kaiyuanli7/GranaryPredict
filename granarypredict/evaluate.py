from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def time_series_cv(
    model: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Tuple[RegressorMixin, Dict[str, float]]:
    """Perform time-series aware cross-validation and return metrics.

    The model is *fitted* on the full data at the end so it can be persisted.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    logger.info("Running TimeSeriesSplit (n_splits=%d) cross-val", n_splits)

    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in splitter.split(X):
        mdl_fold = clone(model)
        mdl_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = mdl_fold.predict(X.iloc[test_idx])

    valid_mask = ~np.isnan(preds)
    mae = mean_absolute_error(y[valid_mask], preds[valid_mask])
    rmse = np.sqrt(mean_squared_error(y[valid_mask], preds[valid_mask]))
    metrics = {"mae_cv": mae, "rmse_cv": rmse}
    logger.info("CV MAE: %.3f, RMSE: %.3f", mae, rmse)

    # Refit on all data for downstream use
    model.fit(X, y)
    return model, metrics


__all__ = ["time_series_cv"] 