import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from granarypredict import cleaning, features, model as model_utils, ingestion
from granarypredict.config import MODELS_DIR, DATA_DIR
from granarypredict.data_utils import comprehensive_sort, assign_group_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


ENV_COLUMNS = [
    "temperature_inside",
    "temperature_outside",
    "humidity_warehouse",
    "humidity_outside",
]

FUTURE_SAFE_EXTRA = [
    "max_temp",
    "min_temp",
    "line_no",
    "layer_no",
]


ALG_CHOICES = {
    "rf": "RandomForest",
    "hgb": "HistGradientBoosting",
    "lgbm": "LightGBM",
}


def load_and_prepare(csv_path: Path, *, future_safe: bool) -> pd.DataFrame:
    """Load a single CSV file and run cleaning + feature engineering."""
    logger.info("Loading %s", csv_path)
    df = ingestion.read_granary_csv(csv_path)
    df = ingestion.standardize_granary_csv(df)
    df = cleaning.basic_clean(df)
    df = cleaning.fill_missing(df)
    df = features.create_time_features(df)
    df = features.create_spatial_features(df)
    df = features.add_sensor_lag(df)
    df = comprehensive_sort(df)
    df = assign_group_id(df)

    if future_safe:
        df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

    # Add dataset identifier derived from file stem (used later for CV grouping)
    df["dataset_id"] = csv_path.stem
    return df



def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Train a single model on multiple grain-temperature CSV datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csvs",
        nargs="+",
        type=str,
        help="Paths or glob patterns to CSV files (relative to the repo root unless absolute).",
    )
    parser.add_argument(
        "--algo",
        choices=list(ALG_CHOICES.keys()),
        default="rf",
        help="Which algorithm to train (rf = RandomForest, hgb = HistGradientBoosting, lgbm = LightGBM).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees/iterations for the model (ignored for LightGBM if < 1).",
    )
    parser.add_argument(
        "--future-safe",
        action="store_true",
        help="Drop environment-only columns so the model can be used for future forecasting.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="global_model.joblib",
        help="Filename for the saved model (placed under the models/ directory).",
    )

    args = parser.parse_args(argv)

    # Resolve all input CSV paths (support wildcards)
    all_paths: List[Path] = []
    for pattern in args.csvs:
        # Use Path.glob if pattern contains wildcard; else treat as literal file
        p = Path(pattern)
        if any(ch in pattern for ch in "*?["):
            all_paths.extend([pp.resolve() for pp in p.parent.glob(p.name)])
        else:
            all_paths.append(p.resolve())

    if not all_paths:
        parser.error("No CSV files matched the supplied patterns.")

    logger.info("Preparing %d CSVs", len(all_paths))
    frames = [load_and_prepare(path, future_safe=args.future_safe) for path in all_paths]
    df_all = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Concatenated frame shape=%s", df_all.shape)

    # Feature matrix / target
    X, y = features.select_feature_target(df_all, target_col="temperature_grain")

    # Perform simple shuffle split for validation (optional)
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # Use physical silo grouping for CV if available else dataset_id fallback
    if "_group_id" in df_all.columns:
        groups = df_all.loc[X.index, "_group_id"]
    else:
        groups = df_all.loc[X.index, "dataset_id"]
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    algo_name = ALG_CHOICES[args.algo]
    if args.algo == "rf":
        mdl, _ = model_utils.train_random_forest(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            test_size=0.0,  # We'll handle our own split above
        )
    elif args.algo == "hgb":
        mdl, _ = model_utils.train_gb_models(
            X_train,
            y_train,
            model_type="hist",
            n_estimators=args.n_estimators,
        )
    else:  # lgbm
        mdl, _ = model_utils.train_lightgbm(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
        )

    # Evaluate on held-out datasets
    preds_test = model_utils.predict(mdl, X_test)
    mae = mean_absolute_error(y_test, preds_test)
    rmse = mean_squared_error(y_test, preds_test, squared=False)
    logger.info("Held-out MAE: %.3f | RMSE: %.3f", mae, rmse)

    # Refit on full data before persisting
    mdl.fit(X, y)

    model_path = model_utils.save_model(mdl, name=args.model_name)
    logger.info("Saved global model to %s", model_path)

    print("\n==== Training Summary ====")
    print(f"Algorithm       : {algo_name}")
    print(f"Datasets used   : {len(all_paths)}")
    print(f"Total rows      : {len(df_all):,}")
    print(f"Held-out MAE    : {mae:.3f}")
    print(f"Held-out RMSE   : {rmse:.3f}")
    print(f"Model saved to  : {model_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main() 