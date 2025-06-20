import pathlib
from datetime import timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from itertools import islice

from granarypredict import cleaning, features, model as model_utils
from granarypredict.config import ALERT_TEMP_THRESHOLD, MODELS_DIR
from granarypredict import ingestion
from granarypredict.data_utils import comprehensive_sort, assign_group_id
from granarypredict.data_organizer import organize_mixed_csv
import tempfile

# Streamlit reload may have stale module; fetch grain thresholds safely
try:
    from granarypredict.config import GRAIN_ALERT_THRESHOLDS  # type: ignore
except ImportError:
    GRAIN_ALERT_THRESHOLDS = {}

from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

# ---------------------------------------------------------------------
# Debug helper – collects messages in session state
# ---------------------------------------------------------------------

# Defined early so it's available everywhere

def _d(msg):
    if not st.session_state.get("debug_mode"):
        return
    import streamlit as _st
    _st.toast(f"🛠️ {msg}")
    log = _st.session_state.setdefault("debug_msgs", [])
    log.append(str(msg))

# Add constant after imports
ENV_COLUMNS = [
    "temperature_inside",
    "temperature_outside",
    "humidity_warehouse",
    "humidity_outside",
]

# Additional columns unavailable for real future dates that should be
# removed when training a "future-safe" model.
FUTURE_SAFE_EXTRA = [
    "max_temp",      # historic max air temp inside silo
    "min_temp",      # historic min inside temp
    "line_no",       # production line identifier (constant, but 0-filled in future)
    "layer_no",      # vertical layer identifier (constant, but 0-filled in future)
]

# Preset horizons (days) for quick selector controls
PRESET_HORIZONS = [7, 14, 30, 90, 180, 365]

# Target column representing daily average grain temperature for evaluation/forecast
TARGET_TEMP_COL = "temperature_grain"  # per-sensor target for model & metrics

# Utility to detect if a model is "future-safe" by filename convention (contains 'fs_')
def is_future_safe_model(model_name: str) -> bool:
    return "fs_" in model_name.lower()

st.set_page_config(page_title="SiloFlow", layout="wide")



# Directory that holds bundled sample CSVs shipped with the repo
PRELOADED_DATA_DIR = pathlib.Path("data/preloaded")

# Directory that holds bundled pre-trained models
PRELOADED_MODEL_DIR = MODELS_DIR / "preloaded"

# Ensure directory exists so globbing is safe
PRELOADED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV safely, rewinding pointer and handling encoding."""
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
        # Convert to internal schema regardless of source CSV variant.
        df = ingestion.standardize_granary_csv(df)
        return df
    except pd.errors.EmptyDataError:
        st.error("Uploaded file appears empty or unreadable. Please verify the CSV.")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_trained_model(path: Optional[str | pathlib.Path] = None):
    """Attempt to load a model from user-saved or preloaded directories."""
    # Default fallback
    if path is None:
        path = MODELS_DIR / "rf_model.joblib"

    path = pathlib.Path(path)

    # If given just a filename, search in both dirs
    if not path.is_absolute() and not path.exists():
        user_path = MODELS_DIR / path
        preload_path = PRELOADED_MODEL_DIR / path
        if user_path.exists():
            path = user_path
        elif preload_path.exists():
            path = preload_path

    if path.exists():
        return model_utils.load_model(path)

    st.warning("Model not found – please train or select another.")
    return None


def plot_3d_grid(df: pd.DataFrame, *, key: str):
    required_cols = {"grid_x", "grid_y", "grid_z", "temperature_grain"}
    if not required_cols.issubset(df.columns):
        st.info("No spatial temperature data present.")
        return

    # Build point hover/label text
    texts = []
    has_pred = "predicted_temp" in df.columns
    for idx, row in df.iterrows():
        if has_pred:
            diff = row["predicted_temp"] - row["temperature_grain"]
            texts.append(
                f"Pred: {row['predicted_temp']:.1f}°C<br>Actual: {row['temperature_grain']:.1f}°C<br>Δ: {diff:+.1f}°C"
            )
        else:
            texts.append(f"Actual: {row['temperature_grain']:.1f}°C")

    fig = go.Figure(data=go.Scatter3d(
        x=df["grid_x"],
        y=df["grid_z"],
        z=df["grid_y"],
        mode="markers",
        marker=dict(
            size=6,
            color=df["predicted_temp"] if has_pred else df["temperature_grain"],
            colorscale="Viridis",
            colorbar=dict(title="Temp (°C)"),
        ),
        text=texts,
        hovertemplate="%{text}<extra></extra>",
    ))
    # Ensure integer ticks on axes
    fig.update_layout(
        scene=dict(
            xaxis=dict(dtick=1, title="grid_x"),
            yaxis=dict(dtick=1, title="grid_y"),
            zaxis=dict(dtick=1, title="grid_z", autorange="reversed"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_time_series(df: pd.DataFrame, *, key: str):
    if "predicted_temp" not in df.columns or "detection_time" not in df.columns:
        return
    tmp = df.copy()
    # Use floor("D") to keep datetime64 dtype; avoids Plotly treating axis as categorical
    tmp["date"] = pd.to_datetime(tmp["detection_time"]).dt.floor("D")

    fig = go.Figure()

    # -------- Actual line --------
    grp_actual = tmp.groupby("date").agg(actual=(TARGET_TEMP_COL, "mean")).reset_index()
    fig.add_trace(
        go.Scatter(
            x=grp_actual["date"],
            y=grp_actual["actual"],
            mode="lines+markers",
            name="Actual Avg",
            line=dict(color="#1f77b4"),
        )
    )

    # ------- Predicted line (continuous across eval & filled) -------
    if "predicted_temp" in tmp.columns:
        grp_pred = (
            tmp.groupby("date").agg(predicted=("predicted_temp", "mean")).reset_index().sort_values("date")
        )

        # Determine cutoff between evaluation (has actual data) and future-only predictions
        actual_mask = tmp[TARGET_TEMP_COL].notna()
        last_actual_date = tmp.loc[actual_mask, "date"].max()

        if pd.isna(last_actual_date):
            pred_eval = pd.DataFrame()
            pred_future = grp_pred
        else:
            pred_eval = grp_pred[grp_pred["date"] <= last_actual_date]
            pred_future = grp_pred[grp_pred["date"] > last_actual_date]

        if not pred_eval.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_eval["date"],
                    y=pred_eval["predicted"],
                    mode="lines+markers",
                    name="Predicted (eval)",
                    line=dict(color="#ff7f0e"),
                    connectgaps=True,
                )
            )

        if not pred_future.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_future["date"],
                    y=pred_future["predicted"],
                    mode="lines+markers",
                    name="Predicted (future)",
                    line=dict(color="#9467bd"),
                    connectgaps=True,
                )
            )

    fig.update_layout(
        title="Average Grain Temperature Over Time",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def list_available_models() -> list[str]:
    """Return unique model filenames from user-saved and preloaded dirs."""
    names = {p.name for p in MODELS_DIR.glob("*.joblib")}
    names.update({p.name for p in PRELOADED_MODEL_DIR.glob("*.joblib")})
    return sorted(names)


def split_train_eval(df: pd.DataFrame, horizon: int = 5):
    """Split by unique date; last 'horizon' dates form evaluation set."""
    df = df.copy()
    df["_date"] = pd.to_datetime(df["detection_time"]).dt.date
    unique_dates = sorted(df["_date"].unique())
    if len(unique_dates) <= horizon + 1:
        return df, pd.DataFrame()  # not enough data
    cutoff_dates = unique_dates[-horizon:]
    df_eval = df[df["_date"].isin(cutoff_dates)].copy()
    df_train = df[~df["_date"].isin(cutoff_dates)].copy()
    # map forecast_day index 1..horizon
    date_to_idx = {date: idx for idx, date in enumerate(cutoff_dates, start=1)}
    df_eval["forecast_day"] = df_eval["_date"].map(date_to_idx)
    df_train.drop(columns=["_date"], inplace=True)
    df_eval.drop(columns=["_date"], inplace=True)
    return df_train, df_eval


# -------------------------------------------------------------------
# NEW – fraction‐based chronological split (May-2025)
# -------------------------------------------------------------------


def split_train_eval_frac(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronologically split *df* by unique date where the **last** fraction
    (``test_frac``) of dates becomes the evaluation set.

    Returns (df_train, df_eval) similar to ``split_train_eval`` but sized by
    proportion instead of fixed horizon.
    """
    df = df.copy()
    df["_date"] = pd.to_datetime(df["detection_time"]).dt.date
    unique_dates = sorted(df["_date"].unique())
    if not unique_dates:
        return df, pd.DataFrame()

    n_test_days = max(1, int(len(unique_dates) * test_frac))
    cutoff_dates = unique_dates[-n_test_days:]

    df_eval = df[df["_date"].isin(cutoff_dates)].copy()
    df_train = df[~df["_date"].isin(cutoff_dates)].copy()

    # map forecast_day index 1..n_test_days
    date_to_idx = {date: idx for idx, date in enumerate(cutoff_dates, start=1)}
    df_eval["forecast_day"] = df_eval["_date"].map(date_to_idx)

    df_train.drop(columns=["_date"], inplace=True)
    df_eval.drop(columns=["_date"], inplace=True)
    return df_train, df_eval


def forecast_summary(df_eval: pd.DataFrame) -> pd.DataFrame:
    if "forecast_day" not in df_eval.columns:
        return pd.DataFrame()

    grp = (
        df_eval.groupby("forecast_day")
        .agg(
            actual_mean=(TARGET_TEMP_COL, "mean"),
            pred_mean=("predicted_temp", "mean"),
        )
        .reset_index()
    )

    # Percent absolute error (only where actual_mean is finite & non-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        grp["pct_error"] = (grp["pred_mean"] - grp["actual_mean"]).abs() / grp["actual_mean"].replace(0, np.nan) * 100

    # Confidence via R² per day (skip NaNs)
    conf_vals: list[float] = []
    for day in grp["forecast_day"]:
        subset = df_eval[df_eval["forecast_day"] == day][[TARGET_TEMP_COL, "predicted_temp"]].dropna()
        if len(subset) > 1:
            r2 = r2_score(subset[TARGET_TEMP_COL], subset["predicted_temp"])
            conf_vals.append(max(0, min(100, r2 * 100)))
        else:
            conf_vals.append(np.nan)
    grp["confidence_%"] = conf_vals

    return grp


def compute_overall_metrics(df_eval: pd.DataFrame) -> tuple[float, float]:
    """Return (confidence %, accuracy %) or (nan, nan) if not computable.
    This helper now drops rows containing NaNs before computing metrics to avoid
    ValueError from scikit-learn when all/any NaNs are present.
    """
    required = {TARGET_TEMP_COL, "predicted_temp"}
    if not required.issubset(df_eval.columns) or df_eval.empty:
        return float("nan"), float("nan")

    valid = df_eval[list(required)].dropna()
    if valid.empty:
        return float("nan"), float("nan")

    r2 = r2_score(valid[TARGET_TEMP_COL], valid["predicted_temp"])
    conf = max(0, min(100, r2 * 100))

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_err = (valid[TARGET_TEMP_COL] - valid["predicted_temp"]).abs() / valid[TARGET_TEMP_COL].replace(0, np.nan)
    avg_pct_err = pct_err.mean(skipna=True) * 100 if not pct_err.empty else float("nan")
    acc = max(0, 100 - avg_pct_err)
    return conf, acc


# --------------------------------------------------
# Helper to build future rows for forecasting
def make_future(df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Generate a future dataframe for the next ``horizon_days`` days.

    For each unique spatial location (grid_x/y/z) present in *df*, this function
    creates ``horizon_days`` duplicated rows with the *detection_time* advanced
    by 1..horizon_days. It also appends a *forecast_day* column (1-indexed).

    The resulting frame is passed through the same feature generators so it can
    be fed directly into the model for prediction.
    """
    if df.empty or "detection_time" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"])
    latest_ts = df["detection_time"].max()

    # Keep spatial coords plus any constant categorical IDs (grain_type / warehouse_type)
    keep_cols = [c for c in df.columns if c in {"grid_x", "grid_y", "grid_z", "granary_id", "heap_id"}]
    sensors = df[keep_cols].drop_duplicates().reset_index(drop=True)

    # Add constant metadata if available (assuming single value across file)
    for const_col in ["grain_type", "warehouse_type"]:
        if const_col in df.columns:
            sensors[const_col] = df[const_col].dropna().iloc[0]

    # Prepare base detection_cycle if present
    max_cycle = df["detection_cycle"].max() if "detection_cycle" in df.columns else None

    frames: List[pd.DataFrame] = []
    for d in range(1, horizon_days + 1):
        tmp = sensors.copy()
        tmp["detection_time"] = latest_ts + timedelta(days=d)
        tmp["forecast_day"] = d
        if max_cycle is not None:
            tmp["detection_cycle"] = max_cycle + d
        frames.append(tmp)

    future_df = pd.concat(frames, ignore_index=True)
    # Feature engineering to match training pipeline
    future_df = features.create_time_features(future_df)
    future_df = features.create_spatial_features(future_df)
    future_df = features.add_sensor_lag(future_df)
    # Ensure both legacy and new target columns exist so downstream feature
    # selection works regardless of current configuration.
    if "temperature_grain" not in future_df.columns:
        future_df["temperature_grain"] = np.nan
    if TARGET_TEMP_COL not in future_df.columns:
        future_df[TARGET_TEMP_COL] = np.nan
    return future_df
# --------------------------------------------------


def main():
    st.session_state.setdefault("evaluations", {})
    st.session_state.setdefault("forecasts", {})  # NEW: container for forecast results

    # Debug toggle – placed at very top so early messages are captured
    st.sidebar.checkbox("Verbose debug mode", key="debug_mode", help="Show detailed internal processing messages")

    with st.sidebar.expander("📂 Data", expanded=("uploaded_file" not in st.session_state)):
        uploaded_file = st.file_uploader("Upload your own CSV", type=["csv"], key="uploader")

        # ------------------------------------------------------------------
        # Offer bundled sample datasets so users can start instantly
        if PRELOADED_DATA_DIR.exists():
            sample_files = sorted(PRELOADED_DATA_DIR.glob("*.csv"))
            if sample_files:
                st.caption("Or pick a bundled sample dataset:")
                sample_names = ["-- Select sample --"] + [p.name for p in sample_files]
                sample_choice = st.selectbox(
                    "Sample dataset",  # non-empty label for accessibility
                    options=sample_names,
                    key="sample_selector",
                    label_visibility="collapsed",  # hide visually but keep for screen readers
                )
                if sample_choice and sample_choice != "-- Select sample --":
                    uploaded_file = PRELOADED_DATA_DIR / sample_choice  # path object -> pd.read_csv works
                    st.info(f"Sample dataset '{sample_choice}' selected.")

    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        with st.expander("Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True)

        # ------------------------------------------------------------------
        # Auto-organise if the upload mixes multiple silos
        # ------------------------------------------------------------------
        if "granary_id" in df.columns and "heap_id" in df.columns:
            uniq_silos = df[["granary_id", "heap_id"]].drop_duplicates().shape[0]
            if uniq_silos > 1 and not st.session_state.get("auto_organised", False) and not _looks_processed(uploaded_file):
                with st.spinner("Detected mixed dataset – organising into per-silo files…"):
                    out_root = "data/raw/by_silo"
                    try:
                        # Persist the upload to a temporary file so organizer can read it
                        if hasattr(uploaded_file, "read") and not isinstance(uploaded_file, (str, pathlib.Path)):
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                            tmp.write(uploaded_file.getvalue())
                            tmp.flush()
                            input_path = tmp.name
                        else:
                            input_path = uploaded_file if isinstance(uploaded_file, (str, pathlib.Path)) else uploaded_file.name
                        written = organize_mixed_csv(input_path, out_dir=out_root)
                        # Read all newly created slices back into a single frame for immediate use
                        import glob, os
                        slice_files = glob.glob(os.path.join(out_root, "**", "*.csv"), recursive=True)
                        df_slices = [pd.read_csv(fp, encoding="utf-8") for fp in slice_files]
                        if df_slices:
                            concat_df = pd.concat(df_slices, ignore_index=True, sort=False)
                            st.session_state["organized_df"] = concat_df

                            # Persist a preprocessed version to data/processed
                            from pathlib import Path
                            processed_dir = Path("data/processed")
                            processed_dir.mkdir(parents=True, exist_ok=True)
                            stem = Path(input_path).stem.replace(" ", "_")
                            out_csv = processed_dir / f"{stem}_processed.csv"
                            _preprocess_df(concat_df).to_csv(out_csv, index=False, encoding="utf-8")
                            st.info(f"Preprocessed CSV written to '{out_csv}'. You can reuse this file directly.")
                        st.success(f"Organised into {written} slice files under '{out_root}'.")
                        st.session_state["auto_organised"] = True
                        # Invalidate previous processed cache
                        st.session_state.pop("processed_df", None)
                    except Exception as exc:
                        st.warning(f"Could not organise mixed CSV: {exc}")

        # Full preprocessing once
        _d("Running full preprocessing on uploaded dataframe (cached)…")
        df = _get_preprocessed_df(uploaded_file)
        _d(f"Preprocessed dataframe shape: {df.shape}")

        # Display sorted table directly below Raw Data
        df_sorted_display = df
        with st.expander("Sorted Data", expanded=False):
            _st_dataframe_safe(df_sorted_display, key="sorted")

        # ------------------------------
        # Global Warehouse → Silo filter
        # ------------------------------

        st.markdown("### 🏢 Location Filter")
        with st.container():
            # Detect possible column names coming from different CSV formats
            wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in df.columns]
            silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in df.columns]

            wh_col = wh_col_candidates[0] if wh_col_candidates else None
            silo_col = silo_col_candidates[0] if silo_col_candidates else None

            # 1️⃣ Warehouse selector – always shown if the column exists
            if wh_col:
                warehouses = sorted(df[wh_col].dropna().unique())
                warehouses_opt = ["All"] + warehouses
                sel_wh_global = st.selectbox(
                    "Warehouse",
                    options=warehouses_opt,
                    key="global_wh",
                )
            else:
                sel_wh_global = "All"

            # 2️⃣ Silo selector – rendered only after a specific warehouse is chosen
            if wh_col and sel_wh_global != "All" and silo_col:
                silos = sorted(df[df[wh_col] == sel_wh_global][silo_col].dropna().unique())
                silos_opt = ["All"] + silos
                sel_silo_global = st.selectbox(
                    "Silo",
                    options=silos_opt,
                    key="global_silo",
                )
            else:
                sel_silo_global = "All"

            # Persist selection in session state for downstream use
            st.session_state["global_filters"] = {
                "wh": sel_wh_global,
                "silo": sel_silo_global,
            }

        with st.sidebar.expander("🏗️ Train / Retrain Model", expanded=False):
            model_choice = st.selectbox(
                "Algorithm",
                ["RandomForest", "HistGradientBoosting", "LightGBM"],
                index=0,
            )
            n_trees = st.slider("Iterations / Trees", 100, 1000, 300, step=100)
            future_safe = st.checkbox("Future-safe (exclude env vars)", value=True)
            train_pct = st.slider("Train split (%)", 50, 95, 80, step=5, help="Percentage of data used for training; the rest is held out for unbiased evaluation.")
            train_pressed = st.button("Train on uploaded CSV")

        if train_pressed and uploaded_file:
            with st.spinner("Training model – please wait..."):
                # -------- Data preparation --------
                df = _get_preprocessed_df(uploaded_file)

                if future_safe:
                    df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

                # Consistent sorting & grouping
                df = comprehensive_sort(df)
                df = assign_group_id(df)

                # Feature matrix / target
                X_all, y_all = features.select_feature_target(df, target_col=TARGET_TEMP_COL)

                # -------- Group-aware hold-out with user-defined split --------
                _d("Preparing train/test split…")
                # Chronological split by date proportion
                test_frac_chrono = max(0.01, 1 - train_pct / 100)
                df_train_tmp, df_eval_tmp = split_train_eval_frac(df, test_frac=test_frac_chrono)
                X_tr, y_tr = features.select_feature_target(df_train_tmp, target_col=TARGET_TEMP_COL)
                X_te, y_te = features.select_feature_target(df_eval_tmp, target_col=TARGET_TEMP_COL)
                perform_validation = not X_te.empty

                # -------- Model selection & training --------
                if model_choice == "RandomForest":
                    mdl = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=42)
                    suffix = "rf"
                elif model_choice == "HistGradientBoosting":
                    mdl = HistGradientBoostingRegressor(max_depth=None, learning_rate=0.1, max_iter=n_trees, random_state=42)
                    suffix = "hgb"
                else:  # LightGBM
                    mdl = LGBMRegressor(n_estimators=n_trees, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
                    suffix = "lgbm"

                _d(f"Instantiated {model_choice} with n_trees={n_trees}")
                mdl.fit(X_tr, y_tr)
                _d("Model fit on training data")

                # Validation on unseen groups (if possible)
                if perform_validation and not X_te.empty:
                    preds = mdl.predict(X_te)
                    _d("Predictions generated on validation split")
                    _d(f"Model {model_choice}: eval predictions shape {preds.shape}")
                    mae_val = mean_absolute_error(y_te, preds)
                    rmse_val = mean_squared_error(y_te, preds) ** 0.5
                    _d(f"Validation metrics → MAE: {mae_val:.3f}, RMSE: {rmse_val:.3f}")
                else:
                    mae_val = rmse_val = float("nan")

                # -------- Persist model --------
                csv_stem = pathlib.Path(uploaded_file.name).stem.replace(" ", "_").lower()
                model_name = f"{csv_stem}_{'fs_' if future_safe else ''}{suffix}_{n_trees}.joblib"
                model_utils.save_model(mdl, name=model_name)

            if np.isnan(mae_val):
                st.sidebar.success(f"{model_choice} trained (validation split contained no ground-truth targets).")
            else:
                st.sidebar.success(f"{model_choice} trained! MAE: {mae_val:.2f}, RMSE: {rmse_val:.2f}")
            # Persist train_pct for evaluation stage
            st.session_state["last_train_pct"] = train_pct

        # Existing model evaluation
        with st.sidebar.expander("🔍 Evaluate Model", expanded=False):
            model_files = list_available_models()
            if not model_files:
                st.write("No saved models yet.")
                eval_pressed = False
                selected_model = None
                eval_fc_pressed = False
                all_models_chk = False
            else:
                selected_model = st.selectbox("Model file", model_files)
                # Checkbox to act on all models
                all_models_chk = st.checkbox("Apply to all models", key="chk_eval_all")

                col_eval, col_evalfc = st.columns([1,1])
                with col_eval:
                    eval_pressed = st.button("Evaluate", key="btn_eval_single", use_container_width=True)
                with col_evalfc:
                    eval_fc_pressed = st.button("Eval & Forecast", key="btn_eval_fc", use_container_width=True)

        if (eval_pressed or eval_fc_pressed):
            if uploaded_file is None:
                st.warning("Please upload a CSV first to evaluate.")
            else:
                # Determine which models to evaluate
                target_models = list_available_models() if all_models_chk else [selected_model]
                with st.spinner("Evaluating model(s) – please wait..."):
                    df = _get_preprocessed_df(uploaded_file)
                    # Use same train/test split fraction recorded during training (default 20%)
                    test_frac = max(0.01, 1 - st.session_state.get("last_train_pct", 80)/100)

                    # ---------- Chronological split (proportional) ----------
                    df_train_base, df_eval_base = split_train_eval_frac(df, test_frac=test_frac)
                    _d(f"Evaluation split – train rows: {len(df_train_base)}, test rows: {len(df_eval_base)}")
                    use_gap_fill = False  # skip calendar gap generation – evaluate only real rows

                    X_train_base, _ = features.select_feature_target(df_train_base, target_col=TARGET_TEMP_COL)

                    for mdl_name in target_models:
                        df_train = df_train_base.copy()
                        df_eval = df_eval_base.copy()

                        mdl = load_trained_model(mdl_name)
                        if not mdl:
                            st.error(f"Could not load {mdl_name}")
                            continue

                        # If model is future-safe, drop env/extra cols from evaluation/training sets to mimic determinate-only input
                        if is_future_safe_model(mdl_name):
                            df_train = df_train.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
                            df_eval = df_eval.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

                        # ---------------- Ensure category codes align with training ----------------
                        cat_cols_train_loop = df_train.select_dtypes(include=["object", "category"]).columns
                        categories_map = {c: pd.Categorical(df_train[c]).categories.tolist() for c in cat_cols_train_loop}

                        X_eval, _ = features.select_feature_target(df_eval, target_col=TARGET_TEMP_COL)
                        # Align features to the model's expected input
                        feature_cols_mdl = get_feature_cols(mdl, X_eval)
                        X_eval_aligned = X_eval.reindex(columns=feature_cols_mdl, fill_value=0)
                        # NEW – generate aligned training design matrix for debugging visualisation
                        X_train, _ = features.select_feature_target(df_train, target_col=TARGET_TEMP_COL)
                        X_train_aligned = X_train.reindex(columns=feature_cols_mdl, fill_value=0)
                        preds = model_utils.predict(mdl, X_eval_aligned)
                        _d(f"Model {mdl_name}: eval predictions shape {preds.shape}")
                        # Assign only to rows actually used in prediction (no-length mismatch)
                        df_eval.loc[X_eval_aligned.index, "predicted_temp"] = preds
                        df_eval["is_forecast"] = False
                        # Combine training (actual only) and evaluation rows for full context time-series
                        df_train_plot = df_train.copy()
                        df_train_plot["is_forecast"] = False
                        df_predplot_all = pd.concat([df_eval, df_train_plot], ignore_index=True)

                        # metrics
                        df_eval_actual = df_eval[df_eval[TARGET_TEMP_COL].notna()].copy()
                        mae = (df_eval_actual[TARGET_TEMP_COL] - df_eval_actual["predicted_temp"]).abs().mean()
                        rmse = ((df_eval_actual[TARGET_TEMP_COL] - df_eval_actual["predicted_temp"]) ** 2).mean() ** 0.5
                        mape = ((df_eval_actual[TARGET_TEMP_COL] - df_eval_actual["predicted_temp"]).abs() / df_eval_actual[TARGET_TEMP_COL]).mean() * 100
                        conf, acc = compute_overall_metrics(df_eval_actual)

                        st.session_state["evaluations"][mdl_name] = {
                            "df_eval": df_eval,
                            "df_predplot_all": df_predplot_all,
                            "confidence": conf,
                            "accuracy": acc,
                            "rmse": rmse,
                            "mae": mae,
                            "mape": mape,
                            "feature_cols": feature_cols_mdl,
                            "categories_map": categories_map,
                            "horizon": len(df_eval["forecast_day"].unique()) if "forecast_day" in df_eval.columns else 0,
                            "df_base": df,
                            "model_name": mdl_name,
                            "future_safe": is_future_safe_model(mdl_name),
                            # NEW debug matrices
                            "X_train": X_train_aligned,
                            "X_eval": X_eval_aligned,
                        }

                    # last evaluated model as active
                    if target_models:
                        st.session_state["active_model"] = target_models[0]

                    st.sidebar.success("Evaluation(s) completed.")

                    # If user requested Eval & Forecast, automatically create forecast for selected model
                    if eval_fc_pressed:
                        st.sidebar.write("Generating forecast…")
                        models_to_fc = target_models  # already respects all_models_chk
                        for mdl in models_to_fc:
                            generate_and_store_forecast(mdl, horizon=7)
                        st.sidebar.success("Forecast(s) created.")

    # Render evaluation tabs if any
    if st.session_state["evaluations"]:
        active_model = st.session_state.get("active_model")
        # Reorder so active_model appears first if present
        eval_keys = list(st.session_state["evaluations"].keys())
        if active_model in eval_keys:
            eval_keys.remove(active_model)
            tab_labels = [active_model] + eval_keys
        else:
            tab_labels = eval_keys
        tabs = st.tabs(tab_labels)
        for tab_label, tab in zip(tab_labels, tabs):
            with tab:
                # Always create both inner tabs to keep layout stable across reruns
                inner_tabs = st.tabs(["Evaluation", "Forecast"])

                # --- Evaluation Tab ---
                with inner_tabs[0]:
                    render_evaluation(tab_label)

                # --- Forecast Tab ---
                with inner_tabs[1]:
                    if tab_label in st.session_state.get("forecasts", {}):
                        render_forecast(tab_label)
                    else:
                        st.info("No forecast generated yet for this model.")
                        if st.button("Generate Forecast", key=f"btn_gen_fc_main_{tab_label}"):
                            with st.spinner("Generating forecast…"):
                                if generate_and_store_forecast(tab_label, horizon=7):
                                    st.success("Forecast generated – switch tabs to view.")

    # --------------------------------------------------
    # Leaderboard (full-width collapsible panel) -----------------------------------
    evals = st.session_state["evaluations"]

    st.markdown("---")
    with st.expander("🏆 Model Leaderboard", expanded=False):
        if not evals:
            st.write("No evaluations yet.")
        else:
            data = []
            for name, d in evals.items():
                data.append(
                    {
                        "model": name,
                        "confidence": d.get("confidence", float("nan")),
                        "accuracy": d.get("accuracy", float("nan")),
                        "rmse": d.get("rmse", float("nan")),
                        "mae": d.get("mae", float("nan")),
                    }
                )
            df_leader = (
                pd.DataFrame(data)
                .sort_values(["confidence", "accuracy"], ascending=False)
                .reset_index(drop=True)
            )
            df_leader.insert(0, "rank", df_leader.index + 1)
            st.dataframe(df_leader, use_container_width=True)

    # Optionally still expose full log
    if st.session_state.get("debug_mode"):
        dbg_log = st.session_state.get("debug_msgs", [])
        if dbg_log:
            with st.expander("🛠️ Debug Log (full)", expanded=False):
                st.code("\n".join(dbg_log), language="text")



# ================== NEW HELPER RENDER FUNCTIONS ==================

def render_evaluation(model_name: str):
    """Render the evaluation view for a given *model_name* inside its tab."""
    res = st.session_state["evaluations"][model_name]
    # Categories captured during initial evaluation (may be empty)
    categories_map = res.get("categories_map", {})
    df_eval = res["df_eval"]
    df_predplot_all = res["df_predplot_all"]

    # Ensure 'forecast_day' exists for downstream UI widgets
    if "forecast_day" not in df_eval.columns and "detection_time" in df_eval.columns:
        date_series = pd.to_datetime(df_eval["detection_time"]).dt.floor("D")
        unique_dates_sorted = sorted(date_series.unique())
        date2idx = {d: idx for idx, d in enumerate(unique_dates_sorted, start=1)}
        df_eval["forecast_day"] = date_series.map(date2idx)
        # Apply same mapping to the combined prediction frame if present
        if "detection_time" in df_predplot_all.columns:
            df_predplot_all["forecast_day"] = pd.to_datetime(df_predplot_all["detection_time"]).dt.floor("D").map(date2idx)

    # ---------------- Warehouse → Silo cascading filters -----------------
    wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in df_eval.columns]
    silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in df_eval.columns]

    wh_col = wh_col_candidates[0] if wh_col_candidates else None
    silo_col = silo_col_candidates[0] if silo_col_candidates else None

    # Apply global location filter (chosen in the sidebar)
    global_filters = st.session_state.get("global_filters", {})
    sel_wh = global_filters.get("wh", "All")
    sel_silo = global_filters.get("silo", "All")

    if wh_col:
        if sel_wh != "All":
            df_eval = df_eval[df_eval[wh_col] == sel_wh]
            df_predplot_all = df_predplot_all[df_predplot_all[wh_col] == sel_wh]

        if silo_col and sel_silo != "All":
            df_eval = df_eval[df_eval[silo_col] == sel_silo]
            df_predplot_all = df_predplot_all[df_predplot_all[silo_col] == sel_silo]

    # -------- Metrics (re-computed on filtered subset) ---------
    conf_val, acc_val = compute_overall_metrics(df_eval)
    if {TARGET_TEMP_COL, "predicted_temp"}.issubset(df_eval.columns) and not df_eval.empty:
        mae_val = (df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]).abs().mean()
        rmse_val = ((df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
        mape_val = ((df_eval[TARGET_TEMP_COL] - df_eval["predicted_temp"]).abs() / df_eval[TARGET_TEMP_COL]).mean() * 100
    else:
        rmse_val = mae_val = mape_val = float("nan")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Confidence (%)", "--" if pd.isna(conf_val) else f"{conf_val:.3f}")
    with metric_cols[1]:
        st.metric("Accuracy (%)", "--" if pd.isna(acc_val) else f"{acc_val:.3f}")
    with metric_cols[2]:
        st.metric("RMSE", "--" if pd.isna(rmse_val) else f"{rmse_val:.3f}")
    with metric_cols[3]:
        st.metric("MAE", "--" if pd.isna(mae_val) else f"{mae_val:.3f}")
    with metric_cols[4]:
        st.metric("MAPE (%)", "--" if pd.isna(mape_val) else f"{mape_val:.3f}")
    st.markdown("---")

    summary_tab, pred_tab, grid_tab, ts_tab, debug_tab = st.tabs(["Summary", "Predictions", "3D Grid", "Time Series", "Debug"])

    with summary_tab:
        if "predicted_temp" in df_eval.columns:
            st.subheader("Forecast Summary (per day)")
            st.dataframe(
                forecast_summary(df_eval),
                use_container_width=True,
                key=f"summary_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}",
            )

            def exceeds(row):
                thresh = GRAIN_ALERT_THRESHOLDS.get(row.get("grain_type"), ALERT_TEMP_THRESHOLD)
                return row["predicted_temp"] >= thresh

            if df_eval.apply(exceeds, axis=1).any():
                st.error("⚠️ High temperature forecast detected for at least one grain type – monitor closely!")
            else:
                st.success("All predicted temperatures within safe limits for their grain types")

    with pred_tab:
        _st_dataframe_safe(df_predplot_all, key=f"pred_df_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")

    with grid_tab:
        day_choice = st.selectbox(
            "Select day",
            options=list(range(1, len(df_eval['forecast_day'].unique()) + 1)),
            key=f"day_{model_name}_grid_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}",
        )
        df_predplot = df_predplot_all[df_predplot_all.get("forecast_day", 1) == day_choice]
        plot_3d_grid(df_predplot, key=f"grid_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")

    with ts_tab:
        plot_time_series(df_predplot_all, key=f"time_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")

    # ------------------ DEBUG TAB ------------------
    with debug_tab:
        st.subheader("⚙️ Feature Matrices (first 100 rows)")
        x_train_dbg = res.get("X_train")
        if x_train_dbg is not None:
            st.write("Training – X_train")
            _st_dataframe_safe(x_train_dbg, key=f"xtrain_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")
        x_eval_dbg = res.get("X_eval")
        if x_eval_dbg is not None:
            st.write("Evaluation – X_eval")
            _st_dataframe_safe(x_eval_dbg, key=f"xeval_{model_name}_{len(df_eval['forecast_day'].unique()) if 'forecast_day' in df_eval.columns else 0}")
        st.write("Model Feature Columns (order)")
        st.code(", ".join(res.get("feature_cols", [])))


def render_forecast(model_name: str):
    """Render the forecast view (if available) for *model_name*."""
    forecast_data = st.session_state.get("forecasts", {}).get(model_name)
    if not forecast_data:
        st.info("No forecast generated for this model yet.")
        return

    # -------- Initial dataframe & warehouse/silo filters --------
    future_df = forecast_data["future_df"]

    # Apply global location filter (chosen in the sidebar)
    global_filters = st.session_state.get("global_filters", {})
    sel_wh_fc = global_filters.get("wh", "All")
    sel_silo_fc = global_filters.get("silo", "All")

    wh_col_candidates = [c for c in ["granary_id", "storepointName"] if c in future_df.columns]
    silo_col_candidates = [c for c in ["heap_id", "storeName"] if c in future_df.columns]

    wh_col = wh_col_candidates[0] if wh_col_candidates else None
    silo_col = silo_col_candidates[0] if silo_col_candidates else None

    if wh_col:
        if sel_wh_fc != "All":
            future_df = future_df[future_df[wh_col] == sel_wh_fc]

        if silo_col and sel_silo_fc != "All":
            future_df = future_df[future_df[silo_col] == sel_silo_fc]

    # Update forecast_data copy with filtered df for downstream plots
    df_plot_base = future_df.copy()

    # -------- Metrics forwarded from last evaluation (confidence etc.) ---------
    res_eval = st.session_state.get("evaluations", {}).get(model_name, {})
    conf_val = res_eval.get("confidence", float("nan"))
    acc_val = res_eval.get("accuracy", float("nan"))
    rmse_val = res_eval.get("rmse", float("nan"))
    mae_val = res_eval.get("mae", float("nan"))
    mape_val = res_eval.get("mape", float("nan"))

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Confidence (%)", "--" if pd.isna(conf_val) else f"{conf_val:.3f}")
    with metric_cols[1]:
        st.metric("Accuracy (%)", "--" if pd.isna(acc_val) else f"{acc_val:.3f}")
    with metric_cols[2]:
        st.metric("RMSE", "--" if pd.isna(rmse_val) else f"{rmse_val:.3f}")
    with metric_cols[3]:
        st.metric("MAE", "--" if pd.isna(mae_val) else f"{mae_val:.3f}")
    with metric_cols[4]:
        st.metric("MAPE (%)", "--" if pd.isna(mape_val) else f"{mape_val:.3f}")

    st.markdown("---")

    # Tabs similar to evaluation (+Debug)
    summary_tab, pred_tab, grid_tab, ts_tab, debug_tab = st.tabs(["Summary", "Predictions", "3D Grid", "Time Series", "Debug"])

    with summary_tab:
        # Only predicted statistics available
        grp = (
            future_df.groupby("forecast_day")
            .agg(pred_mean=("predicted_temp", "mean"), pred_max=("predicted_temp", "max"), pred_min=("predicted_temp", "min"))
            .reset_index()
        )
        st.subheader("Forecast Summary (predicted)")
        _st_dataframe_safe(grp, key=f"forecast_summary_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with pred_tab:
        _st_dataframe_safe(future_df, key=f"future_pred_df_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with grid_tab:
        day_choice = st.selectbox(
            "Select day",
            options=list(range(1, len(future_df['forecast_day'].unique()) + 1)),
            key=f"future_day_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}",
        )
        day_df = future_df[future_df.get("forecast_day", 1) == day_choice]
        plot_3d_grid(day_df, key=f"future_grid_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    with ts_tab:
        plot_time_series(df_plot_base, key=f"future_ts_{model_name}_{len(future_df['forecast_day'].unique()) if 'forecast_day' in future_df.columns else 0}")

    # ------------------ DEBUG TAB ------------------
    with debug_tab:
        st.subheader("⚙️ Future Feature Matrix (first 100 rows)")
        x_future_dbg = st.session_state.get("forecasts", {}).get(model_name, {}).get("X_future")
        if x_future_dbg is not None:
            st.dataframe(x_future_dbg.head(100), use_container_width=True)
            # Compare to evaluation matrix if available
            eval_res = st.session_state["evaluations"].get(model_name, {})
            x_eval_dbg = eval_res.get("X_eval")
            if x_eval_dbg is not None:
                delta = (x_eval_dbg.mean() - x_future_dbg.mean()).abs().sort_values(ascending=False)
                st.subheader("|Mean(X_eval) − Mean(X_future)| (Top 20)")
                st.dataframe(delta.head(20).to_frame(name="abs_diff"), use_container_width=True)
        else:
            st.info("X_future matrix not available yet.")

# --------------------------------------------------
# Helper to create & store forecast
def generate_and_store_forecast(model_name: str, horizon: int) -> bool:
    """Generate future_df for *model_name* and store in session_state['forecasts'].
    Returns True if successful, False otherwise."""
    res_eval = st.session_state.get("evaluations", {}).get(model_name)
    if res_eval is None:
        st.error("Please evaluate the model first.")
        return False

    base_df = res_eval.get("df_base")
    categories_map = res_eval.get("categories_map", {})
    mdl = load_trained_model(model_name)

    if not isinstance(base_df, pd.DataFrame) or mdl is None:
        st.error("Unable to access base data or model for forecasting.")
        return False

    # ---------------- Rolling forecast so lag is updated day-by-day ----------------
    hist_df = base_df.copy()
    all_future_frames: list[pd.DataFrame] = []

    for d in range(1, horizon + 1):
        # Generate placeholder rows for ONE day ahead
        day_df = make_future(hist_df, horizon_days=1)
        day_df = _inject_future_lag(day_df, hist_df)
        day_df["forecast_day"] = d

        # Apply categories levels
        for col, cats in categories_map.items():
            if col in day_df.columns:
                day_df[col] = pd.Categorical(day_df[col], categories=cats)

        X_day, _ = features.select_feature_target(day_df, target_col=TARGET_TEMP_COL)
        model_feats = get_feature_cols(mdl, X_day)
        X_day_aligned = X_day.reindex(columns=model_feats, fill_value=0)
        preds = model_utils.predict(mdl, X_day_aligned)
        day_df["predicted_temp"] = preds
        day_df["temperature_grain"] = preds  # feed back as history for next lag
        day_df[TARGET_TEMP_COL] = preds
        day_df["is_forecast"] = True

        # Append to history for subsequent lag calculation
        hist_df = pd.concat([hist_df, day_df], ignore_index=True, sort=False)
        all_future_frames.append(day_df)

    future_df = pd.concat(all_future_frames, ignore_index=True)

    st.session_state.setdefault("forecasts", {})[model_name] = {
        "future_df": future_df,
        "future_horizon": horizon,
        "X_future": X_day_aligned,  # last horizon step matrix for debug
    }
    return True


# ---------------- Utility to extract feature column order from a model -----------------
def get_feature_cols(model, X_fallback: pd.DataFrame) -> list[str]:
    """Return the exact feature columns the *model* expects.

    1. scikit-learn 1.0+ estimators expose ``feature_names_in_``.
    2. LightGBM exposes ``feature_name_``.
    3. Fallback: use the columns of *X_fallback* (already aligned for current dataset).
    """
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    if hasattr(model, "feature_name_"):
        return list(getattr(model, "feature_name_"))
    return list(X_fallback.columns)


# ----------------- Helper for future lag injection -----------------
def _inject_future_lag(future_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Populate lag_temp_1d in *future_df* using the last known
    temperature_grain for each sensor from *history_df*.  Assumes both frames
    contain grid_x/y/z columns.
    """
    if {"grid_x", "grid_y", "grid_z", "temperature_grain"}.issubset(history_df.columns):
        last_vals = (
            history_df.sort_values("detection_time")
            .dropna(subset=["temperature_grain"])
            .groupby(["grid_x", "grid_y", "grid_z"])["temperature_grain"]
            .last()
        )
        idx = future_df.set_index(["grid_x", "grid_y", "grid_z"]).index
        future_df["lag_temp_1d"] = [last_vals.get(key, np.nan) for key in idx]
    return future_df


# ---------------------------------------------------------------------
# Common dataframe preprocessing (clean → fill → features → lag → sort)
# ---------------------------------------------------------------------

def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full data-prep pipeline exactly once."""
    if df.empty:
        return df
    _d("Starting basic_clean…")
    before_cols = list(df.columns)
    df = cleaning.basic_clean(df)
    _d(f"basic_clean: cols before={len(before_cols)} after={len(df.columns)} rows={len(df)}")

    # -------------------------------------------------------------
    # 1️⃣ Insert missing calendar-day rows first
    # -------------------------------------------------------------
    df = _insert_calendar_gaps(df)
    _d("insert_calendar_gaps: added rows for missing dates")

    # -------------------------------------------------------------
    # 2️⃣ Interpolate numeric columns per sensor across the now-complete
    #    timeline so gap rows take the average of surrounding real values.
    # -------------------------------------------------------------
    df = _interpolate_sensor_numeric(df)
    _d("_interpolate_sensor_numeric: linear interpolation applied per sensor")

    # -------------------------------------------------------------
    # 3️⃣ Final fill_missing to tidy up any residual NaNs (categoricals etc.)
    # -------------------------------------------------------------
    na_before = df.isna().sum().sum()
    df = cleaning.fill_missing(df)
    na_after = df.isna().sum().sum()
    _d(f"fill_missing (final): total NaNs before={na_before} after={na_after}")

    df = features.create_time_features(df)
    _d("create_time_features: added year/month/day/hour cols")
    df = features.create_spatial_features(df)
    _d("create_spatial_features: removed grid_index if present")
    before_lag_na = df["temperature_grain"].isna().sum() if "temperature_grain" in df.columns else 0
    df = features.add_sensor_lag(df)
    after_lag_na = df["lag_temp_1d"].isna().sum() if "lag_temp_1d" in df.columns else 0
    _d(f"add_sensor_lag: lag NaNs={after_lag_na} (target NaNs before={before_lag_na})")
    # Ensure group identifiers available for downstream splitting/evaluation
    df = assign_group_id(df)
    _d("assign_group_id: _group_id column added to dataframe")
    df = comprehensive_sort(df)
    _d("comprehensive_sort: dataframe sorted by granary/heap/grid/date")
    return df


# ---------------------------------------------------------------------
# Helper: insert rows for missing calendar dates per sensor
# ---------------------------------------------------------------------

def _insert_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* where any missing *calendar days* for each sensor are
    back-filled with synthetic rows so models see a continuous timeline.

    • Sensor grouping columns: granary_id, heap_id, grid_x/y/z (subset present).
    • For each missing date, copies the most recent known row for that sensor
      and nulls out numeric, non-static measurement columns so they can be
      filled later (mean, ffill etc.).
    """
    if "detection_time" not in df.columns:
        return df

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    if not group_cols:
        group_cols = []  # treat whole frame as one group

    frames = [df]

    # Helper to decide which numeric cols are *measurements* (varying) vs static
    static_like = set(group_cols + ["granary_id", "heap_id", "grain_type", "warehouse_type"])  # do not null

    for key, sub in df.groupby(group_cols) if group_cols else [(None, df)]:
        sub = sub.sort_values("detection_time")
        date_floor = sub["detection_time"].dt.floor("D")
        full_range = pd.date_range(date_floor.min(), date_floor.max(), freq="D")
        missing_dates = sorted(set(full_range.date) - set(date_floor.dt.date.unique()))
        if not missing_dates:
            continue

        # Use last known row as template (static cols correct)
        template = sub.iloc[-1].copy()

        new_rows = []
        for md in missing_dates:
            row = template.copy()
            row["detection_time"] = pd.Timestamp(md)
            # Null out non-static numeric columns to be filled later
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in static_like:
                    row[col] = np.nan
            new_rows.append(row)

        if new_rows:
            frames.append(pd.DataFrame(new_rows))

    df_full = pd.concat(frames, ignore_index=True)
    return df_full


# ---------------------------------------------------------------------
# Helper to fetch raw (possibly organised) dataframe
# ---------------------------------------------------------------------

def _get_active_df(uploaded_file):
    """Return the raw dataframe – organised slice concat if available."""
    if st.session_state.get("organized_df") is not None:
        return st.session_state["organized_df"].copy()
    return load_uploaded_file(uploaded_file)


# ---------------------------------------------------------------------
# Helper to fetch whichever DataFrame (raw/organised) is active & processed
# ---------------------------------------------------------------------

@st.cache_data(show_spinner="Running full preprocessing…")
def _preprocess_cached(df: pd.DataFrame) -> pd.DataFrame:
    """Cached version of the full preprocessing pipeline."""
    _d("⚙️ _preprocess_cached: CACHE MISS → executing heavy pipeline")
    return _preprocess_df(df)

def _get_preprocessed_df(uploaded_file):
    """Return a fully-preprocessed dataframe (cached in session)."""
    # --------------------------------------------------------
    # 0️⃣ Fast-path: if the uploaded file is already a processed
    #    CSV (name ends with _processed.csv or resides in data/processed),
    #    simply load and return it.
    # --------------------------------------------------------
    if _looks_processed(uploaded_file):
        try:
            _d("✅ Detected preprocessed CSV – loading directly, skipping heavy pipeline")
            df_fast = pd.read_csv(uploaded_file, encoding="utf-8") if isinstance(uploaded_file, (str, pathlib.Path)) else pd.read_csv(uploaded_file.name, encoding="utf-8")
            st.session_state["processed_df"] = df_fast.copy()
            return df_fast
        except Exception as exc:
            _d(f"⚠️ Could not load preprocessed CSV fast-path: {exc}; falling back to pipeline")

    raw_df = _get_active_df(uploaded_file)

    # Use cached preprocessing to avoid repeating heavy work across reruns
    proc = _preprocess_cached(raw_df)
    _d("🔄 Received dataframe from _preprocess_cached (may be cache hit or miss)")

    # --------------------------------------------------------
    # Persist a processed CSV alongside others for future fast-path
    # --------------------------------------------------------
    try:
        if hasattr(uploaded_file, "name"):
            orig_name = pathlib.Path(uploaded_file.name).stem
        else:
            orig_name = pathlib.Path(uploaded_file).stem if isinstance(uploaded_file, (str, pathlib.Path)) else "uploaded"

        processed_dir = pathlib.Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_csv = processed_dir / f"{orig_name}_processed.csv"
        if not out_csv.exists():
            proc.to_csv(out_csv, index=False, encoding="utf-8")
            _d(f"💾 Saved processed CSV to {out_csv}")
    except Exception as exc:
        _d(f"⚠️ Could not persist processed CSV: {exc}")

    st.session_state["processed_df"] = proc.copy()
    return proc


# ---------------------------------------------------------------------
# Helper to safely display DataFrames in Streamlit (Category→str)
# ---------------------------------------------------------------------

def _st_dataframe_safe(df: pd.DataFrame, key: str | None = None):
    """Wrapper around st.dataframe that converts category columns to string
    to avoid pyarrow ArrowInvalid errors when categories mix numeric & text.
    """
    df_disp = df.copy()
    for col in df_disp.select_dtypes(include=["category"]).columns:
        df_disp[col] = df_disp[col].astype(str)
    st.dataframe(df_disp, use_container_width=True, key=key)


# ---------------------------------------------------------------------
# Helper to guess if an uploaded file is already processed
# ---------------------------------------------------------------------

def _looks_processed(upload):
    """Return True if *upload* path or name suggests preprocessed dataset."""
    if isinstance(upload, (str, pathlib.Path)):
        p = pathlib.Path(upload)
        if "data/processed" in p.as_posix() or p.name.endswith("_processed.csv"):
            return True
    elif hasattr(upload, "name"):
        name = upload.name
        if name.endswith("_processed.csv"):
            return True
    return False


# ---------------------------------------------------------------------
# Helper: numeric interpolation per sensor across calendar-completed frame
# ---------------------------------------------------------------------

def _interpolate_sensor_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """For each sensor group, linearly interpolate numeric columns along
    chronological order so values for synthetic gap rows equal the average of
    previous and next real measurements."""

    if "detection_time" not in df.columns:
        return df

    df = df.copy()
    df["detection_time"] = pd.to_datetime(df["detection_time"], errors="coerce")
    df.sort_values("detection_time", inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df

    group_cols = [c for c in ["granary_id", "heap_id", "grid_x", "grid_y", "grid_z"] if c in df.columns]
    if group_cols:
        df[num_cols] = (
            df.groupby(group_cols)[num_cols]
            .apply(lambda g: g.interpolate(method="linear").ffill().bfill())
            .reset_index(level=group_cols, drop=True)
        )
    else:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    return df


if __name__ == "__main__":
    main() 