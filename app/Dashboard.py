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

# Streamlit reload may have stale module; fetch grain thresholds safely
try:
    from granarypredict.config import GRAIN_ALERT_THRESHOLDS  # type: ignore
except ImportError:
    GRAIN_ALERT_THRESHOLDS = {}

from sklearn.metrics import r2_score

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
    "avg_in_temp",   # historic average inside temp
    "min_temp",      # historic min inside temp
    "line_no",       # production line identifier (constant, but 0-filled in future)
    "layer_no",      # vertical layer identifier (constant, but 0-filled in future)
]

# Utility to detect if a model is "future-safe" by filename convention (contains 'fs_')
def is_future_safe_model(model_name: str) -> bool:
    return "fs_" in model_name.lower()

st.set_page_config(page_title="GranaryPredict", layout="wide")



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
        if "temp" in df.columns and "batch" in df.columns:
            df = ingestion.standardize_result147(df)
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
    tmp["date"] = pd.to_datetime(tmp["detection_time"]).dt.date
    grp = tmp.groupby("date").agg(actual=("temperature_grain", "mean"), predicted=("predicted_temp", "mean")).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grp["date"], y=grp["actual"], mode="lines+markers", name="Actual Avg"))
    fig.add_trace(go.Scatter(x=grp["date"], y=grp["predicted"], mode="lines+markers", name="Predicted Avg"))
    fig.update_layout(title="Average Grain Temperature Over Time", xaxis_title="Date", yaxis_title="Temperature (°C)")
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


def forecast_summary(df_eval: pd.DataFrame) -> pd.DataFrame:
    if "forecast_day" not in df_eval.columns:
        return pd.DataFrame()
    grp = (
        df_eval.groupby("forecast_day")
        .agg(actual_mean=("temperature_grain", "mean"), pred_mean=("predicted_temp", "mean"))
        .reset_index()
    )
    grp["pct_error"] = (grp["pred_mean"] - grp["actual_mean"]).abs() / grp["actual_mean"] * 100
    # confidence via R2 per day (fallback 0 if single sample)
    confs = []
    for d in grp["forecast_day"]:
        subset = df_eval[df_eval["forecast_day"] == d]
        if len(subset) > 1:
            r2 = r2_score(subset["temperature_grain"], subset["predicted_temp"])
            confs.append(max(0, min(100, r2 * 100)))
        else:
            confs.append(np.nan)
    grp["confidence_%"] = confs
    return grp


def compute_overall_metrics(df_eval: pd.DataFrame) -> tuple[float, float]:
    """Return (confidence %, accuracy %) or (nan, nan) if not computable."""
    if {"temperature_grain", "predicted_temp"}.issubset(df_eval.columns) and not df_eval.empty:
        r2 = r2_score(df_eval["temperature_grain"], df_eval["predicted_temp"])
        conf = max(0, min(100, r2 * 100))
        avg_pct_err = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs() / df_eval["temperature_grain"]).mean() * 100
        acc = max(0, 100 - avg_pct_err)
        return conf, acc
    return float("nan"), float("nan")


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
    # Add placeholder target column so downstream feature selector does not fail
    if "temperature_grain" not in future_df.columns:
        future_df["temperature_grain"] = np.nan
    return future_df
# --------------------------------------------------


def main():
    st.session_state.setdefault("evaluations", {})
    st.session_state.setdefault("forecasts", {})  # NEW: container for forecast results

    with st.sidebar.expander("📂 Data", expanded=("uploaded_file" not in st.session_state)):
        uploaded_file = st.file_uploader("Upload your own CSV", type=["csv"], key="uploader")

        # ------------------------------------------------------------------
        # Offer bundled sample datasets so users can start instantly
        if PRELOADED_DATA_DIR.exists():
            sample_files = sorted(PRELOADED_DATA_DIR.glob("*.csv"))
            if sample_files:
                st.caption("Or pick a bundled sample dataset:")
                sample_names = ["-- Select sample --"] + [p.name for p in sample_files]
                sample_choice = st.selectbox("", options=sample_names, key="sample_selector")
                if sample_choice and sample_choice != "-- Select sample --":
                    uploaded_file = PRELOADED_DATA_DIR / sample_choice  # path object -> pd.read_csv works
                    st.info(f"Sample dataset '{sample_choice}' selected.")

    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        with st.expander("Raw Data", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Cleaning
        df = cleaning.basic_clean(df)
        df = cleaning.fill_missing(df)

        # Feature engineering
        df = features.create_time_features(df)
        df = features.create_spatial_features(df)

        with st.sidebar.expander("🏗️ Train / Retrain Model", expanded=False):
            model_choice = st.selectbox(
                "Algorithm",
                ["RandomForest", "HistGradientBoosting", "LightGBM"],
                index=0,
            )
            n_trees = st.slider("Iterations / Trees", 100, 1000, 300, step=100)
            future_safe = st.checkbox("Future-safe (exclude env vars)", value=True)
            train_pressed = st.button("Train on uploaded CSV")

        if train_pressed and uploaded_file:
            with st.spinner("Training model – please wait..."):
                df = load_uploaded_file(uploaded_file)
                df = cleaning.basic_clean(df)
                df = cleaning.fill_missing(df)
                df = features.create_time_features(df)
                df = features.create_spatial_features(df)
                if future_safe:
                    df = df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
                X_train, y_train = features.select_feature_target(df)
                csv_stem = pathlib.Path(uploaded_file.name).stem.replace(" ", "_" ).lower()
                if model_choice == "RandomForest":
                    mdl, metrics = model_utils.train_random_forest(X_train, y_train, n_estimators=n_trees)
                    model_name = f"{csv_stem}_{'fs_' if future_safe else ''}rf_{n_trees}.joblib"
                elif model_choice == "HistGradientBoosting":
                    mdl, metrics = model_utils.train_gb_models(X_train, y_train, model_type="hist", n_estimators=n_trees)
                    model_name = f"{csv_stem}_{'fs_' if future_safe else ''}hgb_{n_trees}.joblib"
                else:
                    mdl, metrics = model_utils.train_lightgbm(X_train, y_train, n_estimators=n_trees)
                    model_name = f"{csv_stem}_{'fs_' if future_safe else ''}lgbm_{n_trees}.joblib"
                model_utils.save_model(mdl, name=model_name)
            st.sidebar.success(
                f"{model_choice} trained! MAE: {metrics.get('mae', metrics.get('mae_cv')):.2f}, "
                f"RMSE: {metrics.get('rmse', metrics.get('rmse_cv')):.2f}"
            )

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
                    df = load_uploaded_file(uploaded_file)
                    df = cleaning.basic_clean(df)
                    df = cleaning.fill_missing(df)
                    df = features.create_time_features(df)
                    df = features.create_spatial_features(df)
                    default_horizon = 5

                    # common split & base df saved once
                    df_train_base, df_eval_base = split_train_eval(df, horizon=default_horizon)

                    X_train_base, _ = features.select_feature_target(df_train_base)

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

                        X_eval, _ = features.select_feature_target(df_eval)
                        # Align features to the model's expected input
                        try:
                            feature_cols_mdl = list(mdl.feature_name_)
                        except AttributeError:
                            feature_cols_mdl = list(X_eval.columns)
                        X_eval_aligned = X_eval.reindex(columns=feature_cols_mdl, fill_value=0)
                        preds = model_utils.predict(mdl, X_eval_aligned)
                        df_eval["predicted_temp"] = preds
                        df_predplot_all = pd.concat([df_eval, df_train], ignore_index=True)

                        # metrics
                        mae = (df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs().mean()
                        rmse = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
                        conf, acc = compute_overall_metrics(df_eval)

                        # Capture category levels from training data for consistent encoding later
                        cat_cols_train = df_train_base.select_dtypes(include=["object", "category"]).columns
                        categories_map = {c: pd.Categorical(df_train_base[c]).categories.tolist() for c in cat_cols_train}

                        st.session_state["evaluations"][mdl_name] = {
                            "df_eval": df_eval,
                            "df_predplot_all": df_predplot_all,
                            "confidence": conf,
                            "accuracy": acc,
                            "rmse": rmse,
                            "mae": mae,
                            "feature_cols": feature_cols_mdl,
                            "categories_map": categories_map,
                            "horizon": default_horizon,
                            "df_base": df,
                            "model_name": mdl_name,
                            "future_safe": is_future_safe_model(mdl_name),
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



# ================== NEW HELPER RENDER FUNCTIONS ==================

def render_evaluation(model_name: str):
    """Render the evaluation view for a given *model_name* inside its tab."""
    res = st.session_state["evaluations"][model_name]
    df_eval = res["df_eval"]
    df_predplot_all = res["df_predplot_all"]

    slider_key = f"horizon_slider_{model_name}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(res.get("horizon", 5))

    horizon_sel = st.slider("Evaluation horizon (days)", 1, 30, key=slider_key)

    # If horizon changed, recompute evaluation
    if horizon_sel != res.get("horizon"):
        base_df = res.get("df_base")
        if isinstance(base_df, pd.DataFrame):
            if res.get("future_safe"):
                base_df = base_df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
            with st.spinner("Re-evaluating…"):
                df_train_new, df_eval_new = split_train_eval(base_df, horizon=horizon_sel)
            if res.get("future_safe"):
                df_train_new = df_train_new.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")
                df_eval_new = df_eval_new.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

            X_train_new, _ = features.select_feature_target(df_train_new)
            X_eval_new, _ = features.select_feature_target(df_eval_new)

            mdl_new = load_trained_model(res.get("model_name", model_name))
            if mdl_new:
                try:
                    feature_cols_mdl = list(mdl_new.feature_name_)
                except AttributeError:
                    feature_cols_mdl = list(X_eval_new.columns)
                X_eval_aligned_new = X_eval_new.reindex(columns=feature_cols_mdl, fill_value=0)
                preds_new = model_utils.predict(mdl_new, X_eval_aligned_new)
                df_eval_new["predicted_temp"] = preds_new
                df_predplot_all_new = pd.concat([df_eval_new, df_train_new], ignore_index=True)

                mae_new = (df_eval_new["temperature_grain"] - df_eval_new["predicted_temp"]).abs().mean()
                rmse_new = ((df_eval_new["temperature_grain"] - df_eval_new["predicted_temp"]) ** 2).mean() ** 0.5
                conf_new, acc_new = compute_overall_metrics(df_eval_new)

                cat_cols_train = df_train_new.select_dtypes(include=["object", "category"]).columns
                categories_map = {c: pd.Categorical(df_train_new[c]).categories.tolist() for c in cat_cols_train}

                st.session_state["evaluations"][model_name].update(
                    {
                        "df_eval": df_eval_new,
                        "df_predplot_all": df_predplot_all_new,
                        "confidence": conf_new,
                        "accuracy": acc_new,
                        "rmse": rmse_new,
                        "mae": mae_new,
                        "feature_cols": feature_cols_mdl,
                        "categories_map": categories_map,
                        "horizon": horizon_sel,
                    }
                )

                df_eval = df_eval_new

    # -------- Metrics ---------
    conf_val = res.get("confidence", float("nan"))
    if isinstance(conf_val, tuple):
        conf_val = conf_val[0]
    acc_val = res.get("accuracy", float("nan"))
    if isinstance(acc_val, tuple):
        acc_val = acc_val[1] if len(acc_val) > 1 else acc_val[0]
    rmse_val = res.get("rmse", float("nan"))
    mae_val = res.get("mae", float("nan"))

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Confidence (%)", "--" if pd.isna(conf_val) else f"{conf_val:.3f}")
        if not pd.isna(conf_val):
            st.progress(min(int(conf_val), 100))
    with metric_cols[1]:
        st.metric("Accuracy (%)", "--" if pd.isna(acc_val) else f"{acc_val:.3f}")
        if not pd.isna(acc_val):
            st.progress(min(int(acc_val), 100))
    with metric_cols[2]:
        st.metric("RMSE", "--" if pd.isna(rmse_val) else f"{rmse_val:.3f}")
    with metric_cols[3]:
        st.metric("MAE", "--" if pd.isna(mae_val) else f"{mae_val:.3f}")
    st.markdown("---")

    summary_tab, pred_tab, grid_tab, ts_tab = st.tabs(["Summary", "Predictions", "3D Grid", "Time Series"])

    with summary_tab:
        if "predicted_temp" in df_eval.columns:
            st.subheader("Forecast Summary (per day)")
            st.dataframe(forecast_summary(df_eval), use_container_width=True)

            def exceeds(row):
                thresh = GRAIN_ALERT_THRESHOLDS.get(row.get("grain_type"), ALERT_TEMP_THRESHOLD)
                return row["predicted_temp"] >= thresh

            if df_eval.apply(exceeds, axis=1).any():
                st.error("⚠️ High temperature forecast detected for at least one grain type – monitor closely!")
            else:
                st.success("All predicted temperatures within safe limits for their grain types")

    with pred_tab:
        st.dataframe(df_predplot_all, use_container_width=True)

    with grid_tab:
        day_choice = st.selectbox("Select day", options=list(range(1, horizon_sel + 1)), key=f"day_{model_name}_grid")
        df_predplot = df_predplot_all[df_predplot_all.get("forecast_day", 1) == day_choice]
        plot_3d_grid(df_predplot, key=f"grid_{model_name}")

    with ts_tab:
        plot_time_series(df_predplot_all, key=f"time_{model_name}")


def render_forecast(model_name: str):
    """Render the forecast view (if available) for *model_name*."""
    forecast_data = st.session_state.get("forecasts", {}).get(model_name)
    if not forecast_data:
        st.info("No forecast generated for this model yet.")
        return

    # Allow changing horizon interactively
    horizon_slider_key_tab = f"future_horizon_tab_{model_name}"
    current_default = forecast_data.get("future_horizon", 7)
    future_horizon = st.slider(
        "Forecast days ahead (adjust to regenerate)",
        1,
        30,
        current_default,
        key=horizon_slider_key_tab,
    )

    # Regenerate if horizon differs
    if future_horizon != forecast_data.get("future_horizon"):
        res_eval = st.session_state["evaluations"].get(model_name, {})
        base_df = res_eval.get("df_base")
        categories_map = res_eval.get("categories_map", {})
        mdl = load_trained_model(model_name)
        if isinstance(base_df, pd.DataFrame) and mdl:
            future_df_new = make_future(base_df, horizon_days=future_horizon)
            for col, cats in categories_map.items():
                if col in future_df_new.columns:
                    future_df_new[col] = pd.Categorical(future_df_new[col], categories=cats)
            X_future_new, _ = features.select_feature_target(future_df_new)
            try:
                model_feats = list(mdl.feature_name_)
            except AttributeError:
                model_feats = res_eval.get("feature_cols", list(X_future_new.columns))
            X_future_aligned_new = X_future_new.reindex(columns=model_feats, fill_value=0)
            preds_future_new = model_utils.predict(mdl, X_future_aligned_new)
            future_df_new["predicted_temp"] = preds_future_new
            future_df_new["temperature_grain"] = np.nan

            st.session_state["forecasts"][model_name] = {
                "future_df": future_df_new,
                "future_horizon": future_horizon,
            }
            future_df = future_df_new
        else:
            future_df = forecast_data["future_df"]
    else:
        future_df = forecast_data["future_df"]

    # Combine past eval data with future for time series visualization
    res_eval = st.session_state["evaluations"].get(model_name, {})
    past_df_ts = res_eval.get("df_predplot_all")
    if isinstance(past_df_ts, pd.DataFrame):
        df_plot_ts = pd.concat([past_df_ts, future_df], ignore_index=True, sort=False)
    else:
        df_plot_ts = future_df.copy()

    # -------- Metrics (placeholders) ---------
    metric_cols = st.columns(4)
    for i, label in enumerate(["Confidence (%)", "Accuracy (%)", "RMSE", "MAE"]):
        with metric_cols[i]:
            st.metric(label, "--")

    st.markdown("---")

    # Tabs similar to evaluation
    summary_tab, pred_tab, grid_tab, ts_tab = st.tabs(["Summary", "Predictions", "3D Grid", "Time Series"])

    with summary_tab:
        # Only predicted statistics available
        grp = (
            future_df.groupby("forecast_day")
            .agg(pred_mean=("predicted_temp", "mean"), pred_max=("predicted_temp", "max"), pred_min=("predicted_temp", "min"))
            .reset_index()
        )
        st.subheader("Forecast Summary (predicted)")
        st.dataframe(grp, use_container_width=True)

    with pred_tab:
        st.dataframe(future_df, use_container_width=True)

    with grid_tab:
        day_choice = st.selectbox("Select day", options=list(range(1, future_horizon + 1)), key=f"future_day_{model_name}")
        day_df = future_df[future_df.get("forecast_day", 1) == day_choice]
        plot_3d_grid(day_df, key=f"future_grid_{model_name}")

    with ts_tab:
        plot_time_series(df_plot_ts, key=f"future_ts_{model_name}")

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

    # Respect future-safe constraint for deterministic inputs
    if res_eval.get("future_safe"):
        base_df = base_df.drop(columns=ENV_COLUMNS + FUTURE_SAFE_EXTRA, errors="ignore")

    future_df = make_future(base_df, horizon_days=horizon)

    # Apply training categorical levels
    for col, cats in categories_map.items():
        if col in future_df.columns:
            future_df[col] = pd.Categorical(future_df[col], categories=cats)

    X_future, _ = features.select_feature_target(future_df)
    try:
        model_feats = list(mdl.feature_name_)
    except AttributeError:
        model_feats = res_eval.get("feature_cols", list(X_future.columns))

    X_future_aligned = X_future.reindex(columns=model_feats, fill_value=0)
    preds_future = model_utils.predict(mdl, X_future_aligned)
    future_df["predicted_temp"] = preds_future
    future_df["temperature_grain"] = np.nan

    st.session_state.setdefault("forecasts", {})[model_name] = {
        "future_df": future_df,
        "future_horizon": horizon,
    }
    return True
# --------------------------------------------------


if __name__ == "__main__":
    main() 