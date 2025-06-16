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

from sklearn.metrics import r2_score

st.set_page_config(page_title="GranaryPredict Dashboard", layout="wide")

st.title("🌾 GranaryPredict – Temperature Forecasting")

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
def load_trained_model(path: Optional[str] = None):
    path = path or (MODELS_DIR / "rf_model.joblib")
    if pathlib.Path(path).exists():
        return model_utils.load_model(path)
    st.warning("No trained model found. Please train one first.")
    return None


def plot_3d_grid(df: pd.DataFrame):
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
        y=df["grid_y"],
        z=df["grid_z"],
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
            zaxis=dict(dtick=1, title="grid_z"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
    )
    st.plotly_chart(fig, use_container_width=True)


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


def list_saved_models() -> list[str]:
    return [p.name for p in MODELS_DIR.glob("*.joblib")]


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


def main():
    st.session_state.setdefault("evaluations", {})

    st.sidebar.header("1️⃣ Upload sensor CSV")
    uploaded_file = st.sidebar.file_uploader("Sensor CSV", type=["csv"])

    if uploaded_file:
        df = load_uploaded_file(uploaded_file)
        with st.expander("Raw Data", expanded=False):
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            st.markdown("### Full Data")
            st.dataframe(df, use_container_width=True)

        # Cleaning
        df = cleaning.basic_clean(df)
        df = cleaning.fill_missing(df)

        # Feature engineering
        df = features.create_time_features(df)
        df = features.create_spatial_features(df)
        df_train, df_eval = split_train_eval(df, horizon=5)
        X_train, y_train = features.select_feature_target(df_train)
        X_eval, y_eval = features.select_feature_target(df_eval)

        # Alerts
        if "predicted_temp" in df_eval.columns:
            alerts = df_eval[df_eval["predicted_temp"] >= ALERT_TEMP_THRESHOLD]
            if not alerts.empty:
                st.error(f"⚠️ {len(alerts)} positions predicted above {ALERT_TEMP_THRESHOLD}°C")

        # Summary per forecast day
        if "forecast_day" in df_eval.columns and "predicted_temp" in df_eval.columns:
            st.subheader("Forecast Summary (per day)")
            st.dataframe(forecast_summary(df_eval), use_container_width=True)

            # Overall confidence across evaluation horizon
            if {"temperature_grain", "predicted_temp"}.issubset(df_eval.columns):
                if len(df_eval) > 1:
                    overall_r2 = r2_score(df_eval["temperature_grain"], df_eval["predicted_temp"])
                    overall_conf = max(0, min(100, overall_r2 * 100))
                    st.metric("Overall Confidence (%)", f"{overall_conf:.0f}")

        if "predicted_temp" in df_eval.columns:
            st.subheader("Predictions")
            with st.expander("Show full prediction table"):
                st.dataframe(df_eval, use_container_width=True)

            st.subheader("3D Grid Temperatures")
            plot_3d_grid(df_eval)

            st.subheader("Time Series Comparison")
            plot_time_series(df_predplot_all, key="time_main")

        if {"temperature_grain", "predicted_temp"}.issubset(df_eval.columns):
            mae = (df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs().mean()
            rmse = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
            r2 = r2_score(df_eval["temperature_grain"], df_eval["predicted_temp"])
            conf = max(0, min(100, r2 * 100))
            st.metric("Confidence (%)", f"{conf:.0f}")
            st.caption(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    st.sidebar.header("2️⃣ Model selection & training")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        options=["RandomForest", "HistGradientBoosting", "LightGBM"],
        index=0,
    )
    example_btn = st.sidebar.button("Train model on uploaded CSV")
    if example_btn and uploaded_file:
        df = load_uploaded_file(uploaded_file)
        df = cleaning.basic_clean(df)
        df = cleaning.fill_missing(df)
        df = features.create_time_features(df)
        df = features.create_spatial_features(df)
        df_train, df_eval = split_train_eval(df, horizon=5)
        X_train, y_train = features.select_feature_target(df_train)
        X_eval, y_eval = features.select_feature_target(df_eval)
        if model_choice == "RandomForest":
            mdl, metrics = model_utils.train_random_forest(X_train, y_train)
            model_name = "rf_model.joblib"
        elif model_choice == "HistGradientBoosting":
            mdl, metrics = model_utils.train_gb_models(X_train, y_train, model_type="hist")
            model_name = "hgb_model.joblib"
        else:
            mdl, metrics = model_utils.train_lightgbm(X_train, y_train)
            model_name = "lgbm_model.joblib"
        model_utils.save_model(mdl, name=model_name)
        st.sidebar.success(
            f"{model_choice} trained! MAE: {metrics.get('mae', metrics.get('mae_cv')):.2f}, "
            f"RMSE: {metrics.get('rmse', metrics.get('rmse_cv')):.2f}"
        )

    # Existing model evaluation
    st.sidebar.header("3️⃣ Evaluate existing model")
    model_files = list_saved_models()
    if not model_files:
        st.sidebar.info("No saved models in models/ directory yet.")
    else:
        selected_model = st.sidebar.selectbox("Choose model file", model_files)
        eval_btn = st.sidebar.button("Evaluate selected model", key="eval")
        if eval_btn:
            if uploaded_file is None:
                st.warning("Please upload a CSV first to evaluate.")
            else:
                df = load_uploaded_file(uploaded_file)
                df = cleaning.basic_clean(df)
                df = cleaning.fill_missing(df)
                df = features.create_time_features(df)
                df = features.create_spatial_features(df)
                df_train, df_eval = split_train_eval(df, horizon=5)
                X_train, y_train = features.select_feature_target(df_train)
                X_eval, y_eval = features.select_feature_target(df_eval)

                mdl = load_trained_model(MODELS_DIR / selected_model)
                if mdl:
                    preds = model_utils.predict(mdl, X_eval)
                    df_eval["predicted_temp"] = preds
                    df_predplot_all = pd.concat([df_eval, df_train], ignore_index=True)
                    # store all data; day selection will happen in tab
                    df_predplot = df_predplot_all

                    # Compute metrics if ground truth available
                    if "temperature_grain" in df.columns:
                        mae = (df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs().mean()
                        rmse = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
                        st.caption(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
                    else:
                        st.info("Ground truth temperature_grain not found; showing predictions only.")

                    # Store results in session_state and acknowledge
                    st.session_state["evaluations"][selected_model] = {
                        "df_eval": df_eval,
                        "df_predplot_all": df_predplot_all,
                    }
                    st.sidebar.success(f"Evaluation stored for {selected_model}.")
                else:
                    st.error("Could not load the selected model file.")

    # Render evaluation tabs if any
    if st.session_state["evaluations"]:
        tab_labels = list(st.session_state["evaluations"].keys())
        tabs = st.tabs(tab_labels)
        for tab_label, tab in zip(tab_labels, tabs):
            with tab:
                res = st.session_state["evaluations"][tab_label]
                df_eval = res["df_eval"]
                df_predplot_all = res["df_predplot_all"]
                day_choice = st.selectbox("Select forecast day", options=list(range(1,6)), key=f"day_{tab_label}")
                df_predplot = df_predplot_all[df_predplot_all.get("forecast_day",1) == day_choice]

                summary_tab, pred_tab, grid_tab, ts_tab = st.tabs(["Summary", "Predictions", "3D Grid", "Time Series"])

                with summary_tab:
                    if "predicted_temp" in df_eval.columns:
                        st.subheader("Forecast Summary (per day)")
                        st.dataframe(forecast_summary(df_eval), use_container_width=True)

                        if {"temperature_grain", "predicted_temp"}.issubset(df_eval.columns):
                            overall_r2 = r2_score(df_eval["temperature_grain"], df_eval["predicted_temp"])
                            cols = st.columns(2)
                            with cols[0]:
                                st.metric("Overall Confidence (%)", f"{max(0, min(100, overall_r2 * 100)):.0f}")
                            avg_pct_err = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs() / df_eval["temperature_grain"]).mean() * 100
                            with cols[1]:
                                st.metric("Overall Accuracy (%)", f"{max(0, 100 - avg_pct_err):.1f}")

                            # Alert message collapsible
                            with st.expander("Alerts", expanded=True):
                                if (df_eval["predicted_temp"] >= ALERT_TEMP_THRESHOLD).any():
                                    st.error("⚠️ High temperature forecast detected – monitor closely!")
                                else:
                                    st.success("No high-temperature alerts in forecast window")

                with pred_tab:
                    st.dataframe(df_predplot, use_container_width=True)

                with grid_tab:
                    plot_3d_grid(df_predplot)

                with ts_tab:
                    plot_time_series(df_predplot_all, key=f"time_{tab_label}")


if __name__ == "__main__":
    main() 