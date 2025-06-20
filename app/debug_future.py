# debug_future.py  – run with  `python debug_future.py`
import joblib, pandas as pd
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.Dashboard import load_uploaded_file, make_future         # reuse helpers
from app.Dashboard import features, cleaning, split_train_eval    # idem
from app.Dashboard import compute_overall_metrics
from granarypredict.model import predict, load_model

CSV          = Path("data/preloaded/Data1.csv")           # or your own file
MODEL_FILE   = Path("models/data1_fs_lgbm_500.joblib")     # <-- change
HORIZON_DAYS = 7

# ---------- 1. reproduce the cleaned + engineered dataframe ----------
df = load_uploaded_file(CSV)
df = cleaning.basic_clean(df)
df = cleaning.fill_missing(df)
df = features.create_time_features(df)
df = features.create_spatial_features(df)

# last 5 days used in the UI for evaluation
df_train, df_eval = split_train_eval(df, horizon=5)

# ---------- 2. grab the category map exactly as Dashboard stores it ----------
cat_cols = df_train.select_dtypes(include=["object", "category"]).columns
categories_map = {c: pd.Categorical(df_train[c]).categories.tolist()
                  for c in cat_cols}

# ---------- 3. build the future frame exactly like the UI ----------
future_df = make_future(df_train, horizon_days=HORIZON_DAYS)
for col, cats in categories_map.items():
    if col in future_df.columns:
        future_df[col] = pd.Categorical(future_df[col], categories=cats)

# ---------- 4. prepare feature matrices ----------
X_eval,  _ = features.select_feature_target(df_eval)
X_future, _ = features.select_feature_target(future_df)

model = load_model(MODEL_FILE)
feat_cols = list(model.feature_name_)       # LightGBM / RF / HistGB

X_eval   = X_eval.reindex(columns=feat_cols,   fill_value=0)
X_future = X_future.reindex(columns=feat_cols, fill_value=0)

# ---------- 5a. evaluation predictions & metrics ----------
y_eval_pred = predict(model, X_eval)
df_eval["predicted_temp"] = y_eval_pred
mae_eval  = (df_eval["temperature_grain"] - df_eval["predicted_temp"]).abs().mean()
rmse_eval = ((df_eval["temperature_grain"] - df_eval["predicted_temp"]) ** 2).mean() ** 0.5
conf_eval, acc_eval = compute_overall_metrics(df_eval)
print("\nEvaluation metrics:")
print(f"MAE: {mae_eval:.2f}  RMSE: {rmse_eval:.2f}  Confidence: {conf_eval:.2f}%  Accuracy: {acc_eval:.2f}%")

# ---------- 5b. look at means / prediction comparison ----------
import numpy as np
print("\nColumn-wise mean difference (|Eval - Future|):")
delta = (X_eval.mean() - X_future.mean()).abs().sort_values(ascending=False)
print(delta.head(20))

y_future = predict(model, X_future)
print("\nFuture prediction stats:")
print("min / mean / max :", np.min(y_future), np.mean(y_future), np.max(y_future))
