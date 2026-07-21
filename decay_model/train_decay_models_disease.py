import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import sys
sys.path.append(SCRIPT_DIR)
from windowing import make_windows  # shared across all decay_model scripts -- see windowing.py

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_disease.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

WINDOW = 30  # daily data, same as financial domain

# No ticker column -- data_disease.csv is single-series, so ticker=None.
# TRAIN_END matches train_baseline_disease.py -- see that file for why this
# domain doesn't use the 2015-2022 boundary the other domains use.
TRAIN_END = "2022-04-02"
train_df = df[(df["date"] >= df["date"].min()) & (df["date"] <= TRAIN_END)].reset_index(drop=True)

X_train, y_train, train_sample_dates = make_windows(
    df, df["date"].min(), pd.Timestamp(TRAIN_END), WINDOW
)
X_test, y_test, test_sample_dates = make_windows(
    df, pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1), df["date"].max(), WINDOW
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

weight_files = {
    "fast": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_fast_disease.npy"),
    "medium": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_medium_disease.npy"),
    "slow": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_slow_disease.npy"),
}

train_dates_full = train_df["date"].values
weights_aligned = {}
for name, path in weight_files.items():
    full_weights = np.load(path)
    assert len(full_weights) == len(train_dates_full), "weight array length mismatch"
    date_to_weight = dict(zip(pd.to_datetime(train_dates_full), full_weights))
    aligned = np.array([date_to_weight[pd.Timestamp(d)] for d in train_sample_dates])
    weights_aligned[name] = aligned

# --- Baseline: load Person 1/2's OFFICIAL disease baseline ---
OFFICIAL_BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline_disease.pkl")
PLACEHOLDER_BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline_disease_PLACEHOLDER.pkl")

if os.path.exists(OFFICIAL_BASELINE_PATH):
    with open(OFFICIAL_BASELINE_PATH, "rb") as f:
        baseline_model = pickle.load(f)
    print(f"Loaded OFFICIAL disease baseline: {OFFICIAL_BASELINE_PATH}")
else:
    print("WARNING: official model_baseline_disease.pkl not found. Run train_baseline_disease.py first.")
    print("Falling back to placeholder baseline -- numbers are NOT final until swapped.")
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_train, y_train)
    with open(PLACEHOLDER_BASELINE_PATH, "wb") as f:
        pickle.dump(baseline_model, f)

# --- Decay-weighted models ---
decay_models = {}
for name, S in [("fast", 30), ("medium", 180), ("slow", 365)]:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=weights_aligned[name])
    decay_models[name] = model
    with open(os.path.join(PROJECT_ROOT, "decay_model", "models", f"model_decay_{name}_disease.pkl"), "wb") as f:
        pickle.dump(model, f)

# --- Reality-check baseline: predict zero every time ---
naive_zero_preds = np.zeros_like(y_test)

results = {}
predictions = {"naive_zero": naive_zero_preds}
mae = mean_absolute_error(y_test, naive_zero_preds)
rmse = np.sqrt(mean_squared_error(y_test, naive_zero_preds))
results["naive_zero"] = {"MAE": mae, "RMSE": rmse}
print(f"{'naive_zero':>10} | MAE: {mae:.6f} | RMSE: {rmse:.6f}   <-- reality-check floor")

for name, model in [("baseline", baseline_model)] + list(decay_models.items()):
    try:
        preds = model.predict(X_test)
    except Exception as e:
        print(f"ERROR predicting with '{name}' model: {e}")
        continue
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"MAE": mae, "RMSE": rmse}
    beats_naive = "beats naive" if mae < results["naive_zero"]["MAE"] else "DOES NOT beat naive"
    print(f"{name:>10} | MAE: {mae:.6f} | RMSE: {rmse:.6f}   ({beats_naive})")

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label="Actual", color="black", linewidth=1.2, alpha=0.6)
colors = {"naive_zero": "#4361ee", "baseline": "#888888", "fast": "#e63946", "medium": "#f4a261", "slow": "#2a9d8f"}
for name in ["naive_zero", "baseline", "fast", "medium", "slow"]:
    if name in predictions:
        plt.plot(predictions[name].flatten(), label=name, color=colors[name], linewidth=1.0, alpha=0.8)
plt.xlabel("Test sample index (daily)")
plt.ylabel("Day-over-day change in 7-day-smoothed case count")
plt.title("Disease Domain (US COVID cases): Model Predictions vs Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "decay_model", "results", "disease_comparison.png"), dpi=150)
print("\nSaved results/disease_comparison.png")

import json
with open(os.path.join(PROJECT_ROOT, "decay_model", "results", "disease_mae_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/disease_mae_summary.json")