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

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_consumer.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

WINDOW = 12  # months -- NOT 30. Consumer data is monthly with only ~120 rows total;
             # a 30-period window would leave too few examples to train on.
             # Matches Person 1's baseline window so comparisons are apples-to-apples.

# No ticker column here -- data_consumer.csv is already single-series (PCE), so ticker=None.
train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)

X_train, y_train, train_sample_dates = make_windows(
    df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW
)
X_test, y_test, test_sample_dates = make_windows(
    df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

weight_files = {
    "fast": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_fast_consumer.npy"),
    "medium": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_medium_consumer.npy"),
    "slow": os.path.join(PROJECT_ROOT, "decay_model", "models", "weights_slow_consumer.npy"),
}

train_dates_full = train_df["date"].values
weights_aligned = {}
for name, path in weight_files.items():
    full_weights = np.load(path)
    assert len(full_weights) == len(train_dates_full), "weight array length mismatch"
    # Build a date -> weight lookup, then map each training sample's "current day"
    # date to its weight. Safer than positional slicing now that windows span
    # across the train/test boundary.
    date_to_weight = dict(zip(pd.to_datetime(train_dates_full), full_weights))
    aligned = np.array([date_to_weight[pd.Timestamp(d)] for d in train_sample_dates])
    weights_aligned[name] = aligned

# --- Baseline: load Person 1's OFFICIAL consumer baseline ---
OFFICIAL_BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline_consumer.pkl")
PLACEHOLDER_BASELINE_PATH = os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline_consumer_PLACEHOLDER.pkl")

if os.path.exists(OFFICIAL_BASELINE_PATH):
    with open(OFFICIAL_BASELINE_PATH, "rb") as f:
        baseline_model = pickle.load(f)
    print(f"Loaded OFFICIAL consumer baseline from Person 1: {OFFICIAL_BASELINE_PATH}")
else:
    print("WARNING: Person 1's official model_baseline_consumer.pkl not found.")
    print("Falling back to placeholder baseline -- numbers are NOT final until swapped.")
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_train, y_train)
    with open(PLACEHOLDER_BASELINE_PATH, "wb") as f:
        pickle.dump(baseline_model, f)

# --- Decay-weighted models ---
decay_models = {}
for name, S in [("fast", 30), ("medium", 365), ("slow", 730)]:
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=weights_aligned[name])
    decay_models[name] = model
    with open(os.path.join(PROJECT_ROOT, "decay_model", "models", f"model_decay_{name}_consumer.pkl"), "wb") as f:
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
        print("Check window size (12) and train/test split match across the team.")
        continue
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"MAE": mae, "RMSE": rmse}
    beats_naive = "beats naive" if mae < results["naive_zero"]["MAE"] else "DOES NOT beat naive"
    print(f"{name:>10} | MAE: {mae:.6f} | RMSE: {rmse:.6f}   ({beats_naive})")

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label="Actual", color="black", linewidth=1.5, alpha=0.7, marker="o", markersize=3)
colors = {"naive_zero": "#4361ee", "baseline": "#888888", "fast": "#e63946", "medium": "#f4a261", "slow": "#2a9d8f"}
for name in ["naive_zero", "baseline", "fast", "medium", "slow"]:
    if name in predictions:
        plt.plot(predictions[name].flatten(), label=name, color=colors[name], linewidth=1.2, alpha=0.8)
plt.xlabel("Test sample index (monthly)")
plt.ylabel("Next-month return")
plt.title("Consumer Domain: Model Predictions vs Actual")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "decay_model", "results", "consumer_comparison.png"), dpi=150)
print("\nSaved results/consumer_comparison.png")

import json
with open(os.path.join(PROJECT_ROOT, "decay_model", "results", "day2_consumer_mae_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/day2_consumer_mae_summary.json")