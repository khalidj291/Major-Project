import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), parse_dates=["date"])

# TEAM CONVENTION: data_processed.csv may contain multiple tickers (AAPL,
# BTC-USD, SPY) mixed together. This script is the financial-domain (SPY)
# pipeline, so filter to SPY before anything else touches the data. Without
# this, windowing would build input sequences from interleaved returns
# across different assets.
if "ticker" in df.columns:
    before = len(df)
    df = df[df["ticker"] == "SPY"].reset_index(drop=True)
    print(f"Filtered to ticker=='SPY': {before} rows -> {len(df)} rows")

df = df.sort_values("date").reset_index(drop=True)

WINDOW = 30

def make_windows(full_df, start_date, end_date, window):
    """
    Build windows where the TARGET (y) date falls within [start_date, end_date],
    but the input window can reach back into earlier rows (e.g. into the training
    period, for the first few test-period targets). This matches the convention
    used in the consumer-domain script, and avoids wasting valid test samples
    just because they're near the start of the test period.
    """
    full_df = full_df.sort_values("date").reset_index(drop=True)
    returns = full_df["returns"].values
    dates = full_df["date"].values
    X, y, sample_dates = [], [], []
    for i in range(window, len(returns) - 1):
        target_date = dates[i + 1]
        if start_date <= pd.Timestamp(target_date) <= end_date:
            X.append(returns[i - window:i])
            y.append(returns[i + 1])
            sample_dates.append(dates[i])
    return np.array(X), np.array(y).reshape(-1, 1), np.array(sample_dates)

train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)

X_train, y_train, train_sample_dates = make_windows(
    df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW
)
X_test, y_test, test_sample_dates = make_windows(
    df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Load weight arrays and align them to the windowed samples
# (weights were computed per training-row date; windowing drops the first WINDOW rows)
weight_files = {
    "fast": os.path.join(SCRIPT_DIR, "weights_fast.npy"),
    "medium": os.path.join(SCRIPT_DIR, "weights_medium.npy"),
    "slow": os.path.join(SCRIPT_DIR, "weights_slow.npy"),
}

train_dates_full = train_df["date"].values
weights_aligned = {}
for name, path in weight_files.items():
    full_weights = np.load(path)
    assert len(full_weights) == len(train_dates_full), "weight array length mismatch"
    # sample i in X_train corresponds to train_df row index i+WINDOW (the "current" day used to predict i+WINDOW+1)
    aligned = full_weights[WINDOW: WINDOW + len(X_train)]
    weights_aligned[name] = aligned

# --- Baseline: load Person 1's OFFICIAL model, not a placeholder ---
# This must be the same model object Person 1 trained and validated, so that
# comparisons are apples-to-apples across the whole team.
import os

OFFICIAL_BASELINE_PATH = os.path.join(SCRIPT_DIR, "model_baseline.pkl")
PLACEHOLDER_BASELINE_PATH = os.path.join(SCRIPT_DIR, "model_baseline_PLACEHOLDER.pkl")

if os.path.exists(OFFICIAL_BASELINE_PATH):
    with open(OFFICIAL_BASELINE_PATH, "rb") as f:
        baseline_model = pickle.load(f)
    print(f"Loaded OFFICIAL baseline from Person 1: {OFFICIAL_BASELINE_PATH}")
else:
    print("WARNING: Person 1's official model_baseline.pkl not found.")
    print(f"Falling back to placeholder baseline (train/test split and window logic")
    print("may not exactly match Person 1's -- numbers are NOT final until swapped.")
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
    with open(os.path.join(SCRIPT_DIR, f"model_decay_{name}.pkl"), "wb") as f:
        pickle.dump(model, f)

# --- Reality-check baseline: predict zero change every day ---
# This is the sanity floor. If a "real" model can't beat this, its results are
# likely noise, not signal -- daily stock returns are close to a random walk,
# so "predict nothing changes" is a surprisingly tough benchmark to beat.
naive_zero_preds = np.zeros_like(y_test)

# --- Evaluate naive baseline + all four models on test set ---
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
        print("This usually means input shape or feature construction doesn't match")
        print("what that model was trained on -- confirm window size (30) and")
        print("train/test split boundaries (2015-2022 / 2023-2024) match across the team.")
        continue
    predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"MAE": mae, "RMSE": rmse}
    beats_naive = "beats naive" if mae < results["naive_zero"]["MAE"] else "DOES NOT beat naive"
    print(f"{name:>10} | MAE: {mae:.6f} | RMSE: {rmse:.6f}   ({beats_naive})")

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten()[:150], label="Actual", color="black", linewidth=1.5, alpha=0.7)
colors = {"naive_zero": "#4361ee", "baseline": "#888888", "fast": "#e63946", "medium": "#f4a261", "slow": "#2a9d8f"}
for name in ["naive_zero", "baseline", "fast", "medium", "slow"]:
    plt.plot(predictions[name].flatten()[:150], label=name, color=colors[name], linewidth=1.2, alpha=0.8)
plt.xlabel("Test sample index (first 150 shown)")
plt.ylabel("Next-day return")
plt.title("Model Predictions vs Actual — Baseline vs Decay-Weighted Models")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "results", "four_model_comparison.png"), dpi=150)
print("\nSaved results/four_model_comparison.png")

import json
with open(os.path.join(PROJECT_ROOT, "results", "day1_mae_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Saved results/day1_mae_summary.json")