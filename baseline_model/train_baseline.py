"""
Person 1 - Day 1, Step 4: Train the baseline model on SPY.
Input:  data/data_processed.csv
Output: models/model_baseline.pkl, results/baseline_predictions.png, printed MAE/RMSE

Includes a naive "predict zero change" baseline alongside Ridge Regression.
This matters: stock returns are close to random noise, so if Ridge barely
beats "predict no change", that's a real finding worth reporting honestly —
not a bug in your code.
"""

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

WINDOW = 30
TRAIN_END = "2022-12-31"

def make_windowed_features(returns: pd.Series, window=WINDOW):
    """Row i's features = returns[i-window : i], label = returns[i]."""
    values = returns.values
    X, y, idx = [], [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
        idx.append(i)
    return np.array(X), np.array(y), idx

def train_baseline(processed_path=os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), ticker="SPY"):
    df = pd.read_csv(processed_path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    X, y, idx = make_windowed_features(df["returns"])
    dates = df["date"].values[idx]

    train_mask = dates <= np.datetime64(TRAIN_END)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    test_dates = dates[~train_mask]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Train range: {dates[train_mask].min()} to {dates[train_mask].max()}")
    print(f"Test range:  {test_dates.min()} to {test_dates.max()}")

    # --- Baseline model: Ridge Regression ---
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # --- Naive sanity check: "predict no change" ---
    naive_preds = np.zeros_like(y_test)
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))

    print(f"\nRidge Regression  -> MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    print(f"Naive (predict 0) -> MAE: {naive_mae:.6f}  RMSE: {naive_rmse:.6f}")
    if mae < naive_mae:
        improvement = (naive_mae - mae) / naive_mae * 100
        print(f"Ridge beats naive baseline by {improvement:.1f}% (MAE)")
    else:
        print("WARNING: Ridge does NOT beat the naive 'predict zero' baseline on MAE.")
        print("This is common with raw daily returns — report it honestly, and")
        print("consider it a benchmark decay-weighting needs to clear too.")

    # Save model
    with open(os.path.join(SCRIPT_DIR, "models", "model_baseline.pkl"), "wb") as f:
        pickle.dump(model, f)

    # Save results row for the team
    pd.DataFrame([{
        "domain": "financial", "model": "baseline_ridge", "ticker": ticker,
        "train_period": f"2015-01-01 to {TRAIN_END}",
        "test_period": f"{test_dates.min()} to {test_dates.max()}",
        "MAE": mae, "RMSE": rmse, "naive_MAE": naive_mae, "naive_RMSE": naive_rmse,
    }]).to_csv(os.path.join(SCRIPT_DIR, "results", "baseline_results.csv"), index=False)

    # Plot: actual vs Ridge vs naive, first 150 test days for readability
    n_show = 150
    plt.figure(figsize=(11, 5))
    plt.plot(test_dates[:n_show], y_test[:n_show], label="Actual", linewidth=1.5)
    plt.plot(test_dates[:n_show], preds[:n_show], label="Ridge prediction", linewidth=1.2)
    plt.plot(test_dates[:n_show], naive_preds[:n_show], label="Naive (0)", linestyle="--", linewidth=1)
    plt.title(f"Baseline Ridge Regression — {ticker} next-day return (test set, first {n_show} days)")
    plt.xlabel("Date"); plt.ylabel("Daily return")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "baseline_predictions.png"), dpi=120)
    print("\nSaved: models/model_baseline.pkl, results/baseline_predictions.png, results/baseline_results.csv")

    return model, mae, rmse

if __name__ == "__main__":
    train_baseline()