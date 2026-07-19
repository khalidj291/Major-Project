"""
Person 1 - Day 2, Step 2: Train the baseline model on consumer spending data.
Uses a 12-period (1-year) window instead of 30, since monthly data only has
~119 rows total -- a 30-period window would leave too few rows to train on.
Output: models/model_baseline_consumer.pkl, results/baseline_consumer_predictions.png
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

WINDOW = 12          # 12 months, not 30 -- see note above
TRAIN_END = "2022-12-31"

def make_windowed_features(returns: pd.Series, window=WINDOW):
    values = returns.values
    X, y, idx = [], [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
        idx.append(i)
    return np.array(X), np.array(y), idx

def train_consumer_baseline(processed_path=os.path.join(PROJECT_ROOT, "data", "data_consumer.csv")):
    df = pd.read_csv(processed_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X, y, idx = make_windowed_features(df["returns"])
    dates = df["date"].values[idx]

    train_mask = dates <= np.datetime64(TRAIN_END)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    test_dates = dates[~train_mask]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    if len(X_test) < 10:
        print("WARNING: very few test rows -- results will be noisy, mention this in your report.")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    naive_preds = np.zeros_like(y_test)
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_preds))

    print(f"\nRidge Regression  -> MAE: {mae:.6f}  RMSE: {rmse:.6f}")
    print(f"Naive (predict 0) -> MAE: {naive_mae:.6f}  RMSE: {naive_rmse:.6f}")

    with open(os.path.join(SCRIPT_DIR, "models", "model_baseline_consumer.pkl"), "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame([{
        "domain": "consumer", "model": "baseline_ridge", "window": WINDOW,
        "train_period": f"2015 to {TRAIN_END}",
        "test_period": f"{test_dates.min()} to {test_dates.max()}",
        "MAE": mae, "RMSE": rmse, "naive_MAE": naive_mae, "naive_RMSE": naive_rmse,
    }]).to_csv(os.path.join(SCRIPT_DIR, "results", "baseline_consumer_results.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test, label="Actual", marker="o", linewidth=1.5)
    plt.plot(test_dates, preds, label="Ridge prediction", marker="o", linewidth=1.2)
    plt.plot(test_dates, naive_preds, label="Naive (0)", linestyle="--")
    plt.title("Baseline Ridge Regression -- Consumer Spending (PCE) monthly return")
    plt.xlabel("Date"); plt.ylabel("Monthly return")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "baseline_consumer_predictions.png"), dpi=120)
    print("\nSaved: models/model_baseline_consumer.pkl, results/baseline_consumer_predictions.png")

if __name__ == "__main__":
    train_consumer_baseline()