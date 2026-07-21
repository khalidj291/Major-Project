"""
Person 1/2 - Stretch domain, Step 2: Train the baseline model on disease case data.
Same 30-day window as the financial domain (this is daily data, unlike PCE).
TRAIN_END is NOT 2022-12-31 like the other domains -- see note below.
Output: models/model_baseline_disease.pkl, results/baseline_disease_predictions.png
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

# TEAM CONVENTION NOTE: every other domain uses train=2015-2022 / test=2023-2024.
# That boundary doesn't make sense here -- OWID's US case data only runs from
# 2020-01-22 to 2023-03-09 (see data_disease.csv). Using this domain's own
# natural split instead: TRAIN_END = 2022-04-02 (70% of available dates),
# TEST = everything after. This exact boundary matches the honest
# validation-based check already run and documented in the project log --
# see statistical_significance_disease.py for how that validation worked.
TRAIN_END = "2022-04-02"

def make_windowed_features(returns: pd.Series, window=WINDOW):
    values = returns.values
    X, y, idx = [], [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
        idx.append(i)
    return np.array(X), np.array(y), idx

def train_disease_baseline(processed_path=os.path.join(PROJECT_ROOT, "data", "data_disease.csv")):
    df = pd.read_csv(processed_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X, y, idx = make_windowed_features(df["returns"])
    dates = df["date"].values[idx]

    train_mask = dates <= np.datetime64(TRAIN_END)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    test_dates = dates[~train_mask]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

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
    if mae > naive_mae:
        print("NOTE: baseline does NOT beat naive here -- expected and documented, see project log.")

    with open(os.path.join(SCRIPT_DIR, "models", "model_baseline_disease.pkl"), "wb") as f:
        pickle.dump(model, f)

    pd.DataFrame([{
        "domain": "disease", "model": "baseline_ridge", "window": WINDOW,
        "train_period": f"2020-01-22 to {TRAIN_END}",
        "test_period": f"{test_dates.min()} to {test_dates.max()}",
        "MAE": mae, "RMSE": rmse, "naive_MAE": naive_mae, "naive_RMSE": naive_rmse,
    }]).to_csv(os.path.join(SCRIPT_DIR, "results", "baseline_disease_results.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(test_dates, y_test, label="Actual", linewidth=1.2, alpha=0.8)
    plt.plot(test_dates, preds, label="Ridge prediction", linewidth=1.2, alpha=0.8)
    plt.plot(test_dates, naive_preds, label="Naive (0)", linestyle="--")
    plt.title("Baseline Ridge Regression -- US COVID case data (day-over-day % change)")
    plt.xlabel("Date"); plt.ylabel("Daily change in 7-day-smoothed cases")
    plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "baseline_disease_predictions.png"), dpi=120)
    print("\nSaved: models/model_baseline_disease.pkl, results/baseline_disease_predictions.png")

if __name__ == "__main__":
    train_disease_baseline()