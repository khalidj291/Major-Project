"""
Person 1 - Day 4, Step 3: Baseline robustness test.
Tests the baseline model's MAE/RMSE on: volatile periods, stable periods,
the 2020 COVID crash, and the 2022 rate hike period.

IMPORTANT CAVEAT, read before citing these numbers anywhere:
- The baseline model was trained on windows targeting 2015-01-01 to
  2022-12-31. The COVID crash (Jan-Jun 2020) and the 2022 rate hike
  period (Jan-Dec 2022) both fall INSIDE that training window.
  Numbers for those two subsets are IN-SAMPLE fit quality, not
  held-out generalization -- they answer "does the model fit reasonably
  during turbulent periods it was trained on", not "does it predict well
  on turbulence it's never seen". Don't present these as test performance.
- Volatile/stable period numbers are reported BOTH across the full
  2015-2024 history (mostly in-sample) AND restricted to the genuine
  held-out 2023-2024 test period, so the true out-of-sample numbers are
  available and clearly separated from the in-sample ones.

Output: baseline_model/results/baseline_robustness.csv
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

WINDOW = 30
TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

def make_windows(df, window):
    """Same convention as everywhere else: X = returns[i-window:i], y = returns[i]."""
    df = df.sort_values("date").reset_index(drop=True)
    returns = df["returns"].values
    dates = df["date"].values
    regimes = df["regime"].values if "regime" in df.columns else np.array([None] * len(df))
    X, y, out_dates, out_regimes = [], [], [], []
    for i in range(window, len(returns)):
        X.append(returns[i - window:i])
        y.append(returns[i])
        out_dates.append(dates[i])
        out_regimes.append(regimes[i])
    return np.array(X), np.array(y), pd.to_datetime(out_dates), np.array(out_regimes)

def mae_rmse(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_subset(model, X, y, dates, mask, label):
    n = mask.sum()
    if n == 0:
        return {"subset": label, "n": 0, "MAE": None, "RMSE": None, "sample_type": None}
    preds = model.predict(X[mask])
    mae, rmse = mae_rmse(y[mask], preds)
    in_sample = dates[mask].max() <= pd.Timestamp(TRAIN_END)
    partially_in_sample = (dates[mask].min() <= pd.Timestamp(TRAIN_END)) and (dates[mask].max() > pd.Timestamp(TRAIN_END))
    sample_type = "in-sample" if in_sample else ("mixed" if partially_in_sample else "held-out (test)")
    return {"subset": label, "n": int(n), "MAE": round(mae, 6), "RMSE": round(rmse, 6), "sample_type": sample_type}

def run_robustness_test(processed_path=os.path.join(PROJECT_ROOT, "data", "data_processed.csv"),
                         model_path=os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline.pkl"),
                         ticker="SPY"):
    df = pd.read_csv(processed_path, parse_dates=["date"])
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if "regime" not in df.columns or df["regime"].isna().all():
        raise ValueError("No regime column found -- run add_regime_labels.py first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X, y, dates, regimes = make_windows(df, WINDOW)

    results = []
    # Full-history regime subsets (mostly in-sample -- see caveat above)
    results.append(evaluate_subset(model, X, y, dates, regimes == "volatile", "volatile (full history, 2015-2024)"))
    results.append(evaluate_subset(model, X, y, dates, regimes == "stable", "stable (full history, 2015-2024)"))

    # Same regime subsets, restricted to genuine held-out test period
    test_mask = dates >= pd.Timestamp(TEST_START)
    results.append(evaluate_subset(model, X, y, dates, test_mask & (regimes == "volatile"), "volatile (held-out test period only)"))
    results.append(evaluate_subset(model, X, y, dates, test_mask & (regimes == "stable"), "stable (held-out test period only)"))

    # Named historical events -- both in-sample (training window)
    covid_mask = (dates >= "2020-02-15") & (dates <= "2020-04-15")
    hike_mask = (dates >= "2022-01-01") & (dates <= "2022-12-31")
    results.append(evaluate_subset(model, X, y, dates, covid_mask, "2020 COVID crash (Feb 15 - Apr 15 2020)"))
    results.append(evaluate_subset(model, X, y, dates, hike_mask, "2022 rate hike period (Jan - Dec 2022)"))

    # Full held-out test set, for reference
    results.append(evaluate_subset(model, X, y, dates, test_mask, "full held-out test period (2023-2024)"))

    out_df = pd.DataFrame(results)
    print(out_df.to_string(index=False))

    os.makedirs(os.path.join(SCRIPT_DIR, "results"), exist_ok=True)
    out_path = os.path.join(SCRIPT_DIR, "results", "baseline_robustness.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return out_df

if __name__ == "__main__":
    run_robustness_test()