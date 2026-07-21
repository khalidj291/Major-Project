"""
Memory That Fades — Evaluation Script
Person 3

Usage:
    python evaluate_models.py models/model_baseline.pkl
    python evaluate_models.py models/model_decay_fast.pkl

Loads a model, loads the processed dataset, builds the same 30-day
rolling-window features used at training time, evaluates on the
2023-2024 test window, and appends one row of results to
results/evaluation_metrics.csv.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd

DATA_PATH = "data/data_processed.csv"
RESULTS_PATH = "results/evaluation_metrics.csv"
WINDOW = 30  # 30 days of returns as features, per the data format contract
TEST_START = "2023-01-01"
DEFAULT_TICKER = "SPY"  # data_processed.csv holds multiple tickers stacked together


def load_data(path, ticker=DEFAULT_TICKER):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Waiting on Person 1's data_processed.csv "
            f"(columns: date, ticker, close, volume, returns, regime, rolling_vol)."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    if "ticker" in df.columns:
        if ticker not in df["ticker"].unique():
            raise ValueError(f"Ticker '{ticker}' not found. Available: {df['ticker'].unique().tolist()}")
        df = df[df["ticker"] == ticker].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_features(df, window=WINDOW):
    """Turn a returns series into (n_samples, window) feature rows and
    (n_samples, 1) next-day-return targets, per the data format contract."""
    returns = df["returns"].values
    dates = df["date"].values
    X, y, sample_dates = [], [], []
    for i in range(window, len(returns) - 1):
        X.append(returns[i - window:i])
        y.append(returns[i + 1])
        sample_dates.append(dates[i + 1])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    sample_dates = pd.to_datetime(sample_dates)
    return X, y, sample_dates


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(model_path, ticker=DEFAULT_TICKER):
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    df = load_data(DATA_PATH, ticker=ticker)
    X, y, sample_dates = build_features(df)

    # restrict to test window
    mask = sample_dates >= pd.Timestamp(TEST_START)
    X_test, y_test = X[mask], y[mask]

    if len(X_test) == 0:
        raise ValueError("No samples found in the test window — check date ranges.")

    model = load_model(model_path)
    preds = model.predict(X_test).reshape(-1, 1)

    errors = np.abs(preds - y_test)
    mae = errors.mean()
    rmse = np.sqrt(((preds - y_test) ** 2).mean())

    # "within 10%" = prediction within 10% of the actual value's magnitude
    denom = np.where(np.abs(y_test) < 1e-8, 1e-8, np.abs(y_test))
    within_10pct = (errors / denom <= 0.10).mean() * 100

    row = pd.DataFrame([{
        "model": model_name,
        "ticker": ticker,
        "n_test_samples": len(X_test),
        "mae": round(float(mae), 6),
        "rmse": round(float(rmse), 6),
        "pct_within_10pct": round(float(within_10pct), 2),
    }])

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(RESULTS_PATH, mode="w", header=True, index=False)

    print(f"\n=== {model_name} ===")
    print(f"Test samples : {len(X_test)}")
    print(f"MAE          : {mae:.6f}")
    print(f"RMSE         : {rmse:.6f}")
    print(f"Within 10%   : {within_10pct:.2f}%")
    print(f"Appended to {RESULTS_PATH}\n")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python evaluate_models.py <path_to_model.pkl> [ticker]")
        sys.exit(1)
    ticker_arg = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_TICKER
    evaluate(sys.argv[1], ticker=ticker_arg)
