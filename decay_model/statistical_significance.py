import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy import stats
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from ebbinghaus import ebbinghaus_weight
from windowing import make_windows  # shared across all decay_model scripts -- see windowing.py

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
WINDOW = 30

# data_processed.csv is multi-ticker (AAPL/BTC-USD/SPY) by design --
# ticker="SPY" filters to just this asset before windowing.
train_df = df[(df["ticker"] == "SPY") & (df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
reference_date = train_df["date"].max()

X_train, y_train, train_sample_dates = make_windows(df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW, ticker="SPY")
X_test, y_test, _ = make_windows(df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW, ticker="SPY")

# Best decay model from the sensitivity test
BEST_S = 730
weights = ebbinghaus_weight(train_sample_dates, reference_date, BEST_S)
best_decay_model = Ridge(alpha=1.0)
best_decay_model.fit(X_train, y_train, sample_weight=weights)
decay_preds = best_decay_model.predict(X_test)

# Official baseline
with open(os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_preds = baseline_model.predict(X_test)

# Per-sample absolute errors (this is what gets compared, not the averaged MAE)
decay_errors = np.abs(y_test.flatten() - decay_preds.flatten())
baseline_errors = np.abs(y_test.flatten() - baseline_preds.flatten())

print(f"Best decay model: S={BEST_S}")
print(f"Decay MAE:    {decay_errors.mean():.6f}")
print(f"Baseline MAE: {baseline_errors.mean():.6f}")
print(f"Difference:   {baseline_errors.mean() - decay_errors.mean():.6f}")
print()

# Paired t-test: null hypothesis = no difference in mean error between the two models
t_stat, p_value = stats.ttest_rel(baseline_errors, decay_errors)

print(f"Paired t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print()

alpha = 0.05
if p_value < alpha:
    conclusion = f"SIGNIFICANT at 95% confidence (p={p_value:.4f} < {alpha}). The improvement is unlikely to be due to random chance."
else:
    conclusion = f"NOT significant at 95% confidence (p={p_value:.4f} >= {alpha}). Cannot rule out that the improvement is due to random chance."
print(conclusion)

with open(os.path.join(PROJECT_ROOT, "decay_model", "results", "statistical_significance.txt"), "w") as f:
    f.write(f"Paired t-test: best decay model (S={BEST_S}) vs official baseline\n")
    f.write(f"Financial domain (SPY), test period 2023-01-01 to 2024-12-31, n={len(y_test)}\n\n")
    f.write(f"Decay MAE: {decay_errors.mean():.6f}\n")
    f.write(f"Baseline MAE: {baseline_errors.mean():.6f}\n")
    f.write(f"Mean error difference: {baseline_errors.mean() - decay_errors.mean():.6f}\n\n")
    f.write(f"t-statistic: {t_stat:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n\n")
    f.write(conclusion + "\n")

print("\nSaved results/statistical_significance.txt")