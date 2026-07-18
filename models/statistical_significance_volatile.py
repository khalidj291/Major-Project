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

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_regime.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
WINDOW = 30


def make_windows(full_df, start_date, end_date, window):
    full_df = full_df.sort_values("date").reset_index(drop=True)
    returns = full_df["returns"].values
    dates = full_df["date"].values
    regimes = full_df["regime"].values
    X, y, sample_dates, sample_regimes = [], [], [], []
    for i in range(window, len(returns) - 1):
        target_date = dates[i + 1]
        if start_date <= pd.Timestamp(target_date) <= end_date:
            X.append(returns[i - window:i])
            y.append(returns[i + 1])
            sample_dates.append(dates[i])
            sample_regimes.append(regimes[i + 1])
    return np.array(X), np.array(y).reshape(-1, 1), np.array(sample_dates), np.array(sample_regimes)


train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
reference_date = train_df["date"].max()

X_train, y_train, train_sample_dates, _ = make_windows(df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW)
X_test, y_test, test_sample_dates, test_regimes = make_windows(df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW)

# Isolate the volatile subset only
volatile_mask = test_regimes == "volatile"
X_test_vol = X_test[volatile_mask]
y_test_vol = y_test[volatile_mask]
n_volatile = volatile_mask.sum()
print(f"Volatile subset size: {n_volatile}")

# Fast decay model (the volatile-period winner)
weights = ebbinghaus_weight(train_sample_dates, reference_date, 30)
fast_model = Ridge(alpha=1.0)
fast_model.fit(X_train, y_train, sample_weight=weights)
fast_preds_vol = fast_model.predict(X_test_vol)

# Official baseline
with open(os.path.join(SCRIPT_DIR, "model_baseline.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_preds_vol = baseline_model.predict(X_test_vol)

fast_errors = np.abs(y_test_vol.flatten() - fast_preds_vol.flatten())
baseline_errors = np.abs(y_test_vol.flatten() - baseline_preds_vol.flatten())

print(f"\nFast decay MAE (volatile only): {fast_errors.mean():.6f}")
print(f"Baseline MAE (volatile only):   {baseline_errors.mean():.6f}")
print(f"Difference: {baseline_errors.mean() - fast_errors.mean():.6f}")

t_stat, p_value = stats.ttest_rel(baseline_errors, fast_errors)
print(f"\nPaired t-test (fast vs baseline, VOLATILE subset only, n={n_volatile}):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    conclusion = (f"SIGNIFICANT at 95% confidence (p={p_value:.4f} < {alpha}). "
                   f"Fast decay's advantage over baseline during volatile periods is unlikely to be random chance.")
else:
    conclusion = (f"NOT significant at 95% confidence (p={p_value:.4f} >= {alpha}). "
                   f"With only n={n_volatile} volatile samples, cannot rule out that fast decay's apparent "
                   f"advantage is due to random chance rather than a real effect.")
print(f"\n{conclusion}")

with open(os.path.join(PROJECT_ROOT, "results", "statistical_significance_volatile.txt"), "w") as f:
    f.write(f"Paired t-test: fast decay (S=30) vs official baseline, VOLATILE PERIODS ONLY\n")
    f.write(f"Financial domain (SPY), test period 2023-01-01 to 2024-12-31, n={n_volatile}\n\n")
    f.write(f"Fast decay MAE: {fast_errors.mean():.6f}\n")
    f.write(f"Baseline MAE: {baseline_errors.mean():.6f}\n")
    f.write(f"Mean error difference: {baseline_errors.mean() - fast_errors.mean():.6f}\n\n")
    f.write(f"t-statistic: {t_stat:.4f}\n")
    f.write(f"p-value: {p_value:.4f}\n\n")
    f.write(conclusion + "\n")
print("\nSaved results/statistical_significance_volatile.txt")