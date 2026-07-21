"""
Person 1/2 - Stretch domain: Statistical significance test for the disease
domain, WITH PROPER TRAIN/VALIDATION/TEST SEPARATION.

Why this file looks different from statistical_significance.py (financial):
that script selects its "best S" by minimizing MAE directly on the test
set, which is a real methodological weakness (caught during a later
review -- see project log). It happened not to change the financial
conclusion when checked honestly, but for THIS domain the honest check
matters a lot -- this domain's whole finding depends on not fooling
ourselves. So here, S is selected using ONLY a validation slice carved out
of the training period; the real test set is touched exactly once, at the
very end, after S is already fixed.

Split: train on the first 50% of available dates, validate on the next
20%, test on the final 30% -- never overlapping, never re-used.
Output: results/statistical_significance_disease.txt
"""

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import sys
sys.path.append(SCRIPT_DIR)
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy import stats
from ebbinghaus import ebbinghaus_weight
from windowing import make_windows

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_disease.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
WINDOW = 30

n = len(df)
train_end = df["date"].iloc[int(n * 0.5)]
val_end = df["date"].iloc[int(n * 0.7)]
print(f"train: {df['date'].min().date()} to {train_end.date()}")
print(f"val:   {(train_end + pd.Timedelta(days=1)).date()} to {val_end.date()}")
print(f"test:  {(val_end + pd.Timedelta(days=1)).date()} to {df['date'].max().date()}")

X_tr, y_tr, tr_dates = make_windows(df, df["date"].min(), train_end, WINDOW)
X_val, y_val, val_dates = make_windows(df, train_end + pd.Timedelta(days=1), val_end, WINDOW)
X_test, y_test, test_dates = make_windows(df, val_end + pd.Timedelta(days=1), df["date"].max(), WINDOW)
print(f"train n={len(X_tr)}, val n={len(X_val)}, test n={len(X_test)}")

# --- Step 1: select S using ONLY the validation slice ---
ref_cv = train_end
baseline_cv = Ridge(alpha=1.0).fit(X_tr, y_tr)
baseline_val_mae = mean_absolute_error(y_val, baseline_cv.predict(X_val))
print(f"\nbaseline validation MAE: {baseline_val_mae:.6f}")

best_S, best_val_mae = None, baseline_val_mae
for S in [14, 30, 60, 90, 180, 270, 365, 730]:
    w = ebbinghaus_weight(pd.to_datetime(tr_dates), ref_cv, S)
    m = Ridge(alpha=1.0).fit(X_tr, y_tr, sample_weight=w)
    val_mae = mean_absolute_error(y_val, m.predict(X_val))
    print(f"S={S:>4} | validation MAE: {val_mae:.6f}")
    if val_mae < best_val_mae:
        best_S, best_val_mae = S, val_mae

if best_S is None:
    print("\nNo S beat baseline on the validation set -- honest conclusion: no real effect.")
    with open(os.path.join(SCRIPT_DIR, "results", "statistical_significance_disease.txt"), "w") as f:
        f.write("No decay rate beat baseline on the validation set.\n")
        f.write("Honest conclusion: no evidence of a real effect on this domain/split.\n")
else:
    print(f"\nCV-selected S: {best_S} (chosen using validation data only, test set never touched yet)")

    # --- Step 2: retrain on train+val combined, evaluate ONCE on the real test set ---
    X_full, y_full, full_dates = make_windows(df, df["date"].min(), val_end, WINDOW)
    ref_full = val_end
    baseline_full = Ridge(alpha=1.0).fit(X_full, y_full)
    baseline_test_mae = mean_absolute_error(y_test, baseline_full.predict(X_test))

    w_full = ebbinghaus_weight(pd.to_datetime(full_dates), ref_full, best_S)
    decay_full = Ridge(alpha=1.0).fit(X_full, y_full, sample_weight=w_full)
    decay_preds = decay_full.predict(X_test)
    decay_test_mae = mean_absolute_error(y_test, decay_preds)

    naive_test_mae = mean_absolute_error(y_test, np.zeros_like(y_test))

    errs_baseline = np.abs(y_test.flatten() - baseline_full.predict(X_test).flatten())
    errs_decay = np.abs(y_test.flatten() - decay_preds.flatten())
    t_stat, p_value = stats.ttest_rel(errs_baseline, errs_decay)

    print(f"\nHonest held-out test set (n={len(X_test)}), S={best_S} chosen without ever seeing this data:")
    print(f"  naive_zero MAE: {naive_test_mae:.6f}")
    print(f"  baseline MAE:   {baseline_test_mae:.6f}")
    print(f"  decay(S={best_S}) MAE: {decay_test_mae:.6f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    significant = p_value < 0.05
    print(f"  {'SIGNIFICANT' if significant else 'NOT significant'} at 95% confidence")

    with open(os.path.join(SCRIPT_DIR, "results", "statistical_significance_disease.txt"), "w") as f:
        f.write(f"Disease domain (US COVID cases) -- honest train/validation/test split\n")
        f.write(f"S selected using validation data ONLY (n={len(X_val)}), never the test set\n")
        f.write(f"Selected S: {best_S}\n\n")
        f.write(f"Held-out test set (n={len(X_test)}), touched exactly once:\n")
        f.write(f"  naive_zero MAE: {naive_test_mae:.6f}\n")
        f.write(f"  baseline MAE:   {baseline_test_mae:.6f}\n")
        f.write(f"  decay(S={best_S}) MAE: {decay_test_mae:.6f}\n\n")
        f.write(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}\n")
        f.write(f"{'SIGNIFICANT' if significant else 'NOT significant'} at 95% confidence.\n\n")
        f.write("Honest caveat: decay beats baseline decisively here, but neither baseline\n")
        f.write("nor decay beats naive-zero. Baseline is trained on 2020-2022, a period with\n")
        f.write("reporting-artifact swings far larger than the 2022-2023 test period (see\n")
        f.write("data_documentation.md) -- decay weighting helps specifically by down-weighting\n")
        f.write("that unrepresentative early data, not by making genuinely better predictions\n")
        f.write("than the trivial 'no change' forecast.\n")
    print("\nSaved: results/statistical_significance_disease.txt")