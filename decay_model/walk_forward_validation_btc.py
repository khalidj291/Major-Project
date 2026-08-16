"""
walk_forward_validation_btc.py -- tests whether decay weighting's advantage
over baseline holds up across MANY independent time periods, not just one
train/val/test split.

Why this exists: train_feature_decay_btc.py and train_combined_decay_btc.py
each validate their own hyperparameters honestly (never touching test data
to choose S or alpha) -- but the CHOICE of method itself (power-law vs
exponential, feature-decay vs sample-decay vs combined) was made by
comparing results across those earlier single-split experiments. That is a
subtler form of leakage: the method was picked with some knowledge of which
one "worked."

This script removes that leak entirely. For EVERY 6-month test window, it
selects the best method AND its hyperparameters using ONLY that window's own
preceding 6-month validation slice -- never fixing a method in advance, and
never letting any window's test data influence any choice, for that window
or any other. Training uses an expanding window (all data before that
window's validation period).

Candidate methods searched, per window, on validation only:
  - baseline (no decay)
  - exponential feature decay (ebbinghaus.py's shape, applied to lags)
  - power-law feature decay (powerlaw_decay.py)
  - sample decay alone (ebbinghaus.py, applied to rows)
  - power-law feature decay + sample decay combined

Output: decay_model/results/walk_forward_results_btc.csv
        one row per 6-month window, plus a pooled significance test printed
        across every test day from every window combined.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from powerlaw_decay import powerlaw_weights
from ebbinghaus import ebbinghaus_weight

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")

ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_FEAT_CANDIDATES = [30, 90, 180]      # exponential feature-decay S (lag scale)
S_SAMPLE_CANDIDATES = [180, 365, 730]  # sample-decay S (row/date scale)

def exp_feat_weights(window, S):
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")

results = []
all_baseline_errs, all_selected_errs = [], []

for ts in test_starts:
    test_start = ts
    test_end = min(ts + pd.DateOffset(months=6) - pd.DateOffset(days=1), df["date"].max())
    val_end = test_start - pd.DateOffset(days=1)
    val_start = val_end - pd.DateOffset(months=6) + pd.DateOffset(days=1)
    train_end = val_start - pd.DateOffset(days=1)
    train_start = df["date"].min()
    if test_start > df["date"].max():
        continue

    X_train, y_train, train_dates = make_windows(df, train_start, train_end, WINDOW)
    X_val, y_val, val_dates = make_windows(df, val_start, val_end, WINDOW)
    X_test, y_test, test_dates = make_windows(df, test_start, test_end, WINDOW)
    if len(X_train) < 200 or len(X_val) < 20 or len(X_test) < 5:
        continue

    reference_date = train_dates.max()
    candidates = []  # (name, val_mae, feat_weights_or_None, sample_weights_or_None)

    m = Ridge(alpha=1.0).fit(X_train, y_train)
    candidates.append(("baseline", np.abs(y_val - m.predict(X_val)).mean(), None, None))

    for alpha in ALPHA_CANDIDATES:
        w = powerlaw_weights(WINDOW, alpha)
        m = Ridge(alpha=1.0).fit(X_train * w, y_train)
        candidates.append((f"powerlaw_feat_a{alpha}", np.abs(y_val - m.predict(X_val * w)).mean(), w, None))
        for S in S_SAMPLE_CANDIDATES:
            sw = ebbinghaus_weight(train_dates, reference_date, S)
            m = Ridge(alpha=1.0).fit(X_train * w, y_train, sample_weight=sw)
            candidates.append((f"combined_a{alpha}_S{S}", np.abs(y_val - m.predict(X_val * w)).mean(), w, sw))

    for S in S_FEAT_CANDIDATES:
        w = exp_feat_weights(WINDOW, S)
        m = Ridge(alpha=1.0).fit(X_train * w, y_train)
        candidates.append((f"exp_feat_S{S}", np.abs(y_val - m.predict(X_val * w)).mean(), w, None))

    for S in S_SAMPLE_CANDIDATES:
        sw = ebbinghaus_weight(train_dates, reference_date, S)
        m = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=sw)
        candidates.append((f"sample_S{S}", np.abs(y_val - m.predict(X_val)).mean(), None, sw))

    name, val_mae, w_best, sw_best = min(candidates, key=lambda c: c[1])

    X_train_final = X_train * w_best if w_best is not None else X_train
    X_test_final = X_test * w_best if w_best is not None else X_test
    selected_model = Ridge(alpha=1.0).fit(X_train_final, y_train, sample_weight=sw_best)
    selected_pred = selected_model.predict(X_test_final)

    baseline_model = Ridge(alpha=1.0).fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)

    selected_err = np.abs(y_test.flatten() - selected_pred.flatten())
    baseline_err = np.abs(y_test.flatten() - baseline_pred.flatten())
    all_selected_errs.extend(selected_err.tolist())
    all_baseline_errs.extend(baseline_err.tolist())

    results.append({
        "test_window_start": str(test_start.date()),
        "test_window_end": str(test_end.date()),
        "n_test": len(X_test),
        "selected_method": name,
        "baseline_MAE": baseline_err.mean(),
        "selected_MAE": selected_err.mean(),
        "selected_wins": selected_err.mean() < baseline_err.mean(),
    })

res_df = pd.DataFrame(results)
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
res_df.to_csv(os.path.join(results_dir, "walk_forward_results_btc.csv"), index=False)

t_stat, p_val = stats.ttest_rel(all_selected_errs, all_baseline_errs)

print(res_df.to_string(index=False))
print(f"\nWin rate: {res_df['selected_wins'].sum()}/{len(res_df)} windows")
print(f"Pooled across {len(all_selected_errs)} test days from every window:")
print(f"  mean baseline error : {np.mean(all_baseline_errs):.6f}")
print(f"  mean selected error : {np.mean(all_selected_errs):.6f}")
print(f"  paired t-test        : t={t_stat:.4f}, p={p_val:.4f}")
print(f"\nSaved decay_model/results/walk_forward_results_btc.csv")