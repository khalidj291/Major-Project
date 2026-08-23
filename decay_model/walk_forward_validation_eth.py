"""
walk_forward_validation_eth.py -- same as before, with per-day paired error
logging added (see walk_forward_validation_btc.py docstring for why).
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
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_eth.csv")

ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_FEAT_CANDIDATES = [30, 90, 180]
S_SAMPLE_CANDIDATES = [180, 365, 730]

def exp_feat_weights(window, S):
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
assert "ticker" not in df.columns, "unexpected ticker column in data_eth.csv"

test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")

results = []
per_day_rows = []

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
    candidates = []

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

    for d, be, se in zip(test_dates, baseline_err, selected_err):
        per_day_rows.append({
            "date": pd.Timestamp(d).date(),
            "window_start": str(test_start.date()),
            "baseline_err": be,
            "selected_err": se,
        })

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
res_df.to_csv(os.path.join(results_dir, "walk_forward_results_eth.csv"), index=False)

per_day_df = pd.DataFrame(per_day_rows)
per_day_df.to_csv(os.path.join(results_dir, "walk_forward_per_day_eth.csv"), index=False)

all_baseline_errs = per_day_df["baseline_err"].values
all_selected_errs = per_day_df["selected_err"].values
t_stat, p_val = stats.ttest_rel(all_selected_errs, all_baseline_errs)

print(res_df.to_string(index=False))
print(f"\nWin rate: {res_df['selected_wins'].sum()}/{len(res_df)} windows")
print(f"Pooled across {len(all_selected_errs)} test days from every window:")
print(f"  mean baseline error : {np.mean(all_baseline_errs):.6f}")
print(f"  mean selected error : {np.mean(all_selected_errs):.6f}")
print(f"  paired t-test        : t={t_stat:.4f}, p={p_val:.4f}")
print(f"\nSaved decay_model/results/walk_forward_results_eth.csv")
print(f"Saved decay_model/results/walk_forward_per_day_eth.csv ({len(per_day_df)} rows)")