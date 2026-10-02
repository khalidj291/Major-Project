"""
volatility_switching.py -- Idea #1 from the "what else is worth trying"
list: turn the volatility-regime OBSERVATION (calm windows favor combined
decay, volatile windows favor power-law-only) into an actual predictive
rule, tested honestly.

WHAT THIS TESTS
----------------
The existing walk-forward pipeline picks ONE best method per 6-month
window by searching ~20 candidates against that window's validation MAE.
This script asks a narrower, more interesting question: instead of a
window-level, MAE-search-based pick, can a much simpler DAY-LEVEL rule
using nothing but trailing realized volatility decide, per day, whether
to use the combined-decay or power-law-only-decay prediction -- and does
that switching rule do as well as (or better than) the existing
search-based method selection?

If yes, this is a real mechanism ("volatility-conditioned decay"), not
just a post-hoc explanation of results already seen -- which is exactly
what the earlier volatility-regime analysis was flagged as needing.

LEAKAGE DISCIPLINE
--------------------
1. Trailing volatility for day i = std of returns[i-30:i] (via shift(1)
   before rolling, same convention as ma_crossover_lib.py) -- never
   includes day i itself.
2. Within each window: combined_best and powerlaw_best are each the
   best-in-family candidate by VALIDATION MAE (same search already used
   elsewhere, just restricted to one family at a time).
3. The volatility THRESHOLD that decides which family to use on a given
   day is chosen by testing candidate percentiles (0/25/50/75/100) of
   VALIDATION-day trailing volatility against VALIDATION MAE of the
   switched prediction -- never touches test data.
4. That chosen threshold is then applied, unmodified, to test days.

Compares switched-prediction MAE against: baseline, combined-only,
power-law-only, and the existing search-selected method (already saved
in walk_forward_results_{btc,eth}.csv) -- to see whether day-level
volatility switching adds anything beyond just picking one family for
the whole window.

Run: python volatility_switching.py btc
     python volatility_switching.py eth
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from powerlaw_decay import powerlaw_weights
from ebbinghaus import ebbinghaus_weight

WINDOW = 30
ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_SAMPLE_CANDIDATES = [180, 365, 730]
VOL_LOOKBACK = 30
VOL_THRESHOLD_PERCENTILES = [0, 25, 50, 75, 100]


def add_trailing_volatility(df):
    df = df.sort_values("date").reset_index(drop=True).copy()
    prior_returns = df["returns"].shift(1)  # never includes today
    df["trailing_vol"] = prior_returns.rolling(VOL_LOOKBACK).std()
    return df


def get_vol_for_dates(df_with_vol, dates):
    lookup = df_with_vol.set_index("date")["trailing_vol"]
    return lookup.loc[pd.to_datetime(dates)].values


def pick_best_in_family(X_train, y_train, X_val, y_val, train_dates, reference_date, family):
    """family: 'combined' or 'powerlaw'. Returns (weights, sample_weights)
    for the best-in-family candidate by validation MAE."""
    candidates = []
    if family == "powerlaw":
        for alpha in ALPHA_CANDIDATES:
            w = powerlaw_weights(WINDOW, alpha)
            m = Ridge(alpha=1.0).fit(X_train * w, y_train)
            mae = np.abs(y_val - m.predict(X_val * w)).mean()
            candidates.append((mae, w, None))
    elif family == "combined":
        for alpha in ALPHA_CANDIDATES:
            w = powerlaw_weights(WINDOW, alpha)
            for S in S_SAMPLE_CANDIDATES:
                sw = ebbinghaus_weight(train_dates, reference_date, S)
                m = Ridge(alpha=1.0).fit(X_train * w, y_train, sample_weight=sw)
                mae = np.abs(y_val - m.predict(X_val * w)).mean()
                candidates.append((mae, w, sw))
    best = min(candidates, key=lambda c: c[0])
    return best[1], best[2]


def run(asset):
    label = "BTC-USD" if asset == "btc" else "ETH-USD"
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"data_{asset}.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "ticker" in df.columns:
        df = df[df["ticker"] == "BTC-USD"].reset_index(drop=True)
    df_vol = add_trailing_volatility(df)

    test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")
    per_day_rows = []
    window_rows = []

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

        w_pl, sw_pl = pick_best_in_family(X_train, y_train, X_val, y_val, train_dates, reference_date, "powerlaw")
        w_cb, sw_cb = pick_best_in_family(X_train, y_train, X_val, y_val, train_dates, reference_date, "combined")

        pl_model = Ridge(alpha=1.0).fit(X_train * w_pl, y_train)
        cb_model = Ridge(alpha=1.0).fit(X_train * w_cb, y_train, sample_weight=sw_cb)
        baseline_model = Ridge(alpha=1.0).fit(X_train, y_train)

        pl_val_pred = pl_model.predict(X_val * w_pl).flatten()
        cb_val_pred = cb_model.predict(X_val * w_cb).flatten()
        pl_test_pred = pl_model.predict(X_test * w_pl).flatten()
        cb_test_pred = cb_model.predict(X_test * w_cb).flatten()
        baseline_test_pred = baseline_model.predict(X_test).flatten()

        val_vol = get_vol_for_dates(df_vol, val_dates)
        test_vol = get_vol_for_dates(df_vol, test_dates)
        y_val_actual = y_val.flatten()
        y_test_actual = y_test.flatten()

        best_thresh, best_val_mae = None, np.inf
        for pct in VOL_THRESHOLD_PERCENTILES:
            thresh = np.percentile(val_vol, pct)
            switched_val_pred = np.where(val_vol > thresh, pl_val_pred, cb_val_pred)
            mae = np.abs(y_val_actual - switched_val_pred).mean()
            if mae < best_val_mae:
                best_val_mae, best_thresh = mae, thresh

        switched_test_pred = np.where(test_vol > best_thresh, pl_test_pred, cb_test_pred)

        window_rows.append({
            "test_window_start": str(test_start.date()),
            "vol_threshold": best_thresh,
            "pct_days_using_powerlaw": float(np.mean(test_vol > best_thresh)),
            "baseline_MAE": np.abs(y_test_actual - baseline_test_pred).mean(),
            "combined_only_MAE": np.abs(y_test_actual - cb_test_pred).mean(),
            "powerlaw_only_MAE": np.abs(y_test_actual - pl_test_pred).mean(),
            "switched_MAE": np.abs(y_test_actual - switched_test_pred).mean(),
        })

        for i, d in enumerate(test_dates):
            per_day_rows.append({
                "date": pd.Timestamp(d).date(),
                "actual_return": y_test_actual[i],
                "baseline_err": abs(y_test_actual[i] - baseline_test_pred[i]),
                "combined_only_err": abs(y_test_actual[i] - cb_test_pred[i]),
                "powerlaw_only_err": abs(y_test_actual[i] - pl_test_pred[i]),
                "switched_err": abs(y_test_actual[i] - switched_test_pred[i]),
            })

    window_df = pd.DataFrame(window_rows)
    per_day_df = pd.DataFrame(per_day_rows)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    window_df.to_csv(os.path.join(results_dir, f"volatility_switching_windows_{asset}.csv"), index=False)
    per_day_df.to_csv(os.path.join(results_dir, f"volatility_switching_per_day_{asset}.csv"), index=False)

    print(f"\n{'=' * 100}\n{label} -- volatility-conditioned decay switching\n{'=' * 100}")
    print(window_df.to_string(index=False))

    n = len(per_day_df)
    print(f"\n--- Pooled MAE across {n} test days ---")
    for col in ["baseline_err", "combined_only_err", "powerlaw_only_err", "switched_err"]:
        print(f"  {col:<20}: {per_day_df[col].mean():.6f}")

    t1, p1 = stats.ttest_rel(per_day_df["switched_err"], per_day_df["baseline_err"])
    t2, p2 = stats.ttest_rel(per_day_df["switched_err"], per_day_df["combined_only_err"])
    t3, p3 = stats.ttest_rel(per_day_df["switched_err"], per_day_df["powerlaw_only_err"])
    print(f"\n  paired t-test, switched vs baseline       : t={t1:.4f}, p={p1:.4f}")
    print(f"  paired t-test, switched vs combined-only  : t={t2:.4f}, p={p2:.4f}")
    print(f"  paired t-test, switched vs powerlaw-only  : t={t3:.4f}, p={p3:.4f}")

    existing_path = os.path.join(results_dir, f"walk_forward_results_{asset}.csv")
    if os.path.exists(existing_path):
        existing = pd.read_csv(existing_path)
        print(f"\n--- For reference: existing search-selected method (walk_forward_validation_{asset}.py) ---")
        print(f"  pooled selected_MAE: {existing['selected_MAE'].mean():.6f} (window-mean average, not exactly pooled-by-day)")

    print(f"\nSaved decay_model/results/volatility_switching_windows_{asset}.csv")
    print(f"Saved decay_model/results/volatility_switching_per_day_{asset}.csv ({n} rows)")


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(asset)