"""
established_technique_comparison.py -- compares our decay approach against
a standard moving-average (MA) crossover, on the same leak-free 13-window
walk-forward protocol used everywhere else in this project.

For each window: baseline and decay-selected models are fit and evaluated
exactly as in walk_forward_validation_{btc,eth}.py (same candidate search,
same leakage rules). The MA linear predictor is fit on the same training
data. The MA directional signal needs no fitting at all -- it's a fixed
rule (sign of momentum). All four are evaluated on the identical test days
within each window, which is what makes the paired comparisons
(McNemar's test for direction, paired stats for MAE) valid.

Run: python established_technique_comparison.py btc
     python established_technique_comparison.py eth
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
from ma_crossover_lib import (
    add_ma_features, get_momentum_for_dates, fit_ma_linear_predictor, mcnemar_test,
    SHORT_WINDOW, LONG_WINDOW,
)

WINDOW = 30
ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_FEAT_CANDIDATES = [30, 90, 180]
S_SAMPLE_CANDIDATES = [180, 365, 730]


def exp_feat_weights(window, S):
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)


def run(asset):
    assert asset in ("btc", "eth")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"data_{asset}.csv")
    label = "BTC-USD" if asset == "btc" else "ETH-USD"

    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "ticker" in df.columns:
        df = df[df["ticker"] == "BTC-USD"].reset_index(drop=True)

    df_ma = add_ma_features(df)  # ma_short, ma_long, momentum columns, leak-free

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

        # --- baseline + decay candidate selection (identical to walk_forward_validation_*.py) ---
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
        selected_pred = selected_model.predict(X_test_final).flatten()

        baseline_model = Ridge(alpha=1.0).fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test).flatten()

        # --- MA crossover: linear predictor (fit on train) ---
        train_momentum = get_momentum_for_dates(df_ma, train_dates)
        valid_train = ~np.isnan(train_momentum)  # first LONG_WINDOW rows of the whole series lack momentum
        ma_model = fit_ma_linear_predictor(train_momentum[valid_train], y_train.flatten()[valid_train])
        test_momentum = get_momentum_for_dates(df_ma, test_dates)
        ma_linear_pred = ma_model.predict(test_momentum.reshape(-1, 1)).flatten()

        # --- MA crossover: directional signal (no fitting -- fixed rule) ---
        ma_signal_sign = np.sign(test_momentum)

        y_actual = y_test.flatten()
        actual_sign = np.sign(y_actual)

        baseline_err = np.abs(y_actual - baseline_pred)
        selected_err = np.abs(y_actual - selected_pred)
        ma_linear_err = np.abs(y_actual - ma_linear_pred)

        baseline_correct = np.sign(baseline_pred) == actual_sign
        selected_correct = np.sign(selected_pred) == actual_sign
        ma_signal_correct = ma_signal_sign == actual_sign

        for i, d in enumerate(test_dates):
            per_day_rows.append({
                "date": pd.Timestamp(d).date(),
                "window_start": str(test_start.date()),
                "actual_return": y_actual[i],
                "baseline_pred": baseline_pred[i],
                "selected_pred": selected_pred[i],
                "ma_linear_pred": ma_linear_pred[i],
                "baseline_err": baseline_err[i],
                "selected_err": selected_err[i],
                "ma_linear_err": ma_linear_err[i],
                "baseline_correct": bool(baseline_correct[i]),
                "selected_correct": bool(selected_correct[i]),
                "ma_signal_correct": bool(ma_signal_correct[i]),
            })

        window_rows.append({
            "test_window_start": str(test_start.date()),
            "test_window_end": str(test_end.date()),
            "n_test": len(X_test),
            "selected_method": name,
            "baseline_MAE": baseline_err.mean(),
            "selected_MAE": selected_err.mean(),
            "ma_linear_MAE": ma_linear_err.mean(),
            "baseline_dir_acc": baseline_correct.mean(),
            "selected_dir_acc": selected_correct.mean(),
            "ma_signal_dir_acc": ma_signal_correct.mean(),
        })

    per_day_df = pd.DataFrame(per_day_rows)
    window_df = pd.DataFrame(window_rows)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    per_day_df.to_csv(os.path.join(results_dir, f"ma_comparison_per_day_{asset}.csv"), index=False)
    window_df.to_csv(os.path.join(results_dir, f"ma_comparison_windows_{asset}.csv"), index=False)

    print(f"\n{'=' * 78}\n{label} -- established technique comparison ({SHORT_WINDOW}/{LONG_WINDOW}-day MA crossover)\n{'=' * 78}")
    print(window_df.to_string(index=False))

    n = len(per_day_df)
    print(f"\n--- Pooled MAE across {n} test days ---")
    print(f"  baseline MAE   : {per_day_df['baseline_err'].mean():.6f}")
    print(f"  decay-selected MAE: {per_day_df['selected_err'].mean():.6f}")
    print(f"  MA-linear MAE  : {per_day_df['ma_linear_err'].mean():.6f}")
    t_sel_ma, p_sel_ma = stats.ttest_rel(per_day_df["selected_err"], per_day_df["ma_linear_err"])
    t_base_ma, p_base_ma = stats.ttest_rel(per_day_df["baseline_err"], per_day_df["ma_linear_err"])
    print(f"  paired t-test, decay-selected vs MA-linear: t={t_sel_ma:.4f}, p={p_sel_ma:.4f}")
    print(f"  paired t-test, baseline vs MA-linear       : t={t_base_ma:.4f}, p={p_base_ma:.4f}")

    print(f"\n--- Pooled directional accuracy across {n} test days ---")
    print(f"  baseline       : {per_day_df['baseline_correct'].mean():.4f}")
    print(f"  decay-selected : {per_day_df['selected_correct'].mean():.4f}")
    print(f"  MA-signal      : {per_day_df['ma_signal_correct'].mean():.4f}")

    b1, c1, nd1, p1 = mcnemar_test(per_day_df["baseline_correct"].values, per_day_df["ma_signal_correct"].values)
    print(f"\n  McNemar (baseline vs MA-signal): discordant={nd1} (baseline-only-right={c1}, MA-only-right={b1}), p={p1:.4f}")
    b2, c2, nd2, p2 = mcnemar_test(per_day_df["selected_correct"].values, per_day_df["ma_signal_correct"].values)
    print(f"  McNemar (decay-selected vs MA-signal): discordant={nd2} (decay-only-right={c2}, MA-only-right={b2}), p={p2:.4f}")

    print(f"\nSaved decay_model/results/ma_comparison_per_day_{asset}.csv ({n} rows)")
    print(f"Saved decay_model/results/ma_comparison_windows_{asset}.csv ({len(window_df)} rows)")


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(asset)