"""
trading_cost_backtest.py -- answers "is any of this actually useful once
real trading friction is accounted for," the last of the five originally
proposed rigor improvements. Reuses the exact same leak-free walk-forward
predictions as established_technique_comparison.py, but instead of MAE/
directional-accuracy, converts each method's signal into a long/flat
position and backtests net-of-fees P&L.

Strategies compared:
  - buy_and_hold   : always long (the "do nothing clever" benchmark)
  - baseline       : long when plain Ridge predicts a positive return
  - decay_selected : long when the leak-free-selected decay model predicts positive
  - ma_signal      : long when short-MA > long-MA (true crossover rule,
                      sign of momentum -- NOT the linear predictor's sign,
                      which can differ near zero)

Fee levels swept: 0.05%, 0.1% (primary -- Binance-like spot taker fee,
a concrete real-world reference point), 0.25%, 0.5% per trade, covering
the range real crypto venues actually charge.

Run: python trading_cost_backtest.py btc
     python trading_cost_backtest.py eth
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from powerlaw_decay import powerlaw_weights
from ebbinghaus import ebbinghaus_weight
from ma_crossover_lib import add_ma_features, get_momentum_for_dates
from backtest_lib import build_position, summarize_strategy

WINDOW = 30
ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_FEAT_CANDIDATES = [30, 90, 180]
S_SAMPLE_CANDIDATES = [180, 365, 730]
FEE_LEVELS = [0.0005, 0.001, 0.0025, 0.005]
PRIMARY_FEE = 0.001


def exp_feat_weights(window, S):
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)


def get_predictions(asset):
    """Rerun the same candidate-selection walk-forward loop as
    walk_forward_validation_*.py / established_technique_comparison.py,
    this time keeping the raw momentum value (not just its sign or the
    linear predictor built on it) so the backtest can use the TRUE
    crossover rule for the MA strategy."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"data_{asset}.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "ticker" in df.columns:
        df = df[df["ticker"] == "BTC-USD"].reset_index(drop=True)
    df_ma = add_ma_features(df)

    test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")
    rows = []

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
        selected_pred = selected_model.predict(X_test_final).flatten()

        baseline_model = Ridge(alpha=1.0).fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test).flatten()

        test_momentum = get_momentum_for_dates(df_ma, test_dates)
        y_actual = y_test.flatten()

        for i, d in enumerate(test_dates):
            rows.append({
                "date": pd.Timestamp(d).date(),
                "actual_return": y_actual[i],
                "baseline_pred": baseline_pred[i],
                "selected_pred": selected_pred[i],
                "momentum": test_momentum[i],
            })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def run(asset):
    label = "BTC-USD" if asset == "btc" else "ETH-USD"
    df = get_predictions(asset)
    n = len(df)

    actual_return = df["actual_return"].values
    positions = {
        "buy_and_hold": np.ones(n, dtype=int),
        "baseline": build_position(df["baseline_pred"].values),
        "decay_selected": build_position(df["selected_pred"].values),
        "ma_signal": build_position(df["momentum"].values),
    }

    print(f"\n{'=' * 90}\n{label} -- trading-cost-aware backtest ({n} days, 2020-01-01 to test end)\n{'=' * 90}")

    all_rows = []
    for fee in FEE_LEVELS:
        print(f"\n--- Fee = {fee*100:.2f}% per trade{'  (PRIMARY: Binance-like spot taker fee)' if fee == PRIMARY_FEE else ''} ---")
        header = f"{'strategy':<16} {'n_trades':>9} {'cum_return':>12} {'ann_sharpe':>11} {'max_drawdown':>13}"
        print(header)
        for strat_name, pos in positions.items():
            s = summarize_strategy(strat_name, pos, actual_return, fee)
            print(f"{s['strategy']:<16} {s['n_trades']:>9} {s['cumulative_return']*100:>11.2f}% {s['annualized_sharpe']:>11.3f} {s['max_drawdown']*100:>12.2f}%")
            all_rows.append(s)

    results_df = pd.DataFrame(all_rows)
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"trading_cost_backtest_{asset}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved decay_model/results/trading_cost_backtest_{asset}.csv ({len(results_df)} rows)")

    # Headline comparison at the primary fee level
    primary = results_df[results_df["fee_rate"] == PRIMARY_FEE].set_index("strategy")
    print(f"\n--- Headline at {PRIMARY_FEE*100:.1f}% fee ---")
    for strat in ["buy_and_hold", "baseline", "decay_selected", "ma_signal"]:
        row = primary.loc[strat]
        print(f"  {strat:<16}: {row['cumulative_return']*100:>7.2f}% total return, "
              f"Sharpe {row['annualized_sharpe']:.3f}, {int(row['n_trades'])} trades")


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(asset)