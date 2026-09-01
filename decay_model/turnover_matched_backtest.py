"""
turnover_matched_backtest.py -- trading_cost_backtest.py found decay trades
733-1054 times vs MA crossover's 46-56 times, and MA wins on fees/Sharpe
specifically because of that gap. This asks the direct follow-up: if you
constrain decay's signal to trade about as often as MA crossover does,
does the fee-adjusted gap close?

Two independent ways to reduce turnover, both standard in trading system
design (not invented for this test):
  - deadband: only flip position when the predicted return's MAGNITUDE
    clears a threshold, not just its sign (ignores near-zero noise)
  - minimum holding period: once you enter/exit, ignore the signal for N
    days before allowing another flip

Reuses trading_cost_backtest.py's exact predictions (same walk-forward
run, same signals) -- only how the SIGNAL becomes a POSITION changes.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from trading_cost_backtest import get_predictions, PRIMARY_FEE
from backtest_lib import build_position, summarize_strategy


def build_position_deadband(signal, threshold):
    signal = np.asarray(signal)
    position = np.zeros(len(signal), dtype=int)
    current = 0
    for i, s in enumerate(signal):
        if s > threshold:
            current = 1
        elif s < -threshold:
            current = 0
        position[i] = current
    return position


def build_position_min_hold(signal, min_hold_days):
    signal = np.asarray(signal)
    position = np.zeros(len(signal), dtype=int)
    current = 0
    days_since_change = min_hold_days
    for i, s in enumerate(signal):
        desired = 1 if s > 0 else 0
        if days_since_change >= min_hold_days and desired != current:
            current = desired
            days_since_change = 0
        else:
            days_since_change += 1
        position[i] = current
    return position


def run(asset):
    label = "BTC-USD" if asset == "btc" else "ETH-USD"
    df = get_predictions(asset)
    actual_return = df["actual_return"].values
    signal = df["selected_pred"].values
    momentum = df["momentum"].values

    ma_trades = int((build_position(momentum) != np.concatenate([[0], build_position(momentum)[:-1]])).sum())

    print(f"\n{'='*90}\n{label} -- turnover-matched backtest ({len(df)} days)\n{'='*90}")
    print(f"MA crossover trades {ma_trades} times over this period -- the target to match.\n")

    rows = []
    baseline_pos = build_position(signal)  # unconstrained, for reference
    rows.append(summarize_strategy("decay_unconstrained", baseline_pos, actual_return, PRIMARY_FEE))
    rows.append(summarize_strategy("ma_signal", build_position(momentum), actual_return, PRIMARY_FEE))

    # sweep deadband thresholds (as a fraction of the signal's own std, so it's
    # meaningful regardless of the asset's typical daily return scale)
    sig_std = signal.std()
    for mult in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        pos = build_position_deadband(signal, mult * sig_std)
        rows.append(summarize_strategy(f"decay_deadband_{mult}std", pos, actual_return, PRIMARY_FEE))

    # sweep minimum holding periods (days)
    for hold in [3, 5, 10, 15, 20, 30]:
        pos = build_position_min_hold(signal, hold)
        rows.append(summarize_strategy(f"decay_minhold_{hold}d", pos, actual_return, PRIMARY_FEE))

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    # which turnover-controlled variant landed closest to MA's actual trade count?
    result_df["trade_gap_to_ma"] = (result_df["n_trades"] - ma_trades).abs()
    closest = result_df[result_df["strategy"] != "ma_signal"].sort_values("trade_gap_to_ma").iloc[0]
    ma_row = result_df[result_df["strategy"] == "ma_signal"].iloc[0]
    print(f"\nClosest turnover match to MA crossover ({ma_trades} trades): {closest['strategy']} ({closest['n_trades']} trades)")
    print(f"  its Sharpe    : {closest['annualized_sharpe']:.3f}   vs MA's Sharpe: {ma_row['annualized_sharpe']:.3f}")
    print(f"  its cum. return: {closest['cumulative_return']:+.3%}   vs MA's: {ma_row['cumulative_return']:+.3%}")

    out_path = os.path.join(os.path.dirname(__file__), "results", f"turnover_matched_backtest_{asset}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    return result_df


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(asset)