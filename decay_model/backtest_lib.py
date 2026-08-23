"""
backtest_lib.py -- shared trading-cost-aware backtest mechanics.

STRATEGY DESIGN
----------------
Long/flat only (no shorting): position[i] = 1 (fully invested) if the
signal for day i is positive, else 0 (cash, zero return that day).
Long/flat is chosen over long/short because retail crypto shorting
typically requires margin or derivatives with their own borrow costs,
which would need separate modeling -- long/flat keeps this test honest
about what it's actually measuring.

FEE MODEL
---------
A fee is charged only when the position CHANGES from the previous day
(flat->long costs one trade, long->flat costs one trade; staying in the
same position costs nothing). This is the standard convention -- you pay
to enter and exit, not to hold. fee_rate is expressed as a fraction (e.g.
0.001 = 0.1% per trade), charged against portfolio value on the day of
the trade.

ANNUALIZATION
-------------
Crypto trades every day of the year (no market holidays/weekends unlike
equities), so Sharpe ratios here annualize with sqrt(365), not the
sqrt(252) convention used for stock markets.
"""
import numpy as np


def build_position(signal):
    """signal: array of predicted returns or momentum values (any sign
    convention). Returns 1 where signal > 0 (go long), 0 otherwise (cash).
    Exactly-zero signal (rare) is treated as flat."""
    return (np.asarray(signal) > 0).astype(int)


def backtest_returns(position, actual_return, fee_rate):
    """Daily net strategy returns after fees.

    position[i]: 0 or 1, the position HELD during day i (decided using
    only information available before day i, matching every other
    leak-free convention in this project).
    actual_return[i]: the day's real return.
    fee_rate: fraction charged on days the position changes vs the day
    before (first day is assumed to start from cash, so if position[0]==1
    a fee is charged to enter).
    """
    position = np.asarray(position)
    actual_return = np.asarray(actual_return)
    n = len(position)
    prev_position = np.concatenate([[0], position[:-1]])  # start from cash
    traded = (position != prev_position).astype(int)
    gross_return = position * actual_return
    net_return = gross_return - traded * fee_rate
    return net_return, traded


def equity_curve(net_returns, start_value=1.0):
    return start_value * np.cumprod(1.0 + net_returns)


def max_drawdown(equity):
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return drawdown.min()


def annualized_sharpe(net_returns, periods_per_year=365):
    std = net_returns.std(ddof=1)
    if std == 0:
        return np.nan
    return (net_returns.mean() / std) * np.sqrt(periods_per_year)


def summarize_strategy(name, position, actual_return, fee_rate):
    net_returns, traded = backtest_returns(position, actual_return, fee_rate)
    eq = equity_curve(net_returns)
    return {
        "strategy": name,
        "fee_rate": fee_rate,
        "n_trades": int(traded.sum()),
        "cumulative_return": eq[-1] - 1.0,
        "annualized_sharpe": annualized_sharpe(net_returns),
        "max_drawdown": max_drawdown(eq),
        "mean_daily_net_return": net_returns.mean(),
    }