"""
ma_crossover_lib.py -- shared logic for comparing our decay models against
a standard moving-average (MA) crossover, used by both the BTC and ETH
comparison scripts.

WHY THIS EXISTS
----------------
Every result so far compares our decay approach only against our OWN
baseline (plain Ridge, no decay). A real reviewer's next question is "sure,
but how does this compare to a technique people actually use?" MA crossover
is the standard, simplest answer to that -- short-term average price vs
long-term average price, buy when short > long.

THE COMPARISON PROBLEM
------------------------
MA crossover is naturally a DIRECTIONAL strategy (buy/don't-buy), not a
point predictor of the return's exact size. Our decay models predict a
number and get graded by MAE. To compare fairly we do BOTH:

  1. Directional accuracy: MA crossover's natural metric (is short_MA >
     long_MA the same side as tomorrow's actual return?), compared against
     our baseline and decay models' directional accuracy (sign of their
     predicted return vs actual), on the SAME test days, using McNemar's
     test -- the correct paired test for "did classifier A and classifier B
     get the same days right/wrong," which a plain accuracy-difference
     comparison ignores (it would treat the two classifiers as
     independent samples, which they're not -- they're scored on the
     identical days).

  2. MAE comparison: convert the MA signal into a genuine number predictor
     by fitting a single-feature linear regression: predicted_return =
     a * momentum + b, where momentum = (MA_short - MA_long) / MA_long.
     This is a stretch beyond how MA crossover is normally used (nobody
     actually predicts an exact return size from it), and the writeup
     should say so plainly -- but it lets its MAE sit on the same table as
     everything else.

LEAKAGE
-------
MA_short[i] and MA_long[i] are computed using CLOSE PRICES UP TO DAY i-1
(never including day i), matching windowing.py's convention exactly:
target returns[i] is predicted from data strictly before day i. Standard
window lengths (20-day short, 50-day long) are used, NOT tuned per window
-- tuning them would defeat the point of "how does a standard technique
compare," so they're fixed for every window, same as any textbook use.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SHORT_WINDOW = 20
LONG_WINDOW = 50


def add_ma_features(df):
    """Add ma_short, ma_long, momentum columns to a date-sorted df with a
    'close' column. All three use only data strictly before the row's own
    date (shift(1) before rolling), so momentum[i] is legal to use as a
    feature for predicting returns[i]."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    prior_close = df["close"].shift(1)  # never includes today's close
    df["ma_short"] = prior_close.rolling(SHORT_WINDOW).mean()
    df["ma_long"] = prior_close.rolling(LONG_WINDOW).mean()
    df["momentum"] = (df["ma_short"] - df["ma_long"]) / df["ma_long"]
    return df


def get_momentum_for_dates(df_with_features, dates):
    """Look up momentum values for an array of dates, in order."""
    lookup = df_with_features.set_index("date")["momentum"]
    return lookup.loc[pd.to_datetime(dates)].values


def fit_ma_linear_predictor(train_momentum, train_returns):
    """Single-feature OLS: predicted_return = a*momentum + b. No search,
    no tuning -- this is the simplest possible honest fit, same spirit as
    any of this project's other Ridge fits but with one fixed feature."""
    X = train_momentum.reshape(-1, 1)
    y = train_returns.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model


def mcnemar_test(correct_a, correct_b):
    """McNemar's test for comparing two paired binary classifiers on the
    SAME samples. correct_a, correct_b: boolean arrays, same length, one
    entry per test day, True if that classifier's directional call was
    right that day.

    Only the days where the two classifiers DISAGREE matter -- days they
    both got right or both got wrong provide no information about which
    is better. Of the disagreement days, b_only = A wrong but B right,
    c_only = A right but B wrong. Under H0 (equal accuracy), b_only and
    c_only should be roughly equal; the test asks how likely the observed
    split is under that null.

    Returns (b_only, c_only, chi2_stat, p_value). Uses the exact binomial
    form (appropriate here since discordant counts are in the hundreds,
    not requiring the continuity-corrected chi-square approximation).
    """
    from scipy.stats import binomtest
    both_right = correct_a & correct_b
    both_wrong = (~correct_a) & (~correct_b)
    b_only = int((~correct_a & correct_b).sum())  # A wrong, B right
    c_only = int((correct_a & ~correct_b).sum())  # A right, B wrong
    n_discordant = b_only + c_only
    if n_discordant == 0:
        return b_only, c_only, np.nan, 1.0
    result = binomtest(min(b_only, c_only), n_discordant, 0.5, alternative="two-sided")
    return b_only, c_only, n_discordant, result.pvalue