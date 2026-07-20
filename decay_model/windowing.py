"""
Shared windowing function for the decay_model pipeline.

This used to be copy-pasted independently into 7 different files. That's
exactly how two separate real bugs made it into the project:
  1. A one-day offset between the feature window and the prediction target
     (fixed across all 7 copies individually, one at a time).
  2. Two of the copies (decay_rate_sensitivity.py, statistical_significance.py)
     were missing the ticker filter entirely, silently mixing AAPL/BTC-USD/SPY
     rows together and roughly doubling every reported MAE in those two files.
One shared, tested function means a fix here fixes every caller at once, and
a bug here is only one place to look, not seven.

Convention (matches every other script in the repo):
  feature window = returns[i-window : i]
  target         = returns[i]           (same day the window ends on -- no gap)
"""
import pandas as pd
import numpy as np


def make_windows(full_df, start_date, end_date, window, ticker=None, track_regime=False):
    """
    Build windowed (X, y) samples from a returns dataframe.

    Parameters
    ----------
    full_df : DataFrame with at least 'date' and 'returns' columns.
        May optionally have 'ticker' and/or 'regime' columns.
    start_date, end_date : anything pd.Timestamp() accepts
        Inclusive date range for the TARGET (not the feature window) to fall in.
    window : int
        Number of prior periods used as features.
    ticker : str or None
        If given and the dataframe has a 'ticker' column, filters to that
        ticker BEFORE windowing. data_processed.csv is multi-ticker by
        design (AAPL/BTC-USD/SPY stacked) -- pass ticker="SPY" whenever
        reading from it directly. Files that already read from a
        single-asset source (e.g. data_regime.csv) can leave this as None.
    track_regime : bool
        If True, also returns each sample's regime label. Requires a
        'regime' column in full_df.

    Returns
    -------
    (X, y, sample_dates) or (X, y, sample_dates, sample_regimes) if track_regime
    X : ndarray, shape (n_samples, window)
    y : ndarray, shape (n_samples, 1)
    sample_dates : ndarray, shape (n_samples,)
    sample_regimes : ndarray, shape (n_samples,) -- only if track_regime=True
    """
    df = full_df
    if ticker is not None:
        if "ticker" not in df.columns:
            raise ValueError(
                f"ticker='{ticker}' was passed but this dataframe has no 'ticker' "
                "column -- it's likely already single-asset. Call with ticker=None."
            )
        before = len(df)
        df = df[df["ticker"] == ticker]
        if len(df) == 0:
            raise ValueError(f"No rows found for ticker='{ticker}'")
        print(f"Filtered to ticker=='{ticker}': {before} rows -> {len(df)} rows")

    df = df.sort_values("date").reset_index(drop=True)
    returns = df["returns"].values
    dates = df["date"].values

    if track_regime:
        if "regime" not in df.columns:
            raise ValueError("track_regime=True but no 'regime' column found -- "
                              "run add_regime_labels.py first, or read from data_regime.csv.")
        regimes = df["regime"].values

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    X, y, sample_dates, sample_regimes = [], [], [], []
    for i in range(window, len(returns)):
        target_date = pd.Timestamp(dates[i])
        if start_ts <= target_date <= end_ts:
            X.append(returns[i - window:i])
            y.append(returns[i])
            sample_dates.append(dates[i])
            if track_regime:
                sample_regimes.append(regimes[i])  # regime of the day being PREDICTED

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    sample_dates = np.array(sample_dates)

    if track_regime:
        return X, y, sample_dates, np.array(sample_regimes)
    return X, y, sample_dates


if __name__ == "__main__":
    # Quick self-test with synthetic data -- confirms the no-gap convention
    # and ticker filtering both work as documented.
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date": list(dates) * 2,
        "ticker": ["SPY"] * 50 + ["OTHER"] * 50,
        "returns": list(np.arange(50) * 0.01) + list(np.arange(50) * 99.0),  # OTHER is wildly different
    })
    X, y, d = make_windows(df, "2020-01-01", "2020-02-19", window=5, ticker="SPY")
    assert X.max() < 1.0, "Ticker filter failed -- OTHER's huge values leaked in"
    # X[0] should be returns[0:5], y[0] should be returns[5] -- no gap
    expected_X0 = np.arange(5) * 0.01
    expected_y0 = 5 * 0.01
    assert np.allclose(X[0], expected_X0), f"Feature window wrong: {X[0]} != {expected_X0}"
    assert np.isclose(y[0, 0], expected_y0), f"Target has a gap: {y[0,0]} != {expected_y0}"
    print("Self-test passed: ticker filter works, windowing has no target/feature gap.")