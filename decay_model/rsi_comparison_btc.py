"""
rsi_comparison_btc.py -- scopes "compare against the best known way to
predict BTC" into something concrete: RSI (Relative Strength Index), the
standard mean-reversion technical indicator (Wilder, 1978), used the same
rigorous way ma_crossover_lib.py compares MA crossover -- directional
accuracy via McNemar's test, plus a linear-regression MAE conversion so
it sits on the same table as everything else. This is one legitimate,
well-known technique, not "the" single best predictor (no such agreed-on
thing exists) -- scoped this way so it's actually testable.

RSI signal: RSI < 30 is "oversold" (expect a bounce -> long), RSI > 70 is
"overbought" (expect a pullback -> flat). Standard 14-day RSI window,
NOT tuned per window -- same "use it as a textbook would" convention as
MA crossover's fixed 20/50-day windows.

Leakage: RSI[i] computed from price changes strictly before day i,
matching every other convention in this project.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from powerlaw_decay import powerlaw_weights
from ebbinghaus import ebbinghaus_weight
from ma_crossover_lib import mcnemar_test

RSI_WINDOW = 14
WINDOW = 30
ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 3.0]
S_FEAT_CANDIDATES = [30, 90, 180]
S_SAMPLE_CANDIDATES = [180, 365, 730]


def add_rsi(df):
    """Standard Wilder RSI. Uses price changes strictly before day i (shift(1)
    before the rolling gain/loss averages), same leakage discipline as MA."""
    df = df.sort_values("date").reset_index(drop=True).copy()
    prior_close = df["close"].shift(1)
    delta = prior_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_WINDOW).mean()
    avg_loss = loss.rolling(RSI_WINDOW).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def exp_feat_weights(window, S):
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)


def get_predictions(asset="btc"):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", f"data_{asset}.csv")
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df_rsi = add_rsi(df)

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
        name, val_mae, w_best, sw_best = min(candidates, key=lambda c: c[1])

        X_train_final = X_train * w_best if w_best is not None else X_train
        X_test_final = X_test * w_best if w_best is not None else X_test
        selected_model = Ridge(alpha=1.0).fit(X_train_final, y_train, sample_weight=sw_best)
        selected_pred = selected_model.predict(X_test_final).flatten()

        # RSI signal for these same test dates: centered so 0 = neutral (RSI=50),
        # negative = oversold (expect bounce -> would predict positive return)
        rsi_lookup = df_rsi.set_index("date")["rsi"]
        test_rsi = rsi_lookup.loc[pd.to_datetime(test_dates)].values
        rsi_signal = -(test_rsi - 50)  # oversold (low RSI) -> positive signal -> long

        # fit linear predictor on TRAIN rsi -> returns, same spirit as MA's momentum regression
        train_rsi = rsi_lookup.loc[pd.to_datetime(train_dates)].values
        train_rsi_signal = -(train_rsi - 50)
        valid = ~np.isnan(train_rsi_signal)
        rsi_model = LinearRegression().fit(train_rsi_signal[valid].reshape(-1, 1), y_train[valid])
        rsi_pred = rsi_model.predict(rsi_signal.reshape(-1, 1)).flatten()

        y_actual = y_test.flatten()
        for i, d in enumerate(test_dates):
            rows.append({
                "date": pd.Timestamp(d).date(), "actual_return": y_actual[i],
                "selected_pred": selected_pred[i], "rsi_signal": rsi_signal[i], "rsi_pred": rsi_pred[i],
            })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def run(asset="btc"):
    label = "BTC-USD" if asset == "btc" else "ETH-USD"
    df = get_predictions(asset)
    y = df["actual_return"].values

    decay_mae = np.abs(y - df["selected_pred"].values).mean()
    rsi_mae = np.abs(y - df["rsi_pred"].values).mean()
    t, p = stats.ttest_rel(np.abs(y - df["selected_pred"].values), np.abs(y - df["rsi_pred"].values))

    decay_correct = (np.sign(df["selected_pred"].values) == np.sign(y))
    rsi_correct = (np.sign(df["rsi_signal"].values) == np.sign(y))
    b, c, n_disc, mcnemar_p = mcnemar_test(decay_correct, rsi_correct)

    print(f"\n{'='*80}\n{label} vs RSI(14) mean-reversion -- {len(df)} test days\n{'='*80}")
    print(f"decay-selected MAE : {decay_mae:.6f}")
    print(f"RSI-based MAE       : {rsi_mae:.6f}")
    print(f"paired t-test       : t={t:.4f}, p={p:.4f}")
    print(f"\ndecay directional accuracy: {decay_correct.mean():.4f}")
    print(f"RSI directional accuracy  : {rsi_correct.mean():.4f}")
    print(f"McNemar (discordant days={n_disc}): p={mcnemar_p:.4f}")

    out = pd.DataFrame([{
        "asset": label, "n_days": len(df), "decay_MAE": decay_mae, "rsi_MAE": rsi_mae,
        "mae_t": t, "mae_p": p, "decay_dir_acc": decay_correct.mean(), "rsi_dir_acc": rsi_correct.mean(),
        "mcnemar_p": mcnemar_p,
    }])
    out_path = os.path.join(os.path.dirname(__file__), "results", f"rsi_comparison_{asset}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(asset)