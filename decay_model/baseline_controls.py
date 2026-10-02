"""
baseline_controls.py -- "Is decay actually special, or is it just shrinkage?"

THE QUESTION
------------
Decay beats the plain Ridge baseline. But a reviewer will ask: would ANY
simple fix to that baseline do as well? Three suspects:

  1. Predict-zero / historical-mean  (returns are nearly unpredictable, so
     "predict nothing" may already be hard to beat on MAE).
  2. Ridge with a TUNED penalty. Scaling feature columns by decay weights
     (X * w, w < 1) is mathematically the same kind of thing as penalising
     those features harder -- i.e. decay may just be extra regularisation.
  3. Rolling window (only train on the last L days) -- the standard way
     practitioners "forget" old data.

Every competitor is tuned ONLY on the same validation slice used by the
walk-forward pipeline, then scored on the same 13 test windows. The
existing decay predictions are NOT re-run: we read selected_err from
walk_forward_per_day_{asset}.csv so it is the exact same decay result.

STATS
-----
Paired t-test (as elsewhere) PLUS a Diebold-Mariano style test using
Newey-West (HAC) standard errors, which allows autocorrelated daily errors.

Run: python baseline_controls.py btc
     python baseline_controls.py eth
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows

WINDOW = 30
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4]
ROLLING_LENGTHS = [90, 180, 365, 730]


def newey_west_dm(d, lag=None):
    """DM test on a loss-differential series d (mean>0 means model A worse).
    Returns (mean, t_stat, p_value) using Newey-West variance."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    dc = d - d.mean()
    var = np.dot(dc, dc) / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1.0)
        var += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    se = np.sqrt(var / n)
    t = d.mean() / se
    p = 2 * (1 - stats.norm.cdf(abs(t)))
    return d.mean(), t, p


def run(asset):
    label = "BTC-USD" if asset == "btc" else "ETH-USD"
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "..", "data", f"data_{asset}.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "ticker" in df.columns:
        raise ValueError("unexpected ticker column")

    per_day_existing = pd.read_csv(os.path.join(base, "results", f"walk_forward_per_day_{asset}.csv"),
                                   parse_dates=["date"])
    existing = per_day_existing.set_index("date")

    rows, win_rows = [], []
    test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")

    for ts in test_starts:
        test_start = ts
        test_end = min(ts + pd.DateOffset(months=6) - pd.DateOffset(days=1), df["date"].max())
        val_end = test_start - pd.DateOffset(days=1)
        val_start = val_end - pd.DateOffset(months=6) + pd.DateOffset(days=1)
        train_end = val_start - pd.DateOffset(days=1)
        train_start = df["date"].min()
        if test_start > df["date"].max():
            continue

        X_tr, y_tr, d_tr = make_windows(df, train_start, train_end, WINDOW)
        X_va, y_va, d_va = make_windows(df, val_start, val_end, WINDOW)
        X_te, y_te, d_te = make_windows(df, test_start, test_end, WINDOW)
        if len(X_tr) < 200 or len(X_va) < 20 or len(X_te) < 5:
            continue
        y_va_f, y_te_f = y_va.flatten(), y_te.flatten()

        preds = {}
        # 1a. predict zero
        preds["zero"] = np.zeros(len(y_te_f))
        # 1b. historical mean of training targets
        preds["hist_mean"] = np.full(len(y_te_f), y_tr.mean())

        # 2. Ridge with penalty tuned on validation MAE
        best = min(RIDGE_ALPHA_GRID,
                   key=lambda a: np.abs(y_va_f - Ridge(alpha=a).fit(X_tr, y_tr).predict(X_va).flatten()).mean())
        preds["ridge_tuned_alpha"] = Ridge(alpha=best).fit(X_tr, y_tr).predict(X_te).flatten()

        # 3. rolling window, length tuned on validation MAE (alpha fixed at 1.0)
        def fit_roll(L):
            return Ridge(alpha=1.0).fit(X_tr[-L:], y_tr[-L:])
        bestL = min(ROLLING_LENGTHS,
                    key=lambda L: np.abs(y_va_f - fit_roll(L).predict(X_va).flatten()).mean())
        preds["rolling_window"] = fit_roll(bestL).predict(X_te).flatten()

        # 4. rolling window AND tuned alpha (strongest "ordinary" competitor)
        best_pair = min(((a, L) for a in RIDGE_ALPHA_GRID for L in ROLLING_LENGTHS),
                        key=lambda p: np.abs(y_va_f - Ridge(alpha=p[0]).fit(X_tr[-p[1]:], y_tr[-p[1]:])
                                             .predict(X_va).flatten()).mean())
        preds["rolling_plus_alpha"] = Ridge(alpha=best_pair[0]).fit(
            X_tr[-best_pair[1]:], y_tr[-best_pair[1]:]).predict(X_te).flatten()

        w = {"test_window_start": str(test_start.date()), "tuned_alpha": best,
             "tuned_L": bestL, "pair": str(best_pair)}
        for i, d in enumerate(d_te):
            dd = pd.Timestamp(d)
            row = {"date": dd, "actual": y_te_f[i]}
            for k, p in preds.items():
                row[f"{k}_err"] = abs(y_te_f[i] - p[i])
                row[f"{k}_sq"] = (y_te_f[i] - p[i]) ** 2
            rows.append(row)
        win_rows.append(w)

    pdf = pd.DataFrame(rows).set_index("date")
    # join the EXISTING decay + baseline predictions (read, not recomputed)
    pdf = pdf.join(existing[["baseline_err", "selected_err"]], how="inner").reset_index()
    wdf = pd.DataFrame(win_rows)

    results_dir = os.path.join(base, "results")
    pdf.to_csv(os.path.join(results_dir, f"baseline_controls_per_day_{asset}.csv"), index=False)
    wdf.to_csv(os.path.join(results_dir, f"baseline_controls_windows_{asset}.csv"), index=False)

    print(f"\n{'=' * 90}\n{label} -- is decay special, or just shrinkage? (n={len(pdf)} test days)\n{'=' * 90}")
    print("Tuned hyper-parameters per window (picked on validation only):")
    print(wdf.to_string(index=False))

    names = ["baseline", "zero", "hist_mean", "ridge_tuned_alpha", "rolling_window",
             "rolling_plus_alpha", "selected"]
    print("\nPooled MAE (lower is better):")
    for n in names:
        print(f"  {n:<20}: {pdf[n + '_err'].mean():.6f}")

    print("\nDecay (selected) vs each competitor.  diff = competitor_err - decay_err")
    print("  diff > 0  => decay is BETTER.   NW = Newey-West/Diebold-Mariano p-value")
    print(f"  {'competitor':<20}{'mean diff':>12}{'t-test p':>11}{'NW p':>9}   verdict")
    for n in names[:-1]:
        d = pdf[n + "_err"].values - pdf["selected_err"].values
        _, p_t = stats.ttest_1samp(d, 0.0)
        _, _, p_nw = newey_west_dm(d)
        if p_nw < 0.05:
            v = "decay significantly BETTER" if d.mean() > 0 else "decay significantly WORSE"
        else:
            v = "statistical tie"
        print(f"  {n:<20}{d.mean():>12.6f}{p_t:>11.4f}{p_nw:>9.4f}   {v}")

    # out-of-sample R^2 vs predict-zero (squared error; >0 means beats zero)
    sse_zero = pdf["zero_sq"].sum()
    print("\nOut-of-sample R^2 vs predict-zero (MSE-based; >0 = beats 'predict nothing'):")
    for n in ["hist_mean", "ridge_tuned_alpha", "rolling_window", "rolling_plus_alpha"]:
        print(f"  {n:<20}: {1 - pdf[n + '_sq'].sum() / sse_zero:+.5f}")
    print("  (baseline & decay squared errors not stored in the old per-day files; "
          "MAE comparison above covers them)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")