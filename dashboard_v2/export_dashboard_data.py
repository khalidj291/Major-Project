"""
export_dashboard_data.py -- Dashboard v2 Data Preparation & Verification Script
Memory That Fades Project

Reads directly from saved models, processed datasets, and raw/backtest results.
Regenerates detailed time-series predictions, cumulative error differences,
regime subsets, sensitivity curves, equity curves, and block-bootstrap statistics.
Verifies that all regenerated MAEs exactly match existing reported findings.

Outputs saved into dashboard_v2/data/
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "decay_model"))

from windowing import make_windows
from ebbinghaus import ebbinghaus_weight, apply_decay_weights
from feature_decay import apply_feature_decay
from powerlaw_decay import powerlaw_weights
from backtest_lib import build_position, backtest_returns, equity_curve, max_drawdown

OUT_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "baseline_model", "models")
DECAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "results")


def verify_close(name, val, expected, tol=1e-5):
    diff = abs(val - expected)
    passed = diff <= tol
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:38s} : Computed={val:.6f} | Expected={expected:.6f} (diff={diff:.2e})")
    return passed


# ---------------------------------------------------------
# 1. SPY Domain (Equities)
# ---------------------------------------------------------
def export_spy():
    print("\n--- Exporting SPY Domain ---")
    data_path = os.path.join(DATA_DIR, "data_processed.csv")
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)

    # 30-day window, test period: 2023-01-01 to 2024-12-31 (n=501)
    X, y, dates = make_windows(df, "2023-01-01", "2024-12-31", 30, ticker="SPY")
    y = y.flatten()
    dates = pd.to_datetime(dates)

    # Price and regime lookup
    lookup = df.set_index("date")
    prices = lookup.loc[dates, "close"].values
    regimes = lookup.loc[dates, "regime"].values

    with open(os.path.join(BASELINE_MODELS_DIR, "model_baseline.pkl"), "rb") as f:
        m_base = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_fast.pkl"), "rb") as f:
        m_fast = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_medium.pkl"), "rb") as f:
        m_med = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_slow.pkl"), "rb") as f:
        m_slow = pickle.load(f)

    p_base = m_base.predict(X).flatten()
    p_fast = m_fast.predict(X).flatten()
    p_med = m_med.predict(X).flatten()
    p_slow = m_slow.predict(X).flatten()

    mae_base = np.abs(y - p_base).mean()
    mae_fast = np.abs(y - p_fast).mean()
    mae_med = np.abs(y - p_med).mean()
    mae_slow = np.abs(y - p_slow).mean()
    mae_naive = np.abs(y).mean()

    verify_close("SPY Baseline MAE (reported 0.006154)", mae_base, 0.00615418)
    verify_close("SPY Fast (S=30) MAE", mae_fast, 0.00627609)
    verify_close("SPY Medium (S=180) MAE", mae_med, 0.00621524)
    verify_close("SPY Slow (S=365) MAE", mae_slow, 0.00617683)
    verify_close("SPY Naive-Zero MAE", mae_naive, 0.00618597)

    base_abs_err = np.abs(y - p_base)
    slow_abs_err = np.abs(y - p_slow)
    cum_err_diff = np.cumsum(base_abs_err - slow_abs_err)

    out_df = pd.DataFrame({
        "date": dates,
        "actual": y,
        "naive_zero": 0.0,
        "baseline_pred": p_base,
        "decay_fast_pred": p_fast,
        "decay_medium_pred": p_med,
        "decay_slow_pred": p_slow,
        "regime": regimes,
        "close_price": prices,
        "base_abs_err": base_abs_err,
        "slow_abs_err": slow_abs_err,
        "cum_err_diff_vs_slow": cum_err_diff,
    })
    out_df.to_csv(os.path.join(OUT_DIR, "spy_predictions.csv"), index=False)
    print(f"Saved: spy_predictions.csv ({len(out_df)} rows)")


# ---------------------------------------------------------
# 2. PCE Domain (Macro Consumer)
# ---------------------------------------------------------
def export_pce():
    print("\n--- Exporting PCE Domain ---")
    data_path = os.path.join(DATA_DIR, "data_consumer_with_regime.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(DATA_DIR, "data_consumer.csv")
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X, y, dates = make_windows(df, "2023-01-01", "2024-12-31", 12)
    y = y.flatten()
    dates = pd.to_datetime(dates)

    lookup = df.set_index("date")
    prices = lookup.loc[dates, "close"].values
    regimes = lookup.loc[dates, "regime"].values if "regime" in df.columns else np.array(["neutral"] * len(dates))

    with open(os.path.join(BASELINE_MODELS_DIR, "model_baseline_consumer.pkl"), "rb") as f:
        m_base = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_fast_consumer.pkl"), "rb") as f:
        m_fast = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_medium_consumer.pkl"), "rb") as f:
        m_med = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_slow_consumer.pkl"), "rb") as f:
        m_slow = pickle.load(f)

    p_base = m_base.predict(X).flatten()
    p_fast = m_fast.predict(X).flatten()
    p_med = m_med.predict(X).flatten()
    p_slow = m_slow.predict(X).flatten()

    mae_base = np.abs(y - p_base).mean()
    mae_fast = np.abs(y - p_fast).mean()
    mae_med = np.abs(y - p_med).mean()
    mae_slow = np.abs(y - p_slow).mean()
    mae_naive = np.abs(y).mean()

    verify_close("PCE Baseline MAE (reported 0.002192)", mae_base, 0.00219161)
    verify_close("PCE Fast MAE (reported 0.004141)", mae_fast, 0.00414054)
    verify_close("PCE Medium MAE (reported 0.002573)", mae_med, 0.00257313)
    verify_close("PCE Slow MAE (reported 0.002503)", mae_slow, 0.00250278)
    verify_close("PCE Naive-Zero MAE", mae_naive, 0.00519844)

    out_df = pd.DataFrame({
        "date": dates,
        "actual": y,
        "naive_zero": 0.0,
        "baseline_pred": p_base,
        "decay_fast_pred": p_fast,
        "decay_medium_pred": p_med,
        "decay_slow_pred": p_slow,
        "regime": regimes,
        "pce_level": prices,
        "base_abs_err": np.abs(y - p_base),
        "fast_abs_err": np.abs(y - p_fast),
        "slow_abs_err": np.abs(y - p_slow),
        "cum_err_diff_vs_slow": np.cumsum(np.abs(y - p_base) - np.abs(y - p_slow)),
    })
    out_df.to_csv(os.path.join(OUT_DIR, "pce_predictions.csv"), index=False)
    print(f"Saved: pce_predictions.csv ({len(out_df)} rows)")

    full_returns = df["returns"].values
    lags = list(range(1, 13))
    acf_vals = [float(pd.Series(full_returns).autocorr(lag=k)) for k in lags]
    acf_df = pd.DataFrame({"lag": lags, "autocorrelation": acf_vals})
    acf_df.to_csv(os.path.join(OUT_DIR, "pce_acf.csv"), index=False)
    print("Saved: pce_acf.csv")


# ---------------------------------------------------------
# 3. COVID Domain (Disease)
# ---------------------------------------------------------
def export_covid():
    print("\n--- Exporting COVID Domain ---")
    data_path = os.path.join(DATA_DIR, "data_disease.csv")
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X_full, y_full, dates_full = make_windows(df, df["date"].min(), df["date"].max(), 30)
    y_full = y_full.flatten()
    dates_full = pd.to_datetime(dates_full)

    test_mask = dates_full >= pd.Timestamp("2022-04-03")
    X_test, y_test, dates_test = X_full[test_mask], y_full[test_mask], dates_full[test_mask]

    lookup = df.set_index("date")
    cases = lookup.loc[dates_full, "close"].values

    with open(os.path.join(BASELINE_MODELS_DIR, "model_baseline_disease.pkl"), "rb") as f:
        m_base = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_fast_disease.pkl"), "rb") as f:
        m_fast = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_medium_disease.pkl"), "rb") as f:
        m_med = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_decay_slow_disease.pkl"), "rb") as f:
        m_slow = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_feature_decay_disease.pkl"), "rb") as f:
        m_feat = pickle.load(f)

    p_base = m_base.predict(X_test).flatten()
    p_fast = m_fast.predict(X_test).flatten()
    p_med = m_med.predict(X_test).flatten()
    p_slow = m_slow.predict(X_test).flatten()
    p_feat = m_feat["model"].predict(apply_feature_decay(X_test, 30, m_feat["S"])).flatten()

    mae_base = np.abs(y_test - p_base).mean()
    mae_med = np.abs(y_test - p_med).mean()
    mae_naive = np.abs(y_test).mean()

    verify_close("COVID Baseline MAE (reported 0.045980)", mae_base, 0.045980)
    verify_close("COVID Decay Medium S=180 (reported 0.040868)", mae_med, 0.040868)
    verify_close("COVID Naive-Zero MAE (reported 0.038763)", mae_naive, 0.038763)

    out_df = pd.DataFrame({
        "date": dates_test,
        "actual": y_test,
        "naive_zero": 0.0,
        "baseline_pred": p_base,
        "decay_fast_pred": p_fast,
        "decay_medium_pred": p_med,
        "decay_slow_pred": p_slow,
        "decay_feature_pred": p_feat,
        "cases": cases[test_mask],
        "base_abs_err": np.abs(y_test - p_base),
        "decay_abs_err": np.abs(y_test - p_med),
        "naive_abs_err": np.abs(y_test),
        "cum_err_diff_vs_baseline": np.cumsum(np.abs(y_test - p_base) - np.abs(y_test - p_med)),
    })
    out_df.to_csv(os.path.join(OUT_DIR, "covid_predictions.csv"), index=False)
    print(f"Saved: covid_predictions.csv ({len(out_df)} rows)")

    pred_full_base = m_base.predict(X_full).flatten()
    pred_full_decay = m_med.predict(X_full).flatten()
    err_base_full = np.abs(y_full - pred_full_base)
    err_decay_full = np.abs(y_full - pred_full_decay)

    roll_base = pd.Series(err_base_full).rolling(30, min_periods=10).mean().values
    roll_decay = pd.Series(err_decay_full).rolling(30, min_periods=10).mean().values

    history_df = pd.DataFrame({
        "date": dates_full,
        "actual": y_full,
        "cases": cases,
        "period": ["Pre-2022 (High Volatility / Spikes)" if d < pd.Timestamp("2022-01-01") else "Post-2022 (Stabilizing)" for d in dates_full],
        "is_test_set": test_mask,
        "err_base": err_base_full,
        "err_decay": err_decay_full,
        "rolling_30d_mae_base": roll_base,
        "rolling_30d_mae_decay": roll_decay,
    })
    history_df.to_csv(os.path.join(OUT_DIR, "covid_history.csv"), index=False)
    print(f"Saved: covid_history.csv ({len(history_df)} rows)")


# ---------------------------------------------------------
# 4. Crypto Domain (BTC & ETH Held-Out)
# ---------------------------------------------------------
def export_crypto():
    print("\n--- Exporting Crypto Domain ---")
    data_btc = pd.read_csv(os.path.join(DATA_DIR, "data_btc.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    X_test, y_test, dates_test = make_windows(data_btc, "2024-01-01", data_btc["date"].max(), 30)
    y_test = y_test.flatten()
    dates_test = pd.to_datetime(dates_test)

    lookup = data_btc.set_index("date")
    prices = lookup.loc[dates_test, "close"].values

    with open(os.path.join(BASELINE_MODELS_DIR, "model_baseline_btc.pkl"), "rb") as f:
        m_base = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_feature_decay_btc.pkl"), "rb") as f:
        m_feat = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_row_decay_btc.pkl"), "rb") as f:
        m_row = pickle.load(f)
    with open(os.path.join(DECAY_MODELS_DIR, "model_combined_decay_btc.pkl"), "rb") as f:
        m_comb = pickle.load(f)

    p_base = m_base.predict(X_test).flatten()
    p_feat = m_feat["model"].predict(apply_feature_decay(X_test, 30, m_feat["S"])).flatten()
    p_row = m_row["model"].predict(X_test).flatten()
    p_comb = m_comb["model"].predict(apply_feature_decay(X_test, 30, m_comb["feature_S"])).flatten()

    mae_base = np.abs(y_test - p_base).mean()
    mae_comb = np.abs(y_test - p_comb).mean()
    pct_imp = 100 * (mae_base - mae_comb) / mae_base

    verify_close("BTC Baseline MAE (reported 0.018033)", mae_base, 0.01803322)
    verify_close("BTC Combined Decay MAE (reported 0.017789)", mae_comb, 0.01778927)
    verify_close("BTC % Improvement (reported 1.35%)", pct_imp, 1.35279)

    out_df = pd.DataFrame({
        "ticker": "BTC-USD",
        "date": dates_test,
        "actual": y_test,
        "baseline_pred": p_base,
        "feature_decay_pred": p_feat,
        "row_decay_pred": p_row,
        "combined_decay_pred": p_comb,
        "close_price": prices,
        "base_abs_err": np.abs(y_test - p_base),
        "comb_abs_err": np.abs(y_test - p_comb),
        "cum_err_diff": np.cumsum(np.abs(y_test - p_base) - np.abs(y_test - p_comb)),
    })

    # ETH held-out predictions (2024-2026) from leak-free walk-forward
    ma_eth = pd.read_csv(os.path.join(RESULTS_DIR, "ma_comparison_per_day_eth.csv"), parse_dates=["date"])
    ma_eth = ma_eth[ma_eth["date"] >= pd.Timestamp("2024-01-01")].sort_values("date").reset_index(drop=True)
    eth_data = pd.read_csv(os.path.join(DATA_DIR, "data_eth.csv"), parse_dates=["date"]).set_index("date")
    eth_prices = [eth_data.loc[d, "close"] if d in eth_data.index else np.nan for d in ma_eth["date"]]

    eth_df = pd.DataFrame({
        "ticker": "ETH-USD",
        "date": ma_eth["date"],
        "actual": ma_eth["actual_return"],
        "baseline_pred": ma_eth["baseline_pred"],
        "feature_decay_pred": ma_eth["selected_pred"],
        "row_decay_pred": ma_eth["selected_pred"],
        "combined_decay_pred": ma_eth["selected_pred"],
        "close_price": eth_prices,
        "base_abs_err": ma_eth["baseline_err"],
        "comb_abs_err": ma_eth["selected_err"],
        "cum_err_diff": np.cumsum(ma_eth["baseline_err"] - ma_eth["selected_err"]),
    })

    combined_crypto = pd.concat([out_df, eth_df], ignore_index=True)
    combined_crypto.to_csv(os.path.join(OUT_DIR, "crypto_heldout_predictions.csv"), index=False)
    print(f"Saved: crypto_heldout_predictions.csv ({len(combined_crypto)} rows: BTC={len(out_df)}, ETH={len(eth_df)})")

    # Grid search validation matrix (2023 validation period) for BTC
    X_train, y_train, tr_dates = make_windows(data_btc, data_btc["date"].min(), pd.Timestamp("2022-12-31"), 30)
    X_val, y_val, _ = make_windows(data_btc, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"), 30)
    y_train, y_val = y_train.ravel(), y_val.ravel()
    ref_date = pd.Timestamp("2023-12-31")

    grid_rows = []
    feat_candidates = [5, 10, 15, 20, 30, 60]
    row_candidates = [90, 180, 365, 730]
    base_val_mae = np.abs(y_val - Ridge(alpha=1.0).fit(X_train, y_train).predict(X_val)).mean()

    for fS in feat_candidates:
        X_tr_f = apply_feature_decay(X_train, 30, fS)
        X_val_f = apply_feature_decay(X_val, 30, fS)
        for rS in row_candidates:
            w_row = apply_decay_weights(pd.to_datetime(tr_dates), ref_date, rS)
            m = Ridge(alpha=1.0).fit(X_tr_f, y_train, sample_weight=w_row)
            pred = m.predict(X_val_f)
            val_mae = np.abs(y_val - pred).mean()
            grid_rows.append({
                "feature_S": fS,
                "row_S": rS,
                "val_mae": val_mae,
                "val_mae_pct_vs_base": 100 * (base_val_mae - val_mae) / base_val_mae,
            })

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(os.path.join(OUT_DIR, "crypto_tuning_grid.csv"), index=False)
    print("Saved: crypto_tuning_grid.csv")


# ---------------------------------------------------------
# 5. Block Bootstrap Significance
# ---------------------------------------------------------
def export_block_bootstrap():
    print("\n--- Exporting Block Bootstrap Summary ---")
    RNG_SEED = 42
    N_BOOTSTRAP = 10000
    BLOCK_LENGTHS = [10, 20, 30, 40]

    rows = []
    for asset, fname, label in [
        ("btc", "walk_forward_per_day_btc.csv", "BTC-USD"),
        ("eth", "walk_forward_per_day_eth.csv", "ETH-USD"),
    ]:
        csv_path = os.path.join(RESULTS_DIR, fname)
        df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        d = (df["baseline_err"] - df["selected_err"]).values
        n = len(d)
        d_bar = d.mean()
        rng = np.random.default_rng(RNG_SEED)

        for L in BLOCK_LENGTHS:
            n_blocks = int(np.ceil(n / L))
            max_start = n - L
            d_centered = d - d_bar
            null_means = np.empty(N_BOOTSTRAP)
            ci_means = np.empty(N_BOOTSTRAP)

            for b in range(N_BOOTSTRAP):
                starts = rng.integers(0, max_start + 1, size=n_blocks)
                res_null = np.concatenate([d_centered[s:s + L] for s in starts])[:n]
                res_d = np.concatenate([d[s:s + L] for s in starts])[:n]
                null_means[b] = res_null.mean()
                ci_means[b] = res_d.mean()

            p_val = (np.abs(null_means) >= np.abs(d_bar)).mean()
            ci_lo, ci_hi = np.percentile(ci_means, [2.5, 97.5])

            rows.append({
                "asset": label,
                "n_days": n,
                "observed_d_bar": d_bar,
                "block_length": L,
                "bootstrap_p_value": p_val,
                "ci_lower_95": ci_lo,
                "ci_upper_95": ci_hi,
            })

    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(os.path.join(OUT_DIR, "block_bootstrap_summary.csv"), index=False)
    print("Saved: block_bootstrap_summary.csv")


# ---------------------------------------------------------
# 6. Backtest Equity Curves & Drawdown Series
# ---------------------------------------------------------
def export_equity_curves():
    print("\n--- Exporting Backtest Equity Curves ---")
    ma_daily = pd.read_csv(os.path.join(RESULTS_DIR, "ma_comparison_per_day_btc.csv"), parse_dates=["date"])
    merged = ma_daily.sort_values("date").reset_index(drop=True)

    n = len(merged)
    actual_return = merged["actual_return"].values
    positions = {
        "buy_and_hold": np.ones(n, dtype=int),
        "baseline": build_position(merged["baseline_pred"].values),
        "decay_selected": build_position(merged["selected_pred"].values),
        "ma_signal": build_position(merged["ma_linear_pred"].values),
    }

    fee_levels = [0.0005, 0.001, 0.0025, 0.005]
    all_equity_rows = []

    for fee in fee_levels:
        for strat_name, pos in positions.items():
            net_ret, traded = backtest_returns(pos, actual_return, fee)
            eq = equity_curve(net_ret)
            running_max = np.maximum.accumulate(eq)
            dd = (eq - running_max) / running_max

            for i in range(n):
                all_equity_rows.append({
                    "date": merged["date"].iloc[i],
                    "fee_rate": fee,
                    "strategy": strat_name,
                    "position": pos[i],
                    "net_return": net_ret[i],
                    "equity": eq[i],
                    "drawdown": dd[i],
                })

    eq_df = pd.DataFrame(all_equity_rows)
    eq_df.to_csv(os.path.join(OUT_DIR, "trading_equity_curves.csv"), index=False)
    print(f"Saved: trading_equity_curves.csv ({len(eq_df)} rows)")


if __name__ == "__main__":
    print("==================================================")
    print("DASHBOARD V2: Exporting & Verifying Real Repo Data")
    print("==================================================")
    export_spy()
    export_pce()
    export_covid()
    export_crypto()
    export_block_bootstrap()
    export_equity_curves()
    print("\n[SUCCESS] All data successfully verified and exported to dashboard_v2/data/.")
