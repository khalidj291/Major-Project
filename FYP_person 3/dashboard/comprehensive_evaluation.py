"""
Memory That Fades — Comprehensive Evaluation (Financial / SPY domain)
Person 3 — Day 4 core deliverable

Self-contained: uses data_processed.csv's built-in 'regime' column directly,
so it does not depend on the still-unresolved data_regime.csv question.

Produces:
  results/regime_analysis_financial_v2.csv   (regime-split MAE/RMSE per model)
  results/comprehensive_evaluation.csv        (full evaluation incl. t-tests)
  results/final_comparison_table.csv          (presentation-ready table, financial columns)
  results/final_comparison_table.png
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "data_processed.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TICKER = "SPY"
WINDOW = 30
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-12-31")

MODEL_FILES = {
    "baseline": "model_baseline.pkl",
    "fast": "model_decay_fast.pkl",
    "medium": "model_decay_medium.pkl",
    "slow": "model_decay_slow.pkl",
}


def load_ticker_df(ticker=TICKER):
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    return df


def build_test_windows(df, window=WINDOW):
    """Build (X, y, regime) test-period samples, target date in [TEST_START, TEST_END]."""
    returns = df["returns"].values
    dates = df["date"].values
    regimes = df["regime"].values
    X, y, sample_dates, sample_regimes = [], [], [], []
    for i in range(window, len(returns) - 1):
        target_date = pd.Timestamp(dates[i + 1])
        if TEST_START <= target_date <= TEST_END:
            X.append(returns[i - window:i])
            y.append(returns[i + 1])
            sample_dates.append(target_date)
            sample_regimes.append(regimes[i + 1])
    return (np.array(X), np.array(y), pd.to_datetime(sample_dates), np.array(sample_regimes))


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


def main():
    df = load_ticker_df()
    X_test, y_test, test_dates, test_regimes = build_test_windows(df)
    print(f"Test samples: {len(X_test)}")
    print(f"Regime breakdown: {pd.Series(test_regimes).value_counts().to_dict()}")

    volatile_mask = test_regimes == "volatile"
    stable_mask = test_regimes == "stable"

    predictions = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        with open(path, "rb") as f:
            model = pickle.load(f)
        predictions[name] = model.predict(X_test).flatten()

    # --- regime-split MAE/RMSE per model ---
    rows = []
    for name, preds in predictions.items():
        overall_mae, overall_rmse = mae_rmse(y_test, preds)
        vol_mae, vol_rmse = mae_rmse(y_test[volatile_mask], preds[volatile_mask])
        stab_mae, stab_rmse = mae_rmse(y_test[stable_mask], preds[stable_mask])
        rows.append({
            "model": name,
            "overall_MAE": overall_mae, "overall_RMSE": overall_rmse,
            "volatile_MAE": vol_mae, "volatile_RMSE": vol_rmse, "n_volatile": int(volatile_mask.sum()),
            "stable_MAE": stab_mae, "stable_RMSE": stab_rmse, "n_stable": int(stable_mask.sum()),
        })
    regime_df = pd.DataFrame(rows)
    regime_df.to_csv(os.path.join(RESULTS_DIR, "regime_analysis_financial_v2.csv"), index=False)
    print("\nSaved results/regime_analysis_financial_v2.csv")
    print(regime_df.to_string(index=False))

    volatile_winner = regime_df.loc[regime_df["volatile_MAE"].idxmin(), "model"]
    stable_winner = regime_df.loc[regime_df["stable_MAE"].idxmin(), "model"]
    print(f"\nWinner in VOLATILE periods: {volatile_winner}")
    print(f"Winner in STABLE periods:   {stable_winner}")

    # --- statistical significance: each decay model vs baseline, per regime ---
    baseline_preds = predictions["baseline"]
    sig_rows = []
    for name in ["fast", "medium", "slow"]:
        preds = predictions[name]
        for regime_name, mask in [("overall", np.ones(len(y_test), dtype=bool)),
                                   ("volatile", volatile_mask),
                                   ("stable", stable_mask)]:
            base_err = np.abs(y_test[mask] - baseline_preds[mask])
            dec_err = np.abs(y_test[mask] - preds[mask])
            t_stat, p_val = stats.ttest_rel(base_err, dec_err)
            pct_improvement = (base_err.mean() - dec_err.mean()) / base_err.mean() * 100
            sig_rows.append({
                "model": name, "regime": regime_name, "n": int(mask.sum()),
                "baseline_MAE": base_err.mean(), "model_MAE": dec_err.mean(),
                "pct_improvement_vs_baseline": pct_improvement,
                "t_stat": t_stat, "p_value": p_val,
                "significant_95pct": bool(p_val < 0.05),
            })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_evaluation.csv"), index=False)
    print("\nSaved results/comprehensive_evaluation.csv")
    print(sig_df.to_string(index=False))

    # --- final presentation-ready table (financial columns only; consumer pending) ---
    final_rows = []
    for name in ["baseline", "fast", "medium", "slow"]:
        r = regime_df[regime_df["model"] == name].iloc[0]
        best_regime = "Volatile" if r["volatile_MAE"] < r["stable_MAE"] else "Stable"
        final_rows.append({
            "Model": name.capitalize(),
            "Financial MAE": round(r["overall_MAE"], 5),
            "Financial Volatile MAE": round(r["volatile_MAE"], 5),
            "Financial Stable MAE": round(r["stable_MAE"], 5),
            "Consumer MAE": "pending",
            "Consumer Volatile MAE": "pending",
            "Consumer Stable MAE": "pending",
            "Best Regime (Financial)": best_regime,
        })
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(RESULTS_DIR, "final_comparison_table.csv"), index=False)
    print("\nSaved results/final_comparison_table.csv")

    # render as PNG for slide deck
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.axis("off")
    tbl = ax.table(cellText=final_df.values, colLabels=final_df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    # highlight best financial-volatile-MAE row
    best_idx = final_df["Financial Volatile MAE"].astype(float).idxmin()
    for col in range(len(final_df.columns)):
        tbl[(best_idx + 1, col)].set_facecolor("#d4edda")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_comparison_table.png"), dpi=150, bbox_inches="tight")
    print("Saved results/final_comparison_table.png")

    return regime_df, sig_df, final_df


if __name__ == "__main__":
    main()
