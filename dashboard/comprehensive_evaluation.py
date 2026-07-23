"""
Memory That Fades — Comprehensive Evaluation
Person 3 — Day 4 core deliverable

Fixes applied during integration:
  - Paths now point at the real project structure (baseline_model/models,
    decay_model/models, data/) instead of a self-contained local copy that
    had drifted out of sync with the rest of the team's fixes -- this is
    why n_volatile was showing 44 instead of the correct 62, and baseline
    MAE was showing 0.006138 instead of the correct 0.006154.
  - Same one-day windowing offset bug fixed everywhere else in the project
    was present here too (target was returns[i+1] instead of returns[i]) --
    now uses the shared, tested windowing.py.
  - Consumer domain is now actually computed, not hardcoded as "pending".
  - Disease domain added.

Produces:
  dashboard/results/regime_analysis_financial_v2.csv
  dashboard/results/comprehensive_evaluation.csv
  dashboard/results/final_comparison_table.csv
  dashboard/results/final_comparison_table.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "decay_model"))
from windowing import make_windows

BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "baseline_model", "models")
DECAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def _load_models(suffix):
    files = {
        "baseline": os.path.join(BASELINE_MODELS_DIR, f"model_baseline{suffix}.pkl"),
        "fast": os.path.join(DECAY_MODELS_DIR, f"model_decay_fast{suffix}.pkl"),
        "medium": os.path.join(DECAY_MODELS_DIR, f"model_decay_medium{suffix}.pkl"),
        "slow": os.path.join(DECAY_MODELS_DIR, f"model_decay_slow{suffix}.pkl"),
    }
    models = {}
    for name, path in files.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


def mae_rmse(actual, pred):
    err = actual - pred
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


def evaluate_financial():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_processed.csv"), parse_dates=["date"])
    df = df[df["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    X, y, dates = make_windows(df, "2023-01-01", "2024-12-31", 30, ticker="SPY")
    regimes = df.set_index("date").loc[pd.to_datetime(dates), "regime"].values
    models = _load_models("")
    preds = {name: m.predict(X).flatten() for name, m in models.items()}
    return y.flatten(), preds, regimes


def evaluate_consumer():
    path = os.path.join(DATA_DIR, "data_consumer_with_regime.csv")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, "data_consumer.csv")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X, y, dates = make_windows(df, "2023-01-01", "2024-12-31", 12)
    has_regime = "regime" in df.columns
    regimes = df.set_index("date").loc[pd.to_datetime(dates), "regime"].values if has_regime else None
    models = _load_models("_consumer")
    preds = {name: m.predict(X).flatten() for name, m in models.items()}
    return y.flatten(), preds, regimes


def evaluate_disease():
    df = pd.read_csv(os.path.join(DATA_DIR, "data_disease.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    X, y, dates = make_windows(df, "2022-04-03", df["date"].max(), 30)
    models = _load_models("_disease")
    preds = {name: m.predict(X).flatten() for name, m in models.items()}
    return y.flatten(), preds, None


def regime_split_table(y, preds, regimes, label):
    rows = []
    for name, p in preds.items():
        overall_mae, overall_rmse = mae_rmse(y, p)
        row = {"domain": label, "model": name, "overall_MAE": overall_mae, "overall_RMSE": overall_rmse}
        if regimes is not None:
            vol_mask, stab_mask = regimes == "volatile", regimes == "stable"
            vol_mae, vol_rmse = mae_rmse(y[vol_mask], p[vol_mask])
            stab_mae, stab_rmse = mae_rmse(y[stab_mask], p[stab_mask])
            row.update({
                "volatile_MAE": vol_mae, "volatile_RMSE": vol_rmse, "n_volatile": int(vol_mask.sum()),
                "stable_MAE": stab_mae, "stable_RMSE": stab_rmse, "n_stable": int(stab_mask.sum()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def significance_table(y, preds, regimes, label):
    baseline_preds = preds["baseline"]
    rows = []
    regime_splits = [("overall", np.ones(len(y), dtype=bool))]
    if regimes is not None:
        regime_splits += [("volatile", regimes == "volatile"), ("stable", regimes == "stable")]
    for name in ["fast", "medium", "slow"]:
        for regime_name, mask in regime_splits:
            base_err = np.abs(y[mask] - baseline_preds[mask])
            dec_err = np.abs(y[mask] - preds[name][mask])
            t_stat, p_val = stats.ttest_rel(base_err, dec_err)
            pct = (base_err.mean() - dec_err.mean()) / base_err.mean() * 100
            rows.append({
                "domain": label, "model": name, "regime": regime_name, "n": int(mask.sum()),
                "baseline_MAE": base_err.mean(), "model_MAE": dec_err.mean(),
                "pct_improvement_vs_baseline": pct, "t_stat": t_stat, "p_value": p_val,
                "significant_95pct": bool(p_val < 0.05),
            })
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    y_fin, preds_fin, regimes_fin = evaluate_financial()
    y_con, preds_con, regimes_con = evaluate_consumer()
    y_dis, preds_dis, _ = evaluate_disease()

    regime_df = pd.concat([
        regime_split_table(y_fin, preds_fin, regimes_fin, "financial"),
        regime_split_table(y_con, preds_con, regimes_con, "consumer"),
        regime_split_table(y_dis, preds_dis, None, "disease"),
    ], ignore_index=True)
    regime_df.to_csv(os.path.join(RESULTS_DIR, "regime_analysis_financial_v2.csv"), index=False)
    print("Saved dashboard/results/regime_analysis_financial_v2.csv")
    print(regime_df.to_string(index=False))

    sig_df = pd.concat([
        significance_table(y_fin, preds_fin, regimes_fin, "financial"),
        significance_table(y_con, preds_con, regimes_con, "consumer"),
        significance_table(y_dis, preds_dis, None, "disease"),
    ], ignore_index=True)
    sig_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_evaluation.csv"), index=False)
    print("\nSaved dashboard/results/comprehensive_evaluation.csv")

    # --- final presentation-ready table, all 3 domains, all real numbers ---
    def get_row(df, domain, model):
        r = df[(df["domain"] == domain) & (df["model"] == model)]
        return r.iloc[0] if len(r) else None

    final_rows = []
    for name in ["baseline", "fast", "medium", "slow"]:
        fin = get_row(regime_df, "financial", name)
        con = get_row(regime_df, "consumer", name)
        dis = get_row(regime_df, "disease", name)
        final_rows.append({
            "Model": name.capitalize(),
            "Financial MAE": round(fin["overall_MAE"], 5),
            "Financial Volatile MAE": round(fin["volatile_MAE"], 5),
            "Financial Stable MAE": round(fin["stable_MAE"], 5),
            "Consumer MAE": round(con["overall_MAE"], 5),
            "Consumer Volatile MAE": round(con["volatile_MAE"], 5) if "volatile_MAE" in con else "n/a",
            "Consumer Stable MAE": round(con["stable_MAE"], 5) if "stable_MAE" in con else "n/a",
            "Disease MAE": round(dis["overall_MAE"], 5),
        })
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(RESULTS_DIR, "final_comparison_table.csv"), index=False)
    print("\nSaved dashboard/results/final_comparison_table.csv")
    print(final_df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 2.2))
    ax.axis("off")
    tbl = ax.table(cellText=final_df.values, colLabels=final_df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    best_idx = final_df["Disease MAE"].astype(float).idxmin()
    for col in range(len(final_df.columns)):
        tbl[(best_idx + 1, col)].set_facecolor("#d4edda")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "final_comparison_table.png"), dpi=150, bbox_inches="tight")
    print("Saved dashboard/results/final_comparison_table.png")

    return regime_df, sig_df, final_df


if __name__ == "__main__":
    main()