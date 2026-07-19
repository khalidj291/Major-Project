import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from ebbinghaus import ebbinghaus_weight
from regime_eval import evaluate_by_regime

# Load regime-labeled data (same prices as data_processed.csv, confirmed identical,
# just with 'regime' and 'rolling_vol' columns added by Person 1)
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_regime.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
WINDOW = 30


def make_windows(full_df, start_date, end_date, window):
    full_df = full_df.sort_values("date").reset_index(drop=True)
    returns = full_df["returns"].values
    dates = full_df["date"].values
    regimes = full_df["regime"].values
    X, y, sample_dates, sample_regimes = [], [], [], []
    for i in range(window, len(returns)):
        target_date = dates[i]
        if start_date <= pd.Timestamp(target_date) <= end_date:
            X.append(returns[i - window:i])
            y.append(returns[i])
            sample_dates.append(dates[i])
            sample_regimes.append(regimes[i])  # regime of the day being PREDICTED
    return np.array(X), np.array(y).reshape(-1, 1), np.array(sample_dates), np.array(sample_regimes)


train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
reference_date = train_df["date"].max()

X_train, y_train, train_sample_dates, _ = make_windows(df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW)
X_test, y_test, test_sample_dates, test_regimes = make_windows(df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW)

print(f"Test set size: {len(X_test)}")
print(f"Test regime breakdown: {pd.Series(test_regimes).value_counts().to_dict()}")

# --- Load/train all 4 models ---
with open(os.path.join(PROJECT_ROOT, "baseline_model", "models", "model_baseline.pkl"), "rb") as f:
    baseline_model = pickle.load(f)

decay_models = {}
for name, S in [("fast", 30), ("medium", 180), ("slow", 365)]:
    weights = ebbinghaus_weight(train_sample_dates, reference_date, S)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=weights)
    decay_models[name] = model

naive_preds = np.zeros_like(y_test)

# --- Run regime evaluation for each model ---
all_results = {}
for name, preds in [
    ("naive_zero", naive_preds),
    ("baseline", baseline_model.predict(X_test)),
    ("fast", decay_models["fast"].predict(X_test)),
    ("medium", decay_models["medium"].predict(X_test)),
    ("slow", decay_models["slow"].predict(X_test)),
]:
    result = evaluate_by_regime(preds, y_test, test_regimes)
    all_results[name] = result
    print(f"\n{name}:")
    print(f"  Volatile MAE: {result['volatile_MAE']:.6f} (n={result['n_volatile']})")
    print(f"  Stable MAE:   {result['stable_MAE']:.6f} (n={result['n_stable']})")
    print(f"  Overall MAE:  {result['overall_MAE']:.6f} (n={result['n_overall']})")

# --- Save results table ---
rows = []
for name, r in all_results.items():
    rows.append({
        "model": name,
        "volatile_MAE": r["volatile_MAE"], "volatile_RMSE": r["volatile_RMSE"], "n_volatile": r["n_volatile"],
        "stable_MAE": r["stable_MAE"], "stable_RMSE": r["stable_RMSE"], "n_stable": r["n_stable"],
        "overall_MAE": r["overall_MAE"], "overall_RMSE": r["overall_RMSE"], "n_overall": r["n_overall"],
    })
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(PROJECT_ROOT, "decay_model", "results", "regime_analysis_financial.csv"), index=False)
print("\nSaved results/regime_analysis_financial.csv")

# --- Headline question: which model wins in volatile, which wins in stable? ---
volatile_winner = results_df.loc[results_df["volatile_MAE"].idxmin(), "model"]
stable_winner = results_df.loc[results_df["stable_MAE"].idxmin(), "model"]
print(f"\nWinner in VOLATILE periods: {volatile_winner}")
print(f"Winner in STABLE periods: {stable_winner}")

# --- Grouped bar chart ---
models_order = ["naive_zero", "baseline", "fast", "medium", "slow"]
volatile_maes = [all_results[m]["volatile_MAE"] for m in models_order]
stable_maes = [all_results[m]["stable_MAE"] for m in models_order]

x = np.arange(len(models_order))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - width/2, volatile_maes, width, label="Volatile", color="#e63946")
ax.bar(x + width/2, stable_maes, width, label="Stable", color="#2a9d8f")
ax.set_xlabel("Model")
ax.set_ylabel("MAE (lower is better)")
ax.set_title("Regime Sensitivity — Financial Domain (Headline Finding)")
ax.set_xticks(x)
ax.set_xticklabels(models_order)
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "decay_model", "results", "regime_sensitivity_chart.png"), dpi=150)
print("Saved results/regime_sensitivity_chart.png")