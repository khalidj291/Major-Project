import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)
from ebbinghaus import ebbinghaus_weight

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
WINDOW = 30


def make_windows(full_df, start_date, end_date, window):
    full_df = full_df.sort_values("date").reset_index(drop=True)
    returns = full_df["returns"].values
    dates = full_df["date"].values
    X, y, sample_dates = [], [], []
    for i in range(window, len(returns)):
        target_date = dates[i]
        if start_date <= pd.Timestamp(target_date) <= end_date:
            X.append(returns[i - window:i])
            y.append(returns[i])
            sample_dates.append(dates[i])
    return np.array(X), np.array(y).reshape(-1, 1), np.array(sample_dates)


train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)
reference_date = train_df["date"].max()

X_train, y_train, train_sample_dates = make_windows(df, pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31"), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"), WINDOW)

# Original three + five additional decay rates requested by the role doc
all_S_values = [14, 30, 60, 90, 180, 270, 365, 730]

results = []
for S in all_S_values:
    weights = ebbinghaus_weight(train_sample_dates, reference_date, S)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train, sample_weight=weights)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    results.append({"S": S, "MAE": mae})
    print(f"S={S:>4} | MAE: {mae:.6f}")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(PROJECT_ROOT, "decay_model", "results", "decay_rate_sensitivity.csv"), index=False)

best_row = results_df.loc[results_df["MAE"].idxmin()]
print(f"\nBest S value: {best_row['S']} (MAE: {best_row['MAE']:.6f})")

plt.figure(figsize=(10, 6))
plt.plot(results_df["S"], results_df["MAE"], marker="o", color="#2a9d8f", linewidth=2)
plt.axhline(y=0.006186, color="gray", linestyle="--", label="naive-zero MAE (0.006186)")
plt.scatter([30, 180, 365], results_df[results_df["S"].isin([30, 180, 365])]["MAE"],
            color="#e63946", s=100, zorder=5, label="original fast/medium/slow")
plt.xlabel("Decay rate S (days)")
plt.ylabel("Test MAE")
plt.title("Decay Rate Sensitivity — Financial Domain")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "decay_model", "results", "decay_rate_sensitivity.png"), dpi=150)
print("Saved results/decay_rate_sensitivity.png")
print("Saved results/decay_rate_sensitivity.csv")