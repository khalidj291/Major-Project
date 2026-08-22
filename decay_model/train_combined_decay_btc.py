"""
train_combined_decay_btc.py -- THE actual best BTC result: feature-decay and
row-decay combined, both applied to the same model at once. This is what
should be presented as your headline BTC number.

S for each mechanism is chosen SEPARATELY, each on validation (2023) only --
never on the 2024-2026 test data this script reports on. Requires
train_baseline_btc.py to have been run first (loads its saved model for
the baseline comparison).
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from feature_decay import apply_feature_decay
from ebbinghaus import apply_decay_weights

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")
FEATURE_S_CANDIDATES = [5, 10, 15, 20, 30, 60]
ROW_S_CANDIDATES = [90, 180, 365, 730]
TRAIN_END = "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
TEST_START = "2024-01-01"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, tr_dates = make_windows(df, df["date"].min(), pd.Timestamp(TRAIN_END), WINDOW)
X_val, y_val, val_dates = make_windows(df, pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp(TEST_START), df["date"].max(), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()
ref_date = pd.Timestamp(VAL_END)

print("Selecting feature-decay S on validation:")
best_fS, best_val = None, np.inf
for S in FEATURE_S_CANDIDATES:
    pred = Ridge(alpha=1.0).fit(apply_feature_decay(X_train, WINDOW, S), y_train).predict(apply_feature_decay(X_val, WINDOW, S))
    mae = np.abs(y_val - pred).mean()
    if mae < best_val:
        best_val, best_fS = mae, S
print(f"-> feature-decay S={best_fS}")

print("Selecting row-decay S on validation:")
best_rS, best_val2 = None, np.inf
for S in ROW_S_CANDIDATES:
    w = apply_decay_weights(pd.to_datetime(tr_dates), ref_date, S)
    pred = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=w).predict(X_val)
    mae = np.abs(y_val - pred).mean()
    if mae < best_val2:
        best_val2, best_rS = mae, S
print(f"-> row-decay S={best_rS}\n")

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
trainval_dates = list(tr_dates) + list(val_dates)
w_trainval = apply_decay_weights(pd.to_datetime(trainval_dates), ref_date, best_rS)
X_trainval_decayed = apply_feature_decay(X_trainval, WINDOW, best_fS)
X_test_decayed = apply_feature_decay(X_test, WINDOW, best_fS)

model = Ridge(alpha=1.0).fit(X_trainval_decayed, y_trainval, sample_weight=w_trainval)
pred = model.predict(X_test_decayed)

with open(os.path.join(os.path.dirname(__file__), "..", "baseline_model", "models", "model_baseline_btc.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_pred = baseline_model.predict(X_test)

combo_err = np.abs(y_test - pred)
base_err = np.abs(y_test - baseline_pred)
t_stat, p_val = stats.ttest_rel(combo_err, base_err)

out_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_combined_decay_btc.pkl"), "wb") as f:
    pickle.dump({"model": model, "feature_S": best_fS, "row_S": best_rS, "window": WINDOW}, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pct_improvement = 100 * (base_err.mean() - combo_err.mean()) / base_err.mean()
pd.DataFrame([{
    "domain": "crypto", "model": "combined_decay_ridge", "ticker": "BTC-USD",
    "feature_S": best_fS, "row_S": best_rS,
    "train_period": f"{df['date'].min().date()} to {VAL_END}", "test_period": f"{TEST_START} to {df['date'].max().date()}",
    "MAE": combo_err.mean(), "baseline_MAE": base_err.mean(), "pct_improvement": pct_improvement,
    "t_stat": t_stat, "p_value": p_val,
}]).to_csv(os.path.join(results_dir, "combined_decay_results_btc.csv"), index=False)

print(f"baseline MAE     : {base_err.mean():.6f}")
print(f"combined MAE     : {combo_err.mean():.6f}  (feature_S={best_fS}, row_S={best_rS})")
print(f"relative improvement: {pct_improvement:.2f}%")
print(f"paired t-test    : t={t_stat:.4f}, p={p_val:.4f}")