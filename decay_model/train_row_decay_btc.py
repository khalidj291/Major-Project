"""
train_row_decay_btc.py -- your ORIGINAL mechanism (decay whole training days
by age, ebbinghaus.py), applied to BTC for the first time as its own script.
Built for completeness -- feature_decay alone and this combined together
is train_combined_decay_btc.py, the actual best BTC result.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from ebbinghaus import apply_decay_weights

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")
S_CANDIDATES = [90, 180, 365, 730]
TRAIN_END = "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
TEST_START = "2024-01-01"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, tr_dates = make_windows(df, df["date"].min(), pd.Timestamp(TRAIN_END), WINDOW)
X_val, y_val, val_dates = make_windows(df, pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp(TEST_START), df["date"].max(), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

print("Selecting S on VALIDATION (2023) only:")
ref_date = pd.Timestamp(VAL_END)
best_S, best_val_mae = None, np.inf
for S in S_CANDIDATES:
    w = apply_decay_weights(pd.to_datetime(tr_dates), ref_date, S)
    pred = Ridge(alpha=1.0).fit(X_train, y_train, sample_weight=w).predict(X_val)
    mae = np.abs(y_val - pred).mean()
    if mae < best_val_mae:
        best_val_mae, best_S = mae, S
    print(f"  S={S:3d}: val MAE={mae:.6f}")
print(f"-> S={best_S} selected\n")

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
trainval_dates = list(tr_dates) + list(val_dates)
w_trainval = apply_decay_weights(pd.to_datetime(trainval_dates), ref_date, best_S)

model = Ridge(alpha=1.0).fit(X_trainval, y_trainval, sample_weight=w_trainval)
pred = model.predict(X_test)

with open(os.path.join(os.path.dirname(__file__), "..", "baseline_model", "models", "model_baseline_btc.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_pred = baseline_model.predict(X_test)

decay_err = np.abs(y_test - pred)
base_err = np.abs(y_test - baseline_pred)
t_stat, p_val = stats.ttest_rel(decay_err, base_err)

out_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_row_decay_btc.pkl"), "wb") as f:
    pickle.dump({"model": model, "S": best_S}, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{
    "domain": "crypto", "model": "row_decay_ridge", "ticker": "BTC-USD", "S": best_S,
    "train_period": f"{df['date'].min().date()} to {VAL_END}", "test_period": f"{TEST_START} to {df['date'].max().date()}",
    "MAE": decay_err.mean(), "baseline_MAE": base_err.mean(),
    "t_stat": t_stat, "p_value": p_val,
}]).to_csv(os.path.join(results_dir, "row_decay_results_btc.csv"), index=False)

print(f"baseline MAE  : {base_err.mean():.6f}")
print(f"row-decay MAE : {decay_err.mean():.6f}  (S={best_S})")
print(f"paired t-test : t={t_stat:.4f}, p={p_val:.4f}")