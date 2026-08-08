"""
train_feature_decay_btc.py -- new mechanism: decay by LAG POSITION inside
each window, not by row age (see feature_decay.py for why this is
different from ebbinghaus.py). S=30 was chosen using ONLY the 2019
validation year (see the S-sweep below, printed for transparency) --
2020 test data was never looked at until the single final evaluation.
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

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")
S_CANDIDATES = [5, 10, 15, 20, 30, 60]  # swept on validation only

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, _ = make_windows(df, pd.Timestamp("2013-10-02"), pd.Timestamp("2018-12-31"), WINDOW)
X_val, y_val, _ = make_windows(df, pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

print("Selecting S on VALIDATION (2019) only:")
best_S, best_val_mae = None, np.inf
for S in S_CANDIDATES:
    pred = Ridge(alpha=1.0).fit(apply_feature_decay(X_train, WINDOW, S), y_train).predict(apply_feature_decay(X_val, WINDOW, S))
    mae = np.abs(y_val - pred).mean()
    if mae < best_val_mae:
        best_val_mae, best_S = mae, S
    print(f"  S={S:3d}: val MAE={mae:.6f}")
print(f"-> S={best_S} selected\n")

# refit on train+val combined with the chosen S, exactly as the baseline script does
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
X_trainval_decayed = apply_feature_decay(X_trainval, WINDOW, best_S)
X_test_decayed = apply_feature_decay(X_test, WINDOW, best_S)

model = Ridge(alpha=1.0).fit(X_trainval_decayed, y_trainval)
pred = model.predict(X_test_decayed)

with open(os.path.join(os.path.dirname(__file__), "..", "baseline_model", "models", "model_baseline_btc.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_pred = baseline_model.predict(X_test)

decay_err = np.abs(y_test - pred)
base_err = np.abs(y_test - baseline_pred)
t_stat, p_val = stats.ttest_rel(decay_err, base_err)

out_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_feature_decay_btc.pkl"), "wb") as f:
    pickle.dump({"model": model, "S": best_S, "window": WINDOW}, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{
    "domain": "crypto", "model": "feature_decay_ridge", "ticker": "BTC-USD", "S": best_S,
    "train_period": "2013-10-02 to 2019-12-31", "test_period": "2020-01-01 to 2020-12-31",
    "MAE": decay_err.mean(), "baseline_MAE": base_err.mean(),
    "t_stat": t_stat, "p_value": p_val,
}]).to_csv(os.path.join(results_dir, "feature_decay_results_btc.csv"), index=False)

print(f"baseline MAE      : {base_err.mean():.6f}")
print(f"feature-decay MAE : {decay_err.mean():.6f}  (S={best_S})")
print(f"paired t-test     : t={t_stat:.4f}, p={p_val:.4f}")