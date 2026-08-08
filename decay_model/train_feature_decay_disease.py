"""
train_feature_decay_disease.py -- same new mechanism as train_feature_decay_btc.py,
applied to the disease domain. S=10 chosen using a validation slice carved out
of the tail of the original training period (2021-10-25 to 2022-04-02) --
the real test period (2022-04-03 onward) is never touched until the final
single evaluation below.

Honest note: even at its validated best, this does NOT beat this domain's
existing best techniques (recent-only ~0.039965, sample-decay S=180
~0.040868, RevIN ~0.041183). It's a real, independently-validated win over
baseline via a different mechanism -- worth reporting as converging
evidence, not as this domain's headline number.
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
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_disease.csv")
S_CANDIDATES = [5, 10, 15, 20, 30, 60]
TRAIN_END = "2021-10-24"
VAL_START = "2021-10-25"
VAL_END = "2022-04-02"
TEST_START = "2022-04-03"

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

X_train, y_train, _ = make_windows(df, df["date"].min(), pd.Timestamp(TRAIN_END), WINDOW)
X_val, y_val, _ = make_windows(df, pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp(TEST_START), df["date"].max(), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

print("Selecting S on VALIDATION only:")
best_S, best_val_mae = None, np.inf
for S in S_CANDIDATES:
    pred = Ridge(alpha=1.0).fit(apply_feature_decay(X_train, WINDOW, S), y_train).predict(apply_feature_decay(X_val, WINDOW, S))
    mae = np.abs(y_val - pred).mean()
    if mae < best_val_mae:
        best_val_mae, best_S = mae, S
    print(f"  S={S:3d}: val MAE={mae:.6f}")
print(f"-> S={best_S} selected\n")

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
X_trainval_decayed = apply_feature_decay(X_trainval, WINDOW, best_S)
X_test_decayed = apply_feature_decay(X_test, WINDOW, best_S)

model = Ridge(alpha=1.0).fit(X_trainval_decayed, y_trainval)
pred = model.predict(X_test_decayed)

with open(os.path.join(os.path.dirname(__file__), "..", "baseline_model", "models", "model_baseline_disease.pkl"), "rb") as f:
    baseline_model = pickle.load(f)
baseline_pred = baseline_model.predict(X_test)

decay_err = np.abs(y_test - pred)
base_err = np.abs(y_test - baseline_pred)
t_stat, p_val = stats.ttest_rel(decay_err, base_err)

out_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_feature_decay_disease.pkl"), "wb") as f:
    pickle.dump({"model": model, "S": best_S, "window": WINDOW}, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{
    "domain": "disease", "model": "feature_decay_ridge", "S": best_S,
    "train_period": f"{df['date'].min().date()} to {VAL_END}", "test_period": f"{TEST_START} to {df['date'].max().date()}",
    "MAE": decay_err.mean(), "baseline_MAE": base_err.mean(),
    "t_stat": t_stat, "p_value": p_val,
}]).to_csv(os.path.join(results_dir, "feature_decay_results_disease.csv"), index=False)

print(f"baseline MAE      : {base_err.mean():.6f}")
print(f"feature-decay MAE : {decay_err.mean():.6f}  (S={best_S})")
print(f"paired t-test     : t={t_stat:.4f}, p={p_val:.4f}")
print("(reference: recent-only=0.039965, sample-decay S=180=0.040868, RevIN=0.041183 -- this doesn't beat those, still a real win over baseline)")