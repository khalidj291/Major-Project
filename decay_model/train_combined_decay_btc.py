"""
train_combined_decay_btc.py -- combines two structurally different decay
mechanisms on the same model:
  (1) feature decay (power-law): reweights WHICH LAG inside a window matters
  (2) sample decay (Ebbinghaus, exponential): reweights WHICH TRAINING ROW
      (calendar date) matters

These are not competing techniques -- they answer different questions
("which part of this one window matters" vs "which whole training example
matters") and can be applied together. Both the power-law alpha and the
sample-decay S are selected using ONLY the 2019 validation year; 2020 test
data is never touched until the single final evaluation below.

See powerlaw_decay.py and ebbinghaus.py for the two weighting functions.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy import stats
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from windowing import make_windows
from powerlaw_decay import apply_powerlaw_decay
from ebbinghaus import ebbinghaus_weight

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")
ALPHA_CANDIDATES = [0.5, 1.0, 1.5, 2.0, 3.0]   # power-law feature-decay steepness
S_CANDIDATES = [180, 365, 730, 1460]            # Ebbinghaus sample-decay speed (days)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, train_dates = make_windows(df, pd.Timestamp("2013-10-02"), pd.Timestamp("2018-12-31"), WINDOW)
X_val, y_val, _ = make_windows(df, pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

reference_date = train_dates.max()

print("Selecting (alpha, S) on VALIDATION (2019) only:")
best_params, best_val_mae = None, np.inf
for alpha in ALPHA_CANDIDATES:
    X_train_decayed = apply_powerlaw_decay(X_train, WINDOW, alpha)
    X_val_decayed = apply_powerlaw_decay(X_val, WINDOW, alpha)
    for S in S_CANDIDATES:
        sw = ebbinghaus_weight(train_dates, reference_date, S)
        model = Ridge(alpha=1.0).fit(X_train_decayed, y_train, sample_weight=sw)
        mae = np.abs(y_val - model.predict(X_val_decayed)).mean()
        if mae < best_val_mae:
            best_val_mae, best_params = mae, (alpha, S)
        print(f"  alpha={alpha}, S={S}: val MAE={mae:.6f}")

best_alpha, best_S = best_params
print(f"-> alpha={best_alpha}, S={best_S} selected\n")

# refit on train+val combined with the chosen params, matching the convention
# used in train_feature_decay_btc.py
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])
trainval_dates = np.concatenate([train_dates, make_windows(df, pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), WINDOW)[2]])

X_trainval_decayed = apply_powerlaw_decay(X_trainval, WINDOW, best_alpha)
X_test_decayed = apply_powerlaw_decay(X_test, WINDOW, best_alpha)
sw_trainval = ebbinghaus_weight(trainval_dates, trainval_dates.max(), best_S)

model = Ridge(alpha=1.0).fit(X_trainval_decayed, y_trainval, sample_weight=sw_trainval)
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
    pickle.dump({"model": model, "alpha": best_alpha, "S": best_S, "window": WINDOW}, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{
    "domain": "crypto", "model": "combined_decay_ridge", "ticker": "BTC-USD",
    "alpha": best_alpha, "S": best_S,
    "train_period": "2013-10-02 to 2019-12-31", "test_period": "2020-01-01 to 2020-12-31",
    "MAE": combo_err.mean(), "baseline_MAE": base_err.mean(),
    "t_stat": t_stat, "p_value": p_val,
}]).to_csv(os.path.join(results_dir, "combined_decay_results_btc.csv"), index=False)

print(f"baseline MAE : {base_err.mean():.6f}")
print(f"combined MAE : {combo_err.mean():.6f}  (alpha={best_alpha}, S={best_S})")
print(f"paired t-test: t={t_stat:.4f}, p={p_val:.4f}")