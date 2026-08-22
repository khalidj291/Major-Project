"""
train_baseline_btc.py -- plain Ridge baseline for BTC-USD, no weighting.

UPDATED split (was train<2019/val=2019/test=2020, now uses your full dataset
per your guide's "increase the training dataset" instruction): train through
2022, validate on 2023, test on 2024-2026 (untouched, ~949 days -- the
largest, most robust test period this project has used on any domain).
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decay_model"))
from windowing import make_windows

WINDOW = 30
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_btc.csv")
TRAIN_END = "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
TEST_START = "2024-01-01"

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, _ = make_windows(df, df["date"].min(), pd.Timestamp(TRAIN_END), WINDOW)
X_val, y_val, _ = make_windows(df, pd.Timestamp(VAL_START), pd.Timestamp(VAL_END), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp(TEST_START), df["date"].max(), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

model = Ridge(alpha=1.0).fit(X_trainval, y_trainval)
pred = model.predict(X_test)

mae = np.abs(y_test - pred).mean()
rmse = np.sqrt(((y_test - pred) ** 2).mean())
naive_mae = np.abs(y_test).mean()
naive_rmse = np.sqrt((y_test ** 2).mean())

out_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "model_baseline_btc.pkl"), "wb") as f:
    pickle.dump(model, f)

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
pd.DataFrame([{
    "domain": "crypto", "model": "baseline_ridge", "ticker": "BTC-USD",
    "train_period": f"{df['date'].min().date()} to {VAL_END}", "test_period": f"{TEST_START} to {df['date'].max().date()}",
    "MAE": mae, "RMSE": rmse, "naive_MAE": naive_mae, "naive_RMSE": naive_rmse,
}]).to_csv(os.path.join(results_dir, "baseline_results_btc.csv"), index=False)

print(f"n_train+val={len(X_trainval)}  n_test={len(X_test)}")
print(f"baseline BTC MAE: {mae:.6f}  (naive: {naive_mae:.6f})")