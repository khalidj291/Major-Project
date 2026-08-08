"""
train_baseline_btc.py -- plain Ridge baseline for BTC-USD, no weighting.
New domain: BTC-USD didn't exist in the repo before this. Data source:
RDeconomist/observatory (GitHub), daily close price, 2013-10-02 to 2021-01-04.

Train/val/test matches the split used to validate feature-decay on this
domain (see decay_model/train_feature_decay_btc.py): train < 2019,
val = 2019, test = 2020. Val isn't used here since baseline has no
hyperparameter to select -- it's included so both models are trained on
an identical row set for a fair comparison.
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

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

X_train, y_train, _ = make_windows(df, pd.Timestamp("2013-10-02"), pd.Timestamp("2018-12-31"), WINDOW)
X_val, y_val, _ = make_windows(df, pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"), WINDOW)
X_test, y_test, _ = make_windows(df, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), WINDOW)
y_train, y_val, y_test = y_train.ravel(), y_val.ravel(), y_test.ravel()

# train on train+val combined for the final model (matches feature-decay script's convention,
# since val there is spent choosing S -- baseline has nothing to choose, but we keep the same
# training rows so the comparison in statistical_significance_feature_decay_btc.py is apples-to-apples)
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
    "train_period": "2013-10-02 to 2019-12-31", "test_period": "2020-01-01 to 2020-12-31",
    "MAE": mae, "RMSE": rmse, "naive_MAE": naive_mae, "naive_RMSE": naive_rmse,
}]).to_csv(os.path.join(results_dir, "baseline_results_btc.csv"), index=False)

print(f"n_train+val={len(X_trainval)}  n_test={len(X_test)}")
print(f"baseline BTC MAE: {mae:.6f}  (naive: {naive_mae:.6f})")