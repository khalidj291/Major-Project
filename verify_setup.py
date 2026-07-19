"""
Person 1 - Day 3, Step 1: Verify everything loads correctly before integration.
Run this any time you're not sure something is broken -- it checks both
baseline_model/ and decay_model/ outputs, all 8 models.
"""
import pandas as pd
import pickle
import os

def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition

def verify_setup():
    all_ok = True

    # --- data_processed.csv ---
    if os.path.exists("data/data_processed.csv"):
        df = pd.read_csv("data/data_processed.csv")
        expected_cols = {"date", "ticker", "close", "volume", "returns"}
        all_ok &= check("data_processed.csv exists", True)
        all_ok &= check("data_processed.csv has correct columns", expected_cols.issubset(df.columns))
        all_ok &= check("data_processed.csv has SPY rows", "SPY" in df["ticker"].values)
        all_ok &= check("data_processed.csv has no NaN returns", df["returns"].isna().sum() == 0)
    else:
        all_ok &= check("data_processed.csv exists", False)

    # --- data_consumer.csv ---
    if os.path.exists("data/data_consumer.csv"):
        dfc = pd.read_csv("data/data_consumer.csv")
        expected_cols = {"date", "close", "volume", "returns"}
        all_ok &= check("data_consumer.csv exists", True)
        all_ok &= check("data_consumer.csv has correct columns", expected_cols.issubset(dfc.columns))
    else:
        all_ok &= check("data_consumer.csv exists", False)

    # --- baseline_model/models/*.pkl ---
    baseline_checks = [
        ("baseline_model/models/model_baseline.pkl", 30),
        ("baseline_model/models/model_baseline_consumer.pkl", 12),
    ]
    for path, n_features in baseline_checks:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            test_input = [[0.001] * n_features]
            try:
                pred = model.predict(test_input)
                all_ok &= check(f"{path} loads and predicts", pred.shape == (1,))
            except Exception as e:
                all_ok &= check(f"{path} predict failed: {e}", False)
        else:
            all_ok &= check(f"{path} exists", False)

    # --- decay_model/models/*.pkl ---
    decay_checks = [
        ("decay_model/models/model_decay_fast.pkl", 30),
        ("decay_model/models/model_decay_medium.pkl", 30),
        ("decay_model/models/model_decay_slow.pkl", 30),
        ("decay_model/models/model_decay_fast_consumer.pkl", 12),
        ("decay_model/models/model_decay_medium_consumer.pkl", 12),
        ("decay_model/models/model_decay_slow_consumer.pkl", 12),
    ]
    for path, n_features in decay_checks:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            test_input = [[0.001] * n_features]
            try:
                pred = model.predict(test_input)
                all_ok &= check(f"{path} loads and predicts", pred.shape == (1,))
            except Exception as e:
                all_ok &= check(f"{path} predict failed: {e}", False)
        else:
            all_ok &= check(f"{path} exists", False)

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- fix before Day 5'}")
    return all_ok

if __name__ == "__main__":
    verify_setup()