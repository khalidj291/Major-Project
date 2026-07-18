"""
Person 1 - Day 3, Step 1: Verify everything loads correctly before integration.
Run this any time you're not sure something is broken -- it checks your own
files. Person 2's model files get added to this same script once he shares them.
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

    # --- model_baseline.pkl ---
    if os.path.exists("models/model_baseline.pkl"):
        with open("models/model_baseline.pkl", "rb") as f:
            model = pickle.load(f)
        test_input = [[0.001] * 30]
        try:
            pred = model.predict(test_input)
            all_ok &= check("model_baseline.pkl loads and predicts", pred.shape == (1,))
        except Exception as e:
            all_ok &= check(f"model_baseline.pkl predict failed: {e}", False)
    else:
        all_ok &= check("model_baseline.pkl exists", False)

    # --- model_baseline_consumer.pkl ---
    if os.path.exists("models/model_baseline_consumer.pkl"):
        with open("models/model_baseline_consumer.pkl", "rb") as f:
            model_c = pickle.load(f)
        test_input = [[0.001] * 12]  # 12-period window, not 30, for consumer data
        try:
            pred = model_c.predict(test_input)
            all_ok &= check("model_baseline_consumer.pkl loads and predicts", pred.shape == (1,))
        except Exception as e:
            all_ok &= check(f"model_baseline_consumer.pkl predict failed: {e}", False)
    else:
        all_ok &= check("model_baseline_consumer.pkl exists", False)

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED -- fix before Day 5'}")
    return all_ok

if __name__ == "__main__":
    verify_setup()