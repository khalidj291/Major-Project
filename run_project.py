"""
Person 1 - Day 5: Integration script, filled in.
Loads both domains, loads all 8 models, builds real windowed test-set
features (window=30 financial, window=12 consumer -- consumer is monthly,
NOT the same window as financial), runs every model on its own domain's
test set, and saves comparison_predictions_financial.csv and
comparison_predictions_consumer.csv.

Windowing here matches the corrected convention used everywhere else in
the repo: features = returns[i-window:i], target = returns[i]. No gap.
"""
import pandas as pd
import numpy as np
import pickle
import os

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

def load_data():
    """Load processed data for both domains."""
    financial = pd.read_csv("data/data_processed.csv", parse_dates=["date"])
    financial = financial[financial["ticker"] == "SPY"].sort_values("date").reset_index(drop=True)
    consumer = pd.read_csv("data/data_consumer.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True) \
        if os.path.exists("data/data_consumer.csv") else None
    return financial, consumer

def load_models():
    """Load all 8 models -- 4 financial, 4 consumer."""
    paths = {
        "baseline":      "baseline_model/models/model_baseline.pkl",
        "decay_fast":    "decay_model/models/model_decay_fast.pkl",
        "decay_medium":  "decay_model/models/model_decay_medium.pkl",
        "decay_slow":    "decay_model/models/model_decay_slow.pkl",
        "baseline_consumer":     "baseline_model/models/model_baseline_consumer.pkl",
        "decay_fast_consumer":   "decay_model/models/model_decay_fast_consumer.pkl",
        "decay_medium_consumer": "decay_model/models/model_decay_medium_consumer.pkl",
        "decay_slow_consumer":   "decay_model/models/model_decay_slow_consumer.pkl",
    }
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            models[name] = None
            print(f"[MISSING] {name} not found at {path} yet")
    return models

def make_test_windows(df, window):
    """Build test-set windowed features. Same convention as every training
    script in the repo: X = returns[i-window:i], y = returns[i], no gap."""
    df = df.sort_values("date").reset_index(drop=True)
    returns = df["returns"].values
    dates = df["date"].values
    X, y, out_dates = [], [], []
    for i in range(window, len(returns)):
        d = pd.Timestamp(dates[i])
        if pd.Timestamp(TEST_START) <= d <= pd.Timestamp(TEST_END):
            X.append(returns[i - window:i])
            y.append(returns[i])
            out_dates.append(dates[i])
    return np.array(X), np.array(y), np.array(out_dates)

def run_all_models(models, model_names, X_test):
    """Run every available model in model_names on the same test data."""
    predictions = {}
    for name in model_names:
        model = models.get(name)
        predictions[name] = model.predict(X_test) if model is not None else None
    return predictions

def save_comparison(dates, y_test, predictions, out_path, name_map):
    """Save a comparison_predictions CSV: date, actual, <model>_pred columns."""
    out = pd.DataFrame({"date": dates, "actual": y_test})
    for internal_name, col_name in name_map.items():
        out[col_name] = predictions[internal_name]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")
    return out

def print_mae_summary(df, name_map, label):
    print(f"\n{label} test-set MAE:")
    for internal_name, col_name in name_map.items():
        if df[col_name].notna().all():
            mae = np.mean(np.abs(df["actual"] - df[col_name]))
            print(f"  {internal_name:22s} {mae:.6f}")
        else:
            print(f"  {internal_name:22s} [MISSING]")

def launch_dashboard():
    """Calls Person 3's dashboard.py once it exists."""
    if os.path.exists("dashboard/dashboard.py"):
        print("\nLaunching dashboard...")
        import subprocess
        subprocess.run(["python", "dashboard/dashboard.py"])
    else:
        print("\n[MISSING] dashboard/dashboard.py not built yet -- skipping launch")

def main():
    print("=== Memory That Fades: Integration Run ===\n")
    financial, consumer = load_data()
    print(f"Financial data (SPY): {len(financial)} rows | Consumer data: {len(consumer) if consumer is not None else 'MISSING'}")

    models = load_models()
    print(f"\nModels loaded: {[k for k,v in models.items() if v is not None]}")

    # --- Financial domain: window=30 ---
    fin_names = ["baseline", "decay_fast", "decay_medium", "decay_slow"]
    if all(models.get(n) is not None for n in fin_names):
        X_test, y_test, dates = make_test_windows(financial, window=30)
        preds = run_all_models(models, fin_names, X_test)
        name_map = {"baseline": "baseline_pred", "decay_fast": "decay_fast_pred",
                    "decay_medium": "decay_medium_pred", "decay_slow": "decay_slow_pred"}
        fin_df = save_comparison(dates, y_test, preds,
                                  "decay_model/results/comparison_predictions_financial.csv", name_map)
        print_mae_summary(fin_df, name_map, "Financial (SPY)")
    else:
        print("\n[SKIPPED] Financial comparison -- one or more financial models missing")

    # --- Consumer domain: window=12 ---
    if consumer is not None:
        con_names = ["baseline_consumer", "decay_fast_consumer", "decay_medium_consumer", "decay_slow_consumer"]
        if all(models.get(n) is not None for n in con_names):
            X_test_c, y_test_c, dates_c = make_test_windows(consumer, window=12)
            preds_c = run_all_models(models, con_names, X_test_c)
            name_map_c = {"baseline_consumer": "baseline_pred", "decay_fast_consumer": "decay_fast_pred",
                          "decay_medium_consumer": "decay_medium_pred", "decay_slow_consumer": "decay_slow_pred"}
            con_df = save_comparison(dates_c, y_test_c, preds_c,
                                      "decay_model/results/comparison_predictions_consumer.csv", name_map_c)
            print_mae_summary(con_df, name_map_c, "Consumer (PCE)")
        else:
            print("\n[SKIPPED] Consumer comparison -- one or more consumer models missing")

    launch_dashboard()
    print("\n=== Integration run complete ===")

if __name__ == "__main__":
    main()