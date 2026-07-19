"""
Person 1 - Day 3, Step 3: Integration script skeleton.
This is NOT the final version -- placeholders for Person 2's models,
since they're not finished yet. You'll fill these in on Day 5.
"""
import pandas as pd
import pickle
import os

def load_data():
    """Load processed data for both domains."""
    financial = pd.read_csv("data/data_processed.csv", parse_dates=["date"])
    consumer = pd.read_csv("data/data_consumer.csv", parse_dates=["date"]) \
        if os.path.exists("data/data_consumer.csv") else None
    return financial, consumer

def load_models():
    """Load all four models. Decay models are placeholders until Person 2 shares them."""
    models = {}
    for name, path in [
        ("baseline", "baseline_model/models/model_baseline.pkl"),
        ("decay_fast", "decay_model/models/model_decay_fast.pkl"),
        ("decay_medium", "decay_model/models/model_decay_medium.pkl"),
        ("decay_slow", "decay_model/models/model_decay_slow.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            models[name] = None
            print(f"[MISSING] {name} not found at {path} yet")
    return models

def run_all_models(models, X_test):
    """Run every available model on the same test data. Skips missing ones."""
    predictions = {}
    for name, model in models.items():
        if model is not None:
            predictions[name] = model.predict(X_test)
        else:
            predictions[name] = None
    return predictions

def launch_dashboard():
    """Placeholder -- calls Person 3's dashboard.py once it exists."""
    if os.path.exists("dashboard/dashboard.py"):
        print("Launching dashboard...")
        # actual launch call goes here once dashboard.py exists
    else:
        print("[MISSING] dashboard/dashboard.py not built yet")

def main():
    print("=== Memory That Fades: Integration Run ===\n")
    financial, consumer = load_data()
    print(f"Financial data: {len(financial)} rows | Consumer data: {len(consumer) if consumer is not None else 'MISSING'}")

    models = load_models()
    print(f"\nModels loaded: {[k for k,v in models.items() if v is not None]}")

    # TODO Day 5: build real test-set features (30-day window) and call run_all_models()
    # TODO Day 5: save results to decay_model/results/comparison_predictions.csv
    #             (grouped with the other combined baseline-vs-decay comparison outputs)
    # TODO Day 5: call launch_dashboard()

    print("\nSkeleton run complete. Fill in TODOs on Day 5 once all models exist.")

if __name__ == "__main__":
    main()