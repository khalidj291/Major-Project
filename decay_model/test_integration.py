"""
Day 5 — Integration sanity check (Person 2)

Loads all 8 models COLD (outside the scripts that trained them) and verifies
they behave correctly on dummy input before Person 1's integration script
depends on them.

Root cause note (resolved, do not re-investigate):
Under scikit-learn 1.7.2, fitting Ridge on a 2D y of shape (n,1) silently
squeezes coef_ to (n_features,) and predict() output to (n,) — a behavior
change from 1.5.2, where the same 2D y produced coef_ (1, n_features) and
predict() output (n,1). All 8 real models here were trained on 1.7.2, so
(n,) is the CORRECT expected predict shape, not a bug. This was confirmed
by fitting Ridge under both versions side by side. Only a since-deleted
placeholder file (trained on stale sklearn 1.5.2) ever produced (n,1).
"""

import os
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASELINE_MODELS_DIR = os.path.join(PROJECT_ROOT, "baseline_model", "models")
DECAY_MODELS_DIR = os.path.join(PROJECT_ROOT, "decay_model", "models")

# name -> (directory, filename, expected n_features_in_)
MODELS = {
    "baseline":               (BASELINE_MODELS_DIR, "model_baseline.pkl", 30),
    "decay_fast":              (DECAY_MODELS_DIR, "model_decay_fast.pkl", 30),
    "decay_medium":            (DECAY_MODELS_DIR, "model_decay_medium.pkl", 30),
    "decay_slow":              (DECAY_MODELS_DIR, "model_decay_slow.pkl", 30),
    "baseline_consumer":       (BASELINE_MODELS_DIR, "model_baseline_consumer.pkl", 12),
    "decay_fast_consumer":     (DECAY_MODELS_DIR, "model_decay_fast_consumer.pkl", 12),
    "decay_medium_consumer":   (DECAY_MODELS_DIR, "model_decay_medium_consumer.pkl", 12),
    "decay_slow_consumer":     (DECAY_MODELS_DIR, "model_decay_slow_consumer.pkl", 12),
}

EXPECTED_PREDICT_SHAPE = (10,)  # for dummy input of 10 samples


def run_checks():
    failures = []

    for name, (model_dir, filename, n_features) in MODELS.items():
        path = os.path.join(model_dir, filename)

        # 1. File exists
        if not os.path.exists(path):
            failures.append(f"[{name}] MISSING FILE: {path}")
            continue

        # 2. Loads without error
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            failures.append(f"[{name}] FAILED TO LOAD: {e}")
            continue

        # 3. Feature count matches expected window size
        actual_features = getattr(model, "n_features_in_", None)
        if actual_features != n_features:
            failures.append(
                f"[{name}] FEATURE COUNT MISMATCH: expected {n_features}, "
                f"got {actual_features}"
            )
            continue

        # 4. Predicts on dummy input without error, correct shape
        dummy_X = np.random.randn(10, n_features)
        try:
            preds = model.predict(dummy_X)
        except Exception as e:
            failures.append(f"[{name}] PREDICT FAILED: {e}")
            continue

        if preds.shape != EXPECTED_PREDICT_SHAPE:
            failures.append(
                f"[{name}] UNEXPECTED PREDICT SHAPE: expected "
                f"{EXPECTED_PREDICT_SHAPE}, got {preds.shape}"
            )
            continue

        # 5. Sanity: predictions aren't NaN/inf
        if not np.all(np.isfinite(preds)):
            failures.append(f"[{name}] PREDICTIONS CONTAIN NaN/inf")
            continue

        print(f"[{name}] OK — n_features_in_={actual_features}, "
              f"predict shape={preds.shape}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)
    else:
        print(f"All {len(MODELS)} models passed integration sanity check.")
        print("Predict output shape is consistently (n,) across all models —")
        print("this is the correct, expected shape under sklearn 1.7.2.")


if __name__ == "__main__":
    run_checks()