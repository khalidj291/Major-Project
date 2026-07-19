"""
Regime evaluation framework -- splits model performance by market regime
(volatile vs stable) so we can test whether optimal decay rate varies by regime.

NOTE on framing (per team review from another Claude session):
We are TESTING whether decay rate sensitivity varies by regime, not trying
to "prove" a predetermined answer. Report whatever this function finds,
even if it doesn't match the fast-decay-wins-in-volatile hypothesis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_by_regime(predictions, actuals, regimes):
    """
    Split predictions into volatile/stable subsets and compute MAE/RMSE for each.

    Parameters
    ----------
    predictions : array-like, shape (n_samples,) or (n_samples, 1)
        Model predictions for the test period.
    actuals : array-like, shape (n_samples,) or (n_samples, 1)
        True values for the same test period.
    regimes : array-like of str, shape (n_samples,)
        Regime label for each sample. Expected values: "volatile", "stable"
        (any other label, e.g. "neutral", is included in overall but excluded
        from the volatile/stable breakdown).

    Returns
    -------
    dict with keys:
        volatile_MAE, volatile_RMSE, stable_MAE, stable_RMSE,
        overall_MAE, overall_RMSE, n_volatile, n_stable, n_overall
    """
    predictions = np.asarray(predictions).flatten()
    actuals = np.asarray(actuals).flatten()
    regimes = np.asarray(regimes).flatten()

    assert len(predictions) == len(actuals) == len(regimes), (
        f"Length mismatch: predictions={len(predictions)}, "
        f"actuals={len(actuals)}, regimes={len(regimes)}"
    )

    volatile_mask = regimes == "volatile"
    stable_mask = regimes == "stable"

    result = {
        "overall_MAE": mean_absolute_error(actuals, predictions),
        "overall_RMSE": np.sqrt(mean_squared_error(actuals, predictions)),
        "n_overall": len(actuals),
    }

    if volatile_mask.sum() > 0:
        result["volatile_MAE"] = mean_absolute_error(actuals[volatile_mask], predictions[volatile_mask])
        result["volatile_RMSE"] = np.sqrt(mean_squared_error(actuals[volatile_mask], predictions[volatile_mask]))
    else:
        result["volatile_MAE"] = None
        result["volatile_RMSE"] = None
    result["n_volatile"] = int(volatile_mask.sum())

    if stable_mask.sum() > 0:
        result["stable_MAE"] = mean_absolute_error(actuals[stable_mask], predictions[stable_mask])
        result["stable_RMSE"] = np.sqrt(mean_squared_error(actuals[stable_mask], predictions[stable_mask]))
    else:
        result["stable_MAE"] = None
        result["stable_RMSE"] = None
    result["n_stable"] = int(stable_mask.sum())

    return result


if __name__ == "__main__":
    # --- Test with dummy data before real regime labels arrive ---
    np.random.seed(0)
    n = 200

    actuals = np.random.normal(0, 0.01, n)
    # Simulate a model that's decent overall but worse specifically in volatile periods
    predictions = actuals + np.random.normal(0, 0.003, n)
    regimes = np.random.choice(["volatile", "stable", "neutral"], size=n, p=[0.3, 0.5, 0.2])
    # Inject extra noise into volatile-labeled predictions to make the test meaningful
    predictions[regimes == "volatile"] += np.random.normal(0, 0.01, (regimes == "volatile").sum())

    result = evaluate_by_regime(predictions, actuals, regimes)

    print("Dummy data test of evaluate_by_regime():")
    print(f"  Overall  -> MAE: {result['overall_MAE']:.6f}  RMSE: {result['overall_RMSE']:.6f}  (n={result['n_overall']})")
    print(f"  Volatile -> MAE: {result['volatile_MAE']:.6f}  RMSE: {result['volatile_RMSE']:.6f}  (n={result['n_volatile']})")
    print(f"  Stable   -> MAE: {result['stable_MAE']:.6f}  RMSE: {result['stable_RMSE']:.6f}  (n={result['n_stable']})")
    print()
    if result["volatile_MAE"] > result["stable_MAE"]:
        print("PASS: volatile MAE is higher than stable MAE, as expected from the injected noise.")
    else:
        print("CHECK: volatile MAE is not higher than stable -- review the dummy data setup.")