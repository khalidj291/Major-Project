"""
Ebbinghaus forgetting curve applied as a sample-weighting function for ML training.

Formula: R = e^(-t/S)
  t = days elapsed between the data point's date and the reference (most recent) date
  S = stability constant; controls how fast the weight decays toward 0
  R = weight, always in (0, 1]
"""

import numpy as np
import pandas as pd


def ebbinghaus_weight(date, reference_date, S):
    """
    Compute the Ebbinghaus retention weight for a single date (or array of dates).

    Parameters
    ----------
    date : datetime-like or array-like of datetime-like
        The date(s) of the data point(s).
    reference_date : datetime-like
        The most recent date in the dataset (t=0 here, weight=1.0).
    S : float
        Stability constant. Larger S = slower decay (longer memory).

    Returns
    -------
    float or np.ndarray
        Weight(s) in the range (0, 1].
    """
    date = pd.to_datetime(date)
    reference_date = pd.to_datetime(reference_date)

    t = (reference_date - date)
    # Handle both scalar and Series/array input for t in days
    if isinstance(t, pd.Timedelta):
        t_days = t.days
    else:
        t_days = t.dt.days.values if hasattr(t, "dt") else np.array([x.days for x in t])

    weight = np.exp(-np.asarray(t_days, dtype=float) / S)
    return weight


def apply_decay_weights(dates, reference_date, S):
    """Vectorized helper: apply ebbinghaus_weight across a full date column."""
    return ebbinghaus_weight(dates, reference_date, S)


if __name__ == "__main__":
    reference_date = pd.Timestamp("2022-12-31")  # end of training period, example
    example_ages_days = [30, 180, 365, 365 * 5]
    S_values = {"fast (S=30)": 30, "medium (S=180)": 180, "slow (S=365)": 365}

    print("Numerical examples — weight by age and decay rate")
    print("=" * 60)
    print(f"{'Age':>15} | {'fast (S=30)':>14} | {'medium (S=180)':>16} | {'slow (S=365)':>14}")
    for age in example_ages_days:
        d = reference_date - pd.Timedelta(days=age)
        row = [ebbinghaus_weight(d, reference_date, S) for S in S_values.values()]
        label = f"{age} days ago"
        print(f"{label:>15} | {row[0]:>14.4f} | {row[1]:>16.4f} | {row[2]:>14.4f}")