"""
Power-law decay weighting -- an alternative to Ebbinghaus's exponential
decay, R = e^(-t/S).

Some memory-decay literature (e.g. Wixted & Ebbesen, 1991) argues power-law
decay fits human forgetting data better than pure exponential decay in many
experiments. This module tests that alternative mathematical shape as a
FEATURE decay (applied within a single window's lag positions -- see
feature_decay.py for the sample-weight version and the distinction between
the two).

Power-law: weight(age) = (age + 1) ** (-alpha)
  - age = 0 is the most recent lag (full weight = 1.0)
  - larger alpha = steeper, more aggressive decay (only the very latest
    lags matter); smaller alpha = gentler, closer to uniform weighting
  - unlike exponential decay, power-law has a much "fatter tail" shape at
    low alpha and a much steeper near-term cutoff at high alpha -- it is a
    qualitatively different curve shape, not just a different decay speed
"""

import numpy as np


def powerlaw_weights(window: int, alpha: float) -> np.ndarray:
    """
    Return an array of length `window`, one weight per lag position,
    oldest lag first (index 0) to most recent lag last (index window-1).

    alpha=0 -> all weights equal 1.0 (no decay)
    alpha increasing -> steeper decay, older lags shrink toward zero faster
    """
    age = np.arange(window - 1, -1, -1)  # oldest lag has highest "age"
    return (age + 1.0) ** (-alpha)


def apply_powerlaw_decay(X, window, alpha):
    """X: array of shape (n_samples, window), oldest-to-newest per row.
    Returns X with each column scaled by its lag's power-law weight."""
    w = powerlaw_weights(window, alpha)
    return X * w  # broadcasts across all rows