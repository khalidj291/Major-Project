"""
feature_decay.py -- shared module, used by train_feature_decay_btc.py and
train_feature_decay_disease.py.

Every existing mechanism in this project (fixed-S decay, double-exponential,
ensembling, regime-switching) reweights WHICH ROWS of training data count --
the sample_weight passed into Ridge.fit(). This is a structurally different
mechanism: it reweights WHICH LAGS matter INSIDE a single window, before the
row ever reaches the model.

Per windowing.py's own docstring, a feature window is `returns[i-window:i]`,
which is oldest-to-newest (index 0 = oldest, index -1 = the day right before
the target). So the age of the lag at position j is (window - 1 - j) days,
and we apply the same e^(-age/S) shape as ebbinghaus.py, just to columns
instead of rows.

Found to help on domains where "more recent lag = more relevant lag" is
actually true (disease, BTC-USD) and to actively HARM domains with strong
periodic structure at a lag beyond the recent window (airline passengers,
where lag-12 -- one year ago -- is the single most informative feature and
this decay shrinks it hardest). See PROJECT_CONTEXT for the full domain
breakdown. Only ever validated on disease and BTC-USD -- do not assume it
helps elsewhere without testing first.
"""
import numpy as np


def feature_decay_weights(window, S):
    """Per-lag multiplier, oldest (index 0) to most recent (index -1).
    Most recent lag always gets weight 1.0; older lags shrink toward 0."""
    age = np.arange(window - 1, -1, -1)
    return np.exp(-age / S)


def apply_feature_decay(X, window, S):
    """X: array of shape (n_samples, window), oldest-to-newest per row.
    Returns X with each column scaled by its lag's decay weight."""
    w = feature_decay_weights(window, S)
    return X * w  # broadcasts across all rows