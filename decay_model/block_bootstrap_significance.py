"""
block_bootstrap_significance.py -- moving block bootstrap significance test
on the paired daily errors from walk_forward_validation_{btc,eth}.py.

Why this exists: the paired t-test used everywhere else in this project
assumes each day's error is an independent draw. Financial returns are
known to have short-memory autocorrelation (volatility clustering, etc.),
so consecutive days' prediction errors are not independent -- a plain
t-test can understate the true uncertainty in the estimated mean
difference, making a real effect look more significant than it actually
is (or, less commonly, the reverse).

A moving block bootstrap fixes this by resampling CONTIGUOUS BLOCKS of
days (not individual days) with replacement. Each block keeps whatever
day-to-day dependence exists inside it intact, so the resampling
distribution reflects the real autocorrelation structure instead of
pretending it doesn't exist.

Method
------
Let d_i = baseline_err_i - selected_err_i for each of the n walk-forward
test days (sorted chronologically; the 13 six-month test windows are
contiguous in this project, i.e. one window's test end is the day before
the next window's test start, so treating all n days as a single ordered
series is valid -- verified below).

1. Observed statistic: d_bar = mean(d). If decay genuinely helps,
   d_bar > 0 (baseline error minus selected error, decay is more accurate).
2. Recenter: d' = d - d_bar, so d' has mean exactly 0 -- this is the null
   (H0: true mean difference = 0) resampling population.
3. Moving block bootstrap: for B replicates, build a resampled series of
   length n by drawing overlapping blocks of length L from d' with
   replacement until reaching >= n days, then truncate to n. Compute the
   resampled mean.
4. Two-sided bootstrap p-value = fraction of B replicates where
   |resampled mean| >= |d_bar|.
5. Separately, a percentile bootstrap 95% CI for d_bar itself: resample
   blocks from the UN-recentered d (not d'), B times, take the 2.5th and
   97.5th percentile of the resampled means.

Block length L is not a free parameter we get to pick to make the result
look good -- run across several L values (10, 20, 30, 40 trading days) and
report all of them. If the conclusion only holds for one convenient L,
that itself is worth reporting honestly.
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
N_BOOTSTRAP = 10000
BLOCK_LENGTHS = [10, 20, 30, 40]


def moving_block_bootstrap_mean(series, block_length, n_boot, rng):
    """Resample `series` via overlapping moving blocks of length
    `block_length`, with replacement, B times. Returns array of bootstrap
    means, one per replicate."""
    n = len(series)
    n_blocks_needed = int(np.ceil(n / block_length))
    max_start = n - block_length  # last valid block start index
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        resampled = np.concatenate([series[s:s + block_length] for s in starts])[:n]
        boot_means[b] = resampled.mean()
    return boot_means


def run_block_bootstrap(per_day_csv_path, label):
    df = pd.read_csv(per_day_csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Verify the walk-forward test windows really are contiguous (no gaps,
    # no overlaps) before treating all rows as one ordered series -- if
    # this assumption were wrong, block bootstrap blocks could straddle a
    # discontinuity and mix unrelated periods.
    date_diffs = df["date"].diff().dropna()
    n_gaps = (date_diffs != pd.Timedelta(days=1)).sum()

    d = (df["baseline_err"] - df["selected_err"]).values
    n = len(d)
    d_bar = d.mean()

    rng = np.random.default_rng(RNG_SEED)

    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"n = {n} paired daily errors, contiguity gaps in date sequence: {n_gaps}")
    print(f"Observed mean difference (baseline_err - selected_err): {d_bar:.6f}")
    if n_gaps > 0:
        print("WARNING: date sequence has gaps -- block bootstrap contiguity "
              "assumption violated, results below may not be valid.")

    print(f"\n{'Block len':>10} | {'Bootstrap p-value':>18} | {'95% CI (percentile)':>24}")
    print("-" * 60)
    for L in BLOCK_LENGTHS:
        d_centered = d - d_bar
        null_means = moving_block_bootstrap_mean(d_centered, L, N_BOOTSTRAP, rng)
        p_val = (np.abs(null_means) >= np.abs(d_bar)).mean()

        ci_means = moving_block_bootstrap_mean(d, L, N_BOOTSTRAP, rng)
        ci_lo, ci_hi = np.percentile(ci_means, [2.5, 97.5])

        print(f"{L:>10} | {p_val:>18.4f} | [{ci_lo:.6f}, {ci_hi:.6f}]")

    return {"label": label, "n": n, "d_bar": d_bar}


if __name__ == "__main__":
    results = []
    results.append(run_block_bootstrap("results/walk_forward_per_day_btc.csv", "BTC-USD"))
    results.append(run_block_bootstrap("results/walk_forward_per_day_eth.csv", "ETH-USD"))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for r in results:
        print(f"{r['label']}: n={r['n']}, observed mean diff={r['d_bar']:.6f}")
    print("\nCompare block-bootstrap p-values above against the paired t-test")
    print("p-values already reported (BTC: p=0.0079, ETH: p=0.0002) -- if")
    print("they're in the same ballpark across all block lengths, the t-test")
    print("result is robust to the independence assumption being wrong.")
    print("If block bootstrap p-values are substantially higher, the t-test")
    print("was overstating significance due to ignored autocorrelation.")