# Project Log — Ebbinghaus Decay Weighting
*Person 2 (core model + documentation) — research diary*

**Note for review:** this is a factual reconstruction of what was actually
built, tested, found, and fixed, in order — draft it into your own voice
before final submission, but every number and event below is verified
against the current repository code, not estimated.

---

## Day 1 — Core decay mechanism

Built `ebbinghaus_weight(date, reference_date, S)` implementing
`R = e^(-t/S)`. Generated weight arrays for three decay rates — fast
(S=30), medium (S=180), slow (S=365) — over the financial training period
(2015-2022). Trained three Ridge Regression models using these as
`sample_weight`, same architecture and window (30 days) as Person 1's
baseline so the comparison isolates the weighting mechanism specifically.

First results (financial, full test period, n=501):
- naive-zero: MAE 0.006186
- baseline: MAE 0.006154 (beats naive)
- fast: MAE 0.006276, medium: 0.006215, slow: 0.006177 (none beat baseline)

Baseline wins from day one on the full period. Not discouraging on its
own — the actual hypothesis was always about regime-specific behavior,
not full-period dominance.

## Day 2 — Consumer domain + regime framework

Extended to the consumer (PCE) domain. Important adjustment made here:
consumer data is monthly, not daily, so reusing S=30/180/365 (tuned for
daily granularity) would make "fast" decay collapse to near-zero weight
for all but the single most recent month — not a meaningful test at
monthly scale. Recalibrated consumer-domain decay rates to S=30/365/730
(still representing roughly 1-month / 1-year / 2-year memory horizons,
just expressed correctly for monthly-spaced data).

Consumer full-period results (n=24): baseline MAE 0.002192, clearly ahead
of all three decay variants (fast 0.004141, medium 0.002573, slow
0.002503). Baseline's advantage is much larger here than on financial
data — PCE is simply more predictable month-to-month than daily equity
returns, which matches the much larger baseline-vs-naive gap on this
domain too (naive MAE 0.005198).

Built `evaluate_by_regime()` as a shared, independently-testable function
(validated against synthetic data before real regime labels existed) so
the regime-sensitivity analysis wouldn't be a one-off script.

## Day 3 — Sensitivity and significance testing

Tested 8 decay rates (S=14 through S=730) rather than just the original
three, to check whether fast/medium/slow were near any real optimum or
arbitrary. MAE decreased monotonically as S increased — no interior
sweet spot; S=730 (slowest tested) performed best, at MAE 0.006148,
narrowly ahead of baseline's 0.006154. Ran a paired t-test on this
best-case comparison: p=0.7058, not significant. Honest read: at S=730
the decay weighting is so gradual it's converging toward uniform
weighting rather than representing a meaningfully different model.

## Day 4 — Regime sensitivity (headline finding, revised on Day 5)

Ran the core hypothesis test: does the best decay rate differ between
volatile and stable market periods? Financial domain, test-period split:
62 volatile days, 96 stable days (of 501 total). Also ran the same split
on consumer data (9 volatile, 12 stable months, of 24 total) — small
sample, results here should be read as directional only.

## Day 5 — Two bugs found and fixed; result reversed and re-verified

Two implementation issues surfaced during integration review, both
significant enough to change reported numbers:

1. **Windowing target offset.** The decay-side windowing function set the
   prediction target to the day *after* the feature window ended, while
   the baseline used the day the window ended on — a one-day gap with no
   modeling justification, present in 7 files. Under the original
   (buggy) version, fast decay appeared to win in volatile periods. Fixed
   by aligning all 7 files to baseline's convention (no gap). All models
   retrained after the fix.

2. **Ticker-mixing.** `data_processed.csv` is intentionally multi-ticker
   (AAPL/BTC-USD/SPY stacked). A later code review caught that
   `decay_rate_sensitivity.py` and `statistical_significance.py` were
   still reading it with no ticker filter at all — silently mixing in
   BTC-USD's much larger daily swings, which had been roughly doubling
   every MAE these two scripts reported (baseline showing ~0.0123
   instead of the correct ~0.0062). Fixed by adding the filter, and
   consolidated the windowing logic that had been separately duplicated
   across all 7 files into one shared, tested module (`windowing.py`),
   so this class of bug — a fix applied to some copies but not all —
   can't recur silently again.

**Result after both fixes, verified by re-running every script:**
Baseline wins on the full period, in both domains (unchanged from Day 1-3
— those numbers were already correct). But baseline *also* now wins both
regime splits, in both domains — reversing what looked like the headline
finding under the buggy windowing. Financial volatile subset:
baseline MAE 0.008374 vs fast decay's 0.008608 (p=0.1629, not
significant). Consumer: baseline wins both regimes too.

**This is the final, honest result:** across every test we ran — full
period, regime split, 8 decay rates, both domains — recency-weighted
training via the Ebbinghaus formula did not outperform uniformly-weighted
Ridge regression, at any tested configuration. We're reporting this
directly rather than reframing around the earlier, buggier result that
looked more favorable. The process of catching and correcting two real
implementation bugs, and re-verifying every downstream number after each
fix, is itself part of what this log is documenting.

## Open thread — BTC-USD (stretch goal, not core result)

Proposed testing decay weighting on BTC-USD given its much higher
volatility (daily return std ~3x SPY's). Flagged one design issue before
building anything: reusing the SPY/PCE 2023-2024 test window would put
very few genuinely volatile BTC days in the test set, since most of
BTC's historical turbulence (2018, 2020, 2022) falls before 2023 —
same small-sample problem as the core project, for an avoidable reason.
Proposed an asset-specific split (train 2015-2021, test 2022-2024) so the
2022 crash lands in the test period. Not yet built; parked until the
core project (including this log, `cross_domain_comparison.csv`, and a
refreshed `statistical_significance.txt`) is fully wrapped.