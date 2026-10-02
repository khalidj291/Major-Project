"""
garch_baseline_comparison.py -- adds the baseline a financial-forecasting
reviewer will expect (Ridge alone is a weak baseline by that field's
standards). Uses an AR(1)-GARCH(1,1) model: the AR(1) piece gives a real
point forecast for next-day returns (needed to compute MAE like every
other technique here), the GARCH(1,1) piece models volatility clustering,
which is GARCH's actual purpose -- this is the standard way to get a
comparable point forecast out of a GARCH-family model, not a plain
GARCH-only setup (which would just forecast variance, not a return value).

Reuses the EXACT same 13 six-month walk-forward windows as
walk_forward_validation_btc.py, and reuses that script's saved per-day
baseline/decay-selected errors for a fair, identical-window comparison --
does not require re-running it.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(os.path.join(ROOT, "data", "data_btc.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
per_day = pd.read_csv(os.path.join(os.path.dirname(__file__), "results", "walk_forward_per_day_btc.csv"), parse_dates=["date", "window_start"])

test_starts = pd.date_range("2020-01-01", "2026-01-01", freq="6MS")
rows = []
garch_err_by_date = {}

for i, test_start in enumerate(test_starts):
    test_end = (test_starts[i + 1] - pd.Timedelta(days=1)) if i + 1 < len(test_starts) else df["date"].max()
    train = df[df["date"] < test_start]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    if len(test) == 0 or len(train) < 250:
        continue

    # AR(1) mean + GARCH(1,1) volatility, fit once per window on that window's
    # training data (expanding, same as the decay walk-forward) -- returns
    # scaled to percent for numerical stability, standard practice for arch_model
    am = arch_model(train["returns"] * 100, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=len(test), reindex=False)
    mean_forecast_pct = fc.mean.values[0]  # multi-step analytic mean forecast, scaled back below
    pred = mean_forecast_pct / 100.0

    actual = test["returns"].values
    garch_err = np.abs(actual - pred)
    for d, e in zip(test["date"].values, garch_err):
        garch_err_by_date[pd.Timestamp(d)] = e

    rows.append({
        "test_window_start": test_start, "test_window_end": test_end, "n_test": len(test),
        "garch_MAE": garch_err.mean(),
    })

garch_df = pd.DataFrame(rows)

# merge with the existing per-day results for an identical-window, identical-day comparison
per_day["garch_err"] = per_day["date"].map(garch_err_by_date)
merged = per_day.dropna(subset=["garch_err"])

baseline_mae = merged["baseline_err"].mean()
selected_mae = merged["selected_err"].mean()
garch_mae = merged["garch_err"].mean()

t1, p1 = stats.ttest_rel(merged["selected_err"], merged["garch_err"])
t2, p2 = stats.ttest_rel(merged["baseline_err"], merged["garch_err"])

print(garch_df.to_string(index=False))
print(f"\nPooled across {len(merged)} test days:")
print(f"  Ridge baseline MAE      : {baseline_mae:.6f}")
print(f"  decay-selected MAE      : {selected_mae:.6f}")
print(f"  GARCH(1,1)+AR(1) MAE    : {garch_mae:.6f}")
print(f"\ndecay-selected vs GARCH : t={t1:.4f}, p={p1:.4f}")
print(f"Ridge baseline vs GARCH : t={t2:.4f}, p={p2:.4f}")

out_dir = os.path.join(os.path.dirname(__file__), "results")
garch_df.to_csv(os.path.join(out_dir, "garch_comparison_windows_btc.csv"), index=False)
pd.DataFrame([{
    "n_days": len(merged), "baseline_MAE": baseline_mae, "selected_MAE": selected_mae, "garch_MAE": garch_mae,
    "selected_vs_garch_t": t1, "selected_vs_garch_p": p1,
    "baseline_vs_garch_t": t2, "baseline_vs_garch_p": p2,
}]).to_csv(os.path.join(out_dir, "garch_comparison_summary_btc.csv"), index=False)
print(f"\nSaved decay_model/results/garch_comparison_windows_btc.csv")
print(f"Saved decay_model/results/garch_comparison_summary_btc.csv")