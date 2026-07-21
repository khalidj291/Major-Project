"""
Person 1/2 - Stretch domain, Step 1: Pull real US COVID-19 case data.
Source: Our World in Data (OWID), public GitHub mirror, no API key needed.
This is a THIRD domain, added after core project results showed decay
weighting doesn't beat baseline on financial or consumer data -- disease
case data was tested first in a sandbox check (see project log) because
it has a real, checkable reason to be genuinely non-stationary (pandemic
chaos vs. later endemic behavior), unlike SPY/PCE which are statistically
stable across their whole history.
Output: data/data_disease.csv (date, close, volume, returns)
"""

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import pandas as pd
import numpy as np

OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/jhu/full_data.csv"

def fetch_and_clean_disease_data(location="United States"):
    df = pd.read_csv(OWID_URL, parse_dates=["date"])
    us = df[df["location"] == location].sort_values("date").reset_index(drop=True)

    # 7-day rolling mean of new cases -- reduces weekend/holiday reporting noise,
    # which otherwise dominates the raw daily series (see data_documentation.md
    # for how large those reporting-artifact swings are).
    us["smoothed_cases"] = us["new_cases"].rolling(7, min_periods=1).mean()
    us = us[us["smoothed_cases"] > 0].reset_index(drop=True)

    us["close"] = us["smoothed_cases"]
    us["volume"] = 1.0  # no volume concept here -- placeholder, same convention as PCE
    us["returns"] = us["close"].pct_change()
    us = us.replace([np.inf, -np.inf], np.nan).dropna(subset=["returns"]).reset_index(drop=True)

    out = us[["date", "close", "volume", "returns"]]
    out_path = os.path.join(PROJECT_ROOT, "data", "data_disease.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved data/data_disease.csv -- {len(out)} rows")
    print(f"Date range: {out['date'].min().date()} to {out['date'].max().date()}")
    print(out.head())
    print("\nReturns summary:")
    print(out["returns"].describe()[["mean", "std", "min", "max"]])
    return out

if __name__ == "__main__":
    fetch_and_clean_disease_data()