import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

import pandas as pd
import numpy as np
from ebbinghaus import ebbinghaus_weight

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_consumer.csv"), parse_dates=["date"])

# Training period: 2015-2022 (same convention as financial domain)
train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)

# TEAM CONVENTION -- same locked rule as financial domain: reference date is the
# end of the TRAINING period being used right now, not the full dataset's end.
reference_date = train_df["date"].max()
assert reference_date <= pd.Timestamp("2022-12-31"), (
    "Reference date must be the end of the TRAINING period, not the full dataset."
)

print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"Training rows: {len(train_df)}")
print(f"Reference date (weight=1.0 point): {reference_date.date()}")

# CONSUMER-DOMAIN DECAY RATES -- different from the financial domain.
# This data is MONTHLY, not daily. Reusing S=30/180/365 (tuned for daily data)
# would make "fast" decay forget everything except the single most recent month,
# which isn't a meaningful "fast forgetting" test on data that only updates 12x/year.
# Dates are still real calendar dates, so the same day-based formula works --
# we just need larger S values to represent month-scale memory:
#   fast   (~1 month memory)  -> S=30   (unchanged, 1 month is 1 month either way)
#   medium (~1 year memory)   -> S=365  (was 180 in the daily/financial version)
#   slow   (~2 year memory)   -> S=730  (was 365 in the daily/financial version)
S_values = {"fast": 30, "medium": 365, "slow": 730}

for name, S in S_values.items():
    weights = ebbinghaus_weight(train_df["date"], reference_date, S)
    out_path = os.path.join(SCRIPT_DIR, f"weights_{name}_consumer.npy")
    np.save(out_path, weights)
    print(f"\n{name} (S={S}) -> saved to weights_{name}_consumer.npy")
    print(f"  First 5 (oldest, 2015): {weights[:5]}")
    print(f"  Last 5 (most recent, end of 2022): {weights[-5:]}")