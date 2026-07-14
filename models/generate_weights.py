import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # models/ -> project/
sys.path.append(SCRIPT_DIR)
import pandas as pd
import numpy as np
from ebbinghaus import ebbinghaus_weight

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), parse_dates=["date"])

# Training period: 2015-2022 (matches baseline model split in the master plan)
train_df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2022-12-31")].reset_index(drop=True)

# TEAM CONVENTION -- LOCKED, do not change without telling Person 1 and Person 3:
# "recent" is always measured from the last date of the TRAINING PERIOD being used
# right now, NEVER from the last date of the full dataset (which may extend to 2024).
# Getting this wrong silently breaks the whole decay mechanism -- training-period
# data that should look "recent" would incorrectly look "old" to the formula.
reference_date = train_df["date"].max()
assert reference_date <= pd.Timestamp("2022-12-31"), (
    "Reference date must be the end of the TRAINING period, not the full dataset. "
    "Check that train_df is correctly filtered before computing reference_date."
)

print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"Training rows: {len(train_df)}")
print(f"Reference date (weight=1.0 point): {reference_date.date()}")

S_values = {"fast": 30, "medium": 180, "slow": 365}

for name, S in S_values.items():
    weights = ebbinghaus_weight(train_df["date"], reference_date, S)
    out_path = os.path.join(SCRIPT_DIR, f"weights_{name}.npy")
    np.save(out_path, weights)
    print(f"\n{name} (S={S}) -> saved to weights_{name}.npy")
    print(f"  First 5 (oldest, 2015): {weights[:5]}")
    print(f"  Last 5 (most recent, end of 2022): {weights[-5:]}")