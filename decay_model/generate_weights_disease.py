import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR)

import pandas as pd
import numpy as np
from ebbinghaus import ebbinghaus_weight

df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "data_disease.csv"), parse_dates=["date"])

# Training period matches train_baseline_disease.py's TRAIN_END -- see that
# file for why this domain uses its own boundary instead of 2015-2022.
train_df = df[(df["date"] >= df["date"].min()) & (df["date"] <= "2022-04-02")].reset_index(drop=True)

reference_date = train_df["date"].max()
print(f"Training period: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"Training rows: {len(train_df)}")
print(f"Reference date (weight=1.0 point): {reference_date.date()}")

# DISEASE-DOMAIN DECAY RATES -- daily data like the financial domain, so the
# same S values apply (no monthly rescaling needed, unlike the consumer domain).
S_values = {"fast": 30, "medium": 180, "slow": 365}

for name, S in S_values.items():
    weights = ebbinghaus_weight(train_df["date"], reference_date, S)
    out_path = os.path.join(PROJECT_ROOT, "decay_model", "models", f"weights_{name}_disease.npy")
    np.save(out_path, weights)
    print(f"\n{name} (S={S}) -> saved to weights_{name}_disease.npy")
    print(f"  First 5 (oldest, 2020): {weights[:5]}")
    print(f"  Last 5 (most recent, end of training period): {weights[-5:]}")