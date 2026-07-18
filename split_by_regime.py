"""
Person 1 - Day 4, Step 2: Slice data_processed.csv by regime into three files.
Run this AFTER add_regime_labels.py (needs the 'regime' column to already exist).
Output: data/data_volatile.csv, data/data_stable.csv, data/data_neutral.csv
"""
import pandas as pd

def split_by_regime(processed_path="data/data_processed.csv"):
    df = pd.read_csv(processed_path, parse_dates=["date"])

    if "regime" not in df.columns:
        raise ValueError("No 'regime' column found -- run add_regime_labels.py first.")

    volatile = df[df["regime"] == "volatile"]
    stable = df[df["regime"] == "stable"]
    neutral = df[df["regime"] == "neutral"]

    volatile.to_csv("data/data_volatile.csv", index=False)
    stable.to_csv("data/data_stable.csv", index=False)
    neutral.to_csv("data/data_neutral.csv", index=False)

    print(f"data_volatile.csv: {len(volatile)} rows")
    print(f"data_stable.csv:   {len(stable)} rows")
    print(f"data_neutral.csv:  {len(neutral)} rows")

if __name__ == "__main__":
    split_by_regime()