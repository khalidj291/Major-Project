"""
Person 1 - Day 4, Step 1: Label each day as volatile / stable / neutral.
Uses 30-day rolling std of returns. Top 30% of volatility values = "volatile",
bottom 30% = "stable", the middle 40% = "neutral".
Output: data/data_processed.csv (with a new 'regime' column added)
        results/regime_timeline.png
"""

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def add_regime_labels(processed_path=os.path.join(PROJECT_ROOT, "data", "data_processed.csv"), ticker=None):
    df = pd.read_csv(processed_path, parse_dates=["date"])
    if "ticker" in df.columns and ticker:
        mask = df["ticker"] == ticker
    else:
        mask = pd.Series(True, index=df.index)  # already single-asset file

    sub = df[mask].sort_values("date").reset_index(drop=True)
    sub["rolling_vol"] = sub["returns"].rolling(window=30).std()

    p70 = sub["rolling_vol"].quantile(0.70)
    p30 = sub["rolling_vol"].quantile(0.30)

    def label(v):
        if pd.isna(v):
            return "neutral"
        if v >= p70:
            return "volatile"
        elif v <= p30:
            return "stable"
        return "neutral"

    sub["regime"] = sub["rolling_vol"].apply(label)

    print(sub["regime"].value_counts())
    print(f"\n70th pct threshold: {p70:.5f} | 30th pct threshold: {p30:.5f}")

    # Sanity check: known volatile events should show up as "volatile"
    covid = sub[(sub["date"] >= "2020-02-15") & (sub["date"] <= "2020-04-15")]
    print(f"\nCOVID crash period (Feb-Apr 2020) regime breakdown:\n{covid['regime'].value_counts()}")

    # Save back
    if "ticker" in df.columns and ticker:
        df.loc[mask, "regime"] = sub["regime"].values
        df.loc[mask, "rolling_vol"] = sub["rolling_vol"].values
    else:
        df["regime"] = sub["regime"].values
        df["rolling_vol"] = sub["rolling_vol"].values

    df.to_csv(processed_path, index=False)

    # Plot: price with regime shading
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub["date"], sub["close"], color="black", linewidth=1)
    colors = {"volatile": "red", "stable": "green", "neutral": "lightgray"}
    for regime, color in colors.items():
        mask_r = sub["regime"] == regime
        ax.fill_between(sub["date"], sub["close"].min(), sub["close"].max(),
                         where=mask_r, color=color, alpha=0.15, label=regime)
    ax.set_title("SPY Price with Market Regime Overlay")
    ax.set_xlabel("Date"); ax.set_ylabel("Close price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "results", "regime_timeline.png"), dpi=120)
    print("\nSaved: results/regime_timeline.png")

    return sub

if __name__ == "__main__":
    add_regime_labels()