"""
Person 1 - Day 4, Step 3: Add regime labels to consumer spending data.
Uses a 12-month rolling window (not 30) -- consumer data is monthly, so a
30-period window would span 2.5 years and leave almost no usable rows.
This matches the 12-period window already used in train_baseline_consumer.py.
Output: data/data_consumer.csv (with new 'regime' column added)
        results/regime_timeline_consumer.png
"""
import pandas as pd
import matplotlib.pyplot as plt

WINDOW = 12

def add_consumer_regime_labels(path="data/data_consumer.csv"):
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df["rolling_vol"] = df["returns"].rolling(window=WINDOW).std()

    p70 = df["rolling_vol"].quantile(0.70)
    p30 = df["rolling_vol"].quantile(0.30)

    def label(v):
        if pd.isna(v):
            return "neutral"
        if v >= p70:
            return "volatile"
        elif v <= p30:
            return "stable"
        return "neutral"

    df["regime"] = df["rolling_vol"].apply(label)
    print(df["regime"].value_counts())
    print(f"\n70th pct threshold: {p70:.5f} | 30th pct threshold: {p30:.5f}")

    df.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["date"], df["close"], color="black", linewidth=1.2, marker="o", markersize=3)
    colors = {"volatile": "red", "stable": "green", "neutral": "lightgray"}
    for regime, color in colors.items():
        mask = df["regime"] == regime
        ax.fill_between(df["date"], df["close"].min(), df["close"].max(),
                         where=mask, color=color, alpha=0.15, label=regime)
    ax.set_title("Consumer Spending (PCE) with Regime Overlay (12-month window)")
    ax.set_xlabel("Date"); ax.set_ylabel("PCE value")
    ax.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig("results/regime_timeline_consumer.png", dpi=120)
    print("\nSaved: results/regime_timeline_consumer.png")
    return df

if __name__ == "__main__":
    add_consumer_regime_labels()