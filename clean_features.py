"""
Person 1 - Day 1, Step 3: Clean data_raw.csv and engineer the returns column.
Input:  data/data_raw.csv       (date, ticker, close, volume)
Output: data/data_processed.csv (date, ticker, close, volume, returns)
"""
import pandas as pd

def clean_and_engineer(raw_path="data/data_raw.csv", out_path="data/data_processed.csv"):
    df = pd.read_csv(raw_path, parse_dates=["date"])

    # Drop rows with missing close/volume
    before = len(df)
    df = df.dropna(subset=["close", "volume"])
    print(f"Dropped {before - len(df)} rows with nulls")

    # Returns must be calculated PER TICKER, sorted by date, or you leak
    # yesterday's AAPL price into today's SPY return by accident.
    df = df.sort_values(["ticker", "date"])
    df["returns"] = df.groupby("ticker")["close"].pct_change()

    # First row per ticker has no return (nothing to compare to) — drop it
    df = df.dropna(subset=["returns"])

    df = df[["date", "ticker", "close", "volume", "returns"]].reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {out_path} — {len(df)} rows")
    print("\nReturns summary by ticker:")
    print(df.groupby("ticker")["returns"].describe()[["mean", "std", "min", "max"]])
    return df

if __name__ == "__main__":
    clean_and_engineer()