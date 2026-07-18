"""
Person 1 - Day 1, Step 2: Pull REAL data. Run this on YOUR machine (needs internet).
Requires: pip install yfinance --break-system-packages   (or without the flag on Windows/Mac)

Output: data/data_raw.csv  (date, ticker, close, volume)
This replaces the synthetic placeholder Claude generated to test the pipeline —
everything downstream (clean_features.py, train_baseline.py) works unchanged.
"""
import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "SPY", "BTC-USD"]
START = "2015-01-01"
END = "2024-12-31"

def fetch_all():
    frames = []
    for ticker in TICKERS:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=START, end=END, progress=False)
        df = df.reset_index()[["Date", "Close", "Volume"]]
        df.columns = ["date", "close", "volume"]
        df["ticker"] = ticker
        frames.append(df)
        print(f"  {len(df)} rows")

    raw = pd.concat(frames, ignore_index=True)[["date", "ticker", "close", "volume"]]
    raw.to_csv("data/data_raw.csv", index=False)
    print(f"\nSaved data/data_raw.csv — {len(raw)} total rows")
    print(raw.head())
    return raw

if __name__ == "__main__":
    fetch_all()