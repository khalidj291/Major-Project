"""
fetch_data_btc.py -- Person 1 style: run this on YOUR machine (needs internet).
Requires: pip install yfinance --break-system-packages   (or without the flag on Windows/Mac)

Refreshes data/data_btc.csv with BTC-USD daily prices from START through
TODAY, every time you run it. Re-run this any time you want up-to-date data --
no manual editing needed.

Note: this pulls from Yahoo Finance via yfinance, same source your existing
fetch_data.py already uses for SPY/AAPL. Yahoo's BTC-USD history starts around
2014-09-17 (a bit later than the 2013-10-02 snapshot the validated S=30 result
in train_feature_decay_btc.py was picked on) -- so a fresh pull will have
different exact row counts, and it'll keep growing every time you re-run it.
The feature-decay MECHANISM doesn't change, but if you refresh with a lot of
new data it's worth re-running train_feature_decay_btc.py's validation sweep
rather than assuming S=30 is still the best choice -- more recent years might
genuinely prefer a different S, and that's worth checking rather than assuming.
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
from datetime import date
import yfinance as yf
import pandas as pd

TICKER = "BTC-USD"
START = "2013-10-02"  # yfinance will just start from whenever real data actually begins
END = date.today().strftime("%Y-%m-%d")  # always "today" -- this is what makes re-running it "update"

def fetch_btc():
    print(f"Downloading {TICKER} from {START} to {END} (today)...")
    df = yf.download(TICKER, start=START, end=END, progress=False)
    df = df.reset_index()[["Date", "Close", "Volume"]]
    df.columns = ["date", "close", "volume"]
    df = df.sort_values("date").reset_index(drop=True)
    df["returns"] = df["close"].pct_change()
    df = df.dropna(subset=["returns"]).reset_index(drop=True)

    out_path = os.path.join(PROJECT_ROOT, "data", "data_btc.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} -- {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.tail())
    return df

if __name__ == "__main__":
    fetch_btc()