"""
Person 1 - Day 2, Step 1: Pull real US Personal Consumption Expenditure (PCE) data.
Uses FRED's direct CSV download link -- no API key/signup needed.
Output: data/data_consumer.csv (date, close, volume, returns)
"""
import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCE"

def fetch_and_clean_consumer_data():
    df = pd.read_csv(FRED_URL)
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2024-12-31")].reset_index(drop=True)

    df["volume"] = 1.0  # PCE has no volume concept -- placeholder, per the contract
    df["returns"] = df["close"].pct_change()
    df = df.dropna(subset=["returns"]).reset_index(drop=True)

    df = df[["date", "close", "volume", "returns"]]
    df.to_csv("data/data_consumer.csv", index=False)

    print(f"Saved data/data_consumer.csv -- {len(df)} rows")
    print(df.head())
    print("\nReturns summary:")
    print(df["returns"].describe()[["mean", "std", "min", "max"]])
    return df

if __name__ == "__main__":
    fetch_and_clean_consumer_data()