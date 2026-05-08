"""
Phase 0 smoke test: pull 12 months of daily OHLCV from Kite for one ticker.

Usage:
    python test_kite_data.py            # defaults to RELIANCE
    python test_kite_data.py TCS
"""

import sys
from datetime import date, timedelta

import pandas as pd

from kite_login import get_kite_client
from ticker_utils import to_kite, to_plain


def get_instrument_token(kite, exchange: str, tradingsymbol: str) -> int:
    instruments = kite.instruments(exchange)
    matches = [i for i in instruments if i["tradingsymbol"] == tradingsymbol]
    if not matches:
        sys.exit(f"ERROR: '{tradingsymbol}' not found on {exchange}. Check the ticker symbol.")
    return matches[0]["instrument_token"]


def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    plain = to_plain(ticker)
    kite = get_kite_client()

    print(f"Looking up instrument token for {plain} on NSE...")
    token = get_instrument_token(kite, "NSE", plain)
    print(f"  instrument_token = {token}")

    to_date = date.today()
    from_date = to_date - timedelta(days=365)

    print(f"Fetching daily OHLCV {from_date} -> {to_date}...")
    records = kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval="day",
    )

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    df = fetch_ohlcv(ticker)

    print(f"\n--- {to_plain(ticker)} | Last 10 trading days ---")
    print(df.tail(10).to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")
