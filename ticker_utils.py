"""Ticker normalization between plain, Kite (NSE:X), and yfinance (X.NS) formats."""

import re


def to_plain(ticker: str) -> str:
    """'NSE:RELIANCE' or 'RELIANCE.NS' -> 'RELIANCE'"""
    ticker = ticker.strip().upper()
    if ticker.startswith("NSE:"):
        return ticker[4:]
    if ticker.endswith(".NS"):
        return ticker[:-3]
    return ticker


def to_kite(ticker: str) -> str:
    """Any format -> 'NSE:RELIANCE'"""
    return f"NSE:{to_plain(ticker)}"


def to_yfinance(ticker: str) -> str:
    """Any format -> 'RELIANCE.NS'"""
    return f"{to_plain(ticker)}.NS"


if __name__ == "__main__":
    samples = ["RELIANCE", "NSE:TCS", "HDFCBANK.NS", "nse:infy", "wipro.ns"]
    for s in samples:
        print(f"{s:20s} -> plain={to_plain(s):15s} kite={to_kite(s):20s} yf={to_yfinance(s)}")
