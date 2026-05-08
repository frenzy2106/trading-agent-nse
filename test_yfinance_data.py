"""
Phase 0 smoke test: pull fundamentals from yfinance for one (or more) NSE tickers.

Usage:
    python test_yfinance_data.py                        # RELIANCE only
    python test_yfinance_data.py TCS HDFCBANK INFY      # multiple tickers
    python test_yfinance_data.py --all5                 # 5-sector coverage test
"""

import sys

import yfinance as yf

from ticker_utils import to_plain, to_yfinance

SECTOR_SAMPLE = ["RELIANCE", "TCS", "HDFCBANK", "SUNPHARMA", "TATASTEEL"]

FIELDS_OF_INTEREST = [
    # valuation
    "trailingPE", "forwardPE", "priceToBook",
    # profitability
    "returnOnEquity", "returnOnAssets",
    # debt
    "debtToEquity",
    # size
    "marketCap", "enterpriseValue",
    # growth
    "revenueGrowth", "earningsGrowth",
    # dividends
    "dividendYield",
    # misc
    "sector", "industry", "fullName",
]


def check_ticker(plain: str) -> dict:
    yf_ticker = to_yfinance(plain)
    info = yf.Ticker(yf_ticker).info

    result = {"ticker": plain, "yf_symbol": yf_ticker}
    for field in FIELDS_OF_INTEREST:
        result[field] = info.get(field, "N/A")

    # check income statement availability
    try:
        fin = yf.Ticker(yf_ticker).financials
        result["_income_stmt_rows"] = len(fin) if fin is not None else 0
    except Exception:
        result["_income_stmt_rows"] = "ERROR"

    return result


def print_coverage(tickers: list[str]):
    print(f"\n{'Ticker':<14} {'PE':>7} {'PB':>7} {'ROE':>7} {'D/E':>7} {'RevGrowth':>10} {'Sector'}")
    print("-" * 75)
    for plain in tickers:
        d = check_ticker(plain)
        pe = f"{d['trailingPE']:.1f}" if isinstance(d["trailingPE"], float) else d["trailingPE"]
        pb = f"{d['priceToBook']:.2f}" if isinstance(d["priceToBook"], float) else d["priceToBook"]
        roe = f"{d['returnOnEquity']:.2%}" if isinstance(d["returnOnEquity"], float) else d["returnOnEquity"]
        de = f"{d['debtToEquity']:.1f}" if isinstance(d["debtToEquity"], float) else d["debtToEquity"]
        rg = f"{d['revenueGrowth']:.2%}" if isinstance(d["revenueGrowth"], float) else d["revenueGrowth"]
        sector = str(d.get("sector", "N/A"))[:20]
        print(f"{plain:<14} {pe:>7} {pb:>7} {roe:>7} {de:>7} {rg:>10}  {sector}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--all5" in args:
        tickers = SECTOR_SAMPLE
        print("Running 5-sector coverage test...")
    elif args:
        tickers = [to_plain(a) for a in args]
    else:
        tickers = ["RELIANCE"]

    print_coverage(tickers)
    print("\nNote: 'N/A' means yfinance did not return that field for the ticker.")
