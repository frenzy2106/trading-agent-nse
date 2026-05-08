"""
Phase 1 tool tests — shape + type checks, then a live run.

Usage:
    python test_tools.py              # RELIANCE
    python test_tools.py TCS
"""

import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from tools.technical import get_technical_snapshot
from tools.fundamentals import get_fundamentals_snapshot


def assert_shape_technical(snap: dict, ticker: str):
    assert snap["ticker"] == ticker, f"ticker mismatch: {snap['ticker']}"
    assert isinstance(snap["as_of"], str)

    price = snap["price"]
    assert isinstance(price["current"], float)
    assert isinstance(price["week_52_high"], float)
    assert isinstance(price["week_52_low"], float)
    assert isinstance(price["pct_from_52w_high"], float)
    assert isinstance(price["pct_from_52w_low"], float)

    ind = snap["indicators"]
    # RSI should be 0-100 when data is sufficient
    if ind["rsi_14"] is not None:
        assert 0 <= ind["rsi_14"] <= 100, f"RSI out of range: {ind['rsi_14']}"

    assert isinstance(snap["volume"]["avg_20d"], int)
    assert len(snap["recent_prices"]) <= 10
    assert all("close" in r for r in snap["recent_prices"])
    print(f"  [PASS] technical shape checks for {ticker}")


def assert_shape_fundamentals(snap: dict, ticker: str):
    assert snap["ticker"] == ticker
    assert isinstance(snap["as_of"], str)
    assert isinstance(snap["profile"], dict)
    assert isinstance(snap["valuation"], dict)
    assert isinstance(snap["profitability"], dict)
    assert isinstance(snap["leverage"], dict)
    assert isinstance(snap["quarterly_trend"], list)
    print(f"  [PASS] fundamentals shape checks for {ticker}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    plain = ticker.upper().removeprefix("NSE:").removesuffix(".NS")

    print(f"\n=== Technical snapshot: {plain} ===")
    tech = get_technical_snapshot(plain)
    assert_shape_technical(tech, plain)
    print(json.dumps(tech, indent=2, default=str))

    print(f"\n=== Fundamentals snapshot: {plain} ===")
    fund = get_fundamentals_snapshot(plain)
    assert_shape_fundamentals(fund, plain)
    print(json.dumps(fund, indent=2, default=str))
