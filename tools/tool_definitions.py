"""
LangChain @tool wrappers around the plain-Python tool functions.
Imported by lg_agent.py and bound to the LLM.

Errors are caught and returned as JSON `{"error": ..., "suggestions": [...]}`
so the LLM can see them as tool responses (rather than crashing the graph)
and either pick a suggested ticker or explain the issue to the user.
"""

import json
import logging

from kiteconnect.exceptions import TokenException
from langchain_core.tools import tool

from tools.fundamentals import get_fundamentals_snapshot as _fundamentals
from tools.macro import get_macro_snapshot as _macro
from tools.technical import TickerNotFoundError
from tools.technical import get_technical_snapshot as _technical

logger = logging.getLogger(__name__)


def _serialise_error(exc: Exception) -> str:
    """Convert any caught tool-side exception to a JSON string the LLM can read."""
    if isinstance(exc, TickerNotFoundError):
        msg = f"Ticker '{exc.ticker}' not found on NSE."
        if exc.suggestions:
            msg += f" Possible matches by name: {', '.join(exc.suggestions)}."
        payload = {"error": msg, "suggestions": exc.suggestions, "kind": "ticker_not_found"}
    elif isinstance(exc, TokenException):
        payload = {
            "error": (
                "Kite access token has expired (tokens expire daily at 6 AM IST). "
                "The user must run `python kite_login.py` to refresh before retrying."
            ),
            "kind": "kite_token_expired",
        }
    else:
        payload = {"error": f"{type(exc).__name__}: {exc}", "kind": "unexpected"}

    logger.warning("tool error | kind=%s | %s", payload["kind"], payload["error"])
    return json.dumps(payload)


@tool
def get_technical_snapshot(ticker: str, lookback_days: int = 365) -> str:
    """
    Fetch a technical analysis snapshot for an NSE-listed stock.

    Returns price stats (current, 52-week high/low), momentum indicators
    (RSI-14, MACD), trend indicators (SMA-50, SMA-200, EMA-10), volatility
    (ATR-14, Bollinger Bands), volume averages, and last 10 days of OHLCV.

    On error, returns JSON like {"error": "...", "suggestions": [...]} —
    if you see this, do NOT generate a recommendation; surface the message
    to the user.

    Args:
        ticker: NSE ticker symbol, e.g. RELIANCE, TCS, HDFCBANK
        lookback_days: history window in days (default 365)
    """
    try:
        return json.dumps(_technical(ticker, lookback_days), default=str)
    except (TickerNotFoundError, TokenException) as e:
        return _serialise_error(e)
    except Exception as e:
        return _serialise_error(e)


@tool
def get_macro_snapshot(ticker: str) -> str:
    """
    Benchmark an NSE stock against NIFTY 50 (broad market) and its sector index
    (e.g. NIFTY IT for tech, NIFTY BANK for banks) over 1m/3m/6m/12m windows.

    Returns the stock's returns, the indices' returns, and the relative
    out/underperformance at each window. Use this to judge whether a stock's
    move is sector-driven or stock-specific.

    On error, returns JSON like {"error": "..."} — if you see this, do NOT
    generate analysis; surface the message.

    Args:
        ticker: NSE ticker symbol, e.g. RELIANCE, TCS, HDFCBANK
    """
    try:
        return json.dumps(_macro(ticker), default=str)
    except (TickerNotFoundError, TokenException) as e:
        return _serialise_error(e)
    except Exception as e:
        return _serialise_error(e)


@tool
def get_fundamentals_snapshot(ticker: str) -> str:
    """
    Fetch a fundamentals snapshot for an NSE-listed stock.

    Returns sector/industry, valuation ratios (P/E trailing+forward, P/B),
    profitability (ROE, ROA, ROCE), leverage (D/E), dividend yield, and
    last 4 quarters of revenue and EPS.

    On error, returns JSON like {"error": "..."} — if you see this, do NOT
    generate a recommendation; surface the message to the user.

    Args:
        ticker: NSE ticker symbol, e.g. RELIANCE, TCS, HDFCBANK
    """
    try:
        return json.dumps(_fundamentals(ticker), default=str)
    except Exception as e:
        return _serialise_error(e)
