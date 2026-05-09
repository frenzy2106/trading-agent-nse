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

from tools.commentary import get_management_commentary as _commentary
from tools.fundamentals import get_fundamentals_snapshot as _fundamentals
from tools.macro import get_macro_snapshot as _macro
from tools.news import get_news_and_earnings as _news
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
def get_news_and_earnings(ticker: str) -> str:
    """
    Fetch upcoming + recent earnings dates and recent news headlines for an
    NSE-listed stock.

    Returns:
      - next_earnings_date + days_to_earnings (if available)
      - eps_estimate (consensus, if available)
      - last_earnings_date + days_since_last_earnings + last_eps_surprise_pct
      - recent_news: top 5 headlines with title, short summary, date, provider

    Use this to surface timing risks (e.g. earnings within 7 days = avoid fresh
    positions until after the print) and stock-specific news the price/indicator
    data can't see. The LLM should filter relevance — yfinance sometimes returns
    market-wide headlines for large-caps; ignore items not specific to the company.

    On error, returns JSON like {"error": "..."}.

    Args:
        ticker: NSE ticker symbol, e.g. RELIANCE, TCS, HDFCBANK
    """
    try:
        return json.dumps(_news(ticker), default=str)
    except (TickerNotFoundError, TokenException) as e:
        return _serialise_error(e)
    except Exception as e:
        return _serialise_error(e)


@tool
def get_management_commentary(ticker: str, query: str, k: int = 5) -> str:
    """
    Retrieve top-k quotes from indexed earnings call transcripts (last ~12 months)
    for an NSE-listed stock, semantically matched to your query.

    USE for qualitative dimensions where management's own words add value:
      - growth / margin / capex guidance
      - characterization of demand environment, segment momentum
      - capital allocation intent (buybacks, debt paydown, dividends)
      - explicit risks management names

    DO NOT use for numbers — get_fundamentals_snapshot has those. Concalls are
    management's own framing; treat returned quotes as INPUT, not GROUND TRUTH.

    On error, returns JSON like {"error": "...", "kind": "no_commentary"} —
    say "management commentary unavailable" and do not invent quotes.

    Args:
        ticker: NSE ticker symbol, e.g. RELIANCE
        query: a focused question, e.g. "margin guidance for FY26", "capex priorities"
        k: number of chunks to return (default 5)
    """
    try:
        return json.dumps(_commentary(ticker, query, k), default=str)
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
