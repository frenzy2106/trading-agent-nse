"""
Technical snapshot tool.

get_technical_snapshot(ticker, lookback_days=365) -> dict

Returns price stats, computed indicators, and last 10 days of OHLCV.
Indicators: RSI-14, MACD(12/26/9), SMA-50/200, EMA-10, ATR-14, Bollinger-20.
"""

import logging
import time
from datetime import date, timedelta

import pandas_ta as ta

from cache import fetch_ohlcv_cached, get_nse_instruments
from kite_login import get_kite_client
from ticker_utils import to_plain

logger = logging.getLogger(__name__)


class TickerNotFoundError(Exception):
    """Raised when a ticker is not listed on NSE. Carries up to 3 close-name suggestions."""

    def __init__(self, ticker: str, suggestions: list[str] | None = None):
        self.ticker = ticker
        self.suggestions = suggestions or []
        super().__init__(f"Ticker '{ticker}' not found on NSE")


def _suggest_by_name(instruments: list[dict], query: str, limit: int = 3) -> list[str]:
    """Return up to `limit` NSE equity tradingsymbols whose name contains `query`."""
    q = query.lower()
    out: list[str] = []
    for i in instruments:
        if i.get("instrument_type") != "EQ":
            continue
        name = (i.get("name") or "").lower()
        if q in name:
            out.append(i["tradingsymbol"])
            if len(out) >= limit:
                break
    return out


def _fetch_ohlcv(plain: str, lookback_days: int, as_of_date: date | None = None):
    kite = get_kite_client()
    instruments = get_nse_instruments(kite)
    matches = [i for i in instruments if i["tradingsymbol"] == plain]
    if not matches:
        suggestions = _suggest_by_name(instruments, plain)
        raise TickerNotFoundError(plain, suggestions)
    token = matches[0]["instrument_token"]

    to_date = as_of_date or date.today()
    from_date = to_date - timedelta(days=lookback_days)
    return fetch_ohlcv_cached(kite, token, plain, from_date, to_date)


def _build_volume_block(df, last, val) -> dict:
    """Derived volume signals: VWMA-20 vs SMA-20, today vs 20d avg, OBV trend."""
    avg_20d = int(df["volume"].tail(20).mean())
    today_vol = int(last["volume"])
    vs_avg = round(today_vol / avg_20d, 2) if avg_20d else None

    # OBV trend: compare today vs 20 trading days ago, as a percentage.
    obv_change_pct: float | None = None
    if "OBV" in df.columns and len(df) >= 21:
        obv_now = df["OBV"].iloc[-1]
        obv_then = df["OBV"].iloc[-21]
        if obv_now is not None and obv_then is not None:
            denom = max(abs(float(obv_then)), 1.0)
            obv_change_pct = round((float(obv_now) - float(obv_then)) / denom * 100, 1)

    return {
        "vwma_20": val("VWMA_20"),
        "sma_20": val("SMA_20"),
        "vs_avg_ratio": vs_avg,
        "obv_change_20d_pct": obv_change_pct,
        "avg_20d": avg_20d,
        "avg_full": int(df["volume"].mean()),
    }


def get_technical_snapshot(
    ticker: str,
    lookback_days: int = 365,
    as_of_date: date | None = None,
) -> dict:
    plain = to_plain(ticker)
    t0 = time.time()
    logger.info(
        "get_technical_snapshot start | ticker=%s lookback=%d as_of=%s",
        plain, lookback_days, as_of_date or "today",
    )

    df = _fetch_ohlcv(plain, lookback_days, as_of_date)

    # compute indicators (appended in-place)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)
    df.ta.ema(length=10, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.vwma(length=20, append=True)
    df.ta.obv(append=True)

    last = df.iloc[-1]

    def val(col, decimals=2):
        """Return rounded float or None if column missing / NaN."""
        if col not in df.columns:
            return None
        v = last[col]
        import math
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else round(float(v), decimals)

    # 52-week stats
    high_52w = round(float(df["high"].max()), 2)
    low_52w = round(float(df["low"].min()), 2)
    current_price = round(float(last["close"]), 2)

    # recent 10 days (compact, for LLM context)
    recent = []
    for dt, row in df.tail(10).iterrows():
        recent.append({
            "date": dt.date().isoformat(),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]),
        })

    snapshot = {
        "ticker": plain,
        "as_of": (as_of_date or date.today()).isoformat(),
        "price": {
            "current": current_price,
            "week_52_high": high_52w,
            "week_52_low": low_52w,
            "pct_from_52w_high": round((current_price / high_52w - 1) * 100, 1),
            "pct_from_52w_low": round((current_price / low_52w - 1) * 100, 1),
        },
        "indicators": {
            "rsi_14": val("RSI_14"),
            "macd": {
                "macd": val("MACD_12_26_9"),
                "signal": val("MACDs_12_26_9"),
                "histogram": val("MACDh_12_26_9"),
            },
            "sma_50": val("SMA_50"),
            "sma_200": val("SMA_200"),
            "ema_10": val("EMA_10"),
            "atr_14": val("ATRr_14"),
            "bollinger": {
                "upper": val("BBU_20_2.0_2.0"),
                "mid": val("BBM_20_2.0_2.0"),
                "lower": val("BBL_20_2.0_2.0"),
            },
        },
        "volume": _build_volume_block(df, last, val),
        "recent_prices": recent,
    }

    logger.info(
        "get_technical_snapshot done | ticker=%s rows=%d latency=%.2fs",
        plain, len(df), time.time() - t0,
    )
    return snapshot
