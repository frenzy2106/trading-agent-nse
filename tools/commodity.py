"""
Commodity snapshot tool — fetches spot price, momentum windows, volatility,
and trend regime for a commodity using yfinance futures tickers (with ETF
proxies for inputs that don't have a clean futures contract).

Designed to support input-cost reasoning for stocks with commodity exposure:
  - Jewellers → gold, silver
  - Refiners / OMC / paints / chemicals → crude (WTI/Brent), naphtha proxies
  - City gas distribution / fertilizers → natural gas
  - Chip / cables / EV → copper, gold (bonding wire), palladium, lithium
  - Steel / aluminium players → SLX / aluminium futures
  - Power / nuclear / uranium plays → uranium ETF proxy

The agent first identifies what inputs the company actually flags as material
(via the concall RAG tool — management discloses cost mix in earnings calls),
then calls this tool for the relevant inputs.

Prices are USD-denominated. The LLM should reason about *direction* primarily
(magnitude of the % change is the signal, not absolute INR conversion).
"""

import logging
import time
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


# name → (yfinance ticker, display unit, kind)
# kind = "futures" (direct contract) or "etf_proxy" (ETF used as directional proxy)
COMMODITY_CATALOG: dict[str, tuple[str, str, str]] = {
    # Precious metals
    "gold": ("GC=F", "USD/oz", "futures"),
    "silver": ("SI=F", "USD/oz", "futures"),
    "platinum": ("PL=F", "USD/oz", "futures"),
    "palladium": ("PA=F", "USD/oz", "futures"),

    # Base metals
    "copper": ("HG=F", "USD/lb", "futures"),
    "aluminum": ("ALI=F", "USD/lb", "futures"),
    "aluminium": ("ALI=F", "USD/lb", "futures"),  # spelling alias

    # Energy
    "crude": ("CL=F", "USD/bbl", "futures"),
    "wti": ("CL=F", "USD/bbl", "futures"),
    "wti_crude": ("CL=F", "USD/bbl", "futures"),
    "brent": ("BZ=F", "USD/bbl", "futures"),
    "brent_crude": ("BZ=F", "USD/bbl", "futures"),
    "natural_gas": ("NG=F", "USD/MMBtu", "futures"),
    "ng": ("NG=F", "USD/MMBtu", "futures"),
    "gasoline": ("RB=F", "USD/gal", "futures"),
    "heating_oil": ("HO=F", "USD/gal", "futures"),

    # Soft commodities (textile names)
    "cotton": ("CT=F", "USD/lb", "futures"),

    # ETF proxies — non-futures or thinly-traded inputs
    "lithium": ("LIT", "USD (ETF NAV)", "etf_proxy"),
    "steel": ("SLX", "USD (ETF NAV)", "etf_proxy"),
    "uranium": ("URA", "USD (ETF NAV)", "etf_proxy"),
    "rare_earth": ("REMX", "USD (ETF NAV)", "etf_proxy"),
}

SUPPORTED_NAMES = sorted(set(COMMODITY_CATALOG.keys()))


def _normalize(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _pct(curr: float | None, past: float | None) -> float | None:
    if past in (None, 0) or curr is None:
        return None
    return round((curr / past - 1) * 100, 2)


def _atr_14(df: pd.DataFrame) -> float | None:
    if df.shape[0] < 15:
        return None
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    val = tr.rolling(14).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


def _close_at_offset(closes: pd.Series, days_back: int) -> float | None:
    if len(closes) <= days_back:
        return None
    return float(closes.iloc[-(days_back + 1)])


def get_commodity_snapshot(name: str) -> dict:
    """Fetch a snapshot for a commodity by free-form name (gold, copper, brent, ...)."""
    t0 = time.time()
    key = _normalize(name)
    resolved = COMMODITY_CATALOG.get(key)
    if not resolved:
        logger.warning("get_commodity_snapshot unknown commodity | name=%s", name)
        return {
            "error": f"Commodity '{name}' not in catalog.",
            "kind": "unknown_commodity",
            "supported": SUPPORTED_NAMES,
        }

    ticker, unit, kind = resolved
    logger.info("get_commodity_snapshot start | name=%s ticker=%s", name, ticker)

    try:
        df = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return {"error": f"yfinance fetch failed for {ticker}: {e}", "kind": "fetch_failed"}

    if df.empty:
        return {"error": f"No data returned for {ticker}.", "kind": "no_data"}

    df = df.dropna(subset=["Close"])
    closes = df["Close"].astype(float)
    current = float(closes.iloc[-1])

    changes = {
        "1d_pct": _pct(current, _close_at_offset(closes, 1)),
        "5d_pct": _pct(current, _close_at_offset(closes, 5)),
        "1m_pct": _pct(current, _close_at_offset(closes, 21)),
        "3m_pct": _pct(current, _close_at_offset(closes, 63)),
        "6m_pct": _pct(current, _close_at_offset(closes, 126)),
        "12m_pct": _pct(current, _close_at_offset(closes, min(250, len(closes) - 1))),
    }

    sma50_val = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else None
    sma200_val = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
    sma50 = float(sma50_val) if sma50_val is not None and pd.notna(sma50_val) else None
    sma200 = float(sma200_val) if sma200_val is not None and pd.notna(sma200_val) else None

    if sma50 and sma200:
        regime = (
            "uptrend" if current > sma50 > sma200
            else "downtrend" if current < sma50 < sma200
            else "sideways"
        )
    else:
        regime = "insufficient_history"

    atr14 = _atr_14(df)
    atr_pct = round(atr14 / current * 100, 2) if atr14 and current else None
    rv_series = closes.pct_change().tail(20).std() if len(closes) >= 21 else None
    realized_vol = float(rv_series) * 100 if rv_series is not None and pd.notna(rv_series) else None

    last_year = closes.tail(252)
    high_52w = float(last_year.max()) if not last_year.empty else None
    low_52w = float(last_year.min()) if not last_year.empty else None

    snapshot = {
        "commodity": name,
        "resolved_to": ticker,
        "kind": kind,
        "unit": unit,
        "as_of": date.today().isoformat(),
        "price": round(current, 2),
        "changes_pct": changes,
        "trend": {
            "sma_50": round(sma50, 2) if sma50 else None,
            "sma_200": round(sma200, 2) if sma200 else None,
            "price_vs_sma_50_pct": _pct(current, sma50),
            "price_vs_sma_200_pct": _pct(current, sma200),
            "regime": regime,
        },
        "volatility": {
            "atr_14": round(atr14, 2) if atr14 else None,
            "atr_14_pct_of_price": atr_pct,
            "realized_vol_20d_pct": round(realized_vol, 2) if realized_vol is not None else None,
        },
        "range_52w": {
            "high": round(high_52w, 2) if high_52w else None,
            "low": round(low_52w, 2) if low_52w else None,
            "pct_from_high": _pct(current, high_52w),
            "pct_from_low": _pct(current, low_52w),
        },
        "notes": [],
    }

    if kind == "etf_proxy":
        snapshot["notes"].append(
            f"'{name}' is tracked via ETF proxy {ticker} (not a direct futures contract). "
            "Use price direction as a directional signal; absolute USD/unit numbers are NAV-based, not commodity spot."
        )

    logger.info(
        "get_commodity_snapshot done | name=%s ticker=%s price=%.2f regime=%s latency=%.2fs",
        name, ticker, current, regime, time.time() - t0,
    )
    return snapshot
