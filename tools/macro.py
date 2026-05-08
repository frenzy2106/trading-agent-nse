"""
Macro snapshot tool — benchmark a stock against three indices:

1. Broad market: NIFTY 50 (always)
2. Sector index: NIFTY ENERGY / IT / BANK / etc., mapped from yfinance sector + industry
3. Size-bucket index: NIFTY 100 / MIDCAP 150 / SMLCAP 250 / MICROCAP 250, based on market cap

The size-bucket index matters most for small/mid-caps where the sector mapping
is loose (e.g. SIS → "Industrials" is too broad to map cleanly to a sector index;
NIFTY SMLCAP 250 is a more meaningful peer set).

OHLCV is pulled through the shared cache layer so repeated runs (and other tools)
don't re-hit Kite for index data that hasn't changed today.
"""

import logging
import time
from datetime import date, timedelta

import pandas as pd

from cache import fetch_ohlcv_cached, get_nse_instruments
from kite_login import get_kite_client
from ticker_utils import to_plain, to_yfinance

logger = logging.getLogger(__name__)


SECTOR_TO_NIFTY_INDEX = {
    "Energy": "NIFTY ENERGY",
    "Technology": "NIFTY IT",
    "Healthcare": "NIFTY PHARMA",
    "Consumer Defensive": "NIFTY FMCG",
    "Basic Materials": "NIFTY METAL",
    "Real Estate": "NIFTY REALTY",
    "Communication Services": "NIFTY MEDIA",
    "Consumer Cyclical": "NIFTY AUTO",
    "Industrials": "NIFTY AUTO",
}


def _finserv_index_for_industry(industry: str | None) -> str:
    """Financial Services needs industry-level resolution: bank vs broker vs everything else."""
    if not industry:
        return "NIFTY FIN SERVICE"
    ind = industry.lower()
    if "bank" in ind:
        return "NIFTY BANK"
    if "capital market" in ind:
        return "NIFTY CAPITAL MKT"
    return "NIFTY FIN SERVICE"


# Size-bucket thresholds in INR Crores (approximate; SEBI uses rank, cutoffs shift).
# Numbers reflect typical thresholds for May 2026.
def _size_bucket_index(market_cap_inr: float | None) -> tuple[str | None, str | None]:
    """Return (bucket_label, NIFTY index name) based on market cap."""
    if not market_cap_inr:
        return None, None
    cr = market_cap_inr / 1e7
    if cr >= 50_000:
        return "Large cap", "NIFTY 100"
    if cr >= 15_000:
        return "Mid cap", "NIFTY MIDCAP 150"
    if cr >= 2_000:
        return "Small cap", "NIFTY SMLCAP 250"
    return "Micro cap", "NIFTY MICROCAP 250"


def _yf_metadata(plain: str) -> dict:
    """Single yfinance lookup returning sector, industry, market cap."""
    try:
        import yfinance as yf
        info = yf.Ticker(to_yfinance(plain)).info
        return {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap_inr": info.get("marketCap"),
        }
    except Exception as e:
        logger.warning("yfinance metadata fetch failed for %s: %s", plain, e)
        return {"sector": None, "industry": None, "market_cap_inr": None}


def _resolve_sector_index(meta: dict) -> str | None:
    sector = meta.get("sector")
    industry = meta.get("industry")
    if not sector:
        return None
    if sector == "Financial Services":
        return _finserv_index_for_industry(industry)
    return SECTOR_TO_NIFTY_INDEX.get(sector)


WINDOWS = {"1m": 30, "3m": 90, "6m": 180, "12m": 365}


def _instrument_token(instruments: list[dict], tradingsymbol: str) -> int | None:
    for i in instruments:
        if i["tradingsymbol"] == tradingsymbol:
            return i["instrument_token"]
    return None


def _fetch_close_series(kite, token: int, key: str, lookback_days: int, as_of: date) -> pd.Series:
    if token is None:
        return pd.Series(dtype=float)
    from_date = as_of - timedelta(days=lookback_days + 30)
    df = fetch_ohlcv_cached(kite, token, key, from_date, as_of)
    return df["close"].astype(float) if not df.empty else pd.Series(dtype=float)


def _returns_at_windows(closes: pd.Series, as_of: date) -> dict:
    if closes.empty:
        return {k: None for k in WINDOWS}

    as_of_ts = pd.Timestamp(as_of)
    on_or_before = closes[closes.index <= as_of_ts]
    if on_or_before.empty:
        return {k: None for k in WINDOWS}

    current = float(on_or_before.iloc[-1])
    out: dict = {}
    for label, days in WINDOWS.items():
        target_ts = as_of_ts - pd.Timedelta(days=days)
        prior = closes[closes.index <= target_ts]
        if prior.empty:
            out[label] = None
        else:
            past = float(prior.iloc[-1])
            out[label] = round((current / past - 1) * 100, 2) if past else None
    return out


def _diff(a: float | None, b: float | None) -> float | None:
    return None if (a is None or b is None) else round(a - b, 2)


def _index_block(
    name: str | None,
    closes: pd.Series,
    as_of: date,
    extra: dict | None = None,
) -> dict:
    block = {
        "name": name,
        "available": not closes.empty,
        "returns_pct": _returns_at_windows(closes, as_of),
    }
    if extra:
        block.update(extra)
    return block


def get_macro_snapshot(ticker: str, as_of_date: date | None = None) -> dict:
    plain = to_plain(ticker)
    as_of = as_of_date or date.today()
    t0 = time.time()
    logger.info("get_macro_snapshot start | ticker=%s as_of=%s", plain, as_of)

    kite = get_kite_client()
    instruments = get_nse_instruments(kite)

    stock_token = _instrument_token(instruments, plain)
    if stock_token is None:
        from tools.technical import TickerNotFoundError
        raise TickerNotFoundError(plain, [])

    meta = _yf_metadata(plain)
    sector_index_name = _resolve_sector_index(meta)
    bucket_label, bucket_index_name = _size_bucket_index(meta.get("market_cap_inr"))

    nifty50_token = _instrument_token(instruments, "NIFTY 50")
    sector_token = _instrument_token(instruments, sector_index_name) if sector_index_name else None
    bucket_token = _instrument_token(instruments, bucket_index_name) if bucket_index_name else None

    stock_closes = _fetch_close_series(kite, stock_token, plain, 400, as_of)
    nifty_closes = _fetch_close_series(kite, nifty50_token, "NIFTY 50", 400, as_of)
    sector_closes = _fetch_close_series(kite, sector_token, sector_index_name or "_none_sector", 400, as_of)
    bucket_closes = _fetch_close_series(kite, bucket_token, bucket_index_name or "_none_bucket", 400, as_of)

    stock_returns = _returns_at_windows(stock_closes, as_of)
    broad_returns = _returns_at_windows(nifty_closes, as_of)
    sector_returns = _returns_at_windows(sector_closes, as_of)
    bucket_returns = _returns_at_windows(bucket_closes, as_of)

    snapshot = {
        "ticker": plain,
        "as_of": as_of.isoformat(),
        "sector": meta.get("sector"),
        "industry": meta.get("industry"),
        "market_cap_cr": (
            int(meta["market_cap_inr"] / 1e7) if meta.get("market_cap_inr") else None
        ),
        "stock_returns_pct": stock_returns,
        "broad_index": _index_block("NIFTY 50", nifty_closes, as_of),
        "sector_index": _index_block(
            sector_index_name,
            sector_closes,
            as_of,
            extra={
                "mapped_from_sector": meta.get("sector"),
                "mapped_from_industry": meta.get("industry"),
            },
        ),
        "size_bucket_index": _index_block(
            bucket_index_name,
            bucket_closes,
            as_of,
            extra={"bucket": bucket_label},
        ),
        "relative_pct": {
            "vs_broad": {k: _diff(stock_returns[k], broad_returns[k]) for k in WINDOWS},
            "vs_sector": {k: _diff(stock_returns[k], sector_returns[k]) for k in WINDOWS},
            "vs_size_bucket": {k: _diff(stock_returns[k], bucket_returns[k]) for k in WINDOWS},
        },
        "notes": [],
    }

    # Diagnostic notes for the LLM
    if not sector_index_name:
        snapshot["notes"].append(
            f"No NIFTY sector index mapped (sector={meta.get('sector')!r}). "
            "Lean on size-bucket comparison instead."
        )
    elif sector_index_name == "NIFTY AUTO" and meta.get("sector") in {"Consumer Cyclical", "Industrials"}:
        snapshot["notes"].append(
            f"Sector mapping is imprecise: '{meta.get('sector')}' / '{meta.get('industry')}' → {sector_index_name}. "
            "For non-auto names this comparison is loose; the size-bucket comparison is usually more meaningful."
        )

    if not bucket_index_name:
        snapshot["notes"].append("Could not determine size bucket — market cap missing.")

    logger.info(
        "get_macro_snapshot done | ticker=%s sector_idx=%s bucket=%s latency=%.2fs",
        plain, sector_index_name, bucket_index_name, time.time() - t0,
    )
    return snapshot
