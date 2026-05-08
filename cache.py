"""
Daily-fresh disk cache for Kite API data shared across tools.

What's cached:
1. `kite.instruments("NSE")` — heavy call (~2000+ rows), changes only on corporate
   actions. JSON file refreshed daily.
2. Daily OHLCV per ticker/index — Parquet files, broadest range ever fetched for
   that key. On read, returns the requested date slice if cache covers it AND was
   refreshed today; otherwise re-fetches the missing range and merges.

Not cached:
- yfinance fundamentals (cheap; lightweight)
- Live tick / minute data (irrelevant — we only use daily)

Cache keys are tradingsymbols, sanitised for the filesystem.
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("cache")
INSTRUMENTS_PATH = CACHE_ROOT / "instruments_nse.json"
OHLCV_DIR = CACHE_ROOT / "ohlcv"


def _safe_key(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


def _is_today(ts: float) -> bool:
    return datetime.fromtimestamp(ts).date() == date.today()


# ── Instruments cache ──────────────────────────────────────────────────────


_instruments_mem: list[dict] | None = None


def get_nse_instruments(kite) -> list[dict]:
    """Daily-fresh cache of `kite.instruments('NSE')` list."""
    global _instruments_mem
    if _instruments_mem is not None:
        return _instruments_mem

    if INSTRUMENTS_PATH.exists() and _is_today(INSTRUMENTS_PATH.stat().st_mtime):
        with open(INSTRUMENTS_PATH, "r", encoding="utf-8") as f:
            _instruments_mem = json.load(f)
            logger.info("instruments cache HIT (%d rows)", len(_instruments_mem))
            return _instruments_mem

    logger.info("instruments cache MISS — fetching from Kite")
    t0 = time.time()
    data = kite.instruments("NSE")
    INSTRUMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INSTRUMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    logger.info("instruments fetched + cached (%d rows, %.2fs)", len(data), time.time() - t0)
    _instruments_mem = data
    return data


# ── OHLCV cache ────────────────────────────────────────────────────────────


def _read_cached(key: str) -> pd.DataFrame | None:
    p = OHLCV_DIR / f"{_safe_key(key)}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _write_cached(key: str, df: pd.DataFrame) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    p = OHLCV_DIR / f"{_safe_key(key)}.parquet"
    df.to_parquet(p)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.normalize()
    df = df.set_index("date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_ohlcv_cached(
    kite,
    token: int,
    key: str,
    from_date: date,
    to_date: date,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for `key` (a ticker/index tradingsymbol), using cache.

    The cache stores the broadest range ever fetched for `key`. If the cache is
    today-fresh AND covers [from_date, to_date], returns a slice. Otherwise,
    re-fetches the union of (cached range, requested range) and saves.
    """
    cache_path = OHLCV_DIR / f"{_safe_key(key)}.parquet"

    cached = _read_cached(key)
    cached_today = cache_path.exists() and _is_today(cache_path.stat().st_mtime)

    if cached is not None and cached_today:
        cached_min = cached.index.min().date()
        cached_max = cached.index.max().date()
        # Cover from_date strictly. For to_date, two regimes:
        #   - to_date in the past (backtest): require cached_max >= to_date
        #   - to_date is today: accept any cache written today, since Kite may
        #     not have today's bar yet (trading day in progress / before close).
        live_request = to_date >= date.today()
        covers_back = cached_min <= from_date
        covers_forward = cached_max >= to_date if not live_request else True
        if covers_back and covers_forward:
            logger.info("ohlcv cache HIT | key=%s rows=%d", key, len(cached))
            return cached.loc[pd.Timestamp(from_date):pd.Timestamp(to_date)].copy()

    # Cache miss / stale / range short — refetch the union range.
    fetch_from = from_date
    fetch_to = max(to_date, date.today())
    if cached is not None:
        fetch_from = min(fetch_from, cached.index.min().date())
        fetch_to = max(fetch_to, cached.index.max().date())

    logger.info(
        "ohlcv cache MISS | key=%s fetch=%s->%s",
        key, fetch_from, fetch_to,
    )
    t0 = time.time()
    records = kite.historical_data(
        instrument_token=token,
        from_date=fetch_from,
        to_date=fetch_to,
        interval="day",
    )

    if not records:
        if cached is not None:
            return cached.loc[pd.Timestamp(from_date):pd.Timestamp(to_date)].copy()
        return pd.DataFrame()

    fresh = _normalise(pd.DataFrame(records))

    if cached is not None:
        merged = pd.concat([cached, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh

    _write_cached(key, merged)
    logger.info(
        "ohlcv fetched + cached | key=%s rows=%d latency=%.2fs",
        key, len(merged), time.time() - t0,
    )
    return merged.loc[pd.Timestamp(from_date):pd.Timestamp(to_date)].copy()
