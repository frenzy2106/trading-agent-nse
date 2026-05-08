"""
News + earnings calendar tool — fetches:
  - Next scheduled earnings date (and consensus EPS estimate, if available)
  - Last reported earnings date + EPS surprise vs consensus
  - Recent headlines (top 5) about the ticker

All from yfinance — no separate API key needed. Note that yfinance news quality
varies by name: large-caps get plenty of (sometimes off-topic) coverage; smaller
names may have only a few items.

The LLM is responsible for filtering relevance — we surface what we get and let
it decide which headlines actually pertain to the company.
"""

import logging
import time
from datetime import date

from ticker_utils import to_plain, to_yfinance

logger = logging.getLogger(__name__)


def _earnings_block(yticker, today: date) -> dict:
    """Pull next + last earnings date with consensus / surprise where available."""
    out = {
        "next_earnings_date": None,
        "days_to_earnings": None,
        "eps_estimate": None,
        "last_earnings_date": None,
        "days_since_last_earnings": None,
        "last_eps_surprise_pct": None,
    }

    # Calendar gives the next forward-looking estimate (analyst-covered names).
    try:
        cal = yticker.calendar or {}
        cal_dates = cal.get("Earnings Date") or []
        for d in cal_dates:
            if isinstance(d, date) and d >= today:
                out["next_earnings_date"] = d.isoformat()
                out["days_to_earnings"] = (d - today).days
                break
        eps_avg = cal.get("Earnings Average")
        if isinstance(eps_avg, (int, float)):
            out["eps_estimate"] = round(float(eps_avg), 2)
    except Exception as e:
        logger.warning("calendar fetch failed: %s", e)

    # earnings_dates has historical actuals + occasionally future. Fill gaps.
    try:
        ed = yticker.earnings_dates
        if ed is not None and not ed.empty:
            ed_dates = ed.index.date

            # If calendar didn't have a future date, look here.
            if not out["next_earnings_date"]:
                future_mask = ed_dates >= today
                if future_mask.any():
                    fd = ed.index[future_mask][0].date()
                    out["next_earnings_date"] = fd.isoformat()
                    out["days_to_earnings"] = (fd - today).days

            # Most recent past earnings + surprise
            past_mask = ed_dates < today
            if past_mask.any():
                last_idx = ed.index[past_mask][0]
                out["last_earnings_date"] = last_idx.date().isoformat()
                out["days_since_last_earnings"] = (today - last_idx.date()).days
                surprise = ed.loc[last_idx].get("Surprise(%)")
                if surprise is not None:
                    try:
                        sf = float(surprise)
                        if sf == sf:  # not NaN
                            out["last_eps_surprise_pct"] = round(sf, 2)
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logger.warning("earnings_dates fetch failed: %s", e)

    return out


def _news_block(yticker, limit: int = 5) -> list[dict]:
    """Top N recent headlines with title, summary (truncated), date, provider."""
    items: list[dict] = []
    try:
        news = yticker.news or []
    except Exception as e:
        logger.warning("news fetch failed: %s", e)
        return items

    for raw in news[:limit]:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") or {}
        title = content.get("title")
        if not title:
            continue
        summary = content.get("summary") or content.get("description") or ""
        if len(summary) > 300:
            summary = summary[:300] + "…"
        provider = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else None
        items.append({
            "title": title,
            "summary": summary,
            "date": content.get("pubDate"),
            "provider": provider,
        })
    return items


def get_news_and_earnings(ticker: str, as_of_date: date | None = None) -> dict:
    plain = to_plain(ticker)
    today = as_of_date or date.today()
    t0 = time.time()
    logger.info("get_news_and_earnings start | ticker=%s as_of=%s", plain, today)

    import yfinance as yf
    yticker = yf.Ticker(to_yfinance(plain))

    earnings = _earnings_block(yticker, today)
    news = _news_block(yticker, limit=5)

    notes: list[str] = []
    if not earnings["next_earnings_date"] and not earnings["last_earnings_date"]:
        notes.append("No earnings calendar data found from yfinance for this ticker.")
    if not news:
        notes.append("No recent news headlines returned from yfinance.")

    snapshot = {
        "ticker": plain,
        "as_of": today.isoformat(),
        **earnings,
        "recent_news": news,
        "news_count": len(news),
        "notes": notes,
    }

    logger.info(
        "get_news_and_earnings done | ticker=%s news=%d next_earnings=%s latency=%.2fs",
        plain, len(news), earnings["next_earnings_date"], time.time() - t0,
    )
    return snapshot
