"""
News + earnings calendar tool — fetches:
  - Next scheduled earnings date (and consensus EPS estimate, if available)
  - Last reported earnings date + EPS surprise vs consensus  → yfinance
  - Recent headlines (top 5) about the ticker                → Google News RSS

Why Google News RSS for headlines: tested against yfinance and Tavily News
on Indian small/mid-caps. yfinance returns 0-3 items and often off-topic
(e.g. for MTARTECH it surfaced US-side Bloom Energy news instead of MTAR
stories from Indian financial press). Tavily News doesn't crawl Indian
financial press deeply enough — top relevance scores capped at 0.27.
Google News RSS with en-IN locale + `when:30d` consistently surfaces
Moneycontrol, ET, Business Standard, NDTV Profit, CNBC TV18, Trade Brains.

We keep yfinance for earnings dates because Google News doesn't have that.

The LLM is responsible for filtering relevance — we surface what we get and let
it decide which headlines actually pertain to the company.
"""

import logging
import time
from datetime import date
from urllib.parse import quote_plus

import feedparser

from ticker_utils import to_plain, to_yfinance

logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

# Manual longName overrides for tickers where yfinance .info returns None.
# Add to this as needed (recent IPOs, name changes, etc.).
TICKER_NAME_OVERRIDES = {
    "PNGJL": "P N Gadgil Jewellers",
}


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


def _resolve_company_name(yticker, plain_ticker: str) -> str | None:
    """Get the company's longName for use as a news search query.

    Falls back to manual override map if yfinance .info returns nothing
    (common for recent IPOs like PNGJL).
    """
    try:
        info = yticker.info or {}
        name = info.get("longName") or info.get("shortName")
        if name:
            return name
    except Exception as e:
        logger.warning("longName resolve failed for %s: %s", plain_ticker, e)
    return TICKER_NAME_OVERRIDES.get(plain_ticker)


def _news_block(yticker, plain_ticker: str, limit: int = 5) -> list[dict]:
    """Top N recent headlines from Google News RSS (en-IN locale, last 30 days).

    Falls back to ticker-only query if no company name is resolvable.
    """
    items: list[dict] = []
    company_name = _resolve_company_name(yticker, plain_ticker)

    if company_name:
        query = f"{company_name} share price when:30d"
    else:
        query = f"{plain_ticker} NSE stock when:30d"

    url = f"{GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    logger.info("news query | ticker=%s name=%s", plain_ticker, company_name)

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        logger.warning("Google News RSS fetch failed for %s: %s", plain_ticker, e)
        return items

    for entry in feed.entries[:limit]:
        title = entry.get("title")
        if not title:
            continue

        # Source comes back two ways: dict {title, href} or trailing " - Source" in title.
        provider = None
        src_obj = entry.get("source")
        if isinstance(src_obj, dict):
            provider = src_obj.get("title")
        if not provider and " - " in title:
            provider = title.rsplit(" - ", 1)[-1].strip()

        summary = entry.get("summary") or ""
        if len(summary) > 300:
            summary = summary[:300] + "…"

        items.append({
            "title": title,
            "summary": summary,
            "date": entry.get("published"),
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
    news = _news_block(yticker, plain, limit=5)

    notes: list[str] = []
    if not earnings["next_earnings_date"] and not earnings["last_earnings_date"]:
        notes.append("No earnings calendar data found from yfinance for this ticker.")
    if not news:
        notes.append("No recent news headlines returned from Google News RSS.")

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
