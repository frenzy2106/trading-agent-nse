"""
Analyst consensus tool — sell-side aggregated targets and rating distribution
from yfinance.

get_analyst_consensus(ticker) -> dict

Returns price targets (mean / median / high / low / count), recommendation
distribution (strong_buy / buy / hold / sell / strong_sell), and implied upside
to the mean target.

Coverage is good for large-caps (30+ analysts) and patchy for small/mid-caps
(<5 analysts). Always treat consensus as INPUT, not ground truth — Indian
sell-side ratings skew bullish (~70% buy/hold, ~5% sell on average) and targets
often lag earnings updates.
"""

import logging
import math
import time

import yfinance as yf

from ticker_utils import to_plain, to_yfinance

logger = logging.getLogger(__name__)


def _safe_int(value):
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


def _safe_float(value, decimals=2):
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _coverage_label(n):
    if n is None:
        return "unknown"
    if n >= 15:
        return "heavy"
    if n >= 5:
        return "moderate"
    return "thin"


def get_analyst_consensus(ticker: str) -> dict:
    plain = to_plain(ticker)
    yf_symbol = to_yfinance(plain)
    t0 = time.time()
    logger.info("get_analyst_consensus start | ticker=%s", plain)

    yt = yf.Ticker(yf_symbol)
    info = yt.info or {}

    current = _safe_float(info.get("currentPrice"))
    mean = _safe_float(info.get("targetMeanPrice"))
    median = _safe_float(info.get("targetMedianPrice"))
    high = _safe_float(info.get("targetHighPrice"))
    low = _safe_float(info.get("targetLowPrice"))
    num = _safe_int(info.get("numberOfAnalystOpinions"))

    sb = b = h = s = ss = None
    distribution_history = []
    try:
        recs = yt.recommendations
        if recs is not None and len(recs) > 0:
            for _, row in recs.iterrows():
                row_sb = _safe_int(row.get("strongBuy"))
                row_b = _safe_int(row.get("buy"))
                row_h = _safe_int(row.get("hold"))
                row_s = _safe_int(row.get("sell"))
                row_ss = _safe_int(row.get("strongSell"))
                row_tot = sum(x for x in [row_sb, row_b, row_h, row_s, row_ss] if x is not None) or 0
                bullish_n = (row_sb or 0) + (row_b or 0)
                bullish_pct = round(bullish_n / row_tot * 100, 1) if row_tot > 0 else None
                distribution_history.append({
                    "period": row.get("period"),
                    "strong_buy": row_sb,
                    "buy": row_b,
                    "hold": row_h,
                    "sell": row_s,
                    "strong_sell": row_ss,
                    "total": row_tot,
                    "bullish_pct": bullish_pct,
                })
            # Current snapshot from the first (most recent) row
            head = recs.iloc[0]
            sb = _safe_int(head.get("strongBuy"))
            b = _safe_int(head.get("buy"))
            h = _safe_int(head.get("hold"))
            s = _safe_int(head.get("sell"))
            ss = _safe_int(head.get("strongSell"))
    except Exception as e:
        logger.warning("recommendations fetch failed for %s: %s", plain, e)

    total = sum(x for x in [sb, b, h, s, ss] if x is not None) if any(
        x is not None for x in [sb, b, h, s, ss]
    ) else None

    # Drift summary: compare current period (0m) vs oldest available
    drift = None
    if len(distribution_history) >= 2:
        current_p = distribution_history[0]
        oldest_p = distribution_history[-1]
        cur_bull = current_p.get("bullish_pct")
        old_bull = oldest_p.get("bullish_pct")
        if cur_bull is not None and old_bull is not None:
            drift = {
                "periods_compared": f"{oldest_p['period']} → {current_p['period']}",
                "bullish_pct_now": cur_bull,
                "bullish_pct_oldest": old_bull,
                "bullish_pct_change": round(cur_bull - old_bull, 1),
                "holds_now": current_p.get("hold"),
                "holds_oldest": oldest_p.get("hold"),
                "holds_change": (current_p.get("hold") or 0) - (oldest_p.get("hold") or 0),
            }

    implied_upside_pct = None
    if current and mean:
        implied_upside_pct = round((mean - current) / current * 100, 1)

    snapshot = {
        "ticker": plain,
        "as_of": __import__("datetime").date.today().isoformat(),
        "price_targets": {
            "current": current,
            "mean": mean,
            "median": median,
            "high": high,
            "low": low,
            "num_analysts": num,
            "implied_upside_to_mean_pct": implied_upside_pct,
            "coverage": _coverage_label(num),
        },
        "recommendations": {
            "strong_buy": sb,
            "buy": b,
            "hold": h,
            "sell": s,
            "strong_sell": ss,
            "total": total,
            "key": info.get("recommendationKey"),
            "mean_score": _safe_float(info.get("recommendationMean")),
        },
        "rating_drift": drift,
        "distribution_history": distribution_history,
    }

    logger.info(
        "get_analyst_consensus done | ticker=%s analysts=%s mean_target=%s upside=%s%% latency=%.2fs",
        plain, num, mean, implied_upside_pct, time.time() - t0,
    )
    return snapshot
