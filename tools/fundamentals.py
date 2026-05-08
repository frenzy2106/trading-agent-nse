"""
Fundamentals snapshot tool.

get_fundamentals_snapshot(ticker) -> dict

Returns valuation ratios, profitability metrics, leverage, and last 4 quarters
of revenue + EPS from yfinance.
"""

import logging
import math
import time

import yfinance as yf

from ticker_utils import to_plain, to_yfinance

logger = logging.getLogger(__name__)


def _safe(value, decimals=2):
    """Return rounded float or None for missing/NaN values."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _pct(value, decimals=2):
    """Convert yfinance decimal ratio to readable percentage (0.484 -> 48.4)."""
    v = _safe(value, decimals + 2)
    return None if v is None else round(v * 100, decimals)


def _crore(value):
    """Convert raw INR to Crores (divide by 1e7), return int or None."""
    v = _safe(value, 0)
    return None if v is None else int(v / 1e7)


def _quarterly_trend(ticker_obj) -> list[dict]:
    """Last 4 quarters of revenue and EPS from quarterly_income_stmt."""
    try:
        qi = ticker_obj.quarterly_income_stmt
    except Exception:
        return []

    if qi is None or qi.empty:
        return []

    rev_row = qi.loc["Total Revenue"] if "Total Revenue" in qi.index else None
    eps_row = qi.loc["Basic EPS"] if "Basic EPS" in qi.index else None

    quarters = []
    for col in list(qi.columns)[:4]:
        period = col.date().isoformat() if hasattr(col, "date") else str(col)
        revenue_cr = _crore(rev_row[col] if rev_row is not None else None)
        eps = _safe(eps_row[col] if eps_row is not None else None, 2)
        quarters.append({"period": period, "revenue_cr": revenue_cr, "eps": eps})

    return quarters


def _bs_value(bs, labels):
    """Return first non-null balance-sheet value matching any of the given labels."""
    for label in labels:
        if label in bs.index:
            v = bs.loc[label].iloc[0]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
    return None


def _compute_de_from_bs(ticker_obj) -> float | None:
    """Compute D/E as Total Debt / Total Equity directly from the balance sheet.

    yfinance's info.debtToEquity inflates for asset-light firms (IT services,
    consumer) by including deferred revenue and operating liabilities. Computing
    from the balance sheet keeps it to interest-bearing debt only.
    """
    try:
        bs = ticker_obj.balance_sheet
        if bs is None or bs.empty:
            return None

        total_debt = _bs_value(bs, ["Total Debt"])
        if total_debt is None:
            ltd = _bs_value(bs, ["Long Term Debt"]) or 0
            std = _bs_value(bs, ["Current Debt", "Short Term Debt"]) or 0
            if ltd or std:
                total_debt = ltd + std

        total_equity = _bs_value(
            bs,
            [
                "Stockholders Equity",
                "Common Stock Equity",
                "Total Equity Gross Minority Interest",
            ],
        )

        if total_debt is None or not total_equity:
            return None
        return round(total_debt / total_equity, 2)
    except Exception:
        return None


def _compute_roce(ticker_obj) -> float | None:
    """ROCE = EBIT / (Total Assets - Current Liabilities). Returns None if data unavailable."""
    try:
        fin = ticker_obj.financials
        bs = ticker_obj.balance_sheet
        if fin is None or bs is None or fin.empty or bs.empty:
            return None

        ebit = None
        for label in ["EBIT", "Operating Income"]:
            if label in fin.index:
                ebit = fin.loc[label].iloc[0]
                break

        total_assets = bs.loc["Total Assets"].iloc[0] if "Total Assets" in bs.index else None
        curr_liab = bs.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bs.index else None

        if ebit is None or total_assets is None or curr_liab is None:
            return None

        capital_employed = float(total_assets) - float(curr_liab)
        if capital_employed <= 0:
            return None

        return _safe(float(ebit) / capital_employed, 4)
    except Exception:
        return None


def get_fundamentals_snapshot(ticker: str) -> dict:
    plain = to_plain(ticker)
    yf_symbol = to_yfinance(plain)
    t0 = time.time()
    logger.info("get_fundamentals_snapshot start | ticker=%s", plain)

    yticker = yf.Ticker(yf_symbol)
    info = yticker.info

    snapshot = {
        "ticker": plain,
        "as_of": __import__("datetime").date.today().isoformat(),
        "profile": {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap_cr": _crore(info.get("marketCap")),
        },
        "valuation": {
            "pe_trailing": _safe(info.get("trailingPE")),
            "pe_forward": _safe(info.get("forwardPE")),
            "pb": _safe(info.get("priceToBook")),
        },
        "profitability": {
            "roe_pct": _pct(info.get("returnOnEquity")),
            "roa_pct": _pct(info.get("returnOnAssets")),
            "roce_pct": None if _compute_roce(yticker) is None else round(_compute_roce(yticker) * 100, 2),
            "revenue_growth_yoy_pct": _pct(info.get("revenueGrowth")),
            "earnings_growth_yoy_pct": _pct(info.get("earningsGrowth")),
        },
        "leverage": {
            "debt_to_equity": _compute_de_from_bs(yticker),
        },
        "dividends": {
            "yield": _safe(info.get("dividendYield"), 4),
        },
        "quarterly_trend": _quarterly_trend(yticker),
    }

    logger.info(
        "get_fundamentals_snapshot done | ticker=%s latency=%.2fs",
        plain, time.time() - t0,
    )
    return snapshot
