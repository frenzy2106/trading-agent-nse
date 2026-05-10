"""
Lite screener — deterministic scoring across a watchlist, no LLM calls.

For each ticker pulls technical + fundamentals + news + analyst (skipping commentary
RAG and the LLM synthesis). Scores 0-12 on a transparent rubric and ranks candidates.

Use:
  python screener_lite.py                              # uses watchlist_nifty100.txt
  python screener_lite.py --watchlist my.txt           # custom watchlist
  python screener_lite.py --top 15                     # show top N (default 20)
  python screener_lite.py --workers 6                  # parallel workers (default 4)

Output:
  - Stdout: top-N ranked table
  - File: screener/lite-<run-id>.md (full ranked table for all tickers)
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from tools.analyst import get_analyst_consensus
from tools.fundamentals import get_fundamentals_snapshot
from tools.news import get_news_and_earnings
from tools.technical import TickerNotFoundError, get_technical_snapshot

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING").upper(),
                    format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("screener_lite")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUMMARY_DIR = Path("screener")


def _safe_get(d, *path, default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else None
    return cur if cur is not None else default


def _score_ticker(ticker: str) -> dict:
    """Pull data + score. Returns a row dict including reasons."""
    t0 = time.time()
    row = {"ticker": ticker, "score": 0, "reasons": [], "error": None,
           "rsi": None, "fwd_pe": None, "roe": None, "upside_pct": None,
           "drift_holds_chg": None, "obv_20d_pct": None, "price": None,
           "above_sma200": None, "above_sma50": None}

    try:
        tech = get_technical_snapshot(ticker)
    except TickerNotFoundError as e:
        row["error"] = f"ticker_not_found ({', '.join(e.suggestions[:2]) if e.suggestions else 'no match'})"
        return row
    except Exception as e:
        row["error"] = f"technical: {type(e).__name__}: {str(e)[:80]}"
        return row

    try:
        fund = get_fundamentals_snapshot(ticker)
    except Exception as e:
        fund = {}
        logger.warning("fundamentals failed for %s: %s", ticker, e)

    try:
        news = get_news_and_earnings(ticker)
    except Exception as e:
        news = {}
        logger.warning("news failed for %s: %s", ticker, e)

    try:
        analyst = get_analyst_consensus(ticker)
    except Exception as e:
        analyst = {}
        logger.warning("analyst failed for %s: %s", ticker, e)

    score = 0
    reasons = []

    price = _safe_get(tech, "price", "current")
    sma50 = _safe_get(tech, "indicators", "sma_50")
    sma200 = _safe_get(tech, "indicators", "sma_200")
    rsi = _safe_get(tech, "indicators", "rsi_14")
    macd_hist = _safe_get(tech, "indicators", "macd", "histogram")
    obv_20d = _safe_get(tech, "volume", "obv_change_20d_pct")

    row["price"] = price
    row["rsi"] = rsi
    row["obv_20d_pct"] = obv_20d
    row["above_sma200"] = (price is not None and sma200 is not None and price > sma200)
    row["above_sma50"] = (price is not None and sma50 is not None and price > sma50)

    # Technical health (0-3)
    if row["above_sma200"]:
        score += 1; reasons.append("above SMA-200")
    if row["above_sma50"]:
        score += 1; reasons.append("above SMA-50")
    if rsi is not None and macd_hist is not None:
        if 50 <= rsi <= 70 and macd_hist > 0:
            score += 1; reasons.append("RSI healthy + MACD bullish")
        elif 30 <= rsi < 50 and macd_hist > 0:
            score += 1; reasons.append("recovery setup")

    # Fundamentals (0-3)
    roe = _safe_get(fund, "profitability", "roe_pct")
    roce = _safe_get(fund, "profitability", "roce_pct")
    rev_growth = _safe_get(fund, "profitability", "revenue_growth_yoy_pct")
    row["roe"] = roe

    if roe is not None and roe > 15:
        score += 1; reasons.append(f"ROE {roe:.1f}%")
    if roce is not None and roce > 15:
        score += 1; reasons.append(f"ROCE {roce:.1f}%")
    if rev_growth is not None and rev_growth > 10:
        score += 1; reasons.append(f"rev growth {rev_growth:.1f}%")

    # Valuation (0-2)
    pe_t = _safe_get(fund, "valuation", "pe_trailing")
    pe_f = _safe_get(fund, "valuation", "pe_forward")
    row["fwd_pe"] = pe_f

    if pe_f is not None and pe_t is not None and pe_f < pe_t:
        score += 1; reasons.append(f"fwd<ttm P/E ({pe_f:.1f}x<{pe_t:.1f}x)")
    if pe_f is not None and 5 < pe_f < 30:
        score += 1; reasons.append(f"reasonable fwd P/E {pe_f:.1f}x")

    # Analyst signal (0-2)
    upside = _safe_get(analyst, "price_targets", "implied_upside_to_mean_pct")
    coverage = _safe_get(analyst, "price_targets", "coverage")
    drift = _safe_get(analyst, "rating_drift") or {}
    bullish_chg = drift.get("bullish_pct_change") if isinstance(drift, dict) else None
    holds_chg = drift.get("holds_change") if isinstance(drift, dict) else None
    row["upside_pct"] = upside
    row["drift_holds_chg"] = holds_chg

    # Need at least moderate coverage to weight analyst signal
    if upside is not None and upside > 15 and coverage in ("moderate", "heavy"):
        score += 1; reasons.append(f"analyst upside +{upside:.1f}%")
    if bullish_chg is not None and (bullish_chg > 3 or (holds_chg is not None and holds_chg <= -2)):
        score += 1; reasons.append(f"bullish drift Δ{bullish_chg:+.1f}pp")

    # Tactical (0-2)
    days_to_earnings = _safe_get(news, "days_to_earnings")
    if days_to_earnings is None or days_to_earnings > 30:
        score += 1; reasons.append("no imminent earnings")
    if obv_20d is not None and obv_20d > 0:
        score += 1; reasons.append(f"OBV +{obv_20d:.1f}% 20d")

    row["score"] = score
    row["reasons"] = reasons
    row["latency_s"] = round(time.time() - t0, 1)
    return row


def _render_summary(rows, run_id, elapsed):
    rows_sorted = sorted(rows, key=lambda r: (-r["score"], r["ticker"]))
    n_total = len(rows)
    n_err = sum(1 for r in rows if r["error"])

    parts = [
        f"# Lite Screener — {run_id}",
        "",
        f"**Tickers scanned:** {n_total} | **Errors:** {n_err} | "
        f"**Wall-clock:** {int(elapsed // 60)}m {int(elapsed % 60)}s",
        "",
        "Score = technical (0-3) + fundamentals (0-3) + valuation (0-2) + analyst (0-2) + tactical (0-2). Max 12.",
        "",
        "## Ranked",
        "",
        "| Rank | Ticker | Score | Price | RSI | Fwd P/E | ROE% | Upside% | OBV 20d% | Reasons |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt(v, suffix=""):
        return f"{v:.1f}{suffix}" if isinstance(v, (int, float)) else "—"

    for i, r in enumerate(rows_sorted, 1):
        if r["error"]:
            parts.append(f"| {i} | **{r['ticker']}** | — | — | — | — | — | — | — | ERROR: {r['error']} |")
            continue
        reasons_md = ", ".join(r["reasons"]) if r["reasons"] else "—"
        parts.append(
            f"| {i} | **{r['ticker']}** | **{r['score']}** | "
            f"{fmt(r['price'])} | {fmt(r['rsi'])} | "
            f"{fmt(r['fwd_pe'], 'x')} | {fmt(r['roe'])} | "
            f"{fmt(r['upside_pct'])} | {fmt(r['obv_20d_pct'])} | "
            f"{reasons_md} |"
        )
    return "\n".join(parts)


def _print_top(rows, top_n):
    rows_sorted = sorted([r for r in rows if not r.get("error")],
                        key=lambda r: (-r["score"], r["ticker"]))
    print(f"\n{'='*110}")
    print(f"{'#':>3} {'Ticker':<14} {'Score':>5} {'Price':>10} {'RSI':>5} "
          f"{'FwdPE':>7} {'ROE%':>6} {'Upsd%':>6} {'OBV20':>6}  Reasons")
    print(f"{'-'*110}")
    for i, r in enumerate(rows_sorted[:top_n], 1):
        def f(v, w, fmt="{:>{w}.1f}"):
            if v is None: return f"{'-':>{w}}"
            return fmt.format(v, w=w)
        reasons = ", ".join(r["reasons"][:5])
        if len(reasons) > 60: reasons = reasons[:57] + "…"
        print(f"{i:>3} {r['ticker']:<14} {r['score']:>5} "
              f"{f(r['price'], 10)} {f(r['rsi'], 5)} {f(r['fwd_pe'], 7)} "
              f"{f(r['roe'], 6)} {f(r['upside_pct'], 6)} {f(r['obv_20d_pct'], 6)}  {reasons}")
    print(f"{'='*110}\n")


def _read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"ERROR: watchlist not found at {path}")
    return [
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def main():
    p = argparse.ArgumentParser(description="Lite no-LLM screener over a watchlist")
    p.add_argument("--watchlist", default="watchlist_nifty100.txt")
    p.add_argument("--top", type=int, default=20, help="Print top N to stdout")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = p.parse_args()

    tickers = _read_watchlist(Path(args.watchlist))
    print(f"\n=== Lite Screener ===")
    print(f"Watchlist: {args.watchlist} ({len(tickers)} tickers)")
    print(f"Workers: {args.workers}")
    print()

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score_ticker, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            done += 1
            tag = "ERR" if row.get("error") else f"score={row['score']}"
            print(f"  [{done:>3}/{len(tickers)}] {row['ticker']:<14} {tag}", flush=True)

    elapsed = time.time() - t0
    _print_top(rows, args.top)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"lite-{run_id}.md"
    summary_path.write_text(_render_summary(rows, run_id, elapsed), encoding="utf-8")
    print(f"Summary: {summary_path}")
    print(f"Wall-clock: {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
    main()
