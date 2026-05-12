"""
Watchlist screener — runs the live agent over every ticker in watchlist.txt
and produces a one-glance summary table.

What it does:
  1. Reads the watchlist file (default: watchlist.txt)
  2. Builds the agent ONCE (graph + LLM client are reused across tickers)
  3. For each ticker: runs the agent, persists the full per-ticker report via the
     existing persistence path (reports/<TICKER>/<DATE>.md + .trace.json)
  4. Aggregates ratings + key takeaways into a single screener summary at
     screener/<DATE>-<run-id>.md

Usage:
  python screener.py                            # full scan, all tickers
  python screener.py --watchlist my.txt         # custom watchlist file
  python screener.py --limit 3                  # only first 3 (smoke test)
  python screener.py --rating BUY,OVERWEIGHT    # filter the SUMMARY (still scans all)
  python screener.py --question "..."           # override the per-ticker question

Failures (rate-limits, ticker-not-found, network) on individual tickers are
caught and recorded in the summary as ERROR rows — they don't stop the scan.
"""

import argparse
import logging
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from context_loader import build_context
from lg_agent import build_graph
from llm_factory import get_provider_and_model
from persistence import build_trace, parse_header, save_run

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=log_level, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("screener")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_QUESTION = "Should I buy {ticker} for a 3-month hold?"
SUMMARY_DIR = Path("screener")


# Pull the explanation paragraph from the "Data-Only Conclusion" section
# (Path A format — sits after the **Rating (data only): X** bold line).
_TAKEAWAY_RE = re.compile(
    r"\*\*Rating\s*\(data only\):\s*[A-Z]+\*\*\s*\n+"
    r"(.+?)"
    r"(?=\n###|\n\*This is the data view|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"ERROR: watchlist not found at {path}")
    tickers: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line.upper())
    if not tickers:
        sys.exit(f"ERROR: no tickers in {path}")
    return tickers


def _extract_takeaway(report: str, max_chars: int = 220) -> str:
    """Pull the synthesis paragraph from the Data-Only Conclusion as a key takeaway."""
    m = _TAKEAWAY_RE.search(report)
    if not m:
        return ""
    body = m.group(1).strip()
    # Strip a leading "This is the data view" sentence if it sneaks in
    body = re.sub(r"\s+", " ", body)
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rsplit(" ", 1)[0] + "…"


def _run_ticker(app, provider: str, model: str, question_template: str, ticker: str,
                context_text: str, run_idx: int) -> dict:
    """Run the agent on one ticker. Returns a row dict for the summary table."""
    question = question_template.format(ticker=ticker)
    composed = f"{context_text}\n\n{question}" if context_text else question
    config = {"configurable": {"thread_id": f"screener-{run_idx}-{ticker}"}}

    t0 = time.time()
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=composed)]},
            config=config,
        )
    except Exception as e:
        latency = time.time() - t0
        err = f"{type(e).__name__}: {str(e)[:160]}"
        logger.warning("agent failure | ticker=%s err=%s", ticker, err)
        return {
            "ticker": ticker,
            "rating": "ERROR",
            "takeaway": err,
            "tokens": 0,
            "latency_s": round(latency, 1),
        }

    latency = time.time() - t0
    report = result["messages"][-1].content
    header = parse_header(report)
    trace = build_trace(result["messages"], latency)

    # Persist the full per-ticker report exactly like lg_agent does
    rating = header.get("rating") if header else None
    if header and header.get("ticker"):
        save_run(
            ticker=header["ticker"],
            question=question,
            report=report,
            trace=trace,
            model=f"{provider}/{model}",
            rating=rating,
            confidence=header.get("confidence"),
        )

    return {
        "ticker": header.get("ticker") if header else ticker,
        "rating": rating or "?",
        "takeaway": _extract_takeaway(report),
        "tokens": trace["tokens"]["total"],
        "latency_s": round(latency, 1),
    }


def _render_summary(run_id: str, rows: list[dict], elapsed: float,
                    rating_filter: list[str] | None) -> str:
    total_tokens = sum(r["tokens"] for r in rows)
    n_total = len(rows)
    n_failed = sum(1 for r in rows if r["rating"] == "ERROR")

    rating_order = ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL", "?", "ERROR"]
    sorted_rows = sorted(
        rows,
        key=lambda r: (rating_order.index(r["rating"]) if r["rating"] in rating_order else 99,
                       r["ticker"]),
    )
    if rating_filter:
        filt = {r.upper() for r in rating_filter}
        display_rows = [r for r in sorted_rows if r["rating"].upper() in filt]
    else:
        display_rows = sorted_rows

    parts = [
        f"# Screener Run — {run_id}",
        "",
        f"**Watchlist size:** {n_total} | "
        f"**Failures:** {n_failed} | "
        f"**Wall-clock:** {int(elapsed // 60)}m {int(elapsed % 60)}s | "
        f"**Total tokens:** {total_tokens:,}",
    ]
    if rating_filter:
        parts.append(f"**Filter applied:** {', '.join(rating_filter)} (showing {len(display_rows)} of {n_total})")
    parts += ["", "## Summary", "", "| Ticker | Rating | Latency | Tokens | Key takeaway |",
              "|---|---|---|---|---|"]
    for r in display_rows:
        takeaway = r["takeaway"].replace("|", "\\|")
        parts.append(
            f"| **{r['ticker']}** | {r['rating']} | {r['latency_s']}s | "
            f"{r['tokens']:,} | {takeaway} |"
        )

    parts += ["", "## Detailed reports", ""]
    today = date.today().isoformat()
    for r in display_rows:
        if r["rating"] != "ERROR":
            parts.append(f"- [{r['ticker']}](../reports/{r['ticker']}/{today}.md)")
    return "\n".join(parts)


def _print_console(rows: list[dict], elapsed: float):
    rating_order = ["BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL", "?", "ERROR"]
    sorted_rows = sorted(
        rows,
        key=lambda r: (rating_order.index(r["rating"]) if r["rating"] in rating_order else 99,
                       r["ticker"]),
    )
    print(f"\n{'='*88}")
    print(f"{'Ticker':<12} {'Rating':<13} {'Lat':>6} {'Tokens':>8}  Key takeaway")
    print(f"{'-'*88}")
    for r in sorted_rows:
        takeaway = r["takeaway"][:50] + ("…" if len(r["takeaway"]) > 50 else "")
        print(f"{r['ticker']:<12} {r['rating']:<13} {r['latency_s']:>5.1f}s "
              f"{r['tokens']:>8,}  {takeaway}")
    print(f"{'='*88}")
    print(f"Total: {len(rows)} tickers | "
          f"{int(elapsed // 60)}m {int(elapsed % 60)}s wall-clock | "
          f"{sum(r['tokens'] for r in rows):,} tokens")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSE Trading Analyst — watchlist screener")
    p.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist file")
    p.add_argument("--limit", type=int, default=None, help="Run on first N tickers only")
    p.add_argument(
        "--rating",
        default=None,
        help="Comma-separated ratings to show in summary (e.g. BUY,OVERWEIGHT). Scan still covers all tickers.",
    )
    p.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Question template, must contain {ticker}",
    )
    return p


def main():
    args = _build_argparser().parse_args()

    tickers = _read_watchlist(Path(args.watchlist))
    if args.limit:
        tickers = tickers[: args.limit]

    rating_filter = [r.strip().upper() for r in args.rating.split(",")] if args.rating else None

    provider, model = get_provider_and_model()
    print(f"\n=== Watchlist Screener ===")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Watchlist: {args.watchlist} | Tickers: {len(tickers)}")
    print(f"Question: {args.question}")
    print()

    # Build context once — applied across all tickers in this scan
    ctx_block = build_context()
    context_text = ctx_block.render()
    if context_text:
        print(f"Context: {len(context_text)} chars\n")

    app = build_graph()
    rows: list[dict] = []
    overall_t0 = time.time()
    for i, ticker in enumerate(tickers, start=1):
        print(f"  [{i:>2}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        row = _run_ticker(app, provider, model, args.question, ticker, context_text, i)
        rows.append(row)
        print(f"{row['rating']:<13} ({row['latency_s']}s, {row['tokens']:,} tok)")

    elapsed = time.time() - overall_t0
    _print_console(rows, elapsed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"{run_id}.md"
    summary_path.write_text(
        _render_summary(run_id, rows, elapsed, rating_filter),
        encoding="utf-8",
    )
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
