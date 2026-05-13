"""Print rich per-card data for designing golden eval entries."""
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tools.fact_card import get_fact_card

DEFAULT_TICKERS = ["RELIANCE", "TCS", "INOXINDIA", "HDFCBANK", "ICICIBANK", "HAVELLS",
                   "MTARTECH", "PNGJL", "SBIN", "SHRIRAMFIN", "SIS", "TITAN"]

parser = argparse.ArgumentParser()
parser.add_argument("ticker", nargs="?", help="Specific ticker for full dump; omit for summary of all")
args = parser.parse_args()

tickers = [args.ticker] if args.ticker else DEFAULT_TICKERS
verbose = args.ticker is not None

for t in tickers:
    card = get_fact_card(t)
    if card is None:
        print(f"\n### {t} -- NO CARD\n")
        continue
    print(f"\n### {t} | {card.fiscal_period} | {card.call_date}")
    for fname, label in [("revenue", "Revenue"), ("revenue_growth_yoy", "Rev YoY"),
                         ("ebitda", "EBITDA"), ("ebitda_margin", "EBITDA margin"),
                         ("pat", "PAT"), ("eps", "EPS")]:
        m = getattr(card, fname)
        if m:
            print(f"  {label}: {m.value} {m.unit} ({m.period})")
    print(f"  Segments ({len(card.segments)}): {[s.name for s in card.segments]}")
    if verbose:
        for seg in card.segments:
            print(f"\n  Segment: {seg.name}")
            if seg.revenue:
                print(f"    revenue: {seg.revenue.value} {seg.revenue.unit}")
            if seg.revenue_growth_yoy:
                print(f"    YoY: {seg.revenue_growth_yoy.value} {seg.revenue_growth_yoy.unit}")
            if seg.margin:
                print(f"    margin: {seg.margin.value} {seg.margin.unit}")
            for kd in seg.key_developments:
                print(f"    - {kd}")
    if card.order_book:
        print(f"  Order book: {card.order_book.value} {card.order_book.unit}")
    if card.guidance:
        print(f"  Guidance ({len(card.guidance)}):")
        for g in card.guidance:
            print(f"    - [{g.confidence}] {g.target} ({g.period}): {g.value}")
    if card.capex_plan:
        print(f"  Capex ({len(card.capex_plan)}):")
        for c in card.capex_plan:
            v = f" {c.value.value} {c.value.unit}" if c.value else ""
            print(f"    - {c.description}{v}  [{c.timeline or '?'} / {c.status or '?'}]")
    if card.key_themes:
        print(f"  Themes ({len(card.key_themes)}):")
        for th in card.key_themes:
            print(f"    - {th.title}: {th.summary[:100] if verbose else ''}")
    if card.risks_mentioned:
        print(f"  Risks ({len(card.risks_mentioned)}):")
        for th in card.risks_mentioned:
            print(f"    - {th.title}")
