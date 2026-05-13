"""
Fact card loader + rendering helpers.

Two consumers:
  1. lg_agent.py auto-injects render_headline_layer(card) into the system prompt
     for the ticker being analysed -- the agent always sees the headline financials
     + segments + order book + top guidance without having to retrieve them.

  2. get_fact_card_detail(ticker, section) is exposed as a tool, so the agent
     can pull deeper layers (themes, risks, capex, geographic split) only when
     it needs them. Keeps the system prompt slim while leaving the depth
     accessible.

Stored under:
    data/concalls/<TICKER>/<filename>.facts.json

If a ticker has multiple transcripts (Q3 + Q4), we load the most recent by
call_date.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from ingest.fact_card_schema import FactCard
from ticker_utils import to_plain

logger = logging.getLogger(__name__)

CONCALLS_DIR = Path("data/concalls")

# Sections the agent can request via get_fact_card_detail.
DETAIL_SECTIONS = {
    "themes",            # key_themes
    "risks",             # risks_mentioned
    "segments_detail",   # full segments incl. key_developments + per-segment quotes
    "capex",             # capex_plan
    "geographic",        # geographic_split
    "capacity",          # capacity
    "commentary",        # pricing_commentary + demand_commentary
    "all",               # entire card as JSON
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _list_fact_cards(ticker: str) -> list[Path]:
    """All .facts.json files for a ticker, oldest first."""
    t = to_plain(ticker)
    ticker_dir = CONCALLS_DIR / t
    if not ticker_dir.exists():
        return []
    return sorted(ticker_dir.glob("*.facts.json"))


def get_fact_card(ticker: str) -> Optional[FactCard]:
    """Return the most recent FactCard for a ticker by call_date, or None.

    On parse failure for any individual file, that file is skipped (with a
    warning) so a corrupt extraction doesn't lock out an otherwise-good card.
    """
    paths = _list_fact_cards(ticker)
    if not paths:
        return None

    cards: list[FactCard] = []
    for p in paths:
        try:
            cards.append(FactCard.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("failed to load %s: %s", p, e)
            continue

    if not cards:
        return None
    cards.sort(key=lambda c: c.call_date, reverse=True)
    return cards[0]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_metric(m, label: str) -> Optional[str]:
    if m is None:
        return None
    return f"  - {label}: {m.value} {m.unit} ({m.period}) — \"{m.source_quote}\""


def render_headline_layer(card: FactCard) -> str:
    """Compact text block for system-prompt injection.

    Includes Layer 1 (headline financials) + a slim view of Layer 2/3:
      - segments with name + revenue/margin (no key_developments)
      - order book one-liner
      - top guidance items

    Deeper content (themes, risks, segment narratives, capex, etc.) is left to
    the get_fact_card_detail tool so we don't blow out the system prompt.
    """
    lines: list[str] = []
    lines.append(f"## FactCard | {card.ticker} | {card.fiscal_period} | call_date {card.call_date}")
    lines.append("(Structured extraction from concall transcript. Each fact carries its verbatim source quote. "
                 "Treat these as authoritative for the period above. Call `get_fact_card_detail` for themes/risks/segment narratives/capex/geography.)")

    # ---- Headline financials ----
    lines.append("\n### Headline financials")
    fin_lines = [
        _fmt_metric(card.revenue, "Revenue"),
        _fmt_metric(card.revenue_growth_yoy, "Revenue growth YoY"),
        _fmt_metric(card.ebitda, "EBITDA"),
        _fmt_metric(card.ebitda_margin, "EBITDA margin"),
        _fmt_metric(card.pat, "PAT"),
        _fmt_metric(card.eps, "EPS"),
    ]
    fin_lines = [l for l in fin_lines if l]
    if fin_lines:
        lines.extend(fin_lines)
    else:
        lines.append("  (none extracted)")

    # ---- Segments (slim view) ----
    if card.segments:
        lines.append("\n### Segments")
        for seg in card.segments:
            head = f"  - **{seg.name}**"
            bits = []
            if seg.revenue:
                bits.append(f"revenue {seg.revenue.value} {seg.revenue.unit}")
            if seg.revenue_growth_yoy:
                bits.append(f"YoY {seg.revenue_growth_yoy.value} {seg.revenue_growth_yoy.unit}")
            if seg.margin:
                bits.append(f"margin {seg.margin.value} {seg.margin.unit}")
            if bits:
                head += " — " + ", ".join(bits)
            lines.append(head)

    # ---- Order book ----
    if card.order_book is not None:
        ob = card.order_book
        head = f"\n### Order book\n  - {ob.value} {ob.unit}"
        if ob.as_of_date:
            head += f" (as of {ob.as_of_date})"
        if ob.growth_yoy:
            head += f", YoY {ob.growth_yoy.value} {ob.growth_yoy.unit}"
        if ob.execution_timeline:
            head += f", execution {ob.execution_timeline}"
        lines.append(head)
        lines.append(f"    Quote: \"{ob.source_quote}\"")
        if ob.breakdown:
            for k, m in ob.breakdown.items():
                lines.append(f"    - {k}: {m.value} {m.unit}")

    # ---- Guidance (forward-looking, the alpha) ----
    if card.guidance:
        lines.append("\n### Guidance")
        for g in card.guidance:
            lines.append(
                f"  - **{g.target}** ({g.period}, {g.confidence}): {g.value} "
                f"— \"{g.source_quote}\""
            )

    return "\n".join(lines)


def render_detail(card: FactCard, section: str) -> dict:
    """Return a JSON-friendly dict for one detail section.

    Used by the get_fact_card_detail tool. Each section is a different shape;
    the agent reads the dict and the schema is documented in the tool docstring.
    """
    if section == "themes":
        return {
            "key_themes": [t.model_dump() for t in card.key_themes],
        }
    if section == "risks":
        return {
            "risks_mentioned": [t.model_dump() for t in card.risks_mentioned],
        }
    if section == "segments_detail":
        return {"segments": [s.model_dump() for s in card.segments]}
    if section == "capex":
        return {"capex_plan": [c.model_dump() for c in card.capex_plan]}
    if section == "geographic":
        if card.geographic_split is None:
            return {"geographic_split": None}
        return {
            "geographic_split": {
                k: v.model_dump() for k, v in card.geographic_split.items()
            }
        }
    if section == "capacity":
        return {"capacity": card.capacity.model_dump() if card.capacity else None}
    if section == "commentary":
        return {
            "pricing_commentary": card.pricing_commentary.model_dump() if card.pricing_commentary else None,
            "demand_commentary": card.demand_commentary.model_dump() if card.demand_commentary else None,
        }
    if section == "all":
        return json.loads(card.model_dump_json())
    raise ValueError(f"Unknown section: {section!r}. Known: {sorted(DETAIL_SECTIONS)}")


def get_fact_card_detail(ticker: str, section: str) -> dict:
    """Tool entry point: fetch one section of the most recent fact card."""
    t = to_plain(ticker)
    if section not in DETAIL_SECTIONS:
        return {
            "error": f"Unknown section {section!r}.",
            "kind": "bad_section",
            "supported_sections": sorted(DETAIL_SECTIONS),
        }
    card = get_fact_card(t)
    if card is None:
        return {
            "error": f"No fact card available for {t}. Run ingest/extract_fact_card.py first.",
            "kind": "no_fact_card",
            "ticker": t,
        }
    detail = render_detail(card, section)
    detail["ticker"] = t
    detail["fiscal_period"] = card.fiscal_period
    detail["call_date"] = card.call_date
    detail["section"] = section
    return detail
