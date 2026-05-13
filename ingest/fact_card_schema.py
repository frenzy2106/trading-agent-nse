"""
Fact card schema — structured representation of a single earnings call transcript.

Design choices (locked in v1.0; see project notes for rationale):

  - All numeric/factual fields carry a verbatim `source_quote` (<= 200 chars).
    The quote is the audit trail: a post-extraction validator checks that every
    quote appears in the transcript text. If it doesn't, the field is rejected.

  - `value` / `unit` / `period` are free-form strings, not parsed types or enums.
    Concalls mix units (cr / mn USD / MT / GW / tonnes) and express guidance as
    ranges ("15-17%") or qualitatives ("strong double-digit"). Parsing early
    destroys signal. Normalisation, if needed, happens at query time.

  - `Optional[X]` is used liberally. Most fields are sector-dependent — banks
    don't report order books, IT firms don't report commodity input mix. The
    LLM is instructed to return null rather than hallucinate.

  - `key_themes` capped at 7, `risks_mentioned` at 5. Forces the extractor to
    pick what matters rather than dump everything (which would just re-create
    chunked transcript noise).

  - Per-quarter snapshot, not cross-quarter aggregation. Each transcript = one
    card. Trends are joined at query time, not at extraction time.

  - schema_version is mandatory. When the schema changes, we bump the version
    and re-extract — one LLM call per transcript is cheap.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

# Verbatim quotes are capped to keep system-prompt token cost bounded and to
# force the LLM to pick the narrowest defensible span. Bumped from 200 -> 400
# after two bulk-run failures where the LLM put a natural two-sentence span
# (e.g. "We registered a PAT of INR X cr ... at Shriram Housing Finance") in
# one quote. ~400 chars = roughly two full concall sentences -- still strong
# enough as an audit anchor (the substring match doesn't get weaker with length)
# but doesn't reject natural multi-clause quotes.
MAX_QUOTE_CHARS = 400


class FinancialMetric(BaseModel):
    """A single numeric fact (revenue, margin, growth %, capacity in GW, etc.).

    The combination of `value` + `unit` + `period` carries the full meaning.
    Examples:
        FinancialMetric(value=18.0, unit="%", period="Q4 FY26 YoY",
                        source_quote="Revenue grew 18% year-on-year in Q4...")
        FinancialMetric(value=1485.0, unit="cr",
                        period="as of 31-Mar-26",
                        source_quote="Closing order book stood at 1,485 crore...")
    """

    value: float
    unit: str = Field(description="Free-form unit string: 'cr', '%', 'mn USD', 'MT', 'GW', etc.")
    period: str = Field(description="Free-form period label: 'Q4 FY26', 'FY27 guided', 'YoY', 'as of 31-Mar-26'.")
    source_quote: str = Field(
        max_length=MAX_QUOTE_CHARS,
        description="Verbatim transcript span supporting this fact. <= 200 chars.",
    )


class Segment(BaseModel):
    """One operating segment / division of the business.

    `key_developments` is a list of free-form strings (not nested metrics) because
    segment commentary mixes numbers with qualitatives — e.g. for RELIANCE
    New Energy: "commissioned HJT lines, first in country", "20 GW integrated
    solar capacity expanded". Forcing each into a FinancialMetric loses the
    qualitative half. The block-level `source_quote` is the audit anchor.
    """

    name: str = Field(description="Segment name as used in the transcript, e.g. 'Oil to Chemicals', 'Digital Services'.")
    revenue: Optional[FinancialMetric] = None
    revenue_growth_yoy: Optional[FinancialMetric] = None
    margin: Optional[FinancialMetric] = Field(default=None, description="EBITDA margin or equivalent if disclosed.")
    key_developments: list[str] = Field(default_factory=list, description="3-5 bullet points on what changed this quarter.")
    source_quote: str = Field(
        max_length=MAX_QUOTE_CHARS,
        description="Representative verbatim span from the segment commentary.",
    )


class OrderBook(BaseModel):
    """Outstanding order book / backlog. Critical for capex-cycle and B2B businesses.

    `breakdown` is a free-form dict to handle: exports vs domestic, by segment,
    by customer type, etc. The transcript dictates the breakdown structure;
    forcing a fixed schema breaks across sectors.
    """

    value: float
    unit: str = Field(description="'cr', 'mn USD', etc.")
    as_of_date: Optional[str] = Field(default=None, description="ISO date or 'as of Q4 FY26' if exact date not given.")
    breakdown: Optional[dict[str, FinancialMetric]] = Field(
        default=None,
        description="Splits like {'exports': ..., 'domestic': ...} or {'product_X': ..., 'product_Y': ...}.",
    )
    growth_yoy: Optional[FinancialMetric] = None
    execution_timeline: Optional[str] = Field(default=None, description="Free-form: '12-18 months', 'over the next 2 years'.")
    source_quote: str = Field(max_length=MAX_QUOTE_CHARS)


class Guidance(BaseModel):
    """Forward-looking statement by management.

    `confidence` distinguishes explicit guidance (CFO commits to a number) from
    implicit (management implies, analyst infers). Both matter, but they should
    NOT be conflated. Embeddings cannot capture this distinction; structured
    extraction can.

    `value` is a string to preserve ranges ('15-17%'), inequalities ('>=21%'),
    and qualitatives ('strong double-digit'). The agent reads the string.
    """

    target: str = Field(description="What's being guided: 'revenue growth', 'EBITDA margin', 'capacity', etc.")
    value: str = Field(description="The guided value. Range/inequality/qualitative all allowed.")
    period: str = Field(description="The period the guidance applies to: 'FY27', 'exit FY26', 'medium-term'.")
    confidence: Literal["explicit", "implicit"] = Field(
        description=(
            "'explicit' = management commits to a number/range. "
            "'implicit' = management implies without committing ('comfortable with consensus', "
            "'we feel good about the trajectory')."
        )
    )
    source_quote: str = Field(max_length=MAX_QUOTE_CHARS)


class CapexItem(BaseModel):
    """A specific planned or in-flight capital expenditure.

    Separated from Guidance because capex carries timeline + facility + capacity
    details that don't fit the (target, value, period) shape — e.g. 'commission
    polysilicon line by Q2 FY27 at Jamnagar with 20 GW integrated capacity'.
    """

    description: str = Field(description="What the capex is for. E.g. 'solar polysilicon line', 'new factory in X'.")
    value: Optional[FinancialMetric] = Field(default=None, description="Spend or capacity addition, if quantified.")
    timeline: Optional[str] = Field(default=None, description="'Q2 FY27', 'next 18 months', 'commissioning underway'.")
    status: Optional[str] = Field(default=None, description="'planned', 'under construction', 'commissioning', 'commissioned'.")
    source_quote: str = Field(max_length=MAX_QUOTE_CHARS)


class CapacityStatus(BaseModel):
    """Current production / service capacity for capacity-constrained businesses.

    Applies to manufacturing, refining, hospitality, telecom — anywhere where
    utilisation is a primary growth lever. Skip for asset-light businesses.
    """

    installed: Optional[FinancialMetric] = Field(default=None, description="Total installed / nameplate capacity.")
    utilization_pct: Optional[FinancialMetric] = Field(default=None, description="Current utilisation as a percentage.")
    expansion_planned: Optional[FinancialMetric] = Field(default=None, description="Capacity being added.")
    source_quote: str = Field(max_length=MAX_QUOTE_CHARS)


class Theme(BaseModel):
    """A qualitative theme with provenance.

    Structured (not free-form string) so themes carry source quotes. A theme
    without supporting quotes is hallucination by another name.
    """

    title: str = Field(description="Short label, 2-6 words. E.g. 'AI-led deal wins', 'channel inventory normalising'.")
    summary: str = Field(description="1-2 sentence summary in your own words.")
    source_quotes: list[str] = Field(
        default_factory=list,
        description="Supporting verbatim spans (each <= 200 chars). At least one quote per theme.",
    )


class FactCard(BaseModel):
    """The full structured representation of one earnings call.

    Mirrors the conceptual layers discussed during design:
      Layer 1: headline financials (always relevant, auto-injected into agent)
      Layer 2: structural breakdowns (segments / orders / geo / capacity)
      Layer 3: forward-looking (guidance + capex)
      Layer 4: qualitative themes / risks / commentary
    """

    schema_version: str = SCHEMA_VERSION
    ticker: str = Field(description="NSE ticker symbol.")
    fiscal_period: str = Field(description="'Q4 FY26', 'FY26 H2', etc.")
    call_date: str = Field(description="ISO date the call was held.")

    # ---- Layer 1: headline financials -----------------------------------------
    revenue: Optional[FinancialMetric] = None
    revenue_growth_yoy: Optional[FinancialMetric] = None
    ebitda: Optional[FinancialMetric] = None
    ebitda_margin: Optional[FinancialMetric] = None
    pat: Optional[FinancialMetric] = Field(default=None, description="Profit after tax.")
    eps: Optional[FinancialMetric] = None

    # ---- Layer 2: structural breakdowns ---------------------------------------
    segments: list[Segment] = Field(default_factory=list)
    order_book: Optional[OrderBook] = None
    geographic_split: Optional[dict[str, FinancialMetric]] = Field(
        default=None,
        description="Revenue/order split by geography, e.g. {'India': ..., 'US': ..., 'Europe': ...}.",
    )
    capacity: Optional[CapacityStatus] = None

    # ---- Layer 3: forward-looking ---------------------------------------------
    guidance: list[Guidance] = Field(default_factory=list)
    capex_plan: list[CapexItem] = Field(default_factory=list)

    # ---- Layer 4: qualitative themes ------------------------------------------
    key_themes: list[Theme] = Field(default_factory=list, description="Max 7 themes management emphasised.")
    risks_mentioned: list[Theme] = Field(default_factory=list, description="Max 5 risks management acknowledged.")
    pricing_commentary: Optional[Theme] = Field(default=None, description="Pricing power / pass-through commentary.")
    demand_commentary: Optional[Theme] = Field(default=None, description="Demand environment by geo/segment/customer.")

    # ---- Provenance -----------------------------------------------------------
    extraction_model: str = Field(description="provider/model used, e.g. 'deepseek/deepseek-chat'.")
    extraction_timestamp: str = Field(description="ISO timestamp when the extraction was run.")
    raw_token_count: Optional[int] = Field(default=None, description="Approximate transcript token count, for cost tracking.")


def iter_quotes(card: FactCard):
    """Yield (json_path, quote_string) for every source_quote in the card.

    Used by the validator to check that every quote appears in the transcript.
    Skips quotes that are None or empty (legitimately optional).
    """

    def _emit(path: str, q):
        if isinstance(q, str) and q.strip():
            yield path, q

    # Layer 1 metrics
    for fname in ("revenue", "revenue_growth_yoy", "ebitda", "ebitda_margin", "pat", "eps"):
        m = getattr(card, fname)
        if m is not None:
            yield from _emit(f"{fname}.source_quote", m.source_quote)

    # Layer 2 — segments
    for i, seg in enumerate(card.segments):
        yield from _emit(f"segments[{i}].source_quote", seg.source_quote)
        for sub in ("revenue", "revenue_growth_yoy", "margin"):
            sm = getattr(seg, sub)
            if sm is not None:
                yield from _emit(f"segments[{i}].{sub}.source_quote", sm.source_quote)

    # Layer 2 — order book
    if card.order_book is not None:
        yield from _emit("order_book.source_quote", card.order_book.source_quote)
        if card.order_book.breakdown:
            for k, m in card.order_book.breakdown.items():
                yield from _emit(f"order_book.breakdown.{k}.source_quote", m.source_quote)
        if card.order_book.growth_yoy is not None:
            yield from _emit("order_book.growth_yoy.source_quote", card.order_book.growth_yoy.source_quote)

    # Layer 2 — geographic split
    if card.geographic_split:
        for k, m in card.geographic_split.items():
            yield from _emit(f"geographic_split.{k}.source_quote", m.source_quote)

    # Layer 2 — capacity
    if card.capacity is not None:
        yield from _emit("capacity.source_quote", card.capacity.source_quote)

    # Layer 3 — guidance + capex
    for i, g in enumerate(card.guidance):
        yield from _emit(f"guidance[{i}].source_quote", g.source_quote)
    for i, c in enumerate(card.capex_plan):
        yield from _emit(f"capex_plan[{i}].source_quote", c.source_quote)

    # Layer 4 — themes
    for i, t in enumerate(card.key_themes):
        for j, q in enumerate(t.source_quotes):
            yield from _emit(f"key_themes[{i}].source_quotes[{j}]", q)
    for i, t in enumerate(card.risks_mentioned):
        for j, q in enumerate(t.source_quotes):
            yield from _emit(f"risks_mentioned[{i}].source_quotes[{j}]", q)
    if card.pricing_commentary is not None:
        for j, q in enumerate(card.pricing_commentary.source_quotes):
            yield from _emit(f"pricing_commentary.source_quotes[{j}]", q)
    if card.demand_commentary is not None:
        for j, q in enumerate(card.demand_commentary.source_quotes):
            yield from _emit(f"demand_commentary.source_quotes[{j}]", q)
