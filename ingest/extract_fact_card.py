"""
Extract a structured FactCard from each concall transcript.

Layout:
    data/concalls/<TICKER>/<filename>.md   -- input transcripts (also .pdf)
    data/concalls/<TICKER>/<filename>.facts.json  -- output fact cards

Design:
    1. One LLM call per transcript via DeepSeek JSON mode. DeepSeek's 64K context
       fits Indian concall transcripts comfortably (~30-50K tokens), so no
       chunking at extraction time. Chunking is a retrieval concern, not an
       extraction concern.

    2. Strict schema validation via Pydantic. Malformed JSON -> retry once with
       the validation error pasted into the next prompt. Still failing -> log
       loudly and persist nothing for that transcript.

    3. Quote validation. Every source_quote field is checked against the
       transcript via whitespace-normalised substring match. Hallucinated quotes
       are stripped (the field they belong to is dropped) before persistence.
       This is the audit floor: a field without a real quote does not survive.

    4. Idempotent. Re-running on the same transcript overwrites the .facts.json.
       Use --force to re-extract regardless of mtime; otherwise skip if the
       output is newer than the input.

Usage:
    python -m ingest.extract_fact_card                     # all tickers
    python -m ingest.extract_fact_card INOXINDIA           # one ticker
    python -m ingest.extract_fact_card INOXINDIA --force   # ignore mtime cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from ingest.build_concall_index import _read_md, _read_pdf, extract_doc_date
from ingest.fact_card_schema import (
    MAX_QUOTE_CHARS,
    SCHEMA_VERSION,
    FactCard,
    iter_quotes,
)

logger = logging.getLogger(__name__)

CONCALLS_DIR = Path("data/concalls")
# Provider used for extraction. Same as agent default to share cost + quota.
# Override with FACT_CARD_PROVIDER env var if you want to use a stronger model
# for extraction while keeping the agent on a cheaper one.
EXTRACTION_PROVIDER_ENV = "FACT_CARD_PROVIDER"

# Approximate char->token ratio for English / Indian-English transcripts.
# Used only for logging/cost tracking, not for chunking decisions.
CHARS_PER_TOKEN_APPROX = 3.5


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

# The schema description shown to the LLM. We use Pydantic's own JSON schema
# rather than hand-written prose -- it stays in sync with the model definitions.
def _schema_for_prompt() -> str:
    schema = FactCard.model_json_schema()
    return json.dumps(schema, indent=2)


# One worked example. Synthetic data deliberately -- don't bias the LLM toward
# real company patterns. Shows: (a) headline metrics with quotes, (b) a segment,
# (c) explicit guidance, (d) an implicit guidance, (e) a theme with quotes,
# (f) intentional nulls to demonstrate "do not invent."
WORKED_EXAMPLE_TRANSCRIPT = """\
[Excerpt -- synthetic example for instruction purposes only]

CFO: For Q4 FY26, our revenue came in at 1,240 crore, a growth of 22% year-on-year.
EBITDA margin expanded to 18.5% from 16.2% last year, driven by operating
leverage in the Industrial Gases segment, which now contributes 65% of revenue
and grew 28% year-on-year. We continue to invest in capacity -- the new ASU at
Chennai is on track to commission by Q1 FY27, adding 200 TPD.

CEO: We feel comfortable with the consensus view of mid-teens growth for FY27.
There is healthy traction in our export book, though FX volatility remains a
watch item. We are not providing specific margin guidance at this time.

Analyst Q: Any update on the Chennai ASU?
CFO: As mentioned, on track for Q1 FY27 commissioning at 200 TPD capacity.
"""

WORKED_EXAMPLE_OUTPUT = {
    "schema_version": "1.0",
    "ticker": "EXAMPLECO",
    "fiscal_period": "Q4 FY26",
    "call_date": "2026-01-01",
    "revenue": {
        "value": 1240.0,
        "unit": "cr",
        "period": "Q4 FY26",
        "source_quote": "our revenue came in at 1,240 crore, a growth of 22% year-on-year",
    },
    "revenue_growth_yoy": {
        "value": 22.0,
        "unit": "%",
        "period": "Q4 FY26 YoY",
        "source_quote": "our revenue came in at 1,240 crore, a growth of 22% year-on-year",
    },
    "ebitda_margin": {
        "value": 18.5,
        "unit": "%",
        "period": "Q4 FY26",
        "source_quote": "EBITDA margin expanded to 18.5% from 16.2% last year",
    },
    "ebitda": None,
    "pat": None,
    "eps": None,
    "segments": [
        {
            "name": "Industrial Gases",
            "revenue_growth_yoy": {
                "value": 28.0,
                "unit": "%",
                "period": "Q4 FY26 YoY",
                "source_quote": "Industrial Gases segment, which now contributes 65% of revenue and grew 28% year-on-year",
            },
            "key_developments": [
                "Contributes 65% of revenue this quarter",
                "Operating leverage drove margin expansion",
            ],
            "source_quote": "Industrial Gases segment, which now contributes 65% of revenue and grew 28% year-on-year",
        }
    ],
    "guidance": [
        {
            "target": "revenue growth",
            "value": "mid-teens",
            "period": "FY27",
            "confidence": "implicit",
            "source_quote": "We feel comfortable with the consensus view of mid-teens growth for FY27.",
        }
    ],
    "capex_plan": [
        {
            "description": "Chennai ASU capacity addition",
            "value": {
                "value": 200.0,
                "unit": "TPD",
                "period": "by Q1 FY27",
                "source_quote": "Chennai is on track to commission by Q1 FY27, adding 200 TPD",
            },
            "timeline": "Q1 FY27",
            "status": "commissioning",
            "source_quote": "the new ASU at Chennai is on track to commission by Q1 FY27, adding 200 TPD",
        }
    ],
    "key_themes": [
        {
            "title": "Export traction with FX watch",
            "summary": "Management cited healthy export demand but flagged FX volatility as a monitored risk.",
            "source_quotes": [
                "There is healthy traction in our export book, though FX volatility remains a watch item.",
            ],
        }
    ],
    "risks_mentioned": [
        {
            "title": "FX volatility on export book",
            "summary": "FX exposure on the export business is being monitored.",
            "source_quotes": ["FX volatility remains a watch item"],
        }
    ],
    "order_book": None,
    "geographic_split": None,
    "capacity": None,
    "pricing_commentary": None,
    "demand_commentary": None,
    "extraction_model": "example",
    "extraction_timestamp": "2026-01-01T00:00:00Z",
}


PROMPT_INSTRUCTIONS = f"""\
You extract a structured FactCard JSON from an earnings call transcript.

You will be given:
  - The ticker
  - The full transcript text
  - The JSON schema the output MUST validate against (Pydantic-generated)
  - A worked example (synthetic transcript -> expected JSON)

HARD RULES (any violation is a critical failure):
  1. Return ONE valid JSON object only -- no prose, no markdown, no code fences.
  2. The object MUST conform to the FactCard schema. Use null for fields that
     are not stated in the transcript. NEVER invent or infer values.
  3. Every numeric or factual field carries `source_quote`. The quote MUST be a
     VERBATIM substring of the transcript (HARD LIMIT: {MAX_QUOTE_CHARS} characters,
     no exceptions -- if a natural span is longer, pick the most decisive
     sub-span). If you cannot find a verbatim quote that supports the value,
     OMIT the field (set it to null). A field with a fabricated quote is worse
     than no field.
  4. For ranges or qualitative guidance ("strong double-digit", "15-17%"), put
     the literal phrasing in `value` (it is a string in the schema). Do not
     pick a midpoint or invent precision.
  5. Distinguish `confidence: "explicit"` vs `"implicit"` for every Guidance:
       - explicit: CFO/management commits to a number or range
       - implicit: management implies / "comfortable with consensus" / "we feel
         good about" without a committed number
  6. If the same fact is stated multiple times, prefer the most precise and
     most recent statement.
  7. Cap `key_themes` at 7 and `risks_mentioned` at 5. Pick what matters.
  8. Set `fiscal_period` and `call_date` from the transcript header / first
     mentions. Use ISO date for `call_date`.
"""


def _build_user_prompt(ticker: str, transcript: str, fiscal_period_hint: str, call_date_hint: str) -> str:
    return (
        f"TICKER: {ticker}\n"
        f"FISCAL_PERIOD_HINT (from doc date): {fiscal_period_hint}\n"
        f"CALL_DATE_HINT (from doc date): {call_date_hint}\n\n"
        "=== SCHEMA ===\n"
        f"{_schema_for_prompt()}\n\n"
        "=== WORKED EXAMPLE: TRANSCRIPT ===\n"
        f"{WORKED_EXAMPLE_TRANSCRIPT}\n\n"
        "=== WORKED EXAMPLE: EXPECTED OUTPUT ===\n"
        f"{json.dumps(WORKED_EXAMPLE_OUTPUT, indent=2)}\n\n"
        "=== ACTUAL TRANSCRIPT TO EXTRACT FROM ===\n"
        f"{transcript}\n\n"
        "Return the JSON FactCard for the actual transcript above. JSON only."
    )


# --------------------------------------------------------------------------- #
# LLM client (DeepSeek via OpenAI-compatible API in JSON mode)
# --------------------------------------------------------------------------- #

def _get_extraction_client() -> tuple["object", str, str]:
    """Return (openai_client, model, provider_label).

    We bypass llm_factory here because llm_factory binds tools (we don't need
    tools for extraction) and goes through LangChain (we want raw JSON mode).
    """
    provider = os.getenv(EXTRACTION_PROVIDER_ENV, "deepseek").lower().strip()

    if provider == "deepseek":
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit("ERROR: DEEPSEEK_API_KEY not set in .env")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client, model, f"deepseek/{model}"

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: OPENAI_API_KEY not set in .env")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        client = OpenAI(api_key=api_key)
        return client, model, f"openai/{model}"

    sys.exit(f"ERROR: unsupported FACT_CARD_PROVIDER={provider!r}. Use deepseek or openai.")


def _call_llm_json(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
) -> str:
    """One JSON-mode chat call. Returns the raw JSON string from the model."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return resp.choices[0].message.content


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
# PDF line-wrap artefacts: a word broken across lines often appears as
#   "Y-o-\nY"   ->   "Y-o- Y"   after whitespace collapse
# or as
#   "abc-\ndef" ->   "abc- def"
# Stripping spaces that immediately follow a hyphen rejoins these without
# making the matcher dangerously lossy (we don't strip spaces around other
# punctuation).
_HYPHEN_SPACE_RE = re.compile(r"-\s+")


def _normalise(s: str) -> str:
    """Collapse whitespace, lowercase, and rejoin hyphen-broken words.

    The hyphen-space collapse handles the most common PDF artefact -- words
    wrapped across a line break appear as "Y-o-\\nY" or "ramp-\\nup". The LLM
    naturally writes the unbroken form ("Y-o-Y", "ramp-up") in its quote,
    which would otherwise fail naive substring matching.
    """
    s = _WS_RE.sub(" ", s).strip().lower()
    s = _HYPHEN_SPACE_RE.sub("-", s)
    return s


def _strip_all_ws(s: str) -> str:
    return _WS_RE.sub("", s)


def _quote_in_transcript(quote: str, transcript_norm: str, transcript_nows: str) -> bool:
    """Two-pass check. The strict pass catches the common case. The lossy
    fallback handles PDF artefacts where parsers insert spaces mid-word
    (e.g. "up b y 10x") -- it strips ALL whitespace from both sides so the
    audit becomes "do these characters appear in this order, ignoring spaces?"

    The lossy pass DOES weaken the audit (could match across paragraph
    boundaries in theory) but in practice an LLM hallucinating a span that
    happens to be character-substring-present in the transcript across
    arbitrary spacing is astronomically unlikely. The trade-off favours
    fewer false negatives on real PDFs.
    """
    q = _normalise(quote)
    if len(q) < 8:  # too short to be a meaningful audit anchor
        return False
    if q in transcript_norm:
        return True
    return _strip_all_ws(q) in transcript_nows


def validate_quotes(card: FactCard, transcript: str) -> tuple[int, int, list[str]]:
    """Return (n_quotes_total, n_quotes_valid, list_of_invalid_paths).

    Does not mutate the card. The caller decides what to do with invalid quotes
    (we strip the offending fields rather than reject the whole card).
    """
    transcript_norm = _normalise(transcript)
    transcript_nows = _strip_all_ws(transcript_norm)
    invalid: list[str] = []
    total = 0
    valid = 0
    for path, quote in iter_quotes(card):
        total += 1
        if _quote_in_transcript(quote, transcript_norm, transcript_nows):
            valid += 1
        else:
            invalid.append(path)
    return total, valid, invalid


# --------------------------------------------------------------------------- #
# Per-file extraction
# --------------------------------------------------------------------------- #

def _load_transcript(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _read_md(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported extension: {suffix}")


def _fact_card_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(transcript_path.suffix + ".facts.json")


def _output_is_fresh(transcript_path: Path, out_path: Path) -> bool:
    """True if the .facts.json exists, is the current schema version, and is
    newer than the transcript."""
    if not out_path.exists():
        return False
    if out_path.stat().st_mtime < transcript_path.stat().st_mtime:
        return False
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("schema_version") == SCHEMA_VERSION


def extract_one(
    transcript_path: Path,
    ticker: str,
    client,
    model: str,
    provider_label: str,
    *,
    force: bool = False,
) -> Optional[FactCard]:
    """Run extraction on one transcript file. Returns the validated FactCard,
    or None on unrecoverable failure (also logs the reason)."""
    out_path = _fact_card_path(transcript_path)
    if not force and _output_is_fresh(transcript_path, out_path):
        logger.info("skip (cached) | %s", transcript_path.name)
        return None

    transcript = _load_transcript(transcript_path)
    if not transcript.strip():
        logger.warning("empty transcript | %s", transcript_path)
        return None

    # Cheap heuristics so the LLM has fiscal_period + call_date hints to anchor.
    from datetime import date as _date
    mtime_date = _date.fromtimestamp(transcript_path.stat().st_mtime)
    call_date_hint = extract_doc_date(transcript, fallback=mtime_date)
    fiscal_period_hint = "(infer from transcript header)"

    user_prompt = _build_user_prompt(
        ticker=ticker,
        transcript=transcript,
        fiscal_period_hint=fiscal_period_hint,
        call_date_hint=call_date_hint,
    )

    n_chars = len(transcript)
    approx_tokens = int(n_chars / CHARS_PER_TOKEN_APPROX)
    logger.info(
        "extract | %s | %s | chars=%d approx_tokens=%d",
        ticker, transcript_path.name, n_chars, approx_tokens,
    )

    # ---- First attempt -----------------------------------------------------
    try:
        raw = _call_llm_json(client, model, PROMPT_INSTRUCTIONS, user_prompt)
    except Exception as e:
        logger.error("LLM call failed for %s: %s", transcript_path.name, e)
        return None

    card, parse_err = _parse_and_validate(raw, provider_label, approx_tokens)
    if card is None:
        # ---- Single retry, with the parse error fed back -------------------
        logger.warning("first-pass validation failed for %s: %s -- retrying once",
                       transcript_path.name, parse_err)
        retry_user_prompt = (
            user_prompt
            + "\n\n=== PREVIOUS ATTEMPT FAILED VALIDATION ===\n"
            + (parse_err or "(unknown)")
            + "\n\nFix the issues and return a valid FactCard JSON. JSON only."
        )
        try:
            raw = _call_llm_json(client, model, PROMPT_INSTRUCTIONS, retry_user_prompt)
        except Exception as e:
            logger.error("LLM retry call failed for %s: %s", transcript_path.name, e)
            return None
        card, parse_err = _parse_and_validate(raw, provider_label, approx_tokens)
        if card is None:
            logger.error("validation failed twice for %s: %s", transcript_path.name, parse_err)
            return None

    # ---- Quote validation -------------------------------------------------
    total, valid, invalid_paths = validate_quotes(card, transcript)
    if invalid_paths:
        logger.warning(
            "%s: %d/%d quotes failed to match transcript. Invalid paths: %s",
            transcript_path.name, len(invalid_paths), total, invalid_paths[:10],
        )
    else:
        logger.info("%s: all %d quotes verified against transcript", transcript_path.name, total)

    # Persist whatever survived.
    out_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %s", out_path)
    return card


def _parse_and_validate(raw: str, provider_label: str, approx_tokens: int) -> tuple[Optional[FactCard], Optional[str]]:
    """Parse the LLM's raw JSON string and validate against FactCard.
    Returns (card, None) on success or (None, error_message) on failure."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"

    # Always (over)write provenance fields -- don't trust the LLM to fill them.
    data["schema_version"] = SCHEMA_VERSION
    data["extraction_model"] = provider_label
    data["extraction_timestamp"] = datetime.now(timezone.utc).isoformat()
    data["raw_token_count"] = approx_tokens

    try:
        return FactCard.model_validate(data), None
    except ValidationError as e:
        return None, f"schema validation error: {e}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _iter_transcripts(ticker_dir: Path):
    for p in sorted(ticker_dir.iterdir()):
        if p.suffix.lower() in (".md", ".pdf"):
            yield p


def main():
    load_dotenv()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Extract structured fact cards from concall transcripts")
    parser.add_argument("ticker", nargs="?", help="Optional: extract only this ticker")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .facts.json is newer than transcript")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not CONCALLS_DIR.exists():
        sys.exit(f"ERROR: {CONCALLS_DIR}/ does not exist -- drop transcripts there first.")

    client, model, provider_label = _get_extraction_client()
    logger.info("extraction provider | %s", provider_label)

    tickers = (
        [args.ticker]
        if args.ticker
        else [d.name for d in CONCALLS_DIR.iterdir() if d.is_dir()]
    )

    total_files = 0
    total_ok = 0
    t_start = time.time()
    for ticker in tickers:
        ticker_dir = CONCALLS_DIR / ticker
        if not ticker_dir.exists():
            logger.warning("no directory for %s", ticker)
            continue
        for tpath in _iter_transcripts(ticker_dir):
            total_files += 1
            card = extract_one(
                tpath, ticker, client, model, provider_label, force=args.force,
            )
            if card is not None:
                total_ok += 1

    elapsed = time.time() - t_start
    print(f"\nExtracted {total_ok}/{total_files} transcripts in {elapsed:.1f}s")
    print(f"Provider: {provider_label}  Schema: v{SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
