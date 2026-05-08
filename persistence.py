"""
Report + trace persistence for the agent.

After each successful run, save:
  reports/<TICKER>/<YYYY-MM-DD>.md         — final markdown report
  reports/<TICKER>/<YYYY-MM-DD>.trace.json — tool calls, tokens, latency

Trace is for debugging — replay what the agent did without re-running.
"""

import json
import re
from datetime import date
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

# Path A header, e.g.:
#   "## RELIANCE | 2026-05-07 | Horizon: 3 months"
_TICKER_HEADER_RE = re.compile(
    r"##\s+(?P<ticker>[A-Z][A-Z0-9]{1,15})\s*\|",
)

# Rating now lives in the "Data-Only Conclusion" section as:
#   "**Rating (data only): OVERWEIGHT**"
_RATING_RE = re.compile(
    r"\*\*Rating\s*\(data only\):\s*(?P<rating>BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\*\*",
    re.IGNORECASE,
)

# Legacy patterns — kept so older reports still parse cleanly.
_LEGACY_HEADER_RE = re.compile(
    r"##\s+(?P<ticker>[A-Z][A-Z0-9]{1,15})\s+[—–\-]\s+"
    r"(?P<rating>BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)"
    r"(?:\s*\(Confidence:\s*(?P<confidence>HIGH|MEDIUM|LOW)\s*\))?",
    re.IGNORECASE,
)


def parse_header(content: str) -> dict | None:
    """Extract ticker + rating (and confidence if present in legacy reports)."""
    # New Path A format: ticker in header, rating in body.
    m_ticker = _TICKER_HEADER_RE.search(content)
    m_rating = _RATING_RE.search(content)
    if m_ticker and m_rating:
        return {
            "ticker": m_ticker.group("ticker").upper(),
            "rating": m_rating.group("rating").upper(),
            "confidence": None,
        }

    # Legacy format fallback.
    m_legacy = _LEGACY_HEADER_RE.search(content)
    if m_legacy:
        return {
            "ticker": m_legacy.group("ticker").upper(),
            "rating": m_legacy.group("rating").upper(),
            "confidence": (m_legacy.group("confidence") or "").upper() or None,
        }

    # Last resort: ticker only (rating may be missing if model fumbled).
    if m_ticker:
        return {
            "ticker": m_ticker.group("ticker").upper(),
            "rating": None,
            "confidence": None,
        }
    return None


def extract_ticker(content: str) -> str | None:
    """Backward-compatible helper used by lg_agent.py."""
    h = parse_header(content)
    return h["ticker"] if h else None


def build_trace(messages: list, latency: float) -> dict:
    """Walk the message history and pull out tool calls, responses, token usage."""
    tool_calls: list[dict] = []
    tool_responses: dict[str, str] = {}
    tokens = {"input": 0, "output": 0, "total": 0}

    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls or []:
                tool_calls.append({
                    "id": tc.get("id"),
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                })
            um = getattr(msg, "usage_metadata", None) or {}
            tokens["input"] += um.get("input_tokens", 0)
            tokens["output"] += um.get("output_tokens", 0)
            tokens["total"] += um.get("total_tokens", 0)

        elif isinstance(msg, ToolMessage):
            content = str(msg.content)
            truncated = content[:500] + ("…[truncated]" if len(content) > 500 else "")
            tool_responses[msg.tool_call_id] = truncated

    for tc in tool_calls:
        tc["response"] = tool_responses.get(tc["id"], "<no response>")

    return {
        "tool_calls": tool_calls,
        "tool_call_count": len(tool_calls),
        "tokens": tokens,
        "latency_seconds": round(latency, 2),
    }


def save_run(
    ticker: str,
    question: str,
    report: str,
    trace: dict,
    model: str,
    out_root: str = "reports",
    rating: str | None = None,
    confidence: str | None = None,
) -> tuple[Path, Path]:
    """Write report markdown and trace JSON. Overwrites if same date."""
    today = date.today().isoformat()
    out_dir = Path(out_root) / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{today}.md"
    md_path.write_text(report, encoding="utf-8")

    trace_payload = {
        "ticker": ticker,
        "date": today,
        "model": model,
        "question": question,
        "rating": rating,
        "confidence": confidence,
        **trace,
    }
    trace_path = out_dir / f"{today}.trace.json"
    trace_path.write_text(
        json.dumps(trace_payload, indent=2, default=str),
        encoding="utf-8",
    )
    return md_path, trace_path
