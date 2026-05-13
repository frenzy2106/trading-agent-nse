"""
NSE Trading Analyst — structured-thinking tool for NSE-listed stocks.

Usage:
    # Interactive: prompts for question, no extra context
    python lg_agent.py

    # One-shot with free-text context and sources
    python lg_agent.py "Should I buy RELIANCE for a 3-month hold?" \\
        --context "Iran tensions, oil at 90+" \\
        --pdf research/reliance_q3.pdf \\
        --url https://www.reuters.com/article/...
"""

import argparse
import logging
import os
import re
import sys
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from context_loader import build_context
from llm_factory import get_llm, get_provider_and_model
from persistence import build_trace, parse_header, save_run
from tools.fact_card import get_fact_card, render_headline_layer
from tools.tool_definitions import (
    get_analyst_consensus,
    get_commodity_snapshot,
    get_fact_card_detail,
    get_fundamentals_snapshot,
    get_macro_snapshot,
    get_management_commentary,
    get_news_and_earnings,
    get_technical_snapshot,
)
from prompts import SYSTEM_PROMPT
from ticker_utils import to_plain

load_dotenv()
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("lg_agent")

# Windows terminals default to cp1252; force utf-8 so model output with
# em-dashes, rupee signs, and emoji doesn't crash the print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOOLS = [
    get_technical_snapshot,
    get_macro_snapshot,
    get_fundamentals_snapshot,
    get_news_and_earnings,
    get_analyst_consensus,
    get_management_commentary,
    get_commodity_snapshot,
    get_fact_card_detail,
]


# Cheap regex over the user's question to pick the ticker for fact-card
# injection. We need to do this BEFORE the agent's first tool call so the
# fact card lands in the system prompt.
_TICKER_RE = re.compile(r"\b[A-Z]{2,12}\b")


def _guess_ticker_from_question(question: str) -> str | None:
    """Best-effort ticker detection. Returns None if nothing plausible found.

    The agent does its own ticker parsing too -- this is only used to decide
    which fact card to pre-inject. False positives are harmless (we'll just
    fail to find a card and skip injection).
    """
    candidates = _TICKER_RE.findall(question or "")
    # Filter out common English words that look like tickers.
    stop = {"BUY", "SELL", "HOLD", "I", "A", "ON", "THE", "IT", "IS", "OK", "NSE", "BSE",
            "FY", "Q1", "Q2", "Q3", "Q4", "PE", "PB", "EPS", "USD", "INR", "USA", "AI",
            "AND", "OR", "BUT", "FOR", "TO", "OF", "IN", "AS", "AT", "BY", "AM", "PM"}
    for c in candidates:
        if c in stop or len(c) < 3:
            continue
        return to_plain(c)
    return None


def _system_prompt_for_question(question: str) -> str:
    """Append the fact card headline layer to the base system prompt when a
    plausible ticker is detected and a fact card exists.

    Returns the unchanged SYSTEM_PROMPT when no card is available -- the agent
    still falls back to RAG via get_management_commentary in that case.
    """
    ticker = _guess_ticker_from_question(question)
    if not ticker:
        return SYSTEM_PROMPT
    card = get_fact_card(ticker)
    if card is None:
        return SYSTEM_PROMPT
    headline = render_headline_layer(card)
    logger.info("fact card injected | ticker=%s call_date=%s", ticker, card.call_date)
    return f"{SYSTEM_PROMPT}\n\n## Pre-extracted fact card (authoritative)\n\n{headline}"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph():
    llm = get_llm(tools=TOOLS)

    def chat_node(state: AgentState):
        # Pull the first HumanMessage to derive ticker for fact-card injection.
        # The first message is always the user's question (possibly with a
        # prepended <context> block); _guess_ticker_from_question tolerates both.
        first_human = next(
            (m for m in state["messages"] if isinstance(m, HumanMessage)),
            None,
        )
        question = first_human.content if first_human is not None else ""
        system_prompt = _system_prompt_for_question(question)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=MemorySaver())


def _compose_message(question: str, context_text: str) -> str:
    """Prepend the assembled context block to the user's question."""
    if not context_text:
        return question
    return f"{context_text}\n\n{question}"


def _run_one(app, provider: str, model: str, question: str, context_text: str, thread_id: str):
    run_config = {"configurable": {"thread_id": thread_id}}
    composed = _compose_message(question, context_text)

    logger.info("agent run start | question=%r ctx_chars=%d", question, len(context_text))
    t0 = time.time()
    result = app.invoke(
        {"messages": [HumanMessage(content=composed)]},
        config=run_config,
    )
    latency = time.time() - t0
    report = result["messages"][-1].content
    logger.info("agent run done | latency=%.2fs", latency)

    print(f"\nAgent:\n{report}\n")

    header = parse_header(report)
    if not header or not header.get("ticker"):
        print("[skip persistence] could not extract ticker from response")
        return

    trace = build_trace(result["messages"], latency)
    md_path, trace_path = save_run(
        ticker=header["ticker"],
        question=question,
        report=report,
        trace=trace,
        model=f"{provider}/{model}",
        rating=header.get("rating"),
        confidence=header.get("confidence"),
    )
    print(f"Saved: {md_path}")
    rating_label = header.get("rating") or "?"
    print(
        f"Trace: {trace_path}  "
        f"({rating_label}, "
        f"tools={trace['tool_call_count']}, "
        f"tokens={trace['tokens']['total']}, "
        f"latency={trace['latency_seconds']}s)\n"
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NSE Trading Analyst — structured-thinking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "question",
        nargs="?",
        help="One-shot question (omit for interactive mode)",
    )
    p.add_argument("--context", help="Free-text context to factor in")
    p.add_argument("--pdf", action="append", default=[], help="Path to a PDF source (repeatable)")
    p.add_argument("--md", action="append", default=[], help="Path to a markdown source (repeatable)")
    p.add_argument("--url", action="append", default=[], help="URL of an article source (repeatable)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    app = build_graph()
    provider, model = get_provider_and_model()

    print("\n=== NSE Trading Analyst ===")
    print("NOT FINANCIAL ADVICE. For educational and research purposes only.")
    print(f"Provider: {provider}  |  Model: {model}")

    # Build context once per CLI invocation.
    context_block = build_context(
        free_text=args.context,
        pdfs=args.pdf,
        urls=args.url,
        mds=args.md,
    )
    context_text = context_block.render()
    if not context_block.is_empty():
        bits = []
        if context_block.sources:
            bits.append(f"{len(context_block.sources)} source(s)")
        if context_block.free_text:
            bits.append("free-text context")
        print(f"Context loaded: {', '.join(bits)} ({len(context_text)} chars)")

    if args.question:
        _run_one(app, provider, model, args.question, context_text, thread_id="session-1")
        sys.exit(0)

    print("Ask about any NSE stock, e.g. 'Should I buy RELIANCE for a 3-month hold?'")
    print("Type 'quit' to exit.\n")

    run_idx = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        run_idx += 1
        _run_one(app, provider, model, user_input, context_text, thread_id=f"session-{run_idx}")
