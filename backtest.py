"""
Backtest harness — evaluates the technical-only agent on historical (ticker, date) pairs.

For each (ticker, date) pair:
  1. Run the agent with as_of_date=date, capture rating + confidence
  2. Pull forward returns (T+30, T+60, T+90 days) from Kite
  3. Save row to backtest/<run-id>/results.csv + per-call traces

Limitations:
  - Technical-only (yfinance fundamentals not point-in-time, so excluded)
  - LLM is non-deterministic even at temperature=0; expect minor variance
  - Survivorship bias: tested on currently-listed names

Usage:
    python backtest.py
"""

import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from backtest_prompts import SYSTEM_PROMPT_BACKTEST
from kite_login import get_kite_client
from persistence import build_trace, parse_header
from tools.technical import (
    TickerNotFoundError,
    get_technical_snapshot as _technical,
)

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("backtest")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── Config ──────────────────────────────────────────────────────────────────

NIFTY_10 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HUL", "ITC", "KOTAKBANK", "BAJFINANCE", "SUNPHARMA",
]

TEST_DATES = [
    date(2025, 5, 7),
    date(2025, 8, 7),
    date(2025, 11, 7),
    date(2026, 2, 7),
]

FORWARD_WINDOWS = [30, 60, 90]  # days forward to measure returns


# ── Per-date agent build (bakes in as_of_date) ──────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _make_technical_tool(as_of_date: date):
    """Wrap the underlying technical tool with as_of_date frozen in."""

    @tool
    def get_technical_snapshot(ticker: str, lookback_days: int = 365) -> str:
        """Fetch technical analysis snapshot for an NSE-listed stock as of a fixed historical date."""
        try:
            return json.dumps(_technical(ticker, lookback_days, as_of_date=as_of_date), default=str)
        except TickerNotFoundError as e:
            return json.dumps({
                "error": f"Ticker '{e.ticker}' not found on NSE.",
                "kind": "ticker_not_found",
            })
        except Exception as e:
            return json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "kind": "unexpected",
            })

    return get_technical_snapshot


def build_backtest_graph(as_of_date: date, model: str, api_key: str):
    technical_tool = _make_technical_tool(as_of_date)
    tools = [technical_tool]
    llm = ChatGroq(
        model=model,
        api_key=api_key,
        temperature=0,
    ).bind_tools(tools)

    def chat_node(state: AgentState):
        messages = [SystemMessage(content=SYSTEM_PROMPT_BACKTEST)] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=MemorySaver())


# ── Forward returns from Kite ───────────────────────────────────────────────


def _instrument_token(kite, ticker: str) -> int | None:
    instruments = kite.instruments("NSE")
    matches = [i for i in instruments if i["tradingsymbol"] == ticker]
    return matches[0]["instrument_token"] if matches else None


def fetch_full_history(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """One-shot OHLCV pull covering all backtest dates + forward windows for this ticker."""
    kite = get_kite_client()
    token = _instrument_token(kite, ticker)
    if token is None:
        return pd.DataFrame()

    records = kite.historical_data(
        instrument_token=token,
        from_date=start_date,
        to_date=end_date,
        interval="day",
    )
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None).dt.normalize()
    df = df.set_index("date").sort_index()
    return df


def forward_returns(df: pd.DataFrame, test_date: date, windows: list[int]) -> dict:
    """For a given test_date, compute (close at T+N days / close at T) - 1 for each window N."""
    if df.empty:
        return {f"ret_{w}d": None for w in windows} | {"price_at_test": None}

    test_ts = pd.Timestamp(test_date)
    on_or_after = df[df.index >= test_ts]
    if on_or_after.empty:
        return {f"ret_{w}d": None for w in windows} | {"price_at_test": None}

    test_close = float(on_or_after["close"].iloc[0])
    out: dict = {"price_at_test": round(test_close, 2)}

    for w in windows:
        target_ts = test_ts + pd.Timedelta(days=w)
        forward_slice = df[df.index >= target_ts]
        if forward_slice.empty:
            out[f"ret_{w}d"] = None
        else:
            forward_close = float(forward_slice["close"].iloc[0])
            out[f"ret_{w}d"] = round((forward_close / test_close - 1) * 100, 2)
    return out


# ── Run a single (ticker, date) pair ────────────────────────────────────────


def run_one(ticker: str, test_date: date, model: str, api_key: str, out_dir: Path) -> dict:
    app = build_backtest_graph(test_date, model, api_key)
    config = {"configurable": {"thread_id": f"{ticker}-{test_date}"}}

    user_msg = (
        f"As of {test_date.isoformat()}, what is your call on {ticker} for a 3-month hold? "
        f"Use only the technical snapshot. The current date is {test_date.isoformat()}."
    )

    t0 = time.time()
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )
    except Exception as e:
        logger.warning("agent failure | ticker=%s date=%s err=%s", ticker, test_date, e)
        return {"ticker": ticker, "test_date": test_date.isoformat(), "rating": None,
                "confidence": None, "error": str(e), "latency_s": round(time.time() - t0, 2)}

    latency = time.time() - t0
    report = result["messages"][-1].content
    header = parse_header(report)
    trace = build_trace(result["messages"], latency)

    # Save the per-call markdown + trace under backtest/<run-id>/calls/
    calls_dir = out_dir / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    base = f"{ticker}_{test_date.isoformat()}"
    (calls_dir / f"{base}.md").write_text(report, encoding="utf-8")
    (calls_dir / f"{base}.trace.json").write_text(
        json.dumps({
            "ticker": ticker,
            "test_date": test_date.isoformat(),
            "model": model,
            "rating": header["rating"] if header else None,
            "confidence": header["confidence"] if header else None,
            **trace,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "ticker": ticker,
        "test_date": test_date.isoformat(),
        "rating": header["rating"] if header else None,
        "confidence": header["confidence"] if header else None,
        "tokens": trace["tokens"]["total"],
        "latency_s": round(latency, 2),
    }


# ── Orchestrator ────────────────────────────────────────────────────────────


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("ERROR: GROQ_API_KEY not set in .env")

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("backtest") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Backtest run {run_id} ===")
    print(f"Model: {model} (temperature=0)")
    print(f"Tickers: {len(NIFTY_10)} | Dates: {len(TEST_DATES)} | Total calls: {len(NIFTY_10) * len(TEST_DATES)}")
    print(f"Output: {out_dir}\n")

    # Fetch full price history for each ticker once (covers all test dates + forward windows)
    earliest = min(TEST_DATES)
    latest = max(TEST_DATES) + timedelta(days=max(FORWARD_WINDOWS) + 7)
    print("Fetching price history for forward returns...")
    history: dict[str, pd.DataFrame] = {}
    for ticker in NIFTY_10:
        df = fetch_full_history(ticker, earliest, latest)
        history[ticker] = df
        print(f"  {ticker}: {len(df)} rows {earliest} -> {latest}")

    # Run all agent calls
    rows: list[dict] = []
    total = len(NIFTY_10) * len(TEST_DATES)
    i = 0
    overall_t0 = time.time()
    for test_date in TEST_DATES:
        for ticker in NIFTY_10:
            i += 1
            print(f"  [{i:>2}/{total}] {ticker} @ {test_date}...", end=" ", flush=True)
            row = run_one(ticker, test_date, model, api_key, out_dir)
            row.update(forward_returns(history.get(ticker, pd.DataFrame()), test_date, FORWARD_WINDOWS))
            rows.append(row)
            tag = f"{row.get('rating') or 'ERR'}/{row.get('confidence') or '-'}"
            print(f"{tag}  +30d={row.get('ret_30d')}%  +90d={row.get('ret_90d')}%  ({row['latency_s']}s)")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.time() - overall_t0
    print(f"\n=== Done ===")
    print(f"Wrote {csv_path}")
    print(f"Total wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Per-call traces: {out_dir / 'calls'}/")
    print(f"\nNext: python backtest_analyze.py {run_id}")


if __name__ == "__main__":
    main()
