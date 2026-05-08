# Trading Agent — V1 Implementation Plan

Single-agent recommendation system for NSE-listed stocks at a positional (weeks-to-months) horizon. Recommendation-only — never executes. Inspired by [TradingAgents](https://github.com/TauricResearch/TradingAgents) but stripped to one ReAct loop.

**Scope of v1:** given *"Should I buy RELIANCE for a 3-month hold?"*, return a markdown report with a Buy/Overweight/Hold/Underweight/Sell rating + structured thesis.

**Out of scope for v1:** news, debate, memory across runs, watchlist, scheduled runs, web UI. (See `Implementation-Plan.md` for v2+ roadmap.)

---

## Decisions locked in

| Decision | Choice |
|---|---|
| Architecture | Single-agent ReAct (one chat node + one tool node, like zwigato) |
| State | Just `messages` — no blackboard yet |
| LLM | Groq, `openai/gpt-oss-120b` |
| Prices/technicals | Zerodha Kite Connect |
| Fundamentals | yfinance |
| News | Skipped in v1 |
| Output | Markdown report only |
| User input | Free-text question, agent extracts ticker + horizon + intent |
| Tool granularity | Chunky (`get_technical_snapshot` style) |
| Rating scale | TradingAgents' 5-tier (Buy / Overweight / Hold / Underweight / Sell) |
| Failure mode | Fail-loudly (debuggable while learning) |

---

## Architecture

```
user input → chat_node (Groq) → tool_node → chat_node → markdown report
                ↑                    ↓
                └── system prompt    └── get_technical_snapshot
                    (rating scale,        get_fundamentals_snapshot
                     output format,       (Kite + yfinance under the hood)
                     evidence checklist)
```

State = `messages`. Same shape as zwigato.

---

## Phase 0 — Data foundations (no agent yet)

**Goal:** prove both data sources work before any LangGraph code touches them.

**Deliverables**
- Kite auth flow working (request_token → access_token → cached, daily refresh)
- Standalone script: pull 12 months daily OHLCV for one ticker via Kite, print
- Standalone script: pull yfinance fundamentals for same ticker, print available fields
- Ticker normalization helper: `RELIANCE` ↔ `NSE:RELIANCE` ↔ `RELIANCE.NS`
- `.env` + `.gitignore` + `.env.example` set up before first commit

**Success criteria**
- Both scripts run from a fresh terminal with no agent code involved
- Documented (in repo notes) which fundamental fields yfinance actually returns for 5 tickers across sectors — coverage is patchy
- Verified Kite tokens expire daily and you have a clean re-auth path

**Common traps**
- Kite access_token expires at 6 AM IST daily
- yfinance returns empty `DataFrame()` for less-liquid Indian names — don't assume parity with US tickers
- Kite uses `NSE:RELIANCE` (colon), yfinance uses `RELIANCE.NS` (dot suffix)

---

## Phase 1 — Tool layer (still no agent)

**Goal:** the tools are the contract between deterministic data and the non-deterministic LLM. Get the contract right first.

**Deliverables**
- `get_technical_snapshot(ticker, lookback_days)` → dict with recent prices, computed indicators (RSI, MACD + signal + histogram, 50/200 SMA, 10 EMA, ATR, Bollinger), 52-week stats, average volume
- `get_fundamentals_snapshot(ticker)` → dict with latest income statement summary, key ratios (P/E, P/B, ROE, ROCE, D/E), last 4 quarters revenue/EPS trend
- Functions exposed as plain Python first (not yet `@tool`-decorated) so they're testable from a script
- Sample-input tests asserting return-dict shape + types
- Logging at start/end of each call (ticker, latency, error if any)

**Decisions you make in this phase**
- **Indicator library:** `pandas-ta` (comprehensive, heavier) vs. `stockstats` (lighter, less maintained). Pick one, document why.
- **Output shape:** flat vs. nested dict. Aim for compact and well-typed — the LLM will read this verbatim.
- **What to truncate:** sending 252 daily rows to the LLM wastes tokens. Decide on a summary form (e.g., last 20 days verbatim + statistical summary of the rest).
- **Horizon-aware tool?** Should `get_technical_snapshot` accept the horizon and pre-filter? Or always return the same shape?

**Success criteria**
- Both tools return predictable, documented shapes
- A 5-line script can call them for any NSE ticker and print sensible output
- You can articulate what each field is and why it's included

**Reference:** TradingAgents `agents/utils/agent_utils.py` and `agents/analysts/market_analyst.py` — note how the *prompt* explains what indicators mean, while the tool just returns numbers.

---

## Phase 2 — Bare ReAct agent

**Goal:** prove the LangGraph plumbing end-to-end with a minimal prompt. Output will be bad. That's expected. You're testing wiring.

**Deliverables**
- LangGraph setup mirroring zwigato: `StateGraph(AgentState)`, one `chat_node`, one `tool_node`, `tools_condition`, `MemorySaver` checkpointer
- Phase 1 functions wrapped with `@tool` and bound via `llm.bind_tools(...)`
- Minimal system prompt: just *"You are a trading analyst. Decide which tools to call and return a recommendation."* Nothing about format, rating scale, or sections yet.
- CLI loop like zwigato: input → invoke graph → print last message

**Success criteria**
- `python lg_agent.py`, type the question, agent calls at least one tool, returns text
- Tool errors don't crash the graph — they appear in the output
- Output is vague and uncommitted — **resist the urge to fix this here**, that's Phase 3

**Reference:** zwigato's existing `lg_agent.py` is the structural twin.

---

## Phase 3 — Prompt discipline

**Goal:** turn a generic ReAct loop into an analyst. Most of v1's quality work lives here.

**The prompt should cover** (you write it — don't paste a template)
- **Role.** One line.
- **Intent extraction.** Pull ticker (canonical NSE form), horizon (default if unspecified), specific concerns. What to do if ambiguous: ask, default, or refuse.
- **Evidence checklist.** Mirror TradingAgents' analyst dimensions for a positional call: trend (SMA cross), momentum (RSI, MACD), volatility (ATR, Bollinger), valuation (P/E vs. sector, earnings trajectory), profitability + debt. Make this explicit so the agent doesn't skip dimensions.
- **5-tier rating, defined precisely.** What does each rating mean *at a 3-month horizon*? Borrow TradingAgents' phrasing — reserve `Hold` for genuinely balanced evidence; otherwise force a side.
- **Output sections.** Rating up top, executive summary (2-4 sentences), technical analysis, fundamentals, key risks, then `FINAL RECOMMENDATION: **BUY/HOLD/SELL**` for greppability (TradingAgents' stop-signal convention).
- **Discipline rules.** Cite specific numbers from tool output, don't fabricate, no hedging fluff.

**Iteration plan**
Not one-shot. Plan for 5-15 cycles: run on a sample → critique → tighten prompt → re-run. Keep a `Prompt-Iteration-Log.md` documenting each change + why.

**Success criteria**
- For 3 NSE tickers across sectors (e.g. RELIANCE, TCS, HDFCBANK), reports:
  - Cite specific numbers from tool calls
  - Commit to a rating with reasoning
  - Name at least one risk
  - Have consistent structure across runs
- You can articulate remaining failure modes and what would fix each

**Reference:** all four prompts in `tradingagents/agents/analysts/*.py`, plus the docstring of `tradingagents/agents/schemas.py` for rating-scale phrasing.

---

## Phase 4 — Debuggability

**Goal:** you'll have run the agent dozens of times by now. You need to re-read past runs without re-running.

**Deliverables**
- Each run saves report to `reports/<TICKER>/<YYYY-MM-DD>.md`
- Each run also saves `reports/<TICKER>/<YYYY-MM-DD>.trace.json` with: input question, tool calls (name + args + truncated response), tokens used, latency
- Python `logging` module at boundaries (agent start/end, tool start/end), DEBUG for full payloads, configurable via env var
- Friendly error surfacing: expired Kite token → clear "run kite_login.py" message, not a stack trace

**Success criteria**
- You can hand someone the repo + a sample report and they can find the trace and understand what happened
- Re-running the same input on different days lets you diff outputs cleanly
- Common failures (token expired, ticker not found, network blip) produce one-line readable errors

---

## Phase 5 — GitHub readiness

**Goal:** presentable, usable by someone who isn't you.

**Deliverables**
- `README.md`: what + why, sample input/output, setup steps, **explicit "not financial advice" disclaimer**
- `.env.example` with every required variable + comments on where to get each
- `requirements.txt` with pinned versions
- `samples/` folder with 2-3 committed sample reports
- Pre-commit hook for secret scanning (`detect-secrets` or `gitleaks`) — set up *before* first push
- License (MIT or Apache-2.0)

**Success criteria**
- Friend can clone, follow README, get a working report in <30 minutes
- No secrets in git history
- First impression: "thoughtful learning project"

---

## Risk log

| Risk | Cost | Mitigation |
|---|---|---|
| Kite auth expires daily, you forget the flow | Half a day, repeated | Clean `kite_login.py` in Phase 0; clear error message in Phase 4 |
| `gpt-oss-120b` produces shallow theses | 1-3 days of prompt iteration | Fallback: switch to Claude Sonnet via Anthropic if Phase 3 plateaus after 10 iterations |
| yfinance fundamentals sparse for test ticker | Half a day | Test on 5 tickers in Phase 0, document coverage |
| Rabbit-hole on indicator math | 2-5 days | Use `pandas-ta`, don't reimplement indicators |
| Scope creep into news/debate/memory | Project never finishes | Open a `v2-ideas.md`, dump every "wouldn't it be cool" idea there |
| Secrets accidentally committed | Hours to clean | Pre-commit hook in Phase 0, not Phase 5 |

---

## Working principles

- One concept per phase. Don't mix learning LangGraph state with learning prompt design.
- Run after every meaningful change. Untested 50-line prompt edits cost hours later.
- Document decisions as you make them (this file + `Prompt-Iteration-Log.md` per phase).
- "Not financial advice" — in the README, in the CLI banner, and in your own head. Paper-trade for months before any real capital.
