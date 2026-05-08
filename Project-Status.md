# Project Status — NSE Trading Agent

*Snapshot date: 2026-05-07*

A single-agent recommendation system for NSE-listed stocks at a positional (weeks-to-months) horizon. Recommendation-only; never executes trades.

---

## Phase progress

| Phase | Status | Notes |
|---|---|---|
| 0 — Data foundations | ✅ Done | Kite auth + yfinance both verified live |
| 1 — Tool layer | ✅ Done | Both snapshot tools return well-typed dicts |
| 2 — Bare ReAct agent | ✅ Done | LangGraph wiring confirmed end-to-end |
| 3 — Prompt discipline | ✅ Done | v6 prompt + `openai/gpt-oss-120b` produces clean reports across 3 sectors |
| 4 — Debuggability | ✅ Done | Reports + traces persist; graceful errors with "did you mean" hints |
| 5 — GitHub readiness | ⏳ Not started | No README, no `.env.example`, no samples, no license |

---

## File inventory

### Auth + data
- **`kite_login.py`** — Daily Kite token refresh. Login URL → request_token → access_token cached to `.kite_session.json`.
- **`ticker_utils.py`** — Plain ↔ NSE: ↔ .NS ticker normalization.

### Tool layer (`tools/`)
- **`technical.py`** — `get_technical_snapshot(ticker, lookback_days=365)` returns RSI-14, MACD(12/26/9), SMA-50/200, EMA-10, ATR-14, Bollinger-20, 52-week stats, last 10 days OHLCV.
- **`fundamentals.py`** — `get_fundamentals_snapshot(ticker)` returns P/E, P/B, ROE/ROCE/ROA (as %), D/E, market cap (Cr), revenue (Cr) and EPS for last 4 quarters.
- **`tool_definitions.py`** — LangChain `@tool` wrappers around the plain Python functions; serialise dicts to JSON for the LLM.

### Agent
- **`lg_agent.py`** — LangGraph ReAct loop: `chat_node` ↔ `ToolNode` with `tools_condition`, `MemorySaver` checkpointer, CLI input loop, UTF-8 stdout reconfig for Windows.
- **`prompts.py`** — System prompt v4 (current production). Concise indicator guide + 5-tier rating scale + structured output format + discipline rules.

### Smoke tests
- **`test_kite_data.py`** — Pulls 12 months OHLCV for any NSE ticker; standalone diagnostic.
- **`test_yfinance_data.py`** — Cross-sector coverage check (`--all5`).
- **`test_tools.py`** — Shape + type assertions on both Phase 1 tools, then prints sample output.

### Documentation
- **`V1-Implementation-Plan.md`** — The original plan we're executing against.
- **`Prompt-Iteration-Log.md`** — v1 → v5 prompt history with verdicts.

---

## What works well now

1. **End-to-end pipeline runs cleanly.** Ask "Should I buy RELIANCE for a 3-month hold?" → agent calls both tools → returns structured markdown report with rating + summary table + FINAL RECOMMENDATION line.
2. **Output discipline is strong** with `openai/gpt-oss-120b`. Numbers cited from tool output, sections consistently structured, sector-appropriate framing (e.g. notes that bank D/E isn't meaningful, energy P/E norms differ from IT).
3. **Tool data is clean.** ROE/ROCE in proper percentages, revenue in Crores, EPS in rupees, Bollinger bands populated.
4. **Rating discipline holds across 3 sectors.** RELIANCE → BUY (bullish tech + decent fundamentals), TCS → BUY (cheap valuation overrides bearish tech, justified explicitly), HDFCBANK → Underweight (correctly downgraded from llama's incorrect BUY).

---

## Known issues — sorted by impact

### High impact

**1. TCS-style D/E inflation (yfinance artifact)**
- Issue: yfinance's `debtToEquity` for IT firms includes deferred revenue and operating liabilities, not just interest-bearing debt. TCS shows D/E of 10.39 — in reality TCS is virtually debt-free.
- Symptom: Reports flag "high leverage" for genuinely debt-free firms.
- Fix: In `tools/fundamentals.py`, compute D/E from `balance_sheet` directly: `Total Debt / Total Equity`, ignoring `info.debtToEquity`. Falls back to None gracefully.
- Effort: 1 hour.

**2. No sector-aware valuation context**
- Issue: Prompt has hard-coded sector P/E hints (IT 20-30x, FMCG 40-60x). The model has to memorise these. No live sector median.
- Symptom: Sector calibration is rough. A genuine outlier (e.g. TCS at 17x in IT) gets called "cheap" without comparison data.
- Fix: Either (a) add a `get_sector_benchmarks()` tool that returns sector median P/E from a static table, or (b) leave it for v2 and accept rough calibration.
- Effort: 4 hours for option (a). Defer for v1.

**3. Phase 4 debuggability is missing entirely**
- Issue: No persistence — reports vanish to terminal output. No trace of which tools were called or how long they took.
- Symptom: After 20 iterations, you can't compare runs or diff outputs over time.
- Fix: Add `reports/<TICKER>/<YYYY-MM-DD>.md` writer + `<…>.trace.json` with tool calls, tokens, latency. Wire `logging.INFO` at agent + tool boundaries.
- Effort: 3 hours.

### Medium impact

**4. No graceful failure modes** ✅ FIXED
- Issue: Expired Kite token surfaces as a stack trace. Ticker not found → opaque "instrument not found" error. Network blip → 500 from Groq propagates.
- Fix applied:
  - `tools/technical.py` raises typed `TickerNotFoundError` with up to 3 name-match suggestions from Kite instruments.
  - `tools/tool_definitions.py` wraps both tools with try/except — returns JSON `{"error": ..., "kind": ..., "suggestions": [...]}` so the LLM reads it as a tool response, not a crash.
  - Handles: `TickerNotFoundError`, `kiteconnect.TokenException`, catch-all for everything else.
  - Prompt updated with explicit error-handling guidance.
- Verified: ZOMATO → "I couldn't find ZOMATO on NSE. Did you mean one of: ETERNAL?" (no crash, no persistence).

**5. `requirements.txt` is unpinned**
- Issue: `kiteconnect`, `yfinance`, `pandas-ta` etc. are unpinned. New users get whatever pip resolves today; reproducibility weak.
- Fix: After Phase 5 freeze: `pip freeze | grep -E "kite|yfin|pandas|langgraph|langchain|dotenv" > requirements.txt`.
- Effort: 5 minutes.

**6. No `.env.example` committed**
- Issue: The earlier `.env.example` write didn't land (Windows hidden-file artefact). Friend cloning the repo has no template.
- Fix: Write `.env.example` with `KITE_API_KEY=`, `KITE_API_SECRET=`, `GROQ_API_KEY=`, `GROQ_MODEL=openai/gpt-oss-120b`.
- Effort: 5 minutes.

### Low impact

**7. Memory leak risk in LangGraph state**
- Issue: `MemorySaver` checkpointer keeps full message history in process memory. For a CLI session this is fine; for a long-running daemon it would grow unbounded.
- Fix: Not needed for v1 CLI. Note for v2 if we add scheduled runs.

**8. Random tool-call format errors with `llama-3.3-70b-versatile`**
- Issue: Long system prompt occasionally triggers Groq's `tool_use_failed` 400 (model emits old Llama XML format). Worked around by switching to `openai/gpt-oss-120b`.
- Fix: Already mitigated by model swap. Document in README.

**9. RELIANCE_ohlcv.csv left in working tree**
- Issue: Test artifact from Phase 0 review still in repo root.
- Fix: Delete or move to `.gitignore`'d test directory.

---

## Biggest areas to tackle next — prioritised

### Priority 1 — Fix TCS D/E artifact (#1) + ship Phase 4 debuggability (#3)
These two together unblock the "iterate confidently" loop. Without persisted reports + traces, prompt iteration is guess-and-check; without a clean D/E for IT firms, every TCS-style report has a flagged-but-wrong leverage warning that erodes trust in the output.

Combined effort: ~4 hours.

### Priority 2 — Graceful failures (#4)
Once you start running this daily, the 6 AM IST Kite expiry will burn you. Putting in clean error messages costs little and saves repeated debugging.

Effort: 2 hours.

### Priority 3 — Phase 5 GitHub readiness
Pinned `requirements.txt`, `.env.example`, `README.md` with sample input/output + disclaimer, sample reports under `samples/`, MIT/Apache license, secret-scanning pre-commit hook.

Effort: 3-4 hours total.

### Defer to v2
- Sector benchmarks tool (#2) — accept rough calibration in v1
- News integration
- Memory across runs
- Scheduled runs / web UI

---

## Recommended next session

```
1. Fix tools/fundamentals.py D/E to compute from balance_sheet directly
2. Add reports/ writer in lg_agent.py (markdown + trace.json)
3. Wrap tool errors so the agent sees friendly messages
4. Clean up RELIANCE_ohlcv.csv, write proper .env.example
```

After that, two more iteration runs to confirm the v5 prompt + new D/E hold up, then move to Phase 5 and call v1 done.
