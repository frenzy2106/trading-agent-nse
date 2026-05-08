# Prompt Iteration Log

Each entry records: model, prompt version, tickers tested, what worked, what failed, and what changed.

---

## v1 — Phase 2 minimal prompt | model: llama-3.3-70b-versatile

**Prompt (summary):** "You are a trading analyst specialising in NSE-listed Indian stocks. Use the available tools to gather technical and fundamental data, then return a concise investment recommendation."

**Tickers tested:** RELIANCE

**Output (RELIANCE):**
> Hold. RSI 60.82 indicating slightly overbought. MACD positive, bullish trend. P/E 24.17. ROE 9.14%. D/E 36.65.

**What worked:**
- Agent called both tools automatically
- Produced some numbers from tool output
- Returned a rating

**What failed:**
- Vague language ("slightly overbought") without citing thresholds
- No structured sections
- No evidence checklist — skipped MACD histogram, Bollinger, 52w range
- Hold used as a hedge, not because evidence was balanced
- No summary table, no FINAL RECOMMENDATION signal

---

## v2 — Phase 3 long prompt | model: llama-3.3-70b-versatile

**Prompt (summary):** Full multi-section prompt with detailed indicator explanations (~800 tokens), rating scale definitions, output format, discipline rules.

**Tickers tested:** RELIANCE (success), TCS (fail)

**Output (RELIANCE):** Overweight — structured report with sections, specific numbers cited, summary table, FINAL RECOMMENDATION signal. Good quality.

**Output (TCS):** `groq.BadRequestError 400 — tool_use_failed`. Model generated tool call in old Llama XML format (`<function=get_technical_snapshot{...}>`) instead of JSON.

**Root cause:** System prompt too long (~800 tokens). llama-3.3-70b-versatile falls back to its native function-calling format when prompt is too large.

**Fix:** Shorten prompt significantly, keep indicator explanations as concise bullet points.

---

## v3 — Phase 3 shortened prompt | model: llama-3.3-70b-versatile

**Prompt changes from v2:** Replaced multi-paragraph indicator explanations with single-line bullet points. Removed markdown code fence around output format. ~400 tokens total.

**Tickers tested:** RELIANCE, TCS, HDFCBANK

**Output (RELIANCE):** Overweight
- Correctly above SMA-50 and SMA-200
- MACD histogram cited (+12)
- Revenue shown as raw rupees (unreadable: 2940590000000.0)
- ROE shown as raw decimal (9.14% shown correctly here, but TCS was wrong)

**Output (TCS):** Hold
- ROE shown as 0.484 (should be 48.4%) — factor-of-100 error
- Revenue shown as 706980000000.0 (unreadable)
- D/E 10.39 flagged as "high" — misleading for TCS (virtually debt-free; yfinance includes deferred revenue)
- Rating Hold was correct given bearish technicals vs strong fundamentals

**Output (HDFCBANK):** Overweight
- Bearish technicals (22% off 52w high, below both SMAs) but rated Overweight
- D/E shown as N/A (correct — not meaningful for banks)
- Rating inconsistent with stated technical evidence

**Fixes applied:**
- Added `_pct()` helper: converts yfinance decimal ratios to % (0.484 → 48.4)
- Added `_crore()` helper: converts raw INR to Crores (divide by 1e7)
- Revenue field renamed to `revenue_cr`
- Profitability fields renamed to `roe_pct`, `roa_pct`, `roce_pct`
- Prompt updated: added note that ROE/ROCE are already in %
- Prompt updated: stricter rating rule — bearish technicals require explicit fundamental justification to rate above Hold

---

## v4 — After tool fixes | model: llama-3.3-70b-versatile

**Prompt changes from v3:** Added field name hints (`roe_pct`, `revenue_cr`). Added explicit rule: "If technicals are bearish (price below SMA-50 AND SMA-200, negative MACD), rating should be Hold or lower unless fundamentals provide very strong counter-argument."

**Tickers tested:** RELIANCE, TCS, HDFCBANK

**Output (RELIANCE):** Overweight ✓
- Price 1443 above SMA-50 (1383) and SMA-200 (1433) — correctly bullish
- RSI 60.6, MACD histogram +12.26 — cited correctly
- Revenue 294059 Cr — readable ✓
- EPS 12.54 / 13.78 / 19.95 / 14.34 — cited correctly ✓
- ROE 9.14% — correct ✓
- D/E 36.65 flagged with context note

**Output (TCS):** Hold ✓
- ROE now 48.4% — correct ✓ (was 0.484 in v3)
- ROCE 54.93% — correct ✓
- Price below SMA-50 and SMA-200 — correctly noted as downtrend
- MACD negative — correctly noted as bearish
- Hold rating is appropriate given mixed signals ✓
- D/E 10.39 still cited as "high" — data artifact, not real debt (open issue)
- SMA-200 showing null — needs investigation

**Output (HDFCBANK):** BUY ✗
- Price 794 is 22% below 52w high, below both SMAs — bearish technicals
- Despite this, model rated BUY — rating discipline rule not strong enough
- ROE 13.82% cited correctly
- D/E N/A — handled gracefully ✓
- Revenue in Crores (85346 Cr) — readable ✓

**Open issues going into v5:**
1. HDFCBANK over-rated despite bearish technicals — prompt discipline needs tightening
2. TCS D/E 10.39 is misleading (yfinance deferred-revenue artifact for IT firms)
3. SMA-200 null for TCS — investigate if data issue or model hallucination

---

## v5 — Model switch test | model: openai/gpt-oss-120b

**Same prompt as v4. Only change: model.**

**Output (RELIANCE):** BUY
- Price above SMA-50 (1383.91) and SMA-200 (1433.56) — correctly noted
- MACD histogram +12.26 cited — correct
- Revenue in Crores, EPS in INR — correct
- ROE 9.14%, ROCE 9.18% — correct
- D/E 36.65 with Jio/retail context noted — correct
- Bumped RELIANCE from OVERWEIGHT (v4/llama) to BUY — debatable given ROE below benchmark

**Output (TCS):** BUY
- Price below SMA-50 (2500.87) AND SMA-200 (2947.38) — correctly noted as downtrend
- MACD -27.93 vs signal -20.91, histogram -7.03 — all three cited correctly
- RSI 40.33 — neutral-to-weak, correct
- ROE 48.4%, ROCE 54.93% — correct
- Revenue per quarter in Crores with period labels (Q4 FY2026 etc.) — excellent formatting
- EPS trend cited across 4 quarters — correct
- Rated BUY overriding bearish technicals because of cheap valuation + strong profitability
- Explicitly acknowledged the override: "Key risk is the current bearish technical backdrop"
- This is a defensible call; llama gave Hold (also defensible). Different reasonable views.

**Output (HDFCBANK):** UNDERWEIGHT ✓ (was BUY in v4/llama — this is the key fix)
- Price below SMA-50 and SMA-200 — correctly treated as downtrend
- Noted MACD signal is mixed (MACD > signal = slight bull, but both negative)
- ROE 13.82% correctly flagged as below 15% benchmark
- Revenue decline noted: 95305 Cr → 85346 Cr (-10% sequentially)
- D/E handled correctly: noted bank-specific (no D/E; capital ratios apply instead)
- Earnings growth YoY +7.5% vs revenue growth -1.8% — nuanced read
- Risks specific to HDFCBANK: NPA quality, NIM compression, capital norms

**Verdict vs llama-3.3-70b-versatile:**
| | RELIANCE | TCS | HDFCBANK |
|---|---|---|---|
| llama-3.3-70b | Overweight | Hold | BUY (wrong) |
| gpt-oss-120b | BUY | BUY | Underweight ✓ |

- gpt-oss-120b fixes the HDFCBANK rating error — biggest win
- gpt-oss-120b produces richer prose, better quarter labelling, more explicit risk specificity
- Both models are defensible on TCS (bearish tech vs strong fundamentals — genuine split)
- gpt-oss-120b tendency: slightly more bullish; may need a "burden of proof for BUY" rule in future iterations

**Open issues:**
1. TCS D/E 10.39 still cited as "high" — data artifact (yfinance deferred revenue), not real debt
2. No SMA-200 for TCS in v4 (llama); gpt-oss-120b correctly computed it at 2947.38 — confirms v4 issue was model hallucination, not data

---

## v6 — D/E recomputed from balance sheet | model: openai/gpt-oss-120b

**Tool change:** `tools/fundamentals.py` no longer uses `info.debtToEquity`. New `_compute_de_from_bs()` reads `Total Debt` (or `Long Term Debt + Current Debt`) divided by `Stockholders Equity` directly from the balance sheet. Matches what investors actually mean by D/E for non-financial firms.

**Prompt change:** Removed the "Reliance's D/E reflects Jio capex" hint (no longer accurate — D/E now reads as 0.44 cleanly). Added: ">1.0 = leveraged; >2.0 = high. For banks, D/E is not meaningful — comment briefly and move on."

**Phase 4 added:** Each run now writes `reports/<TICKER>/<DATE>.md` and `<DATE>.trace.json`. Trace contains tool calls (name + args + truncated response), token usage, latency, model.

**D/E values across the three test tickers:**
| Ticker | v5 (info.debtToEquity) | v6 (balance sheet) |
|---|---|---|
| RELIANCE | 36.65 | **0.44** |
| TCS | 10.39 | **0.11** |
| HDFCBANK | N/A | **0.95** |

All three are now sensible. RELIANCE's actual D/E reflects modest leverage; TCS is virtually debt-free as expected; HDFCBANK gets a real number now (though caveat noted that bank D/E should not drive ratings).

**Output (RELIANCE):** Overweight
- D/E 0.44 → "low, comfortable balance-sheet" ✓
- Downgraded from BUY (v5) → Overweight on EPS contraction concern (₹19.95 → ₹12.54). Defensible.

**Output (TCS):** Overweight ✓
- D/E 0.11 → "very low financial risk" ✓
- Bumped from Hold (v4 llama) / BUY (v5) → Overweight: clean acknowledgment of bearish technicals + strong fundamentals override
- Includes explicit rationale paragraph after FINAL RECOMMENDATION
- Best-quality TCS report yet

**Output (HDFCBANK):** Underweight ✓
- D/E 0.95 → "well under 1.0, modest leverage" with bank caveat noted
- Still correctly Underweight despite the (now real) D/E value — bearish technicals + sub-15% ROE drive the rating

**Persistence verified:**
- 3/3 runs created `reports/<TICKER>/2026-05-07.md` + `.trace.json`
- Trace metadata: 6,900-7,300 tokens, ~8-12s latency per run
- Tool args + truncated responses captured for replay

**Open issues going into v7:**
1. None data-side. Output quality is now acceptable for v1.
2. Model occasionally still bumps rating to BUY/Overweight when technicals are clearly bearish + fundamentals strong — this is judgment-call territory; defensible either way.
3. The "Rationale" paragraph TCS produced after FINAL RECOMMENDATION is nice but not in the prompt format. Could either codify or ignore.

---

## v7 — Volume dimension added | model: openai/gpt-oss-120b

**Tool change:** `tools/technical.py` now computes:
- `VWMA_20` (20-day volume-weighted moving average)
- `SMA_20` (20-day simple moving average — reference for VWMA)
- `OBV` (on-balance volume)
- New `_build_volume_block()` derives: `vwma_20`, `sma_20`, `vs_avg_ratio` (today/20d avg), `obv_change_20d_pct` (OBV trend over 20d as %)
- Existing `avg_20d` and `avg_full` retained

**Prompt change:** Added Volume to indicator guide (3 bullets), to the Technical Analysis output template, and to the Summary Table template.

**Output (RELIANCE):** Overweight (unchanged rating)
- Volume bullet cited: vs_avg 0.9, VWMA 1381.08 > SMA-20 1377.82 (buying pressure), OBV +62.1% (strong accumulation)
- Volume row in summary table populated correctly
- Executive summary now references "OBV shows strong accumulation" as an OW driver

**Output (LUPIN):** Overweight (CHANGED from v6 Hold) ⭐
- Volume bullet: vs_avg 1.14, VWMA 2340.7 > SMA-20 2332.83 (buying pressure), OBV +12.5% (accumulation)
- Same technical overbought signals as last run (RSI 70.27, above upper Bollinger)
- BUT: volume confirms the rally is real, not a thin-volume fakeout
- Reasoning explicitly weighs the new evidence: "upside from earnings + accumulation outweighs technical pull-back risk"
- This is the most material rating change yet — and exactly the kind of nuance volume analysis was supposed to add

**Verdict:** Volume earns its place in the prompt. It rebalanced the LUPIN read from "trim the top" to "stay long the trend".

**Token cost:** ~8.6-8.9k tokens (up ~700 from v6 ~7.9k). Acceptable given the quality lift.

---

## v8 — Path A reframe + persistent context | model: openai/gpt-oss-120b

**Trigger:** Backtest (run 20260507-161601, 40 calls) showed:
- HIGH-confidence calls were anti-signal (HIGH-conf bear: -4.87% direction-adjusted at 90d vs MEDIUM bear: +3.09%)
- SELL bucket forward returns +2.71% at 90d (sells went up on average)
- Hit rates 50-62%, barely better than coin flip
- One catastrophic miss: BAJFINANCE 2025-08-07 UNDERWEIGHT/HIGH → +18.78% over 90d

User aligned on Path A: structured-thinking tool, not direct recommender. Also identified the missing-news problem (technicals + fundamentals don't see Iran war, election results, AI launches, etc.).

**Scope of v8:**
1. **Reframed prompt**: agent organises evidence; rating moves to a labelled "Data-Only Conclusion" section at the end.
2. **Persistent macro events log**: `macro_events.md` (markdown, hand-edited) auto-loaded into every run, filtered to last 365 days.
3. **Per-run sources**: `--pdf`, `--url`, `--context` CLI flags. PDFs via `pypdf`, URLs via `trafilatura`. Capped at 10K chars per source.
4. **Prompt-injection defense**: user-supplied content wrapped in `<context>`, `<source label="...">`, `<user_context>` tags. System prompt explicitly tells the LLM to treat that content as reference, not instructions.
5. **Confidence label dropped from display** (kept in trace for analysis). Backtest showed it was anti-signal; surfacing it would mislead.
6. **`persistence.py` updated** to extract rating from the new "Rating (data only): X" pattern in the body. Legacy header pattern kept as fallback.

**Verified output (RELIANCE with `--context "Iran-Israel tensions, oil 80-90 range, refining margin pressure concern"`):**
- Header reads `## RELIANCE | 2026-05-07 | Horizon: 3 months` (no rating)
- "Macro Context" surfaced Claude Opus 4.7 (within 365d); 2024 Iran/election events correctly filtered out
- "User-Supplied Context" cross-referenced the user's margin worry with actual declining-EPS trend in the data — the synthesis Path A is supposed to deliver
- "Considerations Beyond the Data" framed unknowns as questions, not predictions
- "Data-Only Conclusion: HOLD" with explicit hand-off: "weigh it against the macro events, your contextual concerns, and any factors not surfaced above before acting"

**Token cost:** 9.9k (up ~1k from v7 — the context block + macro events). Acceptable.

**What this enables:**
- User can attach research PDFs (broker reports, sectoral notes) and analyst URLs and have them factored into the synthesis
- Persistent events log means major macro shifts only need to be entered once
- Confidence is no longer mis-leadingly surfaced
- The bot's role is now honest: organise evidence, you make the call

**What's still missing (logged for v9+):**
- No macro/sectoral index data (NIFTY, BANKNIFTY, sector indices) — would catch IT mean-reversion misses
- No automated news fetch — relies on user-supplied URLs
- LLM still occasionally confused by long contexts; haven't yet stress-tested with 3+ sources at once

---

## v9 — Sector & broad-market index benchmarking | model: openai/gpt-oss-120b

**New tool:** `get_macro_snapshot(ticker)` (in `tools/macro.py`).

**What it does:**
- Looks up the stock's yfinance sector and maps it to a NIFTY sector index (Energy → NIFTY ENERGY, Technology → NIFTY IT, Financial Services → NIFTY BANK, Healthcare → NIFTY PHARMA, etc.)
- Fetches close-price series for the stock + NIFTY 50 + sector index from Kite (~400 days back)
- Computes returns at 1m / 3m / 6m / 12m for all three
- Computes relative-performance numbers (stock − broad, stock − sector) at each window
- Surfaces sector-mapping caveats in a `notes` field

**Prompt changes:**
- Added Step 4 in workflow: call `get_macro_snapshot` between the technical and fundamentals calls
- Added "Macro / index interpretation" rules to the indicator guide
- Renamed "Macro Context" output section to **"Market & Sector Context"** — now always present, covering both index benchmarking + macro events log
- Added a **Relative Strength** row to the summary table

**Verified output (RELIANCE, no user context):**
- All 3 tools fired (tools=3 in trace)
- Market & Sector Context now reads: *"+4.31% vs NIFTY 50 over 3m... -13.88% vs NIFTY ENERGY over 3m. The stock is modestly outperforming the broad market while lagging sharply behind its own sector, which has been very strong"*
- Key number caught: NIFTY ENERGY +23.41% over 12m vs RELIANCE +4.28% — a **−19.13% relative underperformance** that pure technicals can't see
- Rating shifted from OVERWEIGHT (v7-8 with same data) → **HOLD** specifically citing the sector laggard story
- Death-cross caught explicitly (SMA-50 < SMA-200 even though price > both)

**Cost:**
- Latency: 27.1s (was ~10s) — third tool call + extra Kite OHLCV pull
- Tokens: 15.4k (was ~10k) — macro snapshot adds ~5k tokens

**Sector mapping coverage:**
- Energy → NIFTY ENERGY ✓
- Technology → NIFTY IT ✓
- Financial Services → NIFTY BANK ✓
- Healthcare → NIFTY PHARMA ✓
- Consumer Defensive → NIFTY FMCG ✓
- Basic Materials → NIFTY METAL ✓
- Real Estate → NIFTY REALTY ✓
- Communication Services → NIFTY MEDIA ✓
- Consumer Cyclical / Industrials → NIFTY AUTO (imprecise; flagged in notes) ⚠
- Utilities / unknown → NIFTY 50 only (no sector benchmark)

**What's still missing (logged for v10+):**
- No automated news fetch — still relies on user-supplied URLs
- Sector mapping is sector-level only; some stocks (NBFCs vs banks) deserve finer granularity (NIFTY FINANCIAL SERVICES vs NIFTY BANK)
- No re-run of the backtest with the macro tool included — should be done before next major change to confirm the sector context actually improves hit rates
