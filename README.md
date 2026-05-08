# trading-agent-nse

A LangGraph-based **structured-thinking tool** for analysing NSE-listed Indian stocks. Pulls technicals, fundamentals, sector benchmarks, and your own context (PDFs, URLs, free-text), and returns a markdown report you can read in 2 minutes.

> **NOT a recommender. NOT financial advice.**
> The bot organises evidence so *you* can decide. Backtesting (see below) showed that the technical-only rating is not reliably predictive on a 3-month horizon — use this to **think**, not to **trade automatically**.

---

## What it does

Given a question like *"Should I buy RELIANCE for a 3-month hold?"*, the agent:

1. Pulls 12 months of OHLCV from **Zerodha Kite Connect**
2. Computes technical indicators (RSI-14, MACD, SMA-50/200, EMA-10, Bollinger, ATR-14, OBV, VWMA-20)
3. Pulls fundamentals from **yfinance** (P/E, P/B, ROE/ROCE/ROA, D/E from balance sheet, last-4-quarter revenue + EPS)
4. **Benchmarks against three indices**: NIFTY 50 (broad), the stock's sector index, and its size-bucket index (NIFTY 100 / MIDCAP 150 / SMLCAP 250 / MICROCAP 250)
5. Auto-loads any persistent macro events from `macro_events.md` (filtered to last 365 days)
6. Optionally loads PDFs / URLs / free-text context you supply at runtime
7. Returns a structured markdown report saved to `reports/<TICKER>/<DATE>.md` plus a trace JSON with tool calls, tokens, and latency

---

## Setup

```bash
git clone https://github.com/frenzy2106/trading-agent-nse.git
cd trading-agent-nse
pip install -r requirements.txt
cp .env.example .env  # then edit with your real keys
```

You'll need:
- **Kite Connect** — API key + secret from [developers.kite.trade](https://developers.kite.trade). When asked for a redirect URL, use `http://127.0.0.1`.
- **Groq** — free API key from [console.groq.com](https://console.groq.com).

Daily token refresh (Kite tokens expire at 6 AM IST):
```bash
python kite_login.py
```

Open the printed URL, log in, paste the redirect URL back. Token cached to `.kite_session.json`.

---

## Usage

```bash
# Simple one-shot
python lg_agent.py "Should I buy RELIANCE for a 3-month hold?"

# With user-supplied context
python lg_agent.py "Should I buy TCS for a 3-month hold?" \
    --context "Concerned about US tariff policy on H1B"

# With PDFs and URLs as sources
python lg_agent.py "Should I buy ETERNAL?" \
    --pdf research/zomato_q3.pdf \
    --url https://www.reuters.com/article/...

# Multiple of each
python lg_agent.py "..." --pdf a.pdf --pdf b.pdf --url x.com --url y.com

# Skip the persistent macro events log for this run
python lg_agent.py "..." --no-macro

# Interactive mode (no args)
python lg_agent.py
```

Each run produces:
- `reports/<TICKER>/<DATE>.md` — the markdown report
- `reports/<TICKER>/<DATE>.trace.json` — tool calls (with truncated responses), token usage, latency, model

---

## Persistent macro events

Edit `macro_events.md` to record events that should inform every future analysis (wars, elections, regulatory changes, model launches that affect markets, etc.). The file is auto-loaded into every run, filtered to the last 365 days. Format:

```markdown
## 2025-09-22 — GST 2.0: rate rationalisation (5% / 18%)
Affects: AUTO, FMCG, CONSUMER-DURABLES, PHARMA
Tags: tax-policy, demand
GST Council collapsed slabs from 5/12/18/28% to 5%/18% (40% retained for sin/luxury).
Demand-side tailwind through FY26 for autos, consumer durables, FMCG, pharma...
```

To remove an event, delete its block. To skip the log for a single run, pass `--no-macro`.

---

## Architecture

```
user input + optional context (PDFs / URLs / text / macro events log)
    │
    ▼
LangGraph ReAct agent (Groq, default openai/gpt-oss-120b)
    │
    ├── get_technical_snapshot(ticker)    ← Kite OHLCV → pandas-ta indicators
    ├── get_macro_snapshot(ticker)        ← NIFTY 50 + sector index + size-bucket index
    └── get_fundamentals_snapshot(ticker) ← yfinance ratios + 4-quarter trend
    │
    ▼
Structured markdown report + trace JSON
```

OHLCV data is cached daily in `cache/` for cross-tool and cross-run reuse — running 10 stocks of the same sector hits Kite ~5 times instead of 30+.

User-supplied content (PDFs, URLs, free-text, macro events) is wrapped in `<source>`, `<user_context>`, and `<macro_events>` tags. The system prompt explicitly tells the LLM to treat that content as **reference material, not instructions**, mitigating prompt-injection from external content.

---

## Backtesting

A small backtest harness ships in `backtest.py`:

```bash
python backtest.py                # NIFTY 10 × 4 quarterly dates = 40 calls, ~10 min
python backtest_analyze.py latest # hit rates, confidence calibration, baseline comparison
```

Backtest is **technical-only** because yfinance fundamentals are not point-in-time (using current ratios for past dates would introduce look-ahead bias). Output is saved to `backtest/<run-id>/`.

---

## Honest limitations

- **Sample size**: 40-call backtest is a sanity check, not statistical proof.
- **Hit rates**: 50–62% — barely better than coin flip in the sample window.
- **Confidence label**: was *anti-signal* in the technical-only backtest (HIGH-confidence calls were on average more wrong than MEDIUM). Now hidden from the default output by design.
- **LLM stochasticity**: same data may produce different ratings on different runs, even at temperature=0.
- **No automated news**: relies on you supplying URLs, PDFs, or macro events.
- **Sector mapping is sector-level by default**, with industry-level distinction only for Financial Services (banks vs brokers vs NBFCs vs insurance). Other yfinance sectors map 1:1 to a NIFTY index, sometimes loosely (Industrials → NIFTY AUTO, etc.). When mapping is loose, the agent pivots to the size-bucket comparison instead.
- **Survivorship bias** in the backtest universe (we tested currently-listed names).
- **Indian-market only**: Kite Connect is NSE/BSE; this tool will not work for US/EU equities without rewriting the data layer.

---

## Project status

Current state, planned work, prompt iteration history, and backtest analysis are tracked in:
- [`Project-Status.md`](./Project-Status.md) — what's done, what's next, known issues
- [`Prompt-Iteration-Log.md`](./Prompt-Iteration-Log.md) — every prompt revision with verdicts (v1 through v9)
- [`V1-Implementation-Plan.md`](./V1-Implementation-Plan.md) — the original plan we built against

---

## Repository layout

```
.
├── lg_agent.py               # main agent (LangGraph + Groq + ToolNode)
├── prompts.py                # system prompt (Path A reframe, structured-thinking framing)
├── persistence.py            # save reports + trace JSON
├── context_loader.py         # PDF / URL / macro events loaders, with prompt-injection guards
├── cache.py                  # daily-fresh disk cache (Kite instruments + OHLCV)
├── kite_login.py             # daily Kite token refresh
├── ticker_utils.py           # NSE: / .NS / plain symbol normalisation
├── backtest.py               # technical-only backtest harness
├── backtest_analyze.py       # hit rates, confidence calibration, summary
├── backtest_prompts.py       # technical-only system prompt for backtest
├── tools/
│   ├── technical.py          # OHLCV + indicators (RSI, MACD, SMA, OBV, VWMA, ATR, Bollinger)
│   ├── fundamentals.py       # yfinance valuation + profitability + leverage + earnings trend
│   ├── macro.py              # NIFTY 50, sector index, size-bucket index benchmarking
│   └── tool_definitions.py   # @tool wrappers with structured error responses
├── macro_events.md           # persistent events log (you edit this directly)
├── requirements.txt
└── .env.example
```

---

## License

[MIT](./LICENSE) — including the explicit clause that the software is for educational use only and the authors are not liable for any trading losses.
