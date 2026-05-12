# trading-agent-nse

A LangGraph-based **structured-thinking tool** for analysing NSE-listed Indian stocks. The agent pulls technicals, fundamentals, sector and size-bucket benchmarks, real-time news, sell-side analyst consensus, commodity input-cost data, and management commentary from indexed concall transcripts — then returns a structured markdown report you can read in 2 minutes.

> **NOT a recommender. NOT financial advice.**
> The bot organises evidence so *you* can decide. Backtesting (see below) showed that the technical-only rating is not reliably predictive on a 3-month horizon — use this to **think**, not to **trade automatically**.

---

## What it does

Given a question like *"Should I buy RELIANCE for a 3-month hold?"*, the agent calls the right combination of these tools, then writes a structured report with bull case, bear case, weight verdict, and a labelled data-only rating:

1. **Technicals** — 12 months of OHLCV from **Zerodha Kite Connect**; computes RSI-14, MACD, SMA-50/200, EMA-10, Bollinger, ATR-14, OBV, VWMA-20
2. **Fundamentals** — yfinance: P/E trailing & forward, P/B, ROE/ROCE/ROA, D/E, last-4-quarter revenue + EPS
3. **Macro** — benchmarks the stock against **three indices**: NIFTY 50 (broad), the stock's sector index, and its size-bucket index (NIFTY 100 / MIDCAP 150 / SMLCAP 250 / MICROCAP 250)
4. **News** — last-30-day headlines from **Google News RSS** (en-IN locale), plus earnings calendar from yfinance
5. **Analyst consensus** — sell-side price targets, recommendation distribution, and rating drift from yfinance
6. **Commodities** — yfinance commodity futures (gold, silver, copper, crude/Brent, natural gas, palladium, etc.) plus ETF proxies for lithium/steel/uranium, for input-cost reasoning on stocks with raw-material exposure
7. **Management commentary (RAG)** — semantic search over locally-indexed earnings call transcripts for qualitative inputs
8. Optionally loads your own **PDFs, URLs, markdown files, and free-text context** supplied at runtime
9. Returns a structured markdown report at `reports/<TICKER>/<DATE>.md` plus a trace JSON with tool calls, tokens, and latency

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
- **An LLM provider** — set `LLM_PROVIDER` in `.env` to one of:
  - `deepseek` (default) — get a key from [platform.deepseek.com](https://platform.deepseek.com). Cheap, supports tool calls, 1M context.
  - `groq` — free tier from [console.groq.com](https://console.groq.com); fast but daily cap.
  - `anthropic` — premium, get from [console.anthropic.com](https://console.anthropic.com).

Daily Kite token refresh (tokens expire at 6 AM IST):
```bash
python kite_login.py
```

Open the printed URL, log in, paste the redirect URL back. Token cached to `.kite_session.json`.

---

## Usage

### Single-stock analysis

```bash
# Simple one-shot
python lg_agent.py "Should I buy RELIANCE for a 3-month hold?"

# With user-supplied free-text context
python lg_agent.py "Should I buy TCS for a 3-month hold?" \
    --context "Concerned about US tariff policy on H1B"

# With PDFs and URLs as sources
python lg_agent.py "Should I buy ETERNAL?" \
    --pdf research/zomato_q3.pdf \
    --url https://www.reuters.com/article/...

# Multiple of each
python lg_agent.py "..." --pdf a.pdf --pdf b.pdf --url x.com --url y.com

# Markdown sources work too (e.g. saved concall transcripts)
python lg_agent.py "Should I buy RELIANCE for a 3-month hold?" \
    --md data/concalls/RELIANCE/q4-earnings-call.md

# Interactive mode (no args)
python lg_agent.py
```

Each run produces:
- `reports/<TICKER>/<DATE>.md` — the markdown report
- `reports/<TICKER>/<DATE>.trace.json` — tool calls (with truncated responses), token usage, latency, model

### Watchlist screening

Two screeners ship for scanning multiple tickers in one shot:

```bash
# Full agent run for every ticker — writes individual reports + summary
python screener.py
python screener.py --watchlist watchlist_nifty100.txt
python screener.py --limit 3                         # smoke-test first 3
python screener.py --rating BUY,OVERWEIGHT           # filter summary view

# Lite mode — technicals + fundamentals only, no concall/news/analyst tools
python screener_lite.py
```

Watchlists provided: `watchlist.txt` (default), `watchlist_top15.txt`, `watchlist_nifty100.txt`. Summary lands at `screener/<DATE>-<run-id>.md`.

---

## Management commentary (RAG)

For qualitative dimensions — margin guidance, capex priorities, segment outlook, capital allocation, raw material costs — the agent has an optional `get_management_commentary` tool backed by a local Chroma index over earnings call transcripts.

### Populate the corpus

Drop transcripts here:

```
data/concalls/<TICKER>/<filename>.{md,pdf}
```

Sources I've used: company IR pages (most large-caps publish concall transcript PDFs), BSE filings (search "Analyst Meet / Concall Transcript"), or your own markdown.

### Build the index

```bash
python -m ingest.build_concall_index             # all tickers in data/concalls/
python -m ingest.build_concall_index RELIANCE    # one ticker
```

The script chunks each transcript (1000 chars / 150 overlap), embeds with `sentence-transformers/all-MiniLM-L6-v2` (~80MB downloaded on first run, CPU-only), and upserts to Chroma at `data/vectorstore/`. Idempotent — re-running replaces existing chunks per file.

The `data/` directory is gitignored. Corpus and vectorstore stay local.

### How the agent uses it

When a question warrants qualitative input, the LLM issues **at least three narrowly-focused queries** — one upside-leaning, one downside-leaning, and one targeting whatever specific issue the data raised — and gets the top-5 semantically matching chunks per query (cross-encoder reranked from 20 candidates), filtered to the last 365 days. Quotes must cite source verbatim. If no commentary is indexed for the ticker, the tool returns `no_commentary` and the agent says "management commentary unavailable" rather than fabricating quotes.

The multi-query discipline prevents single-sided retrieval bias — without it the agent retrieves whichever case it was already considering.

---

## Commodity input costs

For stocks with raw-material exposure (jewellers → gold, refiners → crude, EV-makers → lithium/copper, etc.), `get_commodity_snapshot(name)` returns spot price, 1d/5d/1m/3m/6m/12m changes, SMA-50/200 trend regime, ATR-14, and 52-week range.

Supported names: `gold`, `silver`, `platinum`, `palladium`, `copper`, `aluminum`, `crude`/`wti`, `brent`, `natural_gas`, `gasoline`, `heating_oil`, `cotton`, plus ETF proxies for `lithium`, `steel`, `uranium`, `rare_earth`.

The agent decides when to call this — typically after reading what the company itself says about cost drivers in the concall RAG, then cross-referencing with the live commodity price. Example: for a refiner, it pulls Brent and connects the price move to refining-margin pressure that's already visible in EPS.

---

## Architecture

```
user input + optional context (PDFs / URLs / markdown / free-text)
    │
    ▼
LangGraph ReAct agent (DeepSeek default; Groq + Anthropic supported)
    │
    ├── get_technical_snapshot(ticker)     ← Kite OHLCV → pandas-ta indicators
    ├── get_macro_snapshot(ticker)         ← NIFTY 50 + sector index + size-bucket index
    ├── get_fundamentals_snapshot(ticker)  ← yfinance ratios + 4-quarter trend
    ├── get_news_and_earnings(ticker)      ← Google News RSS headlines + yfinance earnings dates
    ├── get_analyst_consensus(ticker)      ← yfinance sell-side targets + rating distribution
    ├── get_commodity_snapshot(name)       ← yfinance futures / ETF proxy for input-cost reasoning
    └── get_management_commentary(t, q, k) ← RAG over indexed concall transcripts (Chroma + MiniLM + cross-encoder)
    │
    ▼
Structured markdown report + trace JSON
```

OHLCV data is cached daily in `cache/` for cross-tool and cross-run reuse — running 10 stocks of the same sector hits Kite ~5 times instead of 30+.

User-supplied content (PDFs, URLs, markdown, free-text) is wrapped in `<source>` and `<user_context>` tags. The system prompt explicitly tells the LLM to treat that content as **reference material, not instructions**, mitigating prompt-injection from external content.

---

## Backtesting

A small backtest harness ships in `backtest.py`:

```bash
python backtest.py                # 10 NIFTY tickers × 4 quarterly dates = 40 calls, ~10 min
python backtest_analyze.py latest # hit rates, confidence calibration, baseline comparison
```

Backtest is **technical-only** because yfinance fundamentals are not point-in-time (using current ratios for past dates would introduce look-ahead bias). Output is saved to `backtest/<run-id>/`.

---

## Honest limitations

- **Sample size**: 40-call backtest is a sanity check, not statistical proof.
- **Confidence label**: was *anti-signal* in the technical-only backtest (HIGH-confidence calls were on average more wrong than MEDIUM). Now hidden from the default output by design.
- **LLM stochasticity**: same data may produce different ratings on different runs, even at temperature=0.
- **News quality varies**: built-in Google News RSS surfaces decent coverage for most NSE names; for thinly-covered tickers you may need to supplement with `--url`/`--pdf`. Article *bodies* are not auto-fetched — the agent sees titles + short descriptions and flags when a headline's full context would change the read.
- **Sector mapping is sector-level by default**, with industry-level distinction only for Financial Services (banks vs brokers vs NBFCs vs insurance). Other yfinance sectors map 1:1 to a NIFTY index, sometimes loosely (Industrials → NIFTY AUTO, etc.). When mapping is loose, the agent pivots to the size-bucket comparison instead.
- **Survivorship bias** in the backtest universe (we tested currently-listed names).
- **Indian-market only**: Kite Connect is NSE/BSE; this tool will not work for US/EU equities without rewriting the data layer.
- **Commodity prices are USD-denominated** continuous futures (or ETF NAVs for proxy commodities). The agent reasons about *direction*, not absolute INR conversion.

---

## Repository layout

```
.
├── lg_agent.py               # main agent (LangGraph ReAct loop)
├── prompts.py                # system prompt (structured-thinking framing, bull/bear/weight discipline)
├── llm_factory.py            # DeepSeek / Groq / Anthropic provider switch
├── persistence.py            # save reports + trace JSON
├── context_loader.py         # PDF / URL / markdown / free-text loaders, with prompt-injection guards
├── cache.py                  # daily-fresh disk cache (Kite instruments + OHLCV)
├── kite_login.py             # daily Kite token refresh
├── ticker_utils.py           # NSE: / .NS / plain symbol normalisation
├── screener.py               # full-agent watchlist scanner
├── screener_lite.py          # technicals+fundamentals-only scanner (faster, cheaper)
├── backtest.py               # technical-only backtest harness
├── backtest_analyze.py       # hit rates, confidence calibration, summary
├── backtest_prompts.py       # technical-only system prompt for backtest
├── watchlist.txt             # default watchlist for screener
├── watchlist_top15.txt       # NIFTY top-15 watchlist
├── watchlist_nifty100.txt    # NIFTY 100 watchlist
├── tools/
│   ├── technical.py          # OHLCV + indicators (RSI, MACD, SMA, OBV, VWMA, ATR, Bollinger)
│   ├── fundamentals.py       # yfinance valuation + profitability + leverage + earnings trend
│   ├── macro.py              # NIFTY 50, sector index, size-bucket index benchmarking
│   ├── news.py               # Google News RSS headlines + yfinance earnings dates
│   ├── analyst.py            # yfinance sell-side targets + rating distribution + drift
│   ├── commodity.py          # yfinance commodity prices (gold, copper, brent, etc.)
│   ├── commentary.py         # RAG retrieval over Chroma-indexed concall transcripts
│   └── tool_definitions.py   # @tool wrappers with structured error responses
├── ingest/
│   └── build_concall_index.py # chunk + embed + upsert concall transcripts to Chroma
├── data/                     # gitignored — concall corpus + Chroma vectorstore (regenerate locally)
├── reports/                  # gitignored — per-run reports + trace JSON
├── backtest/                 # gitignored — backtest outputs
├── screener/                 # gitignored — screener summary files
├── cache/                    # gitignored — daily OHLCV + instruments cache
├── requirements.txt
└── .env.example
```

---

## License

[MIT](./LICENSE) — including the explicit clause that the software is for educational use only and the authors are not liable for any trading losses.
