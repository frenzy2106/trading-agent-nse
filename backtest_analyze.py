"""
Analyse a backtest run.

Usage:
    python backtest_analyze.py <run-id>      # specific run
    python backtest_analyze.py latest        # most recent run

Computes:
  - Hit rate by rating (% of calls where direction matched forward return)
  - Mean forward return by rating bucket
  - Confidence calibration: do HIGH-conf calls outperform LOW-conf?
  - NIFTY-50 baseline comparison (just-buy-and-hold)
  - Overall summary

Writes results to backtest/<run-id>/summary.md.
"""

import sys
from pathlib import Path

import pandas as pd

BACKTEST_ROOT = Path("backtest")

# Bullish vs bearish bucket mapping
BULLISH = {"BUY", "OVERWEIGHT"}
BEARISH = {"UNDERWEIGHT", "SELL"}
NEUTRAL = {"HOLD"}


def resolve_run_id(arg: str) -> str:
    if arg == "latest":
        runs = sorted([d.name for d in BACKTEST_ROOT.iterdir() if d.is_dir()])
        if not runs:
            sys.exit("No backtest runs found.")
        return runs[-1]
    return arg


def hit_rate(df: pd.DataFrame, window_col: str) -> dict:
    """% of calls where the direction matched. Bullish call + positive return = hit."""
    out = {}
    for label, ratings in [("bullish", BULLISH), ("bearish", BEARISH)]:
        sub = df[df["rating"].isin(ratings)].dropna(subset=[window_col])
        if sub.empty:
            out[label] = None
            continue
        if label == "bullish":
            hits = (sub[window_col] > 0).sum()
        else:
            hits = (sub[window_col] < 0).sum()
        out[label] = round(100 * hits / len(sub), 1)
        out[f"{label}_n"] = len(sub)
    return out


def by_rating(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("rating").agg(
        n=("ticker", "count"),
        ret_30d_mean=("ret_30d", "mean"),
        ret_60d_mean=("ret_60d", "mean"),
        ret_90d_mean=("ret_90d", "mean"),
    ).round(2)
    return g


def by_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration check: do HIGH-conf calls beat LOW-conf calls in the rating's expected direction?"""
    df = df.copy()
    df["bucket"] = df["rating"].apply(
        lambda r: "bullish" if r in BULLISH else ("bearish" if r in BEARISH else "neutral")
    )
    # Direction-adjusted return: positive when call is "right"
    df["dir_ret_90d"] = df.apply(
        lambda r: r["ret_90d"] if r["bucket"] == "bullish"
        else (-r["ret_90d"] if r["bucket"] == "bearish" else r["ret_90d"]),
        axis=1,
    )
    g = df.groupby(["confidence", "bucket"]).agg(
        n=("ticker", "count"),
        dir_ret_90d_mean=("dir_ret_90d", "mean"),
    ).round(2)
    return g


def baseline_comparison(df: pd.DataFrame) -> dict:
    """Mean forward return across ALL calls vs. naive 'always long' baseline."""
    return {
        "mean_ret_30d": round(df["ret_30d"].mean(), 2),
        "mean_ret_60d": round(df["ret_60d"].mean(), 2),
        "mean_ret_90d": round(df["ret_90d"].mean(), 2),
        "n": len(df),
    }


def render_summary(run_id: str, df: pd.DataFrame) -> str:
    parts = [
        f"# Backtest summary — {run_id}",
        "",
        f"**Total calls:** {len(df)} | "
        f"**Tickers:** {df['ticker'].nunique()} | "
        f"**Test dates:** {df['test_date'].nunique()}",
        f"**Model:** technical-only, temperature=0, 3-month horizon",
        "",
        "## Rating distribution",
        df["rating"].value_counts(dropna=False).to_string(),
        "",
        "## Confidence distribution",
        df["confidence"].value_counts(dropna=False).to_string(),
        "",
        "## Mean forward return by rating",
        by_rating(df).to_string(),
        "",
        "## Hit rate (direction matched future return)",
    ]
    for w in [30, 60, 90]:
        col = f"ret_{w}d"
        h = hit_rate(df, col)
        parts.append(
            f"- T+{w}d: bullish hit-rate = {h.get('bullish')}% (n={h.get('bullish_n')}), "
            f"bearish hit-rate = {h.get('bearish')}% (n={h.get('bearish_n')})"
        )
    parts += [
        "",
        "## Confidence calibration (90d direction-adjusted return)",
        "*Positive = call was right. Higher = better. If HIGH > LOW, confidence is meaningful.*",
        "",
        by_confidence(df).to_string(),
        "",
        "## Naive baseline (mean return across all dates × tickers)",
        f"- T+30d: {baseline_comparison(df)['mean_ret_30d']}%",
        f"- T+60d: {baseline_comparison(df)['mean_ret_60d']}%",
        f"- T+90d: {baseline_comparison(df)['mean_ret_90d']}%",
        "",
        "## Caveats",
        "- Technical-only — fundamentals were excluded to avoid look-ahead bias.",
        "- Sample is small (NIFTY 10 × 4 quarterly dates = 40 calls). Use as a sanity check, not a statistical conclusion.",
        "- Survivorship bias: we tested currently-listed names only.",
        "- LLM is non-deterministic; same call run twice may differ.",
    ]
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python backtest_analyze.py <run-id|latest>")

    run_id = resolve_run_id(sys.argv[1])
    csv_path = BACKTEST_ROOT / run_id / "results.csv"
    if not csv_path.exists():
        sys.exit(f"Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    summary = render_summary(run_id, df)
    print(summary)

    out_path = BACKTEST_ROOT / run_id / "summary.md"
    out_path.write_text(summary, encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
