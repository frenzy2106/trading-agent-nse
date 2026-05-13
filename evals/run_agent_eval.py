"""
Agent-level eval: ablation of fact-card injection.

For each entry in evals/golden_commentary.json, run the agent twice:
  1. WITH fact-card injection (production behaviour)
  2. WITHOUT fact-card injection (RAG-only baseline)

Both conditions use the EXACT same tools and the same eval-mode system prompt --
only the fact-card injection toggle differs. We capture each answer and score
it on:

  - theme_coverage: fraction of must_contain_themes that appear in the answer
    string (deterministic, judge-free).

  - judge_score: LLM-as-judge correctness vs the golden reference. Single
    DeepSeek JSON-mode call with a fixed rubric (0.0-1.0). One run per
    judgment in v1 -- if you need variance bounds, add --runs N like
    run_commentary_eval.py.

The output is a comparison table per entry plus condition means. The whole
point is to test whether the structured-extraction-as-first-class-citizen
hypothesis pays off vs RAG alone.

Usage:
    python -m evals.run_agent_eval                  # all 9 entries
    python -m evals.run_agent_eval --only INOXINDIA  # filter by ticker substring
    python -m evals.run_agent_eval --limit 3         # first 3 entries only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure we load .env BEFORE importing lg_agent so env-var-based toggles work
# the first time chat_node executes.
load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path("evals/golden_commentary.json")
RESULTS_DIR = Path("evals/results")
JUDGE_PROVIDER_ENV = "JUDGE_PROVIDER"  # optional override; default uses DeepSeek


# --------------------------------------------------------------------------- #
# Theme coverage (deterministic)
# --------------------------------------------------------------------------- #

def theme_coverage(answer: str, themes: list[str]) -> tuple[float, list[str]]:
    """Return (coverage_ratio, missing_themes). Case-insensitive substring."""
    if not themes:
        return 1.0, []
    a = answer.lower()
    missing = [t for t in themes if t.lower() not in a]
    return (len(themes) - len(missing)) / len(themes), missing


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """\
You are a careful evaluator of equity-analyst answers about Indian listed stocks.

You will receive:
  - the QUESTION the analyst was asked
  - the REFERENCE answer (ground truth, written by a human)
  - the CANDIDATE answer the analyst produced

Score the CANDIDATE on its factual alignment with the REFERENCE, on a 0.0-1.0 scale:
  1.0  contains every key fact from the reference (numbers, names, directions)
  0.75 contains most key facts; minor omissions
  0.5  contains some key facts; meaningful gaps OR meaningful additions
  0.25 mostly off-topic OR contradicts the reference
  0.0  unrelated, refuses, or hallucinates contradictory facts

Penalise:
  - missing specific numbers the reference names
  - wrong direction (e.g. reference says margin expanded, candidate says compressed)
  - hallucinated specifics that aren't in the reference

Don't penalise:
  - extra correct context that isn't in the reference but isn't wrong
  - paraphrasing the reference's language
  - rounding differences within 5% on the same metric

Output ONLY a JSON object: {"score": 0.0-1.0, "rationale": "one short sentence"}.
"""


def _build_judge_user_prompt(question: str, reference: str, candidate: str) -> str:
    return (
        f"=== QUESTION ===\n{question}\n\n"
        f"=== REFERENCE ===\n{reference}\n\n"
        f"=== CANDIDATE ===\n{candidate}\n\n"
        "Return the JSON score."
    )


def _judge_client():
    """Return (openai-compatible client, model). DeepSeek by default."""
    provider = os.getenv(JUDGE_PROVIDER_ENV, "deepseek").lower()
    if provider == "deepseek":
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            sys.exit("ERROR: DEEPSEEK_API_KEY not set")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com"), model
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: OPENAI_API_KEY not set")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAI(api_key=api_key), model
    sys.exit(f"unsupported {JUDGE_PROVIDER_ENV}={provider!r}")


def judge_answer(
    client,
    model: str,
    question: str,
    reference: str,
    candidate: str,
    *,
    temperature: float = 0.0,
) -> dict:
    """Single LLM-judge pass. Returns {'score': float, 'rationale': str}."""
    if not candidate or not candidate.strip():
        return {"score": 0.0, "rationale": "empty candidate"}
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": _build_judge_user_prompt(question, reference, candidate)},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        # Clamp to [0, 1] -- judges occasionally hallucinate 1.2 or similar.
        score = max(0.0, min(1.0, score))
        return {"score": score, "rationale": data.get("rationale", "")}
    except Exception as e:
        return {"score": 0.0, "rationale": f"judge parse error: {e}"}


def judge_answer_multi(
    client,
    model: str,
    question: str,
    reference: str,
    candidate: str,
    *,
    n_runs: int,
    temperature: float,
) -> dict:
    """Run the judge n_runs times and return aggregated stats.

    Each run is an independent LLM call. With temperature > 0, the resulting
    score distribution captures judge variance -- which is the dominant noise
    source in LLM-as-judge eval. With n_runs=1 this is equivalent to
    judge_answer() and the result has zero variance.

    Returns {
      'score': mean,        # for backwards compatibility with table printer
      'score_mean': mean,
      'score_std': std,
      'score_min': min, 'score_max': max,
      'runs': [{'score', 'rationale'}, ...]
    }
    """
    runs = []
    for _ in range(n_runs):
        runs.append(judge_answer(
            client, model, question, reference, candidate, temperature=temperature,
        ))
    scores = [r["score"] for r in runs]
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    std = var ** 0.5
    return {
        "score": mean,
        "score_mean": mean,
        "score_std": std,
        "score_min": min(scores),
        "score_max": max(scores),
        "n_runs": n,
        "rationale": runs[0]["rationale"] if runs else "",
        "runs": runs,
    }


# --------------------------------------------------------------------------- #
# Agent invocation
# --------------------------------------------------------------------------- #

def run_agent_for_query(query: str, *, inject_fact_card: bool, thread_id: str) -> str:
    """Invoke the agent for one focused question. Returns the final assistant
    message content as a string.

    Sets env vars BEFORE building the graph so chat_node sees the right toggles:
      AGENT_MODE=eval                        (focused-answer system prompt)
      FACT_CARD_INJECTION=on|off             (the ablation knob)

    The graph is rebuilt per call. That's fine: tool binding is cheap; the
    MemorySaver's thread_id keeps cross-query state isolated.
    """
    os.environ["AGENT_MODE"] = "eval"
    os.environ["FACT_CARD_INJECTION"] = "on" if inject_fact_card else "off"

    # Lazy import to ensure env vars are set first (some imports cache state).
    from lg_agent import build_graph

    app = build_graph()
    result = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content or ""


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "n": 0}
    return {"mean": sum(values) / len(values), "n": len(values)}


def main():
    parser = argparse.ArgumentParser(description="Agent-level eval with fact-card ablation")
    parser.add_argument("--only", help="Filter entries by substring match on ticker or id")
    parser.add_argument("--limit", type=int, help="Run only first N entries")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (theme coverage only)")
    parser.add_argument(
        "--judge-runs", type=int, default=1,
        help="Number of judge runs per (entry, condition). Use >1 with --judge-temp>0 to measure judge variance.",
    )
    parser.add_argument(
        "--judge-temp", type=float, default=0.3,
        help="Sampling temperature for the judge. 0 is deterministic; higher reveals judge variance.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(levelname)s %(name)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    entries = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.only:
        entries = [e for e in entries if args.only.lower() in (e["ticker"] + e["id"]).lower()]
    if args.limit:
        entries = entries[: args.limit]
    print(f"Running {len(entries)} entries × 2 conditions = {2 * len(entries)} agent calls")

    judge_client, judge_model = (None, None)
    if not args.no_judge:
        judge_client, judge_model = _judge_client()
        print(f"Judge: deepseek/{judge_model}  runs={args.judge_runs}  temp={args.judge_temp}")

    results: list[dict] = []
    t_start = time.time()
    for i, entry in enumerate(entries, start=1):
        print(f"\n[{i}/{len(entries)}] {entry['id']} -- {entry['query']!r}")
        # The golden dataset stores `ticker` and `query` separately. The
        # agent's prompt expects the ticker inside the question (its first
        # workflow step parses it out), so we splice them together here.
        agent_query = f"For NSE ticker {entry['ticker']}: {entry['query']}"

        per_condition = {}
        for cond_name, inject in (("with_fc", True), ("without_fc", False)):
            t0 = time.time()
            try:
                answer = run_agent_for_query(
                    agent_query,
                    inject_fact_card=inject,
                    thread_id=f"eval-{entry['id']}-{cond_name}",
                )
            except Exception as e:
                logger.error("agent failed on %s/%s: %s", entry["id"], cond_name, e)
                answer = ""
            elapsed = time.time() - t0

            cov, missing = theme_coverage(answer, entry["must_contain_themes"])
            if judge_client is None:
                judge = {"score": None, "rationale": "judge skipped"}
            else:
                judge = judge_answer_multi(
                    judge_client, judge_model,
                    entry["query"], entry["reference"], answer,
                    n_runs=args.judge_runs, temperature=args.judge_temp,
                )
            per_condition[cond_name] = {
                "answer": answer,
                "theme_coverage": cov,
                "theme_misses": missing,
                "judge": judge,
                "elapsed_seconds": round(elapsed, 1),
            }
            j_summary = (
                f"{judge['score']:.2f}" if judge["score"] is None or args.judge_runs == 1
                else f"{judge['score_mean']:.2f}±{judge['score_std']:.2f}"
            )
            print(
                f"  {cond_name:11s}: themes={cov:.2f}  judge={j_summary}"
                f"  t={elapsed:.1f}s  (chars={len(answer)})"
            )
        results.append({"entry": entry, "conditions": per_condition})

    # ---- summary table ----
    print("\n" + "=" * 110)
    has_std = args.judge_runs > 1
    if has_std:
        header = f"{'ID':30s} {'themes w/fc':>12s} {'themes w/o':>12s} {'judge w/fc':>16s} {'judge w/o':>16s} {'delta_mean':>10s}"
    else:
        header = f"{'ID':30s} {'themes w/fc':>12s} {'themes w/o':>12s} {'judge w/fc':>12s} {'judge w/o':>12s} {'delta':>8s}"
    print(header)
    print("-" * 110)
    deltas_judge: list[float] = []
    deltas_theme: list[float] = []
    with_judge_scores: list[float] = []
    without_judge_scores: list[float] = []
    with_theme_scores: list[float] = []
    without_theme_scores: list[float] = []
    for r in results:
        wfc = r["conditions"]["with_fc"]
        wo = r["conditions"]["without_fc"]
        jw = wfc["judge"]["score"]
        jo = wo["judge"]["score"]
        delta_j = (jw - jo) if (jw is not None and jo is not None) else None
        delta_t = wfc["theme_coverage"] - wo["theme_coverage"]
        if has_std and jw is not None:
            jw_s = f"{wfc['judge']['score_mean']:.2f}±{wfc['judge']['score_std']:.2f}"
            jo_s = f"{wo['judge']['score_mean']:.2f}±{wo['judge']['score_std']:.2f}"
            print(
                f"{r['entry']['id']:30s}"
                f" {wfc['theme_coverage']:>12.2f}"
                f" {wo['theme_coverage']:>12.2f}"
                f" {jw_s:>16s}"
                f" {jo_s:>16s}"
                f" {delta_j:+10.2f}"
            )
        else:
            print(
                f"{r['entry']['id']:30s}"
                f" {wfc['theme_coverage']:>12.2f}"
                f" {wo['theme_coverage']:>12.2f}"
                f" {('n/a' if jw is None else f'{jw:>12.2f}'):>12s}"
                f" {('n/a' if jo is None else f'{jo:>12.2f}'):>12s}"
                f" {('n/a' if delta_j is None else f'{delta_j:+8.2f}'):>8s}"
            )
        with_theme_scores.append(wfc["theme_coverage"])
        without_theme_scores.append(wo["theme_coverage"])
        deltas_theme.append(delta_t)
        if jw is not None:
            with_judge_scores.append(jw)
        if jo is not None:
            without_judge_scores.append(jo)
        if delta_j is not None:
            deltas_judge.append(delta_j)

    print("-" * 100)
    mean_t_with = _agg(with_theme_scores)["mean"] or 0
    mean_t_wo = _agg(without_theme_scores)["mean"] or 0
    mean_j_with = _agg(with_judge_scores).get("mean")
    mean_j_wo = _agg(without_judge_scores).get("mean")

    print(
        f"{'MEAN':30s}"
        f" {mean_t_with:>12.2f}"
        f" {mean_t_wo:>12.2f}"
        f" {('n/a' if mean_j_with is None else f'{mean_j_with:>12.2f}'):>12s}"
        f" {('n/a' if mean_j_wo is None else f'{mean_j_wo:>12.2f}'):>12s}"
    )
    if mean_j_with is not None and mean_j_wo is not None:
        print(f"\nMean judge delta (with_fc - without_fc): {mean_j_with - mean_j_wo:+.3f}")
    print(f"Mean theme delta  (with_fc - without_fc): {mean_t_with - mean_t_wo:+.3f}")
    elapsed = time.time() - t_start
    print(f"\nWall time: {elapsed:.1f}s")

    # ---- persist ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"agent-eval-{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": ts,
        "n_entries": len(results),
        "judge_model": f"deepseek/{judge_model}" if judge_model else None,
        "results": results,
        "summary": {
            "mean_theme_with_fc": mean_t_with,
            "mean_theme_without_fc": mean_t_wo,
            "mean_judge_with_fc": mean_j_with,
            "mean_judge_without_fc": mean_j_wo,
        },
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
