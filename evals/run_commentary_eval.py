"""
Evaluate the management commentary RAG retrieval against the golden dataset.

Two metrics per golden entry:

  Phase A — Theme match (non-LLM, free, instant):
    For each entry, check whether the must_contain_themes appear (as substring,
    case-insensitive) anywhere in the union of retrieved chunk texts. This is a
    smoke detector — if even basic keyword matching fails, retrieval is
    definitely broken and you don't need to burn LLM tokens to confirm.

  Phase B — Context precision with reference (LLM-as-judge via Ragas):
    Uses Ragas's LLMContextPrecisionWithReference metric. For each (query,
    chunks, reference) triple, an LLM judge reads each chunk and decides
    whether it contributes information that supports the reference answer.
    Precision is computed as rank-weighted — chunks ranked higher in retrieval
    matter more (same intuition as nDCG in classical IR).

Run:
    python -m evals.run_commentary_eval               # all entries
    python -m evals.run_commentary_eval --limit 3    # smoke-test 3 entries
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Force UTF-8 stdout on Windows (cp1252 can't handle ₹, em-dashes, arrows).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "WARNING"),
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("eval")

GOLDEN_PATH = Path("evals/golden_commentary.json")
RESULTS_DIR = Path("evals/results")


# ── Phase A: non-LLM theme match ────────────────────────────────────────────

def run_retrieval(entry: dict) -> dict:
    """Call get_management_commentary, return chunks list (empty on error)."""
    from tools.commentary import get_management_commentary
    result = get_management_commentary(entry["ticker"], entry["query"], k=5)
    if "error" in result:
        return {"chunks": [], "error": result.get("error")}
    return {"chunks": result.get("chunks", []), "error": None}


def theme_match(chunks: list[dict], themes: list[str]) -> dict:
    """Substring search: which themes appear in the union of chunk texts?"""
    if not chunks or not themes:
        return {"hits": [], "misses": themes, "ratio": 0.0}
    combined = " ".join(c.get("text", "") for c in chunks).lower()
    hits = [t for t in themes if t.lower() in combined]
    misses = [t for t in themes if t.lower() not in combined]
    return {"hits": hits, "misses": misses, "ratio": len(hits) / len(themes)}


# ── Phase B: Ragas LLM-as-judge ─────────────────────────────────────────────

def build_judge():
    """Build a Ragas-compatible LLM judge using DeepSeek (OpenAI-compatible API).

    Ragas v0.4 deprecated LangchainLLMWrapper in favour of llm_factory, which
    takes a raw OpenAI client. We point that client at DeepSeek's endpoint.
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key.startswith("your_"):
        sys.exit("ERROR: DEEPSEEK_API_KEY not set in .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    return llm_factory(model, client=client)


def run_ragas(rows: list[dict], judge_llm) -> list[float | None]:
    """Run Ragas LLMContextPrecisionWithReference; return per-row scores."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    # The new ragas.metrics.collections module exists but its constructors
    # require per-sample invocation (async). Sticking with the legacy import
    # path that works with the batch evaluate() flow; LLM wrapper is already
    # on the modern llm_factory API in build_judge() above.
    from ragas.metrics import LLMContextPrecisionWithReference

    samples = []
    for r in rows:
        chunk_texts = [c.get("text", "") for c in r["chunks"]]
        if not chunk_texts:
            # Ragas can't score empty contexts — assign None placeholder
            samples.append(None)
            continue
        samples.append(SingleTurnSample(
            user_input=r["query"],
            retrieved_contexts=chunk_texts,
            reference=r["reference"],
        ))

    valid_samples = [s for s in samples if s is not None]
    if not valid_samples:
        return [None] * len(rows)

    dataset = EvaluationDataset(samples=valid_samples)
    result = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference()],
        llm=judge_llm,
        show_progress=True,
    )
    df = result.to_pandas()
    valid_scores = df["llm_context_precision_with_reference"].tolist()

    # Re-interleave: empty-chunk rows get None, others get their score in order
    out: list[float | None] = []
    idx = 0
    for s in samples:
        if s is None:
            out.append(None)
        else:
            out.append(float(valid_scores[idx]) if valid_scores[idx] == valid_scores[idx] else None)
            idx += 1
    return out


# ── Main loop ───────────────────────────────────────────────────────────────

def truncate_chunks_for_save(chunks: list[dict], cap: int = 200) -> list[dict]:
    """Keep saved JSON inspectable but small — truncate chunk text to cap chars."""
    return [
        {
            "source": c.get("source"),
            "date": c.get("date"),
            "text": (c.get("text", "")[:cap] + ("…" if len(c.get("text", "")) > cap else "")),
        }
        for c in chunks
    ]


def main():
    parser = argparse.ArgumentParser(description="Eval RAG retrieval against golden dataset")
    parser.add_argument("--limit", type=int, help="Process only first N entries (smoke test)")
    parser.add_argument("--out", type=str, help="Override output filename")
    args = parser.parse_args()

    entries = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    judge_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    print("\n=== Commentary RAG Eval ===")
    print(f"Golden entries: {len(entries)}")
    print(f"Judge model:    deepseek/{judge_model}")
    print()

    # Phase A: retrieval + theme match
    print("Phase A — retrieval + theme match")
    print("-" * 60)
    rows: list[dict] = []
    for i, entry in enumerate(entries, 1):
        t0 = time.time()
        ret = run_retrieval(entry)
        theme = theme_match(ret["chunks"], entry["must_contain_themes"])
        latency = time.time() - t0
        rows.append({
            **entry,
            "chunks": ret["chunks"],
            "retrieval_error": ret["error"],
            "theme_hits": theme["hits"],
            "theme_misses": theme["misses"],
            "theme_ratio": theme["ratio"],
            "retrieval_latency_s": round(latency, 2),
        })
        n_themes = len(entry["must_contain_themes"])
        n_hits = len(theme["hits"])
        status = "OK" if n_hits == n_themes else f"MISS({','.join(theme['misses'])})"
        print(f"  [{i}/{len(entries)}] {entry['id']:<26s}  themes {n_hits}/{n_themes}  chunks {len(ret['chunks']):>2}  {status}")

    # Phase B: Ragas LLM-as-judge
    print()
    print("Phase B — Ragas LLM-as-judge (context precision with reference)")
    print("-" * 60)
    judge = build_judge()
    cp_scores = run_ragas(rows, judge)
    for row, score in zip(rows, cp_scores):
        row["context_precision"] = score

    # Render summary
    print()
    print("=" * 60)
    print(f"{'ID':<26s}  {'Themes':>8s}  {'CtxPrec':>9s}")
    print("-" * 60)
    for row in rows:
        n_hits = len(row["theme_hits"])
        n_themes = len(row["theme_hits"]) + len(row["theme_misses"])
        themes_str = f"{n_hits}/{n_themes}"
        cp = row.get("context_precision")
        cp_str = f"{cp:.3f}" if cp is not None else "n/a"
        print(f"{row['id']:<26s}  {themes_str:>8s}  {cp_str:>9s}")
    print("-" * 60)

    # Aggregates
    avg_theme = sum(r["theme_ratio"] for r in rows) / len(rows) if rows else 0.0
    cps = [r["context_precision"] for r in rows if r["context_precision"] is not None]
    avg_cp = sum(cps) / len(cps) if cps else None
    cp_str = f"{avg_cp:.3f}" if avg_cp is not None else "n/a"
    print(f"{'MEAN':<26s}  {avg_theme*100:>7.1f}%  {cp_str:>9s}")
    print(f"  (n={len(rows)} entries, judge=deepseek/{judge_model})")

    # Save (with truncated chunks to keep file small/inspectable)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{timestamp}.json"
    payload = {
        "run_id": timestamp,
        "judge_model": f"deepseek/{judge_model}",
        "n_entries": len(rows),
        "aggregates": {
            "mean_theme_ratio": round(avg_theme, 4),
            "mean_context_precision": round(avg_cp, 4) if avg_cp is not None else None,
        },
        "rows": [
            {
                "id": r["id"],
                "ticker": r["ticker"],
                "query": r["query"],
                "reference": r["reference"],
                "must_contain_themes": r["must_contain_themes"],
                "theme_hits": r["theme_hits"],
                "theme_misses": r["theme_misses"],
                "theme_ratio": r["theme_ratio"],
                "context_precision": r["context_precision"],
                "retrieval_latency_s": r["retrieval_latency_s"],
                "retrieval_error": r["retrieval_error"],
                "chunks_truncated": truncate_chunks_for_save(r["chunks"]),
            }
            for r in rows
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
