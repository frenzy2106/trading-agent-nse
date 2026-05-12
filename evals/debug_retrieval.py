"""
Deep-dive retrieval diagnostic. Traces why a specific query fails to surface
known content by walking through the two-stage retrieval pipeline.

Pipeline stages:
  Stage 0 (ground truth): scan all chunks indexed for the ticker; report which
    chunks contain each keyword. Tells you whether the content even exists in
    the corpus.
  Stage 1 (bi-encoder + Chroma): fetch CANDIDATE_K candidates by cosine
    similarity. Tells you whether the embedding model surfaces relevant chunks
    in the top-K.
  Stage 2 (cross-encoder rerank): score all candidates with the cross-encoder.
    Tells you whether the reranker correctly elevates relevant chunks to top-k.

For each stage, ★ marks chunks that contain any of the ground-truth keywords.

Usage:
    python -m evals.debug_retrieval RELIANCE \\
        "Capex plans and new energy ramp-up timelines" \\
        --keywords solar giga polysilicon "Karan Suri" "20 gigawatts" HJT ALMM
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from ticker_utils import to_plain


# Mirror production constants so the diagnostic stays in sync with tools/commentary.py
from tools.commentary import CANDIDATE_K, VECTORSTORE_PATH, COLLECTION_NAME, RECENCY_DAYS


def _mark(text: str, keywords: list[str]) -> str:
    """★ if any keyword (case-insensitive substring) appears in text, else ' '."""
    tl = text.lower()
    return "★" if any(k.lower() in tl for k in keywords) else " "


def _snippet(text: str, keywords: list[str], cap: int = 140) -> str:
    """Show a snippet centred on the first keyword hit if any, else the head."""
    tl = text.lower()
    for k in keywords:
        idx = tl.find(k.lower())
        if idx >= 0:
            start = max(0, idx - 40)
            end = min(len(text), idx + 100)
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return prefix + text[start:end].strip().replace("\n", " ") + suffix
    s = text[:cap].replace("\n", " ").strip()
    return s + ("…" if len(text) > cap else "")


def stage_0_ground_truth(ticker: str, keywords: list[str]) -> dict:
    """Scan ALL indexed chunks for this ticker; count keyword hits."""
    from chromadb import PersistentClient
    client = PersistentClient(path=str(VECTORSTORE_PATH))
    col = client.get_or_create_collection(COLLECTION_NAME)

    # Pull everything for this ticker (date filter mirrors production)
    cutoff_date = date.today() - timedelta(days=RECENCY_DAYS)
    cutoff_ts = int(cutoff_date.strftime("%Y%m%d"))
    raw = col.get(where={"$and": [
        {"ticker": ticker},
        {"doc_date_ts": {"$gte": cutoff_ts}},
    ]})

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    ids = raw.get("ids") or []

    print(f"\nStage 0 — Ground truth scan")
    print(f"  Total indexed chunks for {ticker} (last {RECENCY_DAYS}d): {len(docs)}")
    print()
    print(f"  Keyword hits across the corpus:")
    by_kw: dict[str, list[tuple[str, str, int]]] = {k: [] for k in keywords}
    for i, (doc_id, doc, meta) in enumerate(zip(ids, docs, metas)):
        for k in keywords:
            if k.lower() in doc.lower():
                by_kw[k].append((doc_id, meta.get("source", "?"), i))
    for k in keywords:
        hits = by_kw[k]
        if hits:
            sources = sorted(set(h[1] for h in hits))
            print(f"    '{k}': {len(hits)} chunks across {len(sources)} sources → {', '.join(sources)}")
        else:
            print(f"    '{k}': 0 chunks  ← CORPUS MISS")

    # Build set of doc_ids that contain ANY keyword — these are the "ground truth" chunks
    gt_ids = set()
    for hits in by_kw.values():
        for doc_id, _, _ in hits:
            gt_ids.add(doc_id)
    print(f"\n  Total unique chunks containing ≥1 keyword: {len(gt_ids)}")

    if len(gt_ids) > 0:
        print(f"\n  Sample of ground-truth chunks:")
        shown = 0
        for doc_id, doc, meta in zip(ids, docs, metas):
            if doc_id not in gt_ids:
                continue
            shown += 1
            if shown > 5:
                break
            print(f"    [{shown}] {meta.get('source', '?')}  date={meta.get('doc_date', '?')}")
            print(f"        {_snippet(doc, keywords)}")

    return {"all_ids": list(ids), "all_docs": docs, "all_metas": metas, "gt_ids": gt_ids}


def stage_1_bi_encoder(ticker: str, query: str, keywords: list[str], gt_ids: set) -> list[str]:
    """Run the bi-encoder query and report ranks of ground-truth chunks."""
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=str(VECTORSTORE_PATH))
    col = client.get_or_create_collection(COLLECTION_NAME)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    cutoff_date = date.today() - timedelta(days=RECENCY_DAYS)
    cutoff_ts = int(cutoff_date.strftime("%Y%m%d"))
    q_emb = embedder.encode(query, normalize_embeddings=True).tolist()
    res = col.query(
        query_embeddings=[q_emb],
        n_results=CANDIDATE_K,
        where={"$and": [
            {"ticker": ticker},
            {"doc_date_ts": {"$gte": cutoff_ts}},
        ]},
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    ids = (res.get("ids") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    print(f"\nStage 1 — Bi-encoder retrieval (top {CANDIDATE_K} candidates)")
    print(f"  Query: {query!r}")
    print(f"  {'rank':>4} {'GT':>3} {'dist':>6}  source / date  /  snippet")
    print(f"  {'-'*4} {'-'*3} {'-'*6}  {'-'*60}")
    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances), 1):
        is_gt = "★" if doc_id in gt_ids else " "
        print(f"  {rank:>4} {is_gt:>3} {dist:>6.3f}  {meta.get('source','?')}  ({meta.get('doc_date','?')})")
        print(f"             {_snippet(doc, keywords)}")

    # Which ground-truth chunks made it into top-K?
    found_in_topk = [doc_id for doc_id in ids if doc_id in gt_ids]
    missed = gt_ids - set(ids)
    print(f"\n  Ground-truth chunks in top {CANDIDATE_K}: {len(found_in_topk)} / {len(gt_ids)} total")
    if missed:
        print(f"  Missed ground-truth chunks (not in top {CANDIDATE_K}): {len(missed)} chunks")
    return ids, list(distances)


def stage_2_rerank(query: str, candidate_ids: list[str], candidate_distances: list[float],
                   all_data: dict, gt_ids: set, keywords: list[str], k: int = 5):
    """Cross-encoder rerank — same scoring as production."""
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    id_to_doc = dict(zip(all_data["all_ids"], all_data["all_docs"]))
    id_to_meta = dict(zip(all_data["all_ids"], all_data["all_metas"]))
    candidates = [(cid, id_to_doc[cid], id_to_meta[cid]) for cid in candidate_ids if cid in id_to_doc]
    pairs = [(query, doc) for _, doc, _ in candidates]
    cross_scores = [float(s) for s in reranker.predict(pairs, show_progress_bar=False)]
    reranked = sorted(zip(candidates, cross_scores), key=lambda x: x[1], reverse=True)

    print(f"\nStage 2 — Cross-encoder rerank ({len(candidates)} candidates, top-{k} returned by production)")
    print(f"  {'rank':>4} {'GT':>3} {'bi#':>4} {'score':>7}  source / snippet")
    print(f"  {'-'*4} {'-'*3} {'-'*4} {'-'*7}  {'-'*60}")
    for rank, ((cid, doc, meta), cs) in enumerate(reranked, 1):
        is_gt = "★" if cid in gt_ids else " "
        marker = " ← returned" if rank <= k else ""
        if rank > 10 and cid not in gt_ids:
            continue  # only show GT past rank 10 to keep output readable
        bi_rank = candidate_ids.index(cid) + 1
        print(f"  {rank:>4} {is_gt:>3} {bi_rank:>4} {cs:>7.3f}  {meta.get('source','?')}{marker}")
        print(f"             {_snippet(doc, keywords)}")

    gt_ranks = [rank for rank, ((cid, _, _), _) in enumerate(reranked, 1) if cid in gt_ids]
    in_top_k = [r for r in gt_ranks if r <= k]
    print(f"\n  GT chunk ranks after rerank: {gt_ranks[:10]}{'...' if len(gt_ranks)>10 else ''}")
    print(f"  GT chunks returned to user (top {k}): {len(in_top_k)} / {len(gt_ranks)} in candidates")


def main():
    ap = argparse.ArgumentParser(description="Trace a query through the retrieval pipeline")
    ap.add_argument("ticker", help="NSE ticker, e.g. RELIANCE")
    ap.add_argument("query", help="The query string sent to the retriever")
    ap.add_argument("--keywords", nargs="+", required=True,
                    help="Ground-truth keywords — chunks containing any of these are 'should be retrieved'")
    args = ap.parse_args()

    ticker = to_plain(args.ticker)
    print(f"\n{'='*70}")
    print(f"Retrieval diagnostic: {ticker}")
    print(f"Query:    {args.query!r}")
    print(f"Keywords: {args.keywords}")
    print('='*70)

    s0 = stage_0_ground_truth(ticker, args.keywords)
    if not s0["gt_ids"]:
        print("\n[STOP] No chunks contain any of the keywords. Nothing to rerank.")
        print("       → The content is missing from the corpus (Layer A: chunking / ingestion).")
        return

    candidate_ids, candidate_distances = stage_1_bi_encoder(ticker, args.query, args.keywords, s0["gt_ids"])
    stage_2_rerank(args.query, candidate_ids, candidate_distances, s0, s0["gt_ids"], args.keywords)


if __name__ == "__main__":
    main()
