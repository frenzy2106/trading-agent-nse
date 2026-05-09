"""
Management commentary tool — RAG over indexed concall transcripts.

get_management_commentary(ticker, query, k=5) -> dict

Two-stage retrieval:
  1. Bi-encoder (MiniLM) fetches CANDIDATE_K=20 candidates from Chroma — fast,
     casts a wide net but ranks loosely.
  2. Cross-encoder (ms-marco-MiniLM) reranks the candidates by scoring each
     (query, chunk) pair directly. Cross-encoders consistently outperform
     bi-encoder cosine similarity on retrieval benchmarks because they see
     query and chunk together rather than as independent vectors.

Recency-filtered to last 365 days to avoid stale guidance.

Returns {"error": "no_commentary_indexed", "kind": "no_commentary", ...}
when nothing matches — the agent must say "unavailable" rather than
fabricate quotes.
"""

import logging
import threading
from datetime import date, timedelta
from pathlib import Path

from ticker_utils import to_plain

logger = logging.getLogger(__name__)

VECTORSTORE_PATH = Path("data/vectorstore")
COLLECTION_NAME = "commentary"
RECENCY_DAYS = 365
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_K = 20  # bi-encoder candidates fetched before reranking

_collection = None
_embedder = None
_reranker = None
# Chroma 1.5.x Rust backend is not thread-safe; LangGraph's ToolNode dispatches
# tool calls in parallel when the LLM emits multiple in one turn. Serialise.
_chroma_lock = threading.Lock()


def _get_collection():
    """Lazy-init Chroma collection. Created if missing (empty queries return no chunks)."""
    global _collection
    if _collection is None:
        from chromadb import PersistentClient
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        client = PersistentClient(path=str(VECTORSTORE_PATH))
        _collection = client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def _get_embedder():
    """Lazy-init sentence-transformers model. ~80MB download on first use."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker():
    """Lazy-init cross-encoder reranker. ~80MB download on first use."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_management_commentary(ticker: str, query: str, k: int = 5) -> dict:
    t = to_plain(ticker)
    cutoff_date = date.today() - timedelta(days=RECENCY_DAYS)
    cutoff_ts = int(cutoff_date.strftime("%Y%m%d"))
    logger.info("commentary | ticker=%s query=%r k=%d cutoff=%s", t, query, k, cutoff_date.isoformat())

    with _chroma_lock:
        col = _get_collection()
        embedder = _get_embedder()
        q_emb = embedder.encode(query, normalize_embeddings=True).tolist()
        # Stage 1: bi-encoder retrieval — fetch CANDIDATE_K candidates.
        res = col.query(
            query_embeddings=[q_emb],
            n_results=CANDIDATE_K,
            where={"$and": [
                {"ticker": t},
                {"doc_date_ts": {"$gte": cutoff_ts}},
            ]},
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        if not docs:
            return {
                "error": f"No management commentary indexed for {t} within the last {RECENCY_DAYS} days.",
                "kind": "no_commentary",
                "ticker": t,
                "query": query,
            }

        # Stage 2: cross-encoder rerank — score each (query, chunk) pair directly.
        reranker = _get_reranker()
        pairs = [(query, d) for d in docs]
        scores = reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(docs, metas, scores), key=lambda x: float(x[2]), reverse=True)[:k]

    chunks = [
        {
            "source": m.get("source"),
            "date": m.get("doc_date"),
            "text": d,
        }
        for d, m, _ in ranked
    ]
    top_score = float(ranked[0][2]) if ranked else 0.0
    logger.info(
        "commentary | returned %d/%d candidates reranked for %s (top score=%.3f)",
        len(chunks), len(docs), t, top_score,
    )
    return {"ticker": t, "query": query, "chunks": chunks}
