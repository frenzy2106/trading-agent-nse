"""
Management commentary tool — RAG over indexed concall transcripts.

get_management_commentary(ticker, query, k=5) -> dict

Retrieves top-k semantically relevant chunks from the local Chroma index
populated by `ingest/build_concall_index.py`. Filters by ticker and recency
(default last 365 days) to avoid stale guidance.

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

_collection = None
_embedder = None
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


def get_management_commentary(ticker: str, query: str, k: int = 5) -> dict:
    t = to_plain(ticker)
    cutoff_date = date.today() - timedelta(days=RECENCY_DAYS)
    cutoff_ts = int(cutoff_date.strftime("%Y%m%d"))
    logger.info("commentary | ticker=%s query=%r k=%d cutoff=%s", t, query, k, cutoff_date.isoformat())

    with _chroma_lock:
        col = _get_collection()
        embedder = _get_embedder()
        q_emb = embedder.encode(query, normalize_embeddings=True).tolist()
        res = col.query(
            query_embeddings=[q_emb],
            n_results=k,
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

    chunks = [
        {
            "source": m.get("source"),
            "date": m.get("doc_date"),
            "text": d,
        }
        for d, m in zip(docs, metas)
    ]
    logger.info("commentary | returned %d chunks for %s", len(chunks), t)
    return {"ticker": t, "query": query, "chunks": chunks}
