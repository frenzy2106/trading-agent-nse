"""
Build / refresh the Chroma index over earnings call transcripts.

Layout:
    data/concalls/<TICKER>/<filename>.md   -- transcripts, optional YAML frontmatter
    data/concalls/<TICKER>/<filename>.pdf  -- transcripts as PDFs (pypdf)
    data/vectorstore/                      -- Chroma persistent dir (created)

Idempotent: re-running upserts existing chunks via deterministic IDs.

Usage:
    python -m ingest.build_concall_index             # all tickers
    python -m ingest.build_concall_index RELIANCE    # one ticker
"""

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path

from tools.commentary import COLLECTION_NAME, EMBED_MODEL, VECTORSTORE_PATH

logger = logging.getLogger(__name__)

CONCALLS_DIR = Path("data/concalls")
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150
EMBED_BATCH = 64
# How far back from the ideal endpoint to search for a clean split.
# 200 chars ≈ 30-40 words — generous enough to find a paragraph/sentence break
# without making chunks much shorter than CHUNK_CHARS.
CHUNK_SEARCH_WINDOW = 200

# Split-point patterns in priority order (best → worst).
# Paragraph break beats sentence end beats word boundary beats hard cut.
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[.!?][\s\n]")
_WORD_BREAK_RE = re.compile(r"\s")

_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(20\d{2})\b"
)


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5:].strip()
    return text


def _read_md(path: Path) -> str:
    return _strip_frontmatter(path.read_text(encoding="utf-8"))


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(pages).strip()


def extract_doc_date(text: str, fallback: date) -> str:
    """Find the first 'Month DD, YYYY' in the head of the doc; fall back to the given date."""
    m = _DATE_RE.search(text[:5000])
    if not m:
        return fallback.isoformat()
    try:
        return date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2))).isoformat()
    except ValueError:
        return fallback.isoformat()


def chunk_text(
    text: str,
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
    search_window: int = CHUNK_SEARCH_WINDOW,
) -> list[str]:
    """Sentence-aware sliding-window chunker.

    Walks forward by `size` chars, then looks back up to `search_window` chars
    for a clean split point. Tries in priority order:
      1. Paragraph break (\\n\\n)  — preserves topical coherence
      2. Sentence end (.!? + ws)  — preserves grammatical units
      3. Word boundary (whitespace) — avoids mid-word splits
      4. Hard char cut             — only when nothing better exists

    Replaces the previous pure-character chunker after the eval revealed
    mid-word splits ("issioning of polysilicon...") hurt both bi-encoder
    embedding quality and cross-encoder rerank scores.
    """
    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ideal_end = min(i + size, n)
        if ideal_end >= n:
            chunk = text[i:n].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look back up to search_window chars for the best split point.
        window_start = max(i + 1, ideal_end - search_window)
        best_split = None
        for pat in (_PARAGRAPH_BREAK_RE, _SENTENCE_END_RE, _WORD_BREAK_RE):
            matches = list(pat.finditer(text, window_start, ideal_end))
            if matches:
                best_split = matches[-1].end()  # latest match — closest to ideal
                break

        if best_split is None or best_split <= i:
            best_split = ideal_end  # hard cut fallback

        chunk = text[i:best_split].strip()
        if chunk:
            chunks.append(chunk)

        # Apply overlap, but snap it to the next clean word boundary so the
        # following chunk also STARTS on a clean boundary (not mid-word).
        overlap_target = best_split - overlap
        if overlap_target <= i:
            next_start = best_split
        else:
            ws_match = _WORD_BREAK_RE.search(text, overlap_target, best_split)
            next_start = ws_match.end() if ws_match else best_split
        i = next_start

    return chunks


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return _read_md(path)
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def index_ticker(ticker: str, embedder, collection) -> int:
    """Read all transcripts for a ticker, chunk + embed + upsert. Returns chunk count."""
    ticker_dir = CONCALLS_DIR / ticker
    files = sorted(
        [p for p in ticker_dir.iterdir() if p.suffix.lower() in (".md", ".pdf")]
    )
    if not files:
        logger.warning("no transcripts found for %s", ticker)
        return 0

    total_chunks = 0
    for path in files:
        try:
            text = _load_text(path)
        except Exception as e:
            logger.warning("failed to read %s: %s", path, e)
            continue
        if not text.strip():
            logger.warning("empty text for %s", path)
            continue

        # Use file mtime as the date fallback if no in-content date is found.
        mtime = date.fromtimestamp(path.stat().st_mtime)
        doc_date = extract_doc_date(text, fallback=mtime)
        source_label = f"{ticker} {path.stem}"

        chunks = chunk_text(text)
        ids = [f"{ticker}__{path.stem}__{i:04d}" for i in range(len(chunks))]
        # doc_date_ts as YYYYMMDD int — Chroma's $gte filter requires numeric, not strings.
        doc_date_ts = int(doc_date.replace("-", ""))
        metadatas = [
            {
                "ticker": ticker,
                "source": source_label,
                "doc_date": doc_date,
                "doc_date_ts": doc_date_ts,
                "chunk_idx": i,
            }
            for i in range(len(chunks))
        ]

        # Embed in batches to avoid spikes.
        embeddings = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            embs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(embs.tolist())

        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        logger.info("indexed %s | source=%s date=%s chunks=%d", ticker, source_label, doc_date, len(chunks))
        total_chunks += len(chunks)

    return total_chunks


def main():
    parser = argparse.ArgumentParser(description="Build / refresh concall Chroma index")
    parser.add_argument("ticker", nargs="?", help="Optional: index only this ticker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    if not CONCALLS_DIR.exists():
        sys.exit(f"ERROR: {CONCALLS_DIR}/ does not exist — drop transcripts there first.")

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(VECTORSTORE_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)

    tickers = (
        [args.ticker]
        if args.ticker
        else [d.name for d in CONCALLS_DIR.iterdir() if d.is_dir()]
    )

    grand_total = 0
    for t in tickers:
        n = index_ticker(t, embedder, collection)
        print(f"  {t}: {n} chunks")
        grand_total += n

    print(f"\nTotal chunks indexed: {grand_total}")
    print(f"Collection: {COLLECTION_NAME} @ {VECTORSTORE_PATH}/")


if __name__ == "__main__":
    main()
