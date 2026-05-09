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


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window character chunker. Concall paragraphs are too long for paragraph-based splits."""
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append(text[i:end])
        if end >= n:
            break
        i = end - overlap
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
