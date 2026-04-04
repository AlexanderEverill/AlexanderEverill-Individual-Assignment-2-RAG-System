"""
Build and query a BM25 keyword index using rank_bm25.

The index is persisted as a pickle file alongside ChromaDB so both retrieval
methods share the same chunk corpus.

Usage:
    python src/bm25_index.py          # build index from chunks
    python src/bm25_index.py --info   # print index stats
"""

import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import BM25_INDEX_PATH
from chunking import chunk_all

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Simple tokeniser: lowercase, split on non-alphanumeric, keep rule codes intact
_RE_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenise(text: str) -> list[str]:
    """Lowercase tokenisation preserving dotted rule numbers (e.g. '4.2.1r')."""
    return _RE_TOKEN.findall(text.lower())


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[dict] | None = None) -> tuple[BM25Okapi, list[dict]]:
    """
    Tokenise all chunks, build a BM25Okapi index, and persist it.

    The persisted file stores:
    - 'index': the BM25Okapi object
    - 'chunks': the full chunk list (for retrieving text/metadata by position)
    - 'corpus': the tokenised corpus (for reference)

    Returns (bm25_index, chunks).
    """
    if chunks is None:
        chunks = chunk_all()
    if not chunks:
        raise ValueError("No chunks to index.")

    print(f"\nTokenising {len(chunks)} chunks for BM25...")
    corpus = [tokenise(c["text"]) for c in tqdm(chunks, desc="Tokenising")]

    print("Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus)

    # Persist
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"index": bm25, "chunks": chunks, "corpus": corpus}, f)

    print(f"BM25 index saved to {BM25_INDEX_PATH} "
          f"({BM25_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    return bm25, chunks


def load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Load a previously persisted BM25 index and chunk list."""
    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found at {BM25_INDEX_PATH}. Run bm25_index.py first."
        )
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]


def query_bm25(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """
    Query the BM25 index and return top-k results.

    Returns list of dicts with 'chunk_id', 'text', 'metadata', 'bm25_score'.
    """
    tokens = tokenise(query)
    scores = bm25.get_scores(tokens)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "metadata": chunks[idx]["metadata"],
                "bm25_score": float(scores[idx]),
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_info() -> None:
    """Print stats about the persisted BM25 index."""
    try:
        bm25, chunks = load_bm25_index()
        print(f"BM25 index: {len(chunks)} chunks")
        print(f"File size: {BM25_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

        # Demo query
        demo_q = "financial promotions fair clear not misleading"
        results = query_bm25(demo_q, bm25, chunks, top_k=3)
        print(f"\nDemo query: '{demo_q}'")
        for r in results:
            print(f"  [{r['chunk_id']}] score={r['bm25_score']:.2f} — "
                  f"{r['text'][:100]}...")
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    if "--info" in sys.argv:
        print_info()
    else:
        build_bm25_index()
