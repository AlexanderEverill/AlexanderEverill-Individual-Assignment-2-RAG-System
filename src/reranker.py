# Cross-encoder re-ranking using ms-marco-MiniLM-L-6-v2

"""
Re-ranker module — uses a cross-encoder to re-score (query, chunk) pairs
and return the top-k most relevant chunks.

The cross-encoder reads query and chunk text jointly, producing a more
accurate relevance score than bi-encoder similarity or BM25 alone.
"""

import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config import RERANKER_MODEL, RERANK_TOP_K

# Module-level cache so the model is loaded once per process
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder model (cached after first call)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """
    Re-score candidate chunks with the cross-encoder and return the top-k.

    Args:
        query:      The user's question.
        candidates: List of chunk dicts (must contain "text" and "chunk_id").
        top_k:      Number of top results to return after re-ranking.

    Returns:
        List of chunk dicts with an added "rerank_score" field,
        sorted by descending cross-encoder score.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()

    # Build (query, chunk_text) pairs for the cross-encoder
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    # Attach scores and sort
    scored = []
    for candidate, score in zip(candidates, scores):
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
