# Vector search, BM25 search, and hybrid Reciprocal Rank Fusion

"""
Retrieval module — vector search via ChromaDB, BM25 keyword search, and
hybrid Reciprocal Rank Fusion.

Provides three retrieval modes:
    1. vector_search()  — ChromaDB cosine similarity (baseline)
    2. bm25_search()    — BM25Okapi keyword matching
    3. hybrid_search()  — vector + BM25 merged via Reciprocal Rank Fusion
"""

import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BASELINE_TOP_K,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    OPENAI_API_KEY,
    RERANK_CANDIDATE_K,
    RRF_K,
)
from embedding import get_chroma_client, get_or_create_collection
from bm25_index import load_bm25_index, query_bm25


def _embed_query(query: str, client: OpenAI) -> list[float]:
    """Embed a single query string with OpenAI."""
    resp = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS,
    )
    return resp.data[0].embedding


def vector_search(query: str, top_k: int = BASELINE_TOP_K) -> list[dict]:
    """
    Embed the query and retrieve top-k nearest chunks from ChromaDB.

    Returns a list of dicts:
        {"chunk_id", "text", "metadata", "distance"}
    Sorted by ascending distance (most similar first).
    """
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)

    query_embedding = _embed_query(query, openai_client)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


def bm25_search(query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
    """
    Query the persisted BM25 index and return top-k results.

    Returns a list of dicts:
        {"chunk_id", "text", "metadata", "bm25_score"}
    Sorted by descending BM25 score.
    """
    bm25, chunks = load_bm25_index()
    return query_bm25(query, bm25, chunks, top_k=top_k)


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    top_n: int = RERANK_CANDIDATE_K,
) -> list[dict]:
    """
    Merge vector and BM25 result lists using Reciprocal Rank Fusion.

    RRF_score(doc) = Σ 1 / (k + rank(doc))

    where rank is 1-based position in each result list.
    Returns the top-n candidates sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    # Score vector results
    for rank, hit in enumerate(vector_results, start=1):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_lookup[cid] = hit

    # Score BM25 results
    for rank, hit in enumerate(bm25_results, start=1):
        cid = hit["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_lookup:
            chunk_lookup[cid] = hit

    # Sort by RRF score descending, take top_n
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]

    fused = []
    for cid in sorted_ids:
        entry = {
            "chunk_id": cid,
            "text": chunk_lookup[cid]["text"],
            "metadata": chunk_lookup[cid]["metadata"],
            "rrf_score": rrf_scores[cid],
        }
        fused.append(entry)

    return fused


def hybrid_search(
    query: str,
    vector_k: int = HYBRID_TOP_K,
    bm25_k: int = HYBRID_TOP_K,
    top_n: int = RERANK_CANDIDATE_K,
) -> list[dict]:
    """
    Run vector search and BM25 search in parallel, then merge via RRF.

    Returns top-n candidate chunks (default 20) sorted by RRF score,
    ready for cross-encoder re-ranking.
    """
    vec_results = vector_search(query, top_k=vector_k)
    bm25_results = bm25_search(query, top_k=bm25_k)
    return reciprocal_rank_fusion(vec_results, bm25_results, top_n=top_n)
