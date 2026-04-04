# Orchestrates full RAG pipeline — exposes run_baseline() and run_enhanced()

"""
Pipeline module — wires retrieval and generation into end-to-end functions.

    run_baseline()  — vector search → baseline prompt → GPT-4o
    run_enhanced()  — hybrid retrieval → cross-encoder re-rank → enhanced prompt
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BASELINE_TOP_K, RERANK_CANDIDATE_K, RERANK_TOP_K
from retrieval import vector_search, hybrid_search
from reranker import rerank
from generation import generate_baseline, generate_enhanced


def run_baseline(question: str, top_k: int = BASELINE_TOP_K, verbose: bool = False) -> dict:
    """
    Baseline RAG pipeline: embed query → vector top-k → vanilla prompt → GPT-4o.

    Returns:
        {
            "question": str,
            "answer": str,
            "chunks": list[dict],
            "pipeline": "baseline",
            "latency_ms": float,
        }
    """
    start = time.perf_counter()

    chunks = vector_search(question, top_k=top_k)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[BASELINE] Question: {question}")
        print(f"Retrieved {len(chunks)} chunks:")
        for c in chunks:
            print(f"  - {c['chunk_id']}  (dist={c['distance']:.4f})")
        print(f"{'='*60}\n")

    answer = generate_baseline(question, chunks)
    latency = (time.perf_counter() - start) * 1000

    if verbose:
        print(f"Answer:\n{answer}\n")
        print(f"Latency: {latency:.0f} ms\n")

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "pipeline": "baseline",
        "latency_ms": round(latency, 1),
    }


def run_enhanced(
    question: str,
    candidate_k: int = RERANK_CANDIDATE_K,
    final_k: int = RERANK_TOP_K,
    verbose: bool = False,
) -> dict:
    """
    Enhanced RAG pipeline:
        hybrid retrieval (vector + BM25 → RRF) → cross-encoder re-rank → enhanced prompt.

    Returns:
        {
            "question": str,
            "answer": str,
            "chunks": list[dict],        # final top-k after re-ranking
            "candidates": list[dict],     # pre-rerank candidates from RRF
            "pipeline": "enhanced",
            "latency_ms": float,
        }
    """
    start = time.perf_counter()

    # Step 1: Hybrid retrieval (vector k=10 + BM25 k=10 → RRF top-20)
    candidates = hybrid_search(question, top_n=candidate_k)

    if verbose:
        print(f"\n{'='*60}")
        print(f"[ENHANCED] Question: {question}")
        print(f"Hybrid retrieval returned {len(candidates)} candidates:")
        for c in candidates:
            print(f"  - {c['chunk_id']}  (rrf={c['rrf_score']:.4f})")
        print(f"{'='*60}")

    # Step 2: Cross-encoder re-ranking → top-5
    reranked = rerank(question, candidates, top_k=final_k)

    if verbose:
        print(f"Re-ranked to {len(reranked)} chunks:")
        for c in reranked:
            print(f"  - {c['chunk_id']}  (rerank={c['rerank_score']:.4f})")
        print(f"{'='*60}\n")

    # Step 3: Enhanced generation with citation instructions
    answer = generate_enhanced(question, reranked)
    latency = (time.perf_counter() - start) * 1000

    if verbose:
        print(f"Answer:\n{answer}\n")
        print(f"Latency: {latency:.0f} ms\n")

    return {
        "question": question,
        "answer": answer,
        "chunks": reranked,
        "candidates": candidates,
        "pipeline": "enhanced",
        "latency_ms": round(latency, 1),
    }


# ---------------------------------------------------------------------------
# CLI — run baseline or enhanced via command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FCA RAG Pipeline")
    parser.add_argument("--mode", choices=["baseline", "enhanced"], default="enhanced")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    test_queries = [args.query] if args.query else [
        "What are the FCA Principles for Businesses?",
        "What does COBS 4.2.1R say about financial promotions?",
        "What are the requirements for complaint handling under SYSC?",
    ]

    runner = run_enhanced if args.mode == "enhanced" else run_baseline

    for q in test_queries:
        result = runner(q, verbose=True)
        print("-" * 60)
