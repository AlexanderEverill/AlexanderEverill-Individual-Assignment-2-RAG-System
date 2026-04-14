# Runs evaluation metrics (P@5, R@5, P@10, R@10, MRR, generation quality) on the test set

"""
Evaluation module — runs all 6 ablation configurations against the test set,
computes retrieval metrics (P@5, R@5, P@10, R@10, MRR) and LLM-as-judge
generation scores, then writes consolidated results to
outputs/evaluation_results.json.

Ablation configurations:
    0. No RAG        — No retrieval, LLM answers from training data only
    1. Baseline      — Vector only (k=5), no re-ranking, vanilla prompt
    2. +Prompt       — Vector only (k=5), no re-ranking, enhanced prompt
    3. +Hybrid       — Hybrid BM25+Vector (k=20→top 5 by RRF), no re-ranking, enhanced prompt
    4. +Rerank       — Vector only (k=20→top 5 by cross-encoder), enhanced prompt
    5. Enhanced      — Hybrid (k=20) → cross-encoder (→top 5) → enhanced prompt
"""

import json
import re
import sys
import time
from pathlib import Path

from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_EVALUATION_DIR,
    GENERATION_MODEL,
    OPENAI_API_KEY,
    OUTPUTS_DIR,
    RERANK_CANDIDATE_K,
    RERANK_TOP_K,
)
from retrieval import vector_search, hybrid_search
from reranker import rerank
from generation import generate_no_rag, generate_baseline, generate_enhanced


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

def load_test_set() -> list[dict]:
    """Load the evaluation test set from data/evaluation/test_set.json."""
    path = DATA_EVALUATION_DIR / "test_set.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Ablation pipeline runners
# ---------------------------------------------------------------------------

def run_config_no_rag(question: str) -> dict:
    """Config 0: No retrieval — LLM answers from training data only."""
    answer = generate_no_rag(question)
    return {"answer": answer, "chunks": [], "candidates": []}


def run_config_baseline(question: str) -> dict:
    """Config 1: Vector only (k=5) → vanilla prompt."""
    chunks = vector_search(question, top_k=5)
    answer = generate_baseline(question, chunks)
    return {"answer": answer, "chunks": chunks, "candidates": chunks}


def run_config_prompt_only(question: str) -> dict:
    """Config 2: Vector only (k=5) → enhanced prompt."""
    chunks = vector_search(question, top_k=5)
    answer = generate_enhanced(question, chunks)
    return {"answer": answer, "chunks": chunks, "candidates": chunks}


def run_config_hybrid(question: str) -> dict:
    """Config 3: Hybrid BM25+Vector (k=20 → top 5 by RRF) → enhanced prompt."""
    candidates = hybrid_search(question, top_n=RERANK_CANDIDATE_K)
    top5 = candidates[:RERANK_TOP_K]
    answer = generate_enhanced(question, top5)
    return {"answer": answer, "chunks": top5, "candidates": candidates}


def run_config_rerank(question: str) -> dict:
    """Config 4: Vector only (k=20 → cross-encoder → top 5) → enhanced prompt."""
    candidates = vector_search(question, top_k=RERANK_CANDIDATE_K)
    reranked = rerank(question, candidates, top_k=RERANK_TOP_K)
    answer = generate_enhanced(question, reranked)
    return {"answer": answer, "chunks": reranked, "candidates": candidates}


def run_config_enhanced(question: str) -> dict:
    """Config 5: Hybrid (k=20) → cross-encoder (→ top 5) → enhanced prompt."""
    candidates = hybrid_search(question, top_n=RERANK_CANDIDATE_K)
    reranked = rerank(question, candidates, top_k=RERANK_TOP_K)
    answer = generate_enhanced(question, reranked)
    return {"answer": answer, "chunks": reranked, "candidates": candidates}


CONFIGS = {
    "no_rag":       run_config_no_rag,
    "baseline":     run_config_baseline,
    "+prompt":      run_config_prompt_only,
    "+hybrid":      run_config_hybrid,
    "+rerank":      run_config_rerank,
    "enhanced":     run_config_enhanced,
}


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def chunk_is_relevant(chunk: dict, relevant_rule_codes: list[str]) -> bool:
    """
    Determine whether a retrieved chunk is relevant to the query.

    A chunk is relevant if any of the ground-truth rule codes appear in
    the chunk text or in its metadata (rule_number field).
    """
    if not relevant_rule_codes:
        return False

    text = chunk.get("text", "").upper()
    meta = chunk.get("metadata", {})
    rule_number = str(meta.get("rule_number", "")).upper()

    for code in relevant_rule_codes:
        code_upper = code.upper()
        code_no_space = code_upper.replace(" ", "")
        # Also try without the trailing type letter (R/G/D/E) since some
        # sourcebooks store "PRIN 2.1.1" rather than "PRIN 2.1.1R"
        code_base = code_upper.rstrip("RGDE")
        code_base_no_space = code_base.replace(" ", "")

        candidates = {code_upper, code_no_space, code_base, code_base_no_space}
        for variant in candidates:
            if variant in text or variant in rule_number:
                return True
    return False


def precision_at_k(chunks: list[dict], relevant_rule_codes: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    if not relevant_rule_codes:
        return 0.0
    top = chunks[:k]
    relevant_count = sum(1 for c in top if chunk_is_relevant(c, relevant_rule_codes))
    return relevant_count / k


def recall_at_k(chunks: list[dict], relevant_rule_codes: list[str], k: int = 5) -> float:
    """Fraction of relevant rule codes found in top-k chunks."""
    if not relevant_rule_codes:
        return 0.0
    top = chunks[:k]
    found = set()
    for code in relevant_rule_codes:
        for c in top:
            if chunk_is_relevant(c, [code]):
                found.add(code)
                break
    return len(found) / len(relevant_rule_codes)


def reciprocal_rank(chunks: list[dict], relevant_rule_codes: list[str]) -> float:
    """1 / rank of the first relevant chunk (0 if none found)."""
    if not relevant_rule_codes:
        return 0.0
    for i, c in enumerate(chunks):
        if chunk_is_relevant(c, relevant_rule_codes):
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# LLM-as-judge generation scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an expert evaluator for a Retrieval-Augmented Generation system that answers \
questions about UK FCA financial regulations.

You will be given:
- A QUESTION
- A GROUND TRUTH answer (the ideal/reference answer)
- A GENERATED answer (the system's output)
- The RETRIEVED CONTEXT chunks that were provided to the system

Score the generated answer on these four dimensions, each from 1 (worst) to 5 (best):

1. **Correctness**: Does the generated answer accurately convey the same information as \
the ground truth? (1 = completely wrong, 5 = fully correct)
2. **Groundedness**: Is every claim in the answer supported by the retrieved context? \
(1 = mostly unsupported/hallucinated, 5 = every claim traceable to context)
3. **Completeness**: Does the answer cover all the key points from the ground truth? \
(1 = misses almost everything, 5 = covers all key points)
4. **Citation accuracy**: Are specific FCA rule codes (e.g. COBS 4.2.1R) cited \
correctly and do they match what appears in the retrieved context? \
(1 = wrong/fabricated codes, 5 = all citations accurate and present in context)

Respond with ONLY a JSON object (no markdown, no explanation):
{{"correctness": <int>, "groundedness": <int>, "completeness": <int>, "citation_accuracy": <int>}}

---

QUESTION: {question}

GROUND TRUTH: {ground_truth}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER: {answer}

JSON scores:"""


def judge_answer(
    question: str,
    ground_truth: str,
    answer: str,
    chunks: list[dict],
    client: OpenAI,
) -> dict:
    """Use GPT-4o as a judge to score the generated answer on 4 dimensions."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", f"chunk_{i}")
        context_parts.append(f"[{cid}]\n{c['text'][:500]}")
    context_str = "\n---\n".join(context_parts)

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=context_str,
        answer=answer,
    )

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            break
        except RateLimitError:
            wait = 2 ** attempt
            print(f"\n    [rate limited, retrying in {wait}s]", end="", flush=True)
            time.sleep(wait)
    raw = response.choices[0].message.content.strip()

    # Parse JSON from the response — handle potential markdown wrapping
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {
            "correctness": 0,
            "groundedness": 0,
            "completeness": 0,
            "citation_accuracy": 0,
            "_parse_error": raw,
        }
    return scores


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(verbose: bool = True) -> dict:
    """
    Run all 5 ablation configs against the full test set.

    Returns a results dict and writes it to outputs/evaluation_results.json.
    """
    test_set = load_test_set()
    client = OpenAI(api_key=OPENAI_API_KEY)

    results = {
        "metadata": {
            "test_set_size": len(test_set),
            "configs": list(CONFIGS.keys()),
            "metrics": ["P@5", "R@5", "P@10", "R@10", "MRR",
                        "correctness", "groundedness",
                        "completeness", "citation_accuracy"],
        },
        "per_query": [],
        "aggregated": {},
    }

    # Per-config accumulators
    accum = {
        cfg: {
            "p5": [], "r5": [], "p10": [], "r10": [], "mrr": [],
            "correctness": [], "groundedness": [],
            "completeness": [], "citation_accuracy": [],
            "latency_ms": [],
        }
        for cfg in CONFIGS
    }

    for qi, test_case in enumerate(test_set):
        qid = test_case["id"]
        query = test_case["query"]
        category = test_case["category"]
        ground_truth = test_case["ground_truth_answer"]
        relevant_codes = test_case["relevant_rule_codes"]

        if verbose:
            print(f"\n{'='*70}")
            print(f"[{qi+1}/{len(test_set)}] {qid} ({category}): {query}")
            print(f"{'='*70}")

        query_result = {
            "id": qid,
            "query": query,
            "category": category,
            "relevant_rule_codes": relevant_codes,
            "configs": {},
        }

        for cfg_name, runner in CONFIGS.items():
            if verbose:
                print(f"  Running config: {cfg_name} ...", end=" ", flush=True)

            start = time.perf_counter()
            for attempt in range(5):
                try:
                    output = runner(query)
                    break
                except RateLimitError:
                    wait = 2 ** attempt
                    print(f"\n    [rate limited, retrying in {wait}s]", end="", flush=True)
                    time.sleep(wait)
            latency = (time.perf_counter() - start) * 1000

            chunks = output["chunks"]
            candidates = output["candidates"]
            answer = output["answer"]

            # -- Retrieval metrics --
            p5 = precision_at_k(chunks, relevant_codes, k=5)
            r5 = recall_at_k(chunks, relevant_codes, k=5)
            p10 = precision_at_k(candidates, relevant_codes, k=10)
            r10 = recall_at_k(candidates, relevant_codes, k=10)
            mrr = reciprocal_rank(chunks, relevant_codes)

            # -- Generation metrics (LLM-as-judge) --
            gen_scores = judge_answer(query, ground_truth, answer, chunks, client)

            if verbose:
                print(f"P@5={p5:.2f}  R@5={r5:.2f}  P@10={p10:.2f}  R@10={r10:.2f}  MRR={mrr:.2f}  "
                      f"Corr={gen_scores.get('correctness', '?')}  "
                      f"Ground={gen_scores.get('groundedness', '?')}  "
                      f"({latency:.0f}ms)")

            cfg_result = {
                "answer": answer,
                "chunk_ids": [c.get("chunk_id", "") for c in chunks],
                "retrieval": {"P@5": p5, "R@5": r5, "P@10": p10, "R@10": r10, "MRR": mrr},
                "generation": gen_scores,
                "latency_ms": round(latency, 1),
            }
            query_result["configs"][cfg_name] = cfg_result

            # Accumulate (exclude edge cases from retrieval metrics as they
            # have no relevant rule codes, making P/R/MRR undefined)
            if relevant_codes:
                accum[cfg_name]["p5"].append(p5)
                accum[cfg_name]["r5"].append(r5)
                accum[cfg_name]["p10"].append(p10)
                accum[cfg_name]["r10"].append(r10)
                accum[cfg_name]["mrr"].append(mrr)
            accum[cfg_name]["latency_ms"].append(latency)
            for metric in ["correctness", "groundedness", "completeness", "citation_accuracy"]:
                val = gen_scores.get(metric, 0)
                if isinstance(val, (int, float)):
                    accum[cfg_name][metric].append(val)

        results["per_query"].append(query_result)

    # -- Aggregate across all queries --
    for cfg_name, acc in accum.items():
        n_retrieval = len(acc["p5"]) or 1
        n_gen = len(acc["correctness"]) or 1
        n_total = len(acc["latency_ms"]) or 1
        results["aggregated"][cfg_name] = {
            "retrieval": {
                "P@5": round(sum(acc["p5"]) / n_retrieval, 4),
                "R@5": round(sum(acc["r5"]) / n_retrieval, 4),
                "P@10": round(sum(acc["p10"]) / n_retrieval, 4),
                "R@10": round(sum(acc["r10"]) / n_retrieval, 4),
                "MRR": round(sum(acc["mrr"]) / n_retrieval, 4),
            },
            "generation": {
                "correctness": round(sum(acc["correctness"]) / n_gen, 2),
                "groundedness": round(sum(acc["groundedness"]) / n_gen, 2),
                "completeness": round(sum(acc["completeness"]) / n_gen, 2),
                "citation_accuracy": round(sum(acc["citation_accuracy"]) / n_gen, 2),
            },
            "avg_latency_ms": round(sum(acc["latency_ms"]) / n_total, 1),
            "num_queries_retrieval": n_retrieval,
            "num_queries_generation": n_gen,
        }

    # -- Archive previous results, then write new results --
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "evaluation_results.json"
    drafts_dir = OUTPUTS_DIR / "drafts"
    drafts_dir.mkdir(exist_ok=True)

    if output_path.exists():
        # Find the next sequential number
        existing = sorted(drafts_dir.glob("evaluation_results_*.json"))
        next_num = 1
        if existing:
            last = existing[-1].stem  # e.g. "evaluation_results_3"
            try:
                next_num = int(last.rsplit("_", 1)[1]) + 1
            except (ValueError, IndexError):
                next_num = len(existing) + 1
        archive_path = drafts_dir / f"evaluation_results_{next_num}.json"
        output_path.rename(archive_path)
        if verbose:
            print(f"\nPrevious results archived to {archive_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\n{'='*70}")
        print("AGGREGATED RESULTS")
        print(f"{'='*70}")
        for cfg_name, agg in results["aggregated"].items():
            ret = agg["retrieval"]
            gen = agg["generation"]
            print(f"\n  {cfg_name:12s}  |  "
                  f"P@5={ret['P@5']:.3f}  R@5={ret['R@5']:.3f}  "
                  f"P@10={ret['P@10']:.3f}  R@10={ret['R@10']:.3f}  MRR={ret['MRR']:.3f}  |  "
                  f"Corr={gen['correctness']:.1f}  Grnd={gen['groundedness']:.1f}  "
                  f"Comp={gen['completeness']:.1f}  Cite={gen['citation_accuracy']:.1f}  |  "
                  f"{agg['avg_latency_ms']:.0f}ms")

        print(f"\nResults written to {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation(verbose=True)
