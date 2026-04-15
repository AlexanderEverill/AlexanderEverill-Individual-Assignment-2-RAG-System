# FCA Regulatory RAG Pipeline — Project Plan

## IB9AU0 Individual Assignment 2 | Warwick Business School 2025–2026

---

## 1. Project overview

**Domain:** UK Financial Conduct Authority (FCA) regulatory documents.

**Why RAG is necessary (not just prompting):**

- FCA rules are updated monthly via Handbook Notices and legal instruments. LLMs trained on static data quickly become stale on specific rule wording, effective dates, and newly introduced obligations (e.g. Consumer Duty rules effective July 2023, ongoing updates through 2024–2025). Note: a RAG corpus is also a point-in-time snapshot, but unlike model weights it is an explicit, inspectable data layer that can be refreshed by re-running the ingestion pipeline — minutes of work rather than months of retraining. See Section 12 (Update Strategy) for how this would work in practice.
- Regulatory text is extremely precise — a single word ("must" vs "should", "rule" vs "guidance") changes legal obligation. LLMs paraphrase and lose this precision without grounding. RAG retrieves exact source text, so the LLM can quote rather than guess.
- Rule codes (e.g. COBS 4.2.1R, PRIN 2A.1.1R, SYSC 10.1.8G) are opaque identifiers that LLMs cannot reliably map to content without retrieval.
- Hallucination risk is unacceptable in compliance — a wrong rule reference could lead to regulatory breach. RAG grounds every answer in traceable source chunks, making claims verifiable.

**Advanced techniques implemented:**

1. **Hybrid Search (BM25 + Vector)** — Retrieval stage. Combines keyword precision (for exact rule codes, defined terms) with semantic recall (for conceptual questions).
2. **Cross-Encoder Re-ranking** — Post-retrieval stage. Uses a cross-encoder model to re-score and reorder retrieved chunks for precise relevance before passing to the LLM.

---

## 2. Data sourcing

**Source:** FCA Handbook — freely available at https://www.handbook.fca.org.uk/

**Specific sourcebooks to ingest (scope):**

| Sourcebook | Full name | Why included |
|---|---|---|
| PRIN | Principles for Businesses | Core principles (11 + Consumer Duty PRIN 2A). Foundation of FCA regulation. |
| COBS | Conduct of Business Sourcebook | Client interaction rules. Rich in specific obligations, rule codes, and defined terms. |
| SYSC | Senior Management Arrangements, Systems and Controls | Governance rules. Good for testing cross-referencing questions. |
| Consumer Duty (cross-cutting) | PRIN 2A + related rules | Recent regulation (2023–2025). Unlikely to be well-represented in LLM training data. |

**Data collection method:**

- Download HTML/text content from the FCA Handbook website for each sourcebook section.
- Store raw text files in `data/raw/` with metadata (sourcebook, chapter, section, effective date).
- Also collect 2–3 recent FCA Policy Statements (PDFs) related to Consumer Duty for additional niche content.

**Target corpus size:** ~200–400 pages of regulatory text, yielding approximately 1,000–2,500 chunks.

---

## 3. Tech stack

| Component | Technology | Version/Model |
|---|---|---|
| Language | Python | 3.11+ |
| Embeddings | OpenAI | `text-embedding-3-small` (1536 dims) |
| Vector store | ChromaDB | Latest (open-source, local) |
| BM25 keyword search | `rank_bm25` | Python package |
| Re-ranker | `sentence-transformers` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (generation) | OpenAI | `gpt-4o` (both baseline and enhanced) |
| PDF parsing | `PyMuPDF` (fitz) | For policy statement PDFs |
| Orchestration | Custom Python | No framework dependency (LangChain optional) |

**Environment:** Local Python environment. API key required for OpenAI only.

---

## 4. Project structure

```
fca-rag-pipeline/
├── README.md                    # Setup instructions & how to run
├── PIPELINE_PLAN.md             # This document
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for API keys
│
├── data/
│   ├── raw/                     # Raw downloaded FCA text/HTML files
│   ├── processed/               # Cleaned text ready for chunking
│   └── evaluation/              # Test set queries + ground truth (JSON)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Central config (paths, model names, hyperparams)
│   ├── data_ingestion.py        # Download/scrape FCA handbook sections
│   ├── preprocessing.py         # Clean text, normalise formatting
│   ├── chunking.py              # Recursive chunking with section awareness
│   ├── embedding.py             # Embed chunks via OpenAI, store in ChromaDB
│   ├── bm25_index.py            # Build and query BM25 index
│   ├── retrieval.py             # Vector search, BM25 search, hybrid fusion
│   ├── reranker.py              # Cross-encoder re-ranking
│   ├── generation.py            # LLM generation with prompt templates
│   ├── pipeline.py              # Orchestrates full RAG pipeline (baseline + enhanced)
│   └── evaluation.py            # Runs eval metrics on test set
│
├── outputs/
│   ├── evaluation_results.json  # Full evaluation output
│   ├── demo_log.md              # Sample inputs + outputs + commentary
│   └── report.docx              # 1,500-word report
```

---

## 5. Pipeline architecture

### 5A. Baseline pipeline

```
User Query
    │
    ▼
[Embed query] ──► OpenAI text-embedding-3-small
    │
    ▼
[Vector search] ──► ChromaDB top-k (k=5) cosine similarity
    │
    ▼
[Stuff into prompt] ──► Vanilla prompt template + retrieved chunks
    │
    ▼
[Generate] ──► GPT-4o
    │
    ▼
Answer
```

- No query transformation.
- Pure vector retrieval only.
- Simple prompt: "Answer the question based on the context provided."
- No re-ranking.

### 5B. Enhanced pipeline

```
User Query
    │
    ▼
[Embed query] ──► OpenAI text-embedding-3-small
    │
    ├──► [Vector search] ──► ChromaDB top-k (k=10)
    │
    ├──► [BM25 search] ──► rank_bm25 top-k (k=10)
    │
    ▼
[Reciprocal Rank Fusion] ──► Merge + deduplicate, top-20 candidates
    │
    ▼
[Cross-Encoder Re-ranking] ──► ms-marco-MiniLM scores each (query, chunk) pair
    │
    ▼
[Select top-5 by re-rank score]
    │
    ▼
[Stuff into prompt] ──► Enhanced prompt template (grounding instructions,
    │                      source citations, CoT reasoning)
    ▼
[Generate] ──► GPT-4o
    │
    ▼
Answer (with source references)
```

**Key differences from baseline:**

1. Dual retrieval (vector + BM25) with Reciprocal Rank Fusion.
2. Wider initial candidate pool (k=20 combined vs k=5).
3. Cross-encoder re-ranking to precision-filter candidates.
4. Enhanced prompt template with grounding instructions and citation requirements.

---

## 6. Module specifications

### 6.1 Data ingestion (`data_ingestion.py`)

- Fetch FCA Handbook sections via HTTPS (handbook.fca.org.uk).
- Parse HTML to extract rule text, stripping navigation/boilerplate.
- For PDF policy statements: extract text via PyMuPDF.
- Output: one `.txt` file per section with YAML-style frontmatter metadata:

```
---
sourcebook: COBS
chapter: 4
section: 4.2
title: Fair, clear and not misleading communications
effective_date: 2024-01-01
url: https://www.handbook.fca.org.uk/handbook/COBS/4/2.html
---
[rule text here]
```

### 6.2 Preprocessing (`preprocessing.py`)

- Normalise whitespace and Unicode.
- Preserve rule numbering (e.g. "4.2.1R", "4.2.2G") — these are critical identifiers.
- Remove repeated headers/footers from PDF extractions.
- Tag each paragraph with its rule type: R (Rule), G (Guidance), D (Direction), E (Evidential provision).

### 6.3 Chunking (`chunking.py`)

**Strategy:** Recursive chunking with section-awareness.

- Primary split: by regulatory section boundaries (e.g. each numbered rule/guidance block = one chunk).
- If a section exceeds 512 tokens: recursively split by paragraph, then sentence.
- Chunk overlap: 50 tokens (to preserve cross-sentence context).
- Each chunk stores metadata: `{sourcebook, chapter, section, rule_number, rule_type, effective_date}`.

**Rationale:** FCA rules are naturally structured with clear section numbers. Splitting at these boundaries preserves semantic coherence. Recursive fallback handles unusually long sections.

### 6.4 Embedding & vector store (`embedding.py`)

- Embed all chunks using OpenAI `text-embedding-3-small` (1536 dimensions).
- Store in ChromaDB persistent collection with metadata filters enabled.
- Collection name: `fca_handbook`.
- Batch embedding calls (max 100 chunks per API call) for efficiency.

### 6.5 BM25 index (`bm25_index.py`)

- Tokenise all chunk texts (lowercase, basic punctuation handling).
- Build a `BM25Okapi` index from `rank_bm25`.
- Persist tokenised corpus alongside ChromaDB so both are queryable.
- At query time: tokenise query, retrieve top-k scores.

### 6.6 Retrieval (`retrieval.py`)

Three retrieval modes:

1. **Vector only** (baseline): `ChromaDB.query(query_embedding, n_results=5)`.
2. **BM25 only** (for ablation): `bm25.get_top_n(tokenised_query, n=10)`.
3. **Hybrid** (enhanced): Run both vector (k=10) and BM25 (k=10), merge via Reciprocal Rank Fusion.

**Reciprocal Rank Fusion (RRF):**

```
RRF_score(doc) = Σ 1 / (k + rank(doc))
```

where `k = 60` (standard constant). Take union of results, compute RRF score for each, sort descending.

### 6.7 Re-ranking (`reranker.py`)

- Load `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers.
- For each candidate chunk: score `(query, chunk_text)` pair.
- Sort by cross-encoder score descending.
- Return top-5 chunks.

**Performance note:** The cross-encoder processes ~20 chunks in ~0.3s on CPU. Acceptable latency for this project.

### 6.8 Generation (`generation.py`)

**No-RAG mode:** Sends the question directly to GPT-4o with no retrieved context, serving as the true baseline for the ablation study.

**Baseline prompt template:**

```
Answer the following question based on the provided context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:
```

**Enhanced prompt template:**

```
You are an expert UK financial regulation assistant specialising in the FCA Handbook.
Answer the question using ONLY the retrieved regulatory context below. Follow these rules:
1. Cite specific rule numbers (e.g. COBS 4.2.1R) when referencing obligations.
2. Distinguish between Rules (R — mandatory) and Guidance (G — non-binding).
3. If the context does not contain enough information to answer fully, state what is missing.
4. Be precise with regulatory language — do not paraphrase "must" as "should".

Retrieved context (ordered by relevance):
---
{context_with_sources}
---

Question: {question}

Provide a clear, grounded answer with rule references:
```

### 6.9 Pipeline orchestration (`pipeline.py`)

Exposes two callable functions:

```python
def run_baseline(query: str) -> dict:
    """Vector-only retrieval + vanilla prompt."""

def run_enhanced(query: str) -> dict:
    """Hybrid retrieval + re-ranking + enhanced prompt."""
```

Both return:

```python
{
    "query": str,
    "retrieved_chunks": list[dict],  # id, text, metadata, score
    "answer": str,
    "latency_ms": float,
    "pipeline": "baseline" | "enhanced"
}
```

---

## 7. Evaluation framework

### 7.1 Test set design (`data/evaluation/test_set.json`)

Minimum **15 queries** across these categories:

| Category | Count | Example |
|---|---|---|
| Simple fact lookup | 3–4 | "What does PRIN 2 (Skill, Care and Diligence) require?" |
| Specific rule reference | 3–4 | "What does COBS 4.2.1R say about financial promotions?" |
| Cross-section reasoning | 2–3 | "How do the Consumer Duty rules in PRIN 2A relate to COBS 4?" |
| Ambiguous / vague query | 2–3 | "What are the rules about treating customers fairly?" |
| Edge case / out-of-scope | 2 | "What are the FCA rules on cryptocurrency staking?" (may not be in corpus) |
| Exact-term keyword match | 2 | "What is an 'eligible counterparty' under COBS?" |

Each query has:

```json
{
    "id": "Q01",
    "query": "...",
    "category": "simple_fact | rule_reference | cross_section | ambiguous | edge_case | keyword",
    "ground_truth_answer": "...",
    "relevant_rule_codes": ["COBS 4.2.1R", "COBS 4.2.2G"],
    "relevant_chunk_ids": ["chunk_042", "chunk_043"]
}
```

### 7.2 Retrieval metrics

| Metric | What it measures | How computed |
|---|---|---|
| Precision@5 | Fraction of top-5 retrieved chunks that are relevant | `relevant_in_top5 / 5` |
| Recall@5 | Fraction of all relevant chunks found in top-5 | `relevant_in_top5 / total_relevant` |
| MRR (Mean Reciprocal Rank) | How high the first relevant chunk ranks | `1 / rank_of_first_relevant` |

### 7.3 Generation metrics

| Metric | What it measures | How evaluated |
|---|---|---|
| Correctness | Does the answer match ground truth? | LLM-as-judge (GPT-4o scores 1–5) + manual spot-check |
| Groundedness | Is every claim supported by retrieved context? | LLM-as-judge: check for unsupported statements |
| Completeness | Does the answer cover all relevant aspects? | Compare against ground truth answer components |
| Citation accuracy | Are rule codes cited correctly? | Automated: check cited codes exist in retrieved chunks |

### 7.4 Ablation study

Run 6 configurations to isolate contributions:

| Config | Retrieval | Re-ranking | Prompt |
|---|---|---|---|
| No RAG | None (LLM only) | None | Direct question |
| Baseline | Vector only (k=5) | None | Vanilla |
| +Prompt | Vector only (k=5) | None | Enhanced |
| +Hybrid | Hybrid BM25+Vector (k=20→5) | None | Enhanced |
| +Rerank | Vector only (k=20→5) | Cross-encoder | Enhanced |
| Enhanced (full) | Hybrid (k=20) | Cross-encoder (→top 5) | Enhanced |

The no-RAG config establishes a true baseline (LLM from training data only), demonstrating the value added by retrieval-augmented generation. The remaining configs let us attribute improvements: prompt engineering vs hybrid retrieval vs re-ranking.

---

## 8. Deliverables

### 8.1 Code (`src/` directory)

- Fully modular, documented Python code.
- `requirements.txt` with pinned versions.
- `README.md` with setup + run instructions.
- Runnable end-to-end: `python -m src.pipeline --mode enhanced --query "..."`.

### 8.2 Report (`outputs/report.docx`) — ~1,500 words

**Structure:**

1. **Introduction & domain justification** (~200 words) — Why FCA regulations need RAG. Why standard LLMs fall short.
2. **System design** (~400 words) — Architecture overview, chunking strategy rationale, why hybrid search + re-ranking were chosen (reference the technique comparison analysis).
3. **Results & evaluation** (~500 words) — Retrieval metrics table (P@5, R@5, MRR for each config). Generation quality scores. Ablation analysis showing where each technique contributes.
4. **Analysis & reflection** (~400 words) — Failure modes (e.g. cross-sourcebook questions, ambiguous queries, missing data). Trade-offs (latency, cost, complexity). Proposed improvements (query expansion, metadata filtering, keeping corpus updated). Limitations (static corpus, no table/chart understanding).

### 8.3 Demo log (`outputs/demo_log.md`)

5–8 example interactions showing:

```
## Query 1: [category label]
**Input:** "What does COBS 4.2.1R require for financial promotions?"
**Pipeline:** Enhanced

### Retrieved chunks (top 3):
1. [COBS 4.2.1R] "A firm must ensure that a communication..." (score: 0.94)
2. [COBS 4.2.2G] "In the FCA's view, the fair, clear and..." (score: 0.88)
3. [COBS 4.1.1R] "This chapter applies to..." (score: 0.72)

### Generated answer:
"COBS 4.2.1R requires that a firm must ensure that a communication
or financial promotion is fair, clear and not misleading..."

### Commentary:
- Retrieval correctly identified the exact rule and supporting guidance.
- Baseline missed COBS 4.2.2G (guidance) because it relies on vector similarity
  which favoured a chunk from COBS 6 about disclosure instead.
- Enhanced pipeline cited the correct rule number with exact wording.
```

Include at least one failure case with analysis of why it failed.

---

## 9. Implementation phases

| Phase | Tasks | Estimated effort |
|---|---|---|
| **Phase 1: Data** | Download FCA sections, parse HTML/PDF, store raw files | 1 session |
| **Phase 2: Preprocessing** | Clean text, build chunks, generate embeddings, populate ChromaDB + BM25 | 1 session |
| **Phase 3: Baseline** | Implement vector retrieval + vanilla generation. Verify end-to-end. | 1 session |
| **Phase 4: Enhanced** | Add BM25, RRF fusion, cross-encoder re-ranking, enhanced prompts | 1 session |
| **Phase 5: Evaluation** | Build test set, run all 4 configs, compute metrics, generate results | 1 session |
| **Phase 6: Deliverables** | Write report, produce demo log, final code cleanup | 1 session |

---

## 10. Key design decisions (for report reference)

1. **Recursive chunking over fixed-size:** FCA rules have natural section boundaries. Splitting at these preserves semantic units. Fixed-size chunking would arbitrarily cut rules mid-sentence.

2. **Hybrid search over pure vector:** FCA queries often contain exact rule codes (COBS 4.2.1R) that vector search cannot match precisely. BM25 catches these while vectors handle conceptual similarity.

3. **Cross-encoder re-ranking over no re-ranking:** Initial retrieval (especially BM25) produces false positives. The cross-encoder reads query + chunk together for a more accurate relevance judgement, ensuring the LLM receives only the most pertinent context.

4. **Two techniques (not four):** Implementing one from every RAG stage would make it impossible to attribute improvements. Two complementary techniques from different stages allow clean ablation.

5. **Enhanced prompt with citation instructions:** Regulatory answers without rule references are useless to compliance professionals. The prompt enforces citation and distinguishes between mandatory rules (R) and non-binding guidance (G).

---

## 11. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| FCA website structure changes | Cannot scrape data | Download and cache all raw data locally early. Manual copy as fallback. |
| OpenAI API rate limits / cost | Slow evaluation, high spend | Batch embedding calls. Cache results. |
| Cross-encoder model too large | Slow on CPU | Use MiniLM variant (~80MB). Acceptable latency (~0.3s for 20 chunks). |
| Corpus too small for recall metrics | Unreliable evaluation | Ensure minimum 1,000 chunks across 3+ sourcebooks. |
| Ground truth labelling is subjective | Inconsistent evaluation | Use multiple annotators (self + LLM-as-judge). Report inter-rater agreement. |

---

## 12. Update strategy (corpus freshness)

A RAG corpus is a point-in-time snapshot. Left unmanaged, it becomes stale just as an LLM's training data does — the critical difference is that refreshing a RAG corpus is a lightweight, controllable operation rather than a full model retrain. Below is the realistic maintenance approach for a production version of this system.

**Refresh cycle:** The FCA publishes Handbook Notices monthly, summarising all rule changes. A monthly refresh aligned to these notices would keep the corpus current.

**Refresh pipeline:**

1. **Detect changes:** Download the latest Handbook Notice PDF. Parse it to identify which sourcebook sections have been amended, added, or revoked.
2. **Re-ingest affected sections:** Re-run `data_ingestion.py` for only the changed sections (not the full corpus).
3. **Re-chunk and re-embed:** Process the updated raw text through `preprocessing.py`, `chunking.py`, and `embedding.py`. Replace the affected chunks in ChromaDB and rebuild the BM25 index.
4. **Version tag:** Store a `corpus_version` metadata field (e.g. `2025-06-HN112`) so queries can report which version of the rules they are grounded in.

**Estimated effort per refresh:** ~10 minutes of compute time for a typical monthly update affecting 5–15 sections. No model retraining, no downtime.

**Scope for this assignment:** The submitted system uses a static snapshot (appropriate for a coursework deliverable). The architecture above is documented as a realistic production extension and is discussed in the report's limitations section.

---

## 13. Dependencies (`requirements.txt`)

```
openai>=1.30.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
rank-bm25>=0.2.2
pymupdf>=1.24.0
beautifulsoup4>=4.12.0
requests>=2.31.0
tiktoken>=0.7.0
python-dotenv>=1.0.0
tqdm>=4.66.0
numpy>=1.26.0
pandas>=2.2.0
```

---

*This plan is the authoritative reference for building the FCA RAG pipeline. All implementation chats should follow this structure, file layout, and architectural decisions.*
