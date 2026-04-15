# FCA Handbook RAG Pipeline

A Retrieval-Augmented Generation system for answering questions about UK Financial Conduct Authority (FCA) regulatory documents. Built as part of IB9AU0 coursework at Warwick Business School (2025-26).

The system ingests FCA Handbook sourcebooks (PRIN, COBS, SYSC), chunks and indexes them, then retrieves relevant passages to generate grounded answers via GPT-4o. Six pipeline configurations — from a **no-RAG** baseline through to a fully **enhanced** pipeline — enable an ablation study isolating the contribution of each retrieval and generation technique.

## Architecture

```
FCA PDFs ──▶ Ingestion ──▶ Preprocessing ──▶ Chunking ──▶ Embedding
                                                             │
                                              ChromaDB ◀─────┤
                                              BM25 Index ◀───┘
                                                   │
                          Query ──▶ Retrieval (Vector / BM25 / Hybrid RRF)
                                        │
                                   Re-ranking (Cross-Encoder)
                                        │
                                   Generation (GPT-4o)
                                        │
                                     Answer
```

### Baseline vs Enhanced

| Component | Baseline | Enhanced |
|-----------|----------|----------|
| **Retrieval** | Vector-only (cosine, k=5) | Hybrid BM25 + Vector via Reciprocal Rank Fusion |
| **Re-ranking** | None | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Prompt** | Vanilla "answer from context" | Expert FCA persona with citation rules, Rule/Guidance distinction, step-by-step reasoning |

## Project Structure

```
├── src/
│   ├── config.py            # Central configuration (paths, models, hyperparams)
│   ├── data_ingestion.py    # Download FCA PDFs, spatial PDF extraction with rule-type rejoining
│   ├── preprocessing.py     # Unicode normalisation, artefact removal, rule tagging
│   ├── chunking.py          # Section-aware recursive chunking with overlap
│   ├── embedding.py         # OpenAI embeddings + ChromaDB storage
│   ├── bm25_index.py        # BM25Okapi index building and querying
│   ├── retrieval.py         # Vector, BM25, and hybrid RRF retrieval
│   ├── reranker.py          # Cross-encoder re-ranking
│   ├── generation.py        # No-RAG, baseline, and enhanced prompt templates + GPT-4o
│   ├── pipeline.py          # Orchestrates baseline/enhanced runs (CLI entry point)
│   └── evaluation.py        # 6-config ablation with retrieval + generation metrics
├── data/
│   ├── raw/                 # Ingested source text with YAML frontmatter
│   ├── processed/           # Cleaned and rule-tagged text
│   ├── chroma/              # ChromaDB persistent vector store
│   ├── bm25_index.pkl       # Pickled BM25 index
│   └── evaluation/
│       └── test_set.json    # 25 evaluation queries with ground truth
├── outputs/
│   └── evaluation_results.json
├── requirements.txt
└── PIPELINE_PLAN.md         # Detailed project plan and design document
```

## Setup

**Prerequisites:** Python 3.10+, an OpenAI API key.

```bash
# Clone the repository
git clone https://github.com/AlexanderEverill/Individual-Assignment-2-RAG-System.git
cd Individual-Assignment-2-RAG-System

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Build the index (first time only)

Run each stage sequentially to ingest, preprocess, and index the FCA Handbook:

```bash
python src/data_ingestion.py    # Download PDFs and extract text
python src/preprocessing.py     # Clean and tag text
python src/embedding.py         # Chunk, embed, and store in ChromaDB + BM25
```

### Query the system

```bash
# Enhanced pipeline (hybrid retrieval + re-ranking + expert prompt)
python -m src.pipeline --mode enhanced --query "What does COBS 4.2.1R require?"

# Baseline pipeline (vector-only retrieval + vanilla prompt)
python -m src.pipeline --mode baseline --query "What are the FCA Principles for Businesses?"
```

### Run the evaluation

Runs all 6 ablation configurations against the 25-query test set:

```bash
python src/evaluation.py
```

Results are written to `outputs/evaluation_results.json`.

## Evaluation

The evaluation framework tests 6 configurations in an ablation study across 25 queries spanning 6 categories (fact lookup, rule reference, cross-section reasoning, ambiguous, edge case, keyword). A no-RAG baseline (LLM only, no retrieval) is included to demonstrate the value of the RAG pipeline itself.

**Retrieval metrics:** Precision@5, Recall@5, Precision@10, Recall@10, MRR

**Generation metrics (LLM-judge, 1-5 scale):** Correctness, Groundedness, Completeness, Citation Accuracy

### Results Summary

| Configuration | P@5 | R@5 | MRR | Correctness | Groundedness | Citation Acc. |
|---------------|------|------|------|-------------|--------------|---------------|
| `no_rag` | — | — | — | 3.08 | 1.00 | 1.00 |
| `baseline` | 0.270 | 0.457 | 0.568 | 1.60 | 1.72 | 1.68 |
| `+prompt` | 0.270 | 0.457 | **0.568** | 3.36 | 3.84 | 3.88 |
| `+hybrid` | 0.310 | 0.506 | 0.567 | 3.44 | 3.68 | 3.76 |
| `+rerank` | 0.330 | 0.503 | 0.458 | 3.36 | 3.72 | 3.64 |
| `enhanced` | **0.350** | **0.542** | 0.499 | 3.28 | **3.88** | **3.92** |

**Key findings:**
- The no-RAG baseline achieves reasonable correctness (3.08) from LLM training data alone, but scores 1.0 on groundedness and citation accuracy — confirming that RAG is essential for verifiable, source-grounded answers.
- The enhanced prompt alone provides the largest single improvement to generation quality (correctness 1.60 &rarr; 3.36).
- Hybrid retrieval (BM25 + Vector with RRF) yields meaningful recall gains over vector-only retrieval.
- The fully enhanced pipeline achieves the highest groundedness (3.88) and citation accuracy (3.92).

## Changes Since Initial Pipeline

1. **Spatial PDF extraction with rule-type rejoining** — FCA Handbook PDFs place rule type letters (R/G/D/E) in a separate column from the rule number. Standard text extraction lost this association. `data_ingestion.py` now uses PyMuPDF's dictionary-mode extraction to spatially match type letters to their rule headings based on x/y coordinates, producing cleaner source text (e.g. `COBS 4.2.1R` instead of `COBS 4.2.1` with an orphaned `R`).
2. **Improved rule-type fallback in chunking** — `chunking.py` now falls back to the trailing letter on the rule number itself (e.g. the `R` in `COBS 4.2.1R`) when no `[RULE:X]` preprocessing tag is present, improving rule-type metadata coverage.
3. **Manually validated evaluation questions** — The 25-query test set was reviewed and corrected to ensure ground-truth answers and relevant rule codes accurately reflect the FCA Handbook content.
4. **No-RAG baseline config** — Added a 6th ablation configuration (`no_rag`) that queries GPT-4o directly without any retrieval, serving as a true baseline to demonstrate the value of the RAG pipeline.
5. **Re-run evaluation** — All data was re-ingested, re-embedded, and re-evaluated with the improved pipeline across all 6 configurations, producing updated metrics.

## Tech Stack

- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims)
- **Vector Store:** ChromaDB (cosine distance)
- **Keyword Search:** BM25Okapi (`rank-bm25`)
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers)
- **Generation:** GPT-4o (temperature=0.2)
- **PDF Extraction:** PyMuPDF
- **Tokenisation:** tiktoken (cl100k_base)

## Data Sources

Three FCA Handbook sourcebooks retrieved from the FCA API:

| Sourcebook | Full Name |
|------------|-----------|
| **PRIN** | Principles for Businesses |
| **COBS** | Conduct of Business Sourcebook |
| **SYSC** | Senior Management Arrangements, Systems and Controls |
