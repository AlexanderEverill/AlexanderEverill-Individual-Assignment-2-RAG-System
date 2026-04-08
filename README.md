# FCA Handbook RAG Pipeline

A Retrieval-Augmented Generation system for answering questions about UK Financial Conduct Authority (FCA) regulatory documents. Built as part of IB9AU0 coursework at Warwick Business School (2025-26).

The system ingests FCA Handbook sourcebooks (PRIN, COBS, SYSC), chunks and indexes them, then retrieves relevant passages to generate grounded answers via GPT-4o. Two pipeline configurations — **baseline** and **enhanced** — enable an ablation study comparing retrieval and generation strategies.

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
│   ├── data_ingestion.py    # Download FCA PDFs, extract text via PyMuPDF
│   ├── preprocessing.py     # Unicode normalisation, artefact removal, rule tagging
│   ├── chunking.py          # Section-aware recursive chunking with overlap
│   ├── embedding.py         # OpenAI embeddings + ChromaDB storage
│   ├── bm25_index.py        # BM25Okapi index building and querying
│   ├── retrieval.py         # Vector, BM25, and hybrid RRF retrieval
│   ├── reranker.py          # Cross-encoder re-ranking
│   ├── generation.py        # Baseline and enhanced prompt templates + GPT-4o
│   ├── pipeline.py          # Orchestrates baseline/enhanced runs (CLI entry point)
│   └── evaluation.py        # 5-config ablation with retrieval + generation metrics
├── data/
│   ├── raw/                 # Ingested source text with YAML frontmatter
│   ├── processed/           # Cleaned and rule-tagged text
│   ├── chroma/              # ChromaDB persistent vector store
│   ├── bm25_index.pkl       # Pickled BM25 index
│   └── evaluation/
│       └── test_set.json    # 25 evaluation queries with ground truth
├── outputs/
│   └── evaluation_results.json
├── notebooks/
│   └── exploration.ipynb
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

Runs all 5 ablation configurations against the 25-query test set:

```bash
python src/evaluation.py
```

Results are written to `outputs/evaluation_results.json`.

## Evaluation

The evaluation framework tests 5 configurations in an ablation study across 25 queries spanning 6 categories (fact lookup, rule reference, cross-section reasoning, ambiguous, edge case, keyword).

**Retrieval metrics:** Precision@5, Recall@5, Precision@10, Recall@10, MRR

**Generation metrics (LLM-judge, 1-5 scale):** Correctness, Groundedness, Completeness, Citation Accuracy

### Results Summary

| Configuration | P@5 | R@5 | MRR | Correctness | Groundedness | Citation Acc. |
|---------------|------|------|------|-------------|--------------|---------------|
| `baseline` | 0.224 | 0.368 | 0.399 | 1.52 | 1.64 | 1.52 |
| `+prompt` | 0.224 | 0.368 | 0.399 | 3.36 | 3.92 | 4.04 |
| `+hybrid` | 0.256 | 0.424 | 0.536 | 3.36 | 3.80 | 3.60 |
| `+rerank` | 0.280 | 0.390 | 0.425 | 3.44 | 3.92 | 3.72 |
| `enhanced` | **0.320** | **0.524** | 0.446 | 3.40 | **4.04** | **4.00** |

**Key findings:**
- The enhanced prompt alone provides the largest single improvement to generation quality (correctness 1.52 &rarr; 3.36).
- Hybrid retrieval (BM25 + Vector with RRF) yields the best MRR (0.536) and meaningful recall gains.
- The full enhanced pipeline achieves the highest precision, recall, and groundedness overall.

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
