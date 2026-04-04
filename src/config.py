import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_EVALUATION_DIR = ROOT_DIR / "data" / "evaluation"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
BM25_INDEX_PATH = ROOT_DIR / "data" / "bm25_index.pkl"

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536
GENERATION_MODEL = "gpt-4o"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- ChromaDB ---
CHROMA_COLLECTION_NAME = "fca_handbook"

# --- Chunking ---
CHUNK_MAX_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

# --- Retrieval ---
BASELINE_TOP_K = 5
HYBRID_TOP_K = 10        # per retriever (vector + BM25) before fusion
RERANK_CANDIDATE_K = 20  # pool size fed into cross-encoder
RERANK_TOP_K = 5         # final chunks passed to LLM

# --- BM25 ---
RRF_K = 60               # standard RRF constant

# --- Embedding batching ---
EMBEDDING_BATCH_SIZE = 100

# --- FCA Handbook PDF download targets ---
FCA_SOURCEBOOKS = {
    "PRIN": "https://api-handbook.fca.org.uk/files/sourcebook/PRIN.pdf",
    "COBS": "https://api-handbook.fca.org.uk/files/sourcebook/COBS.pdf",
    "SYSC": "https://api-handbook.fca.org.uk/files/sourcebook/SYSC.pdf",
}