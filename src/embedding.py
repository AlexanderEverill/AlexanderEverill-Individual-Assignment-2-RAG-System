"""
Embed chunks via OpenAI text-embedding-3-small and store in a ChromaDB
persistent collection ('fca_handbook').

Usage:
    python src/embedding.py          # chunk + embed + store
    python src/embedding.py --info   # print collection stats
"""

import sys
from pathlib import Path

import chromadb
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from chunking import chunk_all


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client rooted at CHROMA_DIR."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the fca_handbook collection with cosine distance."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    resp = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMS,
    )
    return [item.embedding for item in resp.data]


def build_vector_store(chunks: list[dict] | None = None) -> chromadb.Collection:
    """
    Embed all chunks and upsert into ChromaDB.

    If chunks is None, runs chunk_all() to generate them.
    Returns the populated ChromaDB collection.
    """
    if chunks is None:
        chunks = chunk_all()
    if not chunks:
        raise ValueError("No chunks to embed.")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)

    # Check if already populated with the same number of chunks
    existing_count = collection.count()
    if existing_count >= len(chunks):
        print(f"Collection already has {existing_count} items (>= {len(chunks)} chunks). "
              "Delete data/chroma/ to re-index.")
        return collection

    print(f"\nEmbedding {len(chunks)} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    for i in tqdm(range(0, len(chunks), EMBEDDING_BATCH_SIZE), desc="Embedding"):
        batch = chunks[i : i + EMBEDDING_BATCH_SIZE]

        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Stringify any None values in metadata (ChromaDB requires str/int/float/bool)
        for meta in metadatas:
            for k, v in meta.items():
                if v is None:
                    meta[k] = ""

        embeddings = embed_texts(texts, openai_client)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    final_count = collection.count()
    print(f"\nChromaDB collection '{CHROMA_COLLECTION_NAME}' now has {final_count} items.")
    return collection


def print_info() -> None:
    """Print stats about the existing ChromaDB collection."""
    chroma_client = get_chroma_client()
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION_NAME)
        print(f"Collection: {CHROMA_COLLECTION_NAME}")
        print(f"Items: {collection.count()}")
        # Peek at a sample
        sample = collection.peek(limit=3)
        if sample["ids"]:
            print(f"\nSample IDs: {sample['ids']}")
            for i, doc in enumerate(sample["documents"][:2]):
                print(f"\n--- {sample['ids'][i]} ---")
                print(doc[:200] + "..." if len(doc) > 200 else doc)
    except Exception as e:
        print(f"Collection not found: {e}")


if __name__ == "__main__":
    if "--info" in sys.argv:
        print_info()
    else:
        build_vector_store()
