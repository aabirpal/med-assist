"""
knowledge_base/vectorstore.py
Builds a persistent ChromaDB collection from DOCUMENTS with chunking.

CHUNKING STRATEGY:
Each document is split into overlapping word-based chunks before embedding.
This means retrieval returns the specific paragraph that answers the question
rather than an entire 1000-word document, improving faithfulness scores.

PERSISTENCE:
Collection is stored on disk at CHROMA_PATH (./chroma_db by default).
First run builds the index. Subsequent runs load it instantly.

TO FORCE REBUILD:
Delete the ./chroma_db folder and restart.
This is necessary after editing knowledge base documents.
"""

import os
import hashlib
import json
import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, COLLECTION, CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from knowledge_base.loader import DOCUMENTS


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:       Full document text.
        chunk_size: Target number of words per chunk.
        overlap:    Number of words to repeat between adjacent chunks.
                    Prevents answers from being cut mid-sentence at boundaries.
    Returns:
        List of chunk strings.
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max(1, chunk_size - overlap)   # step forward, keeping overlap

    return chunks


def _documents_fingerprint(docs: list) -> str:
    """
    Compute an MD5 fingerprint of the documents list.
    Used to detect whether the knowledge base documents have changed since the last build.
    """
    raw = json.dumps([{"id": d["id"], "topic": d["topic"], "text": d["text"]}
                      for d in docs], sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _fingerprint_path() -> str:
    return os.path.join(CHROMA_PATH, ".doc_fingerprint")


def _saved_fingerprint() -> str:
    path = _fingerprint_path()
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ""


def _save_fingerprint(fp: str) -> None:
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(_fingerprint_path(), "w") as f:
        f.write(fp)


# ── Build or load ─────────────────────────────────────────────────────────────

def build_collection(force_rebuild: bool = False) -> tuple:
    """
    Load the persistent ChromaDB collection, or build it if:
      - It doesn't exist yet, OR
      - documents have changed (fingerprint mismatch), OR
      - force_rebuild=True

    Returns:
        (embedder, collection)
    """
    current_fp = _documents_fingerprint(DOCUMENTS)
    saved_fp   = _saved_fingerprint()

    needs_build = (
        force_rebuild
        or current_fp != saved_fp
        or not os.path.exists(CHROMA_PATH)
    )

    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if needs_build:
        reason = "force rebuild" if force_rebuild else (
            "documents changed" if current_fp != saved_fp else "first run"
        )
        print(f"Building ChromaDB index ({reason})...")

        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

        collection = client.create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        all_texts      = []
        all_ids        = []
        all_metadatas  = []

        for doc in DOCUMENTS:
            chunks = _chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
            for j, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{j:03d}"
                all_texts.append(chunk)
                all_ids.append(chunk_id)
                all_metadatas.append({
                    "topic":    doc["topic"],
                    "doc_id":   doc["id"],
                    "chunk":    j,
                    "n_chunks": len(chunks),
                })

        # Batch encode — faster than one-by-one
        print(f"Embedding {len(all_texts)} chunks from {len(DOCUMENTS)} documents...")
        embeddings = embedder.encode(all_texts, show_progress_bar=True).tolist()

        # ChromaDB recommends batches ≤ 5000
        batch_size = 500
        for i in range(0, len(all_texts), batch_size):
            collection.add(
                documents=all_texts[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                ids=all_ids[i:i + batch_size],
                metadatas=all_metadatas[i:i + batch_size],
            )

        _save_fingerprint(current_fp)
        print(f"✅ Index built: {collection.count()} chunks from {len(DOCUMENTS)} documents")

    else:
        collection = client.get_collection(COLLECTION)
        print(f"✅ Loaded existing index: {collection.count()} chunks from {len(DOCUMENTS)} documents")

    return embedder, collection


# ── Build once on import ──────────────────────────────────────────────────────
embedder, collection = build_collection()
