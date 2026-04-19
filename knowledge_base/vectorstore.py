"""
knowledge_base/vectorstore.py
Builds the ChromaDB in-memory collection from DOCUMENTS.
Imported once at startup; shared across notebook, eval, and app.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, COLLECTION
from knowledge_base.documents import DOCUMENTS


def build_collection() -> tuple:
    """
    Encode all documents and load them into a fresh ChromaDB collection.
    Returns (embedder, collection).
    """
    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.Client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    texts     = [d["text"]  for d in DOCUMENTS]
    ids       = [d["id"]    for d in DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in DOCUMENTS]

    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=ids,
        metadatas=metadatas,
    )

    print(f"✅ Knowledge base ready: {collection.count()} documents")
    return embedder, collection


embedder, collection = build_collection()
