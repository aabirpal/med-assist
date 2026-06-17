"""knowledge_base — public exports."""
from knowledge_base.loader import DOCUMENTS
from knowledge_base.vectorstore import embedder, collection, build_collection

__all__ = ["DOCUMENTS", "embedder", "collection", "build_collection"]
