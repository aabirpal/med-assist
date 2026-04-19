"""knowledge_base — public exports."""
from knowledge_base.documents import DOCUMENTS
from knowledge_base.vectorstore import embedder, collection

__all__ = ["DOCUMENTS", "embedder", "collection"]
