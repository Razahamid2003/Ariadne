"""Retrieval package.

Purpose
-------
Finds the most relevant evidence for a query by combining keyword search, semantic
vector search, optional reranking, and a document-type signal, then merging them
into one ranked list.
"""

from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.keyword_index import KeywordSearchIndex
from backend.app.retrieval.models import HybridSearchRequest, HybridSearchResponse, RetrievalCandidate

__all__ = [
    "HybridRetriever",
    "KeywordSearchIndex",
    "HybridSearchRequest",
    "HybridSearchResponse",
    "RetrievalCandidate",
]
