"""PDF loading, chunking, embeddings, and retrieval.

This package is a fixed workflow, not an agent: every file goes through
load -> chunk -> embed -> store. The agent in Phase 3 will call retrieve().
"""

from studypilot.rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
from studypilot.rag.loader import load_directory, load_file
from studypilot.rag.store import RetrievedChunk, ingest_documents, retrieve

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "RetrievedChunk",
    "chunk_documents",
    "ingest_documents",
    "load_directory",
    "load_file",
    "retrieve",
]
