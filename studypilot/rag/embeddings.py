"""Embedding functions used by the vector store.

The default is Chroma's ONNX MiniLM model: same idea as
sentence-transformers/all-MiniLM-L6-v2, without installing PyTorch.
Tests pass a tiny keyword embedding instead so pytest never downloads a model.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

# 384-dimensional MiniLM. First call may download ~80MB into the Chroma cache.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings() -> Embeddings:
    """Return the embedding model used for real ingest and retrieve."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return _ChromaDefaultEmbeddings(DefaultEmbeddingFunction())


class _ChromaDefaultEmbeddings(Embeddings):
    """Adapt Chroma's embedding function to LangChain's Embeddings interface."""

    def __init__(self, function: object) -> None:
        self._function = function

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Chroma's ONNX model returns numpy float32. Cast to Python float so
        # langchain-chroma will accept the vectors.
        vectors = self._function(texts)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return [float(value) for value in self._function([text])[0]]
