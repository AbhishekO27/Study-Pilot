"""Write chunks to Chroma and retrieve the nearest ones with citations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from studypilot.rag.chunking import chunk_documents
from studypilot.rag.embeddings import get_embeddings

COLLECTION_NAME = "studypilot"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "chroma_db"
DEFAULT_K = 4


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit. source and page are what the later answer node will cite."""

    text: str
    source: str
    page: int
    score: float


def ingest_documents(
    documents: list[Document],
    persist_directory: Path = DEFAULT_PERSIST_DIR,
    embeddings: Embeddings | None = None,
    rebuild: bool = True,
) -> int:
    """Chunk documents, embed them, and write them to Chroma. Returns chunk count.

    rebuild=True deletes the existing collection first so a second ingest of
    the same folder does not duplicate vectors.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    chunks = chunk_documents(documents)
    if not chunks:
        raise ValueError("Nothing to ingest: no text found in the documents.")

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    if rebuild:
        _delete_collection(persist_directory)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_directory),
    )
    return len(chunks)


def retrieve(
    query: str,
    persist_directory: Path = DEFAULT_PERSIST_DIR,
    embeddings: Embeddings | None = None,
    k: int = DEFAULT_K,
) -> list[RetrievedChunk]:
    """Return the k nearest chunks, each labelled with source filename and page.

    score is Chroma's distance: smaller means closer, so more relevant.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )
    pairs = store.similarity_search_with_score(query, k=k)
    return [_to_chunk(document, score) for document, score in pairs]


def _to_chunk(document: Document, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=document.page_content,
        source=str(document.metadata.get("source", "unknown")),
        page=int(document.metadata.get("page", 0)),
        score=float(score),
    )


def index_count(persist_directory: Path = DEFAULT_PERSIST_DIR) -> int:
    """How many chunks are already stored. Zero when the folder or collection is missing."""
    path = Path(persist_directory)
    if not path.exists():
        return 0
    client = chromadb.PersistentClient(path=str(path))
    names = {item.name for item in client.list_collections()}
    if COLLECTION_NAME not in names:
        return 0
    return int(client.get_collection(COLLECTION_NAME).count())


def ensure_sample_index(persist_directory: Path = DEFAULT_PERSIST_DIR) -> int:
    """Build the sample lecture index when it is empty. Returns the chunk count."""
    count = index_count(persist_directory)
    if count > 0:
        return count
    from studypilot.rag.loader import load_file

    sample = Path(__file__).resolve().parents[2] / "data" / "sample" / "os_lecture_notes.pdf"
    if not sample.is_file():
        raise FileNotFoundError(
            f"Missing {sample}. Run: python scripts/make_sample_pdf.py"
        )
    return ingest_documents(load_file(sample), persist_directory=persist_directory, rebuild=True)


def ingest_sample_and_uploads(
    upload_dir: Path,
    persist_directory: Path = DEFAULT_PERSIST_DIR,
) -> int:
    """Rebuild the index from the sample PDF plus every pdf/txt in upload_dir.

    rebuild replaces the collection, so a second upload does not duplicate chunks.
    The sample notes stay in the index together with the uploaded files.
    """
    from studypilot.rag.loader import SUPPORTED_SUFFIXES, load_file

    documents: list[Document] = []
    sample = Path(__file__).resolve().parents[2] / "data" / "sample" / "os_lecture_notes.pdf"
    if sample.is_file():
        documents.extend(load_file(sample))
    upload_dir = Path(upload_dir)
    if upload_dir.is_dir():
        for path in sorted(upload_dir.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                documents.extend(load_file(path))
    return ingest_documents(documents, persist_directory=persist_directory, rebuild=True)


def _delete_collection(persist_directory: Path) -> None:
    """Remove only our collection. Other Chroma data in the same folder is left alone."""
    client = chromadb.PersistentClient(path=str(persist_directory))
    existing = {item.name for item in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
