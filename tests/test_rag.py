"""Offline RAG tests. No API key and no embedding-model download."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from studypilot.rag.chunking import CHUNK_OVERLAP, CHUNK_SIZE, chunk_documents
from studypilot.rag.loader import load_file
from studypilot.rag.store import ingest_documents, retrieve

# Words that get their own axis in the fake embedding. A query containing
# "deadlock" lands near chunks that also contain "deadlock".
_KEYWORDS = ("deadlock", "coffman", "quantum", "amdahl", "thread")


class KeywordEmbeddings(Embeddings):
    """Tiny bag-of-keywords embedding so retrieval tests stay offline."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("'", "")
        return [1.0 if word in lowered else 0.0 for word in _KEYWORDS]


def test_load_txt_sets_source_and_page(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("A process is a program in execution.", encoding="utf-8")
    docs = load_file(path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["page"] == 1
    assert "program in execution" in docs[0].page_content


def test_load_pdf_keeps_page_numbers() -> None:
    pdf_path = Path(__file__).resolve().parent.parent / "data" / "sample" / "os_lecture_notes.pdf"
    docs = load_file(pdf_path)
    assert len(docs) >= 5
    pages = [doc.metadata["page"] for doc in docs]
    assert pages == list(range(1, len(docs) + 1))
    assert all(doc.metadata["source"] == "os_lecture_notes.pdf" for doc in docs)
    deadlock_pages = [doc.metadata["page"] for doc in docs if "Coffman" in doc.page_content]
    assert deadlock_pages == [5]


def test_chunking_copies_metadata_and_stays_near_size() -> None:
    long_text = "Deadlock needs four conditions. " * 80
    docs = [Document(page_content=long_text, metadata={"source": "x.pdf", "page": 3})]
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "x.pdf"
        assert chunk.metadata["page"] == 3
        # Overlap can make a chunk a bit larger than CHUNK_SIZE; stay within one extra window.
        assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP


def test_retrieve_returns_source_and_page(tmp_path: Path) -> None:
    documents = [
        Document(
            page_content="A deadlock needs the four Coffman conditions.",
            metadata={"source": "os.pdf", "page": 5},
        ),
        Document(
            page_content="Round Robin uses a time quantum.",
            metadata={"source": "os.pdf", "page": 4},
        ),
        Document(
            page_content="Amdahl's law estimates speedup from extra cores.",
            metadata={"source": "os.pdf", "page": 7},
        ),
    ]
    persist = tmp_path / "chroma"
    embeddings = KeywordEmbeddings()
    n_chunks = ingest_documents(documents, persist_directory=persist, embeddings=embeddings)
    assert n_chunks == 3

    hits = retrieve(
        "What are the Coffman conditions for deadlock?",
        persist_directory=persist,
        embeddings=embeddings,
        k=2,
    )
    assert hits
    assert hits[0].source == "os.pdf"
    assert hits[0].page == 5
    assert "Coffman" in hits[0].text


def test_rebuild_does_not_duplicate_chunks(tmp_path: Path) -> None:
    documents = [
        Document(page_content="A thread shares the process address space.", metadata={"source": "t.txt", "page": 1})
    ]
    persist = tmp_path / "chroma"
    embeddings = KeywordEmbeddings()
    first = ingest_documents(documents, persist_directory=persist, embeddings=embeddings)
    second = ingest_documents(documents, persist_directory=persist, embeddings=embeddings)
    assert first == second == 1
    hits = retrieve("thread", persist_directory=persist, embeddings=embeddings, k=5)
    assert len(hits) == 1
