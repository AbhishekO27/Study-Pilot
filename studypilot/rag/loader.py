"""Turn a PDF or text file into LangChain Documents with source and page."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_file(path: Path) -> list[Document]:
    """Load one file. Each PDF page becomes one Document so page numbers survive chunking."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".txt":
        return _load_txt(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf or .txt.")


def load_directory(folder: Path) -> list[Document]:
    """Load every .pdf and .txt in a folder, in filename order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"No folder at {folder}")

    documents: list[Document] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.extend(load_file(path))
    return documents


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        # Blank pages have no text to embed; skipping them avoids empty vectors.
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_number},
            )
        )
    return documents


def _load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # A .txt file has no real pages. We store page=1 so retrieve() always
    # has the same metadata shape as a PDF.
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]
