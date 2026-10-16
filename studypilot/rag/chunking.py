"""Split loaded documents into overlapping chunks.

Chunk size and overlap are the two knobs interviewers ask about. The
defaults live here so DESIGN_DECISIONS can point at one place.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 800 characters is roughly a short lecture paragraph plus a bit of the next.
# Small enough that one chunk is usually one idea; large enough that a
# sentence is rarely cut off from the sentence that defines it.
CHUNK_SIZE = 800

# 150 characters of overlap (~one or two sentences) so a fact that sits on
# a split boundary appears in both neighbouring chunks.
CHUNK_OVERLAP = 150


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents. Metadata (source, page) is copied onto every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to break at paragraph, then line, then sentence, then word.
        # The last "" split is a last resort so a long word still fits.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
