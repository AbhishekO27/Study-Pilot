"""Keyword search over a plain-text notes file.

This is deliberately simple: split the file into paragraphs and count how
many query words each paragraph contains. Phase 2 replaces it with embeddings
and a vector store. Keeping this version around makes the upgrade easy to
explain: same tool shape, smarter matching.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NOTES_PATH = REPO_ROOT / "data" / "sample" / "study_notes.txt"

TOP_K = 3

# Very common words carry no information about which paragraph is relevant.
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "how", "why", "when", "does", "do", "did", "of", "in", "on", "to", "and",
    "or", "for", "with", "it", "this", "that", "be", "by", "as", "at", "from",
    "explain", "describe", "define", "tell", "me", "about",
}


def _words(text: str) -> set[str]:
    """Lowercase alphabetic words, minus stop words."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP_WORDS}


def load_paragraphs(path: Path = DEFAULT_NOTES_PATH) -> list[str]:
    """Return the non-empty paragraphs of the notes file, in order."""
    text = path.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def search_paragraphs(query: str, paragraphs: list[str], top_k: int = TOP_K) -> list[tuple[int, str]]:
    """Return up to top_k (paragraph_number, paragraph) pairs that best match the query.

    Score = number of distinct query words that appear in the paragraph.
    Paragraphs with a score of zero are never returned.
    """
    query_words = _words(query)
    scored: list[tuple[int, int, str]] = []
    for number, paragraph in enumerate(paragraphs, start=1):
        score = len(query_words & _words(paragraph))
        if score > 0:
            scored.append((score, number, paragraph))

    # Highest score first; ties keep file order so results are stable.
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(number, paragraph) for _, number, paragraph in scored[:top_k]]


@tool
def search_notes(query: str) -> str:
    """Search the student's course notes for paragraphs related to the query.

    Use this whenever a question is about course content. Returns the most
    relevant paragraphs, each labelled with its paragraph number so you can
    cite it.
    """
    if not DEFAULT_NOTES_PATH.is_file():
        return f"Error: notes file not found at {DEFAULT_NOTES_PATH}"

    hits = search_paragraphs(query, load_paragraphs())
    if not hits:
        return "No matching paragraphs found in the notes."

    return "\n\n".join(f"[paragraph {number}] {text}" for number, text in hits)
