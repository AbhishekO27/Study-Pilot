"""Run the Phase 3 LangGraph on demo questions, or on one you pass in.

    python scripts/run_phase3.py
    python scripts/run_phase3.py "What are the Coffman conditions?"

Q&A questions need the Phase 2 index. If chroma_db/ is missing, this script
builds it from data/sample/os_lecture_notes.pdf (no Gemini call for that step).
The answer step does call Gemini.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studypilot.graph.qa_graph import run_qa  # noqa: E402
from studypilot.rag.loader import load_file  # noqa: E402
from studypilot.rag.store import COLLECTION_NAME, DEFAULT_PERSIST_DIR, ingest_documents  # noqa: E402

SAMPLE_PDF = ROOT / "data" / "sample" / "os_lecture_notes.pdf"

DEMO_QUESTIONS = [
    "What are the four Coffman conditions for deadlock?",
    "Quiz me on deadlock.",
    "Build a revision schedule before my exam date.",
]


def index_is_ready() -> bool:
    if not DEFAULT_PERSIST_DIR.exists():
        return False
    client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIR))
    names = {item.name for item in client.list_collections()}
    if COLLECTION_NAME not in names:
        return False
    return client.get_collection(COLLECTION_NAME).count() > 0


def ensure_index() -> None:
    if index_is_ready():
        return
    if not SAMPLE_PDF.is_file():
        raise FileNotFoundError(
            f"Missing {SAMPLE_PDF}. Run: python scripts/make_sample_pdf.py"
        )
    documents = load_file(SAMPLE_PDF)
    n_chunks = ingest_documents(documents)
    print(f"Built the index: {n_chunks} chunks from {SAMPLE_PDF.name}.\n")


def main() -> int:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    try:
        ensure_index()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    for question in questions:
        print("=" * 70)
        print(f"QUESTION: {question}")
        try:
            state = run_qa(question)
        except RuntimeError as exc:
            print(f"\n{exc}")
            return 1
        print(f"ROUTE: {state['route']}")
        if state["route"] == "qa":
            for chunk in state.get("chunks") or []:
                print(f"  retrieved {chunk['source']}, p. {chunk['page']}")
        print(f"\nANSWER:\n{state['answer']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
