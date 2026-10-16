"""Ingest the sample course files and print retrieved chunks with citations.

    python scripts/run_phase2.py
    python scripts/run_phase2.py --query "What is a deadlock?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studypilot.rag.loader import load_file  # noqa: E402
from studypilot.rag.store import ingest_documents, retrieve  # noqa: E402

# Phase 1 still uses study_notes.txt for keyword search. This script indexes
# the PDF so retrieved citations have real page numbers.
SAMPLE_PDF = ROOT / "data" / "sample" / "os_lecture_notes.pdf"
DEFAULT_QUERIES = [
    "What are the four Coffman conditions for deadlock?",
    "How does Round Robin scheduling work?",
    "What is Amdahl's law used for?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 RAG ingest and retrieve")
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="A question to retrieve for. Repeat to ask several. Default: three demos.",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Reuse the existing chroma_db instead of rebuilding it.",
    )
    args = parser.parse_args()

    if not args.skip_ingest:
        if not SAMPLE_PDF.is_file():
            print(f"Missing {SAMPLE_PDF}. Run: python scripts/make_sample_pdf.py")
            return 1
        documents = load_file(SAMPLE_PDF)
        n_chunks = ingest_documents(documents)
        print(f"Ingested {len(documents)} pages/files into {n_chunks} chunks.\n")

    queries = args.queries or DEFAULT_QUERIES
    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        hits = retrieve(query)
        if not hits:
            print("No chunks returned.")
            continue
        for index, hit in enumerate(hits, start=1):
            preview = hit.text if len(hit.text) <= 280 else hit.text[:280] + "..."
            print(f"\n[{index}] {hit.source}, page {hit.page}  (distance {hit.score:.3f})")
            print(preview)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
