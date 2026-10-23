"""Run the Phase 4 graph: a real question, then one the notes do not contain.

    python scripts/run_phase4.py
    python scripts/run_phase4.py "What is a mutex?"

The second demo question is about Paris. The nearest PDF chunks are still
operating-systems notes, so the grader should reject the answer and, after
two retries, return the fallback sentence. That path calls the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_phase3  # noqa: E402
from studypilot.graph.qa_graph import run_qa  # noqa: E402

DEMO_QUESTIONS = [
    "What are the four Coffman conditions for deadlock?",
    "What is the capital of France?",
]


def main() -> int:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    try:
        run_phase3.ensure_index()
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
        print(f"RETRIES: {state.get('retries', 0)}  GRADE: {state.get('grade_action', '')}")
        if state.get("grade_reason"):
            print(f"REASON: {state['grade_reason']}")
        if state.get("search_query"):
            print(f"LAST SEARCH: {state['search_query']}")
        print(f"\nANSWER:\n{state['answer']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
