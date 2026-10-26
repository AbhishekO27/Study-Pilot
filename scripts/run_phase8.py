"""Ask the sample questions and print correctness, citations, and hallucinations.

    python scripts/run_phase8.py

Each question calls the graph, which calls the model. The scores themselves
are string checks in studypilot/evals/score.py, so the same numbers can be
unit-tested without an API key. A refusal that comes back as any other
sentence counts as a hallucination. A supported answer that misses a phrase
counts as incorrect, not as a hallucination.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studypilot.evals import format_report, load_cases, score_case, summarize  # noqa: E402
from studypilot.graph.qa_graph import run_qa  # noqa: E402
from studypilot.rag.store import ensure_sample_index  # noqa: E402


def main() -> int:
    try:
        count = ensure_sample_index()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Index chunks: {count}")

    cases = load_cases()
    scores = []
    for case in cases:
        print(f"\n--- {case.id}: {case.question}")
        try:
            state = run_qa(case.question)
        except RuntimeError as exc:
            print(exc)
            return 1
        score = score_case(case, state)
        scores.append(score)
        print(state.get("answer", ""))
        print(
            f"correct={score.correct} citation={score.citation_ok} "
            f"hallucinated={score.hallucinated} retrieval={score.retrieval_hit}"
        )

    report = summarize(scores, cases)
    print("\n" + format_report(scores, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
