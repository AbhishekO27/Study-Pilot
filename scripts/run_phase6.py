"""Draft a revision plan and wait for approval.

    python scripts/run_phase6.py
    python scripts/run_phase6.py "Build a revision schedule before my exam date 2026-12-01."

The demo records one missed quiz topic in memory, then asks the model for a
RevisionPlan. Python drops any topic that is not on that weak list and forces
the exam date to the YYYY-MM-DD in the question. The graph then pauses.
Type approve to keep the schedule, or decline to discard it.

Nothing is written to disk. Phase 7 is the one that saves a session.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from studypilot.graph.qa_graph import (  # noqa: E402
    MAX_GRAPH_STEPS,
    build_qa_graph,
    initial_state,
)
from studypilot.models.schemas import AnswerGrade  # noqa: E402
from studypilot.quiz import WeakTopicTracker  # noqa: E402

DEMO_QUESTION = "Build a revision schedule before my exam date 2026-12-01."
DEMO_TOPIC = "deadlock"


def _weak_topics() -> list[str]:
    tracker = WeakTopicTracker()
    tracker.record(
        AnswerGrade(
            topic=DEMO_TOPIC,
            correct=False,
            score=0.0,
            feedback="Demo miss, so the topic starts weak.",
        )
    )
    return [stat.topic for stat in tracker.weak_topics()]


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or DEMO_QUESTION
    weak_topics = _weak_topics()
    print(f"QUESTION: {question}")
    print("WEAK TOPICS: " + ", ".join(weak_topics))

    app = build_qa_graph(checkpointer=InMemorySaver())
    config = {
        "recursion_limit": MAX_GRAPH_STEPS,
        "configurable": {"thread_id": "phase6"},
    }
    try:
        paused = app.invoke(initial_state(question, weak_topics), config)
    except RuntimeError as exc:
        print(f"\n{exc}")
        return 1

    print(f"ROUTE: {paused['route']}")
    print(f"\n{paused['answer']}\n")
    interrupts = paused.get("__interrupt__") or []
    if not interrupts:
        return 0

    try:
        decision = input("Approve this plan? Type approve or decline: ")
    except EOFError:
        print("No reply. The plan is still pending.")
        return 1

    finished = app.invoke(Command(resume=decision), config)
    print(f"PLAN: {finished['plan_status']}")
    print(f"\n{finished['answer']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
