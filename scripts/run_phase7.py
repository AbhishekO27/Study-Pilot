"""Route with the supervisor and keep the session in SQLite.

    python scripts/run_phase7.py
    python scripts/run_phase7.py "Build a revision schedule before my exam date 2026-12-01."

The first run records one missed topic when the session file has no weak
topics, then saves it. A later run loads that list instead of starting over.

The supervisor picks qa, quiz, or plan. If that call fails, keyword rules
are used. A plan that is waiting for approval is stored in the same SQLite
file. Run the script again and it resumes that pause without drafting a
second plan.

The file is sessions/studypilot.sqlite. It is gitignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.types import Command  # noqa: E402

from studypilot.graph.qa_graph import (  # noqa: E402
    MAX_GRAPH_STEPS,
    build_qa_graph,
    initial_state,
)
from studypilot.models.schemas import AnswerGrade  # noqa: E402
from studypilot.quiz import format_weak_topics  # noqa: E402
from studypilot.session import (  # noqa: E402
    DEFAULT_DB_PATH,
    DEFAULT_SESSION_ID,
    SessionStore,
    open_checkpointer,
)

DEMO_QUESTION = "Build a revision schedule before my exam date 2026-12-01."
DEMO_TOPIC = "deadlock"


def _ask() -> str:
    return input("Approve this plan? Type approve or decline: ")


def _finish(store: SessionStore, finished: dict) -> None:
    status = finished.get("plan_status", "")
    print(f"PLAN: {status}")
    print(f"\n{finished.get('answer', '')}\n")
    if status in {"approved", "declined"}:
        store.save_plan(
            status,
            finished.get("revision_plan") or {},
            finished.get("answer", ""),
        )
        print(f"Saved the plan in {DEFAULT_DB_PATH}")


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or DEMO_QUESTION
    store = SessionStore()
    checkpointer = open_checkpointer()
    try:
        tracker = store.load_tracker()
        if tracker.get(DEMO_TOPIC) is None and not tracker.weak_topics():
            tracker.record(
                AnswerGrade(
                    topic=DEMO_TOPIC,
                    correct=False,
                    score=0.0,
                    feedback="Demo miss, saved in the session file.",
                )
            )
            store.save_tracker(tracker)
            print("Saved a demo miss. The next run loads it from SQLite.")
        else:
            print("Loaded weak topics from the session file.")
        print(f"WEAK TOPICS:\n{format_weak_topics(tracker)}")
        print(f"SESSION: {DEFAULT_DB_PATH}\n")

        app = build_qa_graph(checkpointer=checkpointer, use_supervisor=True)
        config = {
            "recursion_limit": MAX_GRAPH_STEPS,
            "configurable": {"thread_id": DEFAULT_SESSION_ID},
        }
        if app.get_state(config).next:
            print("A plan is still waiting. The model will not draft a new one.")
            print(f"\n{app.get_state(config).values.get('answer', '')}\n")
            try:
                decision = _ask()
            except EOFError:
                print("No reply. The pause stays in the session file.")
                return 1
            finished = app.invoke(Command(resume=decision), config)
            _finish(store, finished)
            return 0

        weak = [stat.topic for stat in tracker.weak_topics()]
        print(f"QUESTION: {question}")
        try:
            state = app.invoke(initial_state(question, weak), config)
        except RuntimeError as exc:
            print(f"\n{exc}")
            return 1
        print(f"ROUTE: {state['route']}  ({state.get('route_reason', '')})")
        print(f"\n{state['answer']}\n")
        if not state.get("__interrupt__"):
            return 0
        try:
            decision = _ask()
        except EOFError:
            print("No reply. The pause stays in the session file.")
            return 1
        finished = app.invoke(Command(resume=decision), config)
        _finish(store, finished)
        return 0
    finally:
        store.close()
        checkpointer.conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
