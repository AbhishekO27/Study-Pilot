"""Actions the Streamlit page calls. This module does not import Streamlit.

The page and a test can both open the session file, ask the graph, and
record a quiz grade. The correct quiz index stays in the quiz dict that
the page holds; the rendered text is built separately and omits it.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from studypilot.graph.qa_graph import MAX_GRAPH_STEPS, build_qa_graph, initial_state
from studypilot.models.schemas import QuizQuestion
from studypilot.quiz import grade_quiz_answer
from studypilot.rag.store import ensure_sample_index, ingest_sample_and_uploads
from studypilot.session import DEFAULT_DB_PATH, DEFAULT_SESSION_ID, SessionStore, open_checkpointer

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "uploads"


def thread_config() -> dict:
    return {
        "recursion_limit": MAX_GRAPH_STEPS,
        "configurable": {"thread_id": DEFAULT_SESSION_ID},
    }


def weak_topic_lines(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    store = SessionStore(db_path)
    try:
        return [
            f"{stat.topic}: {stat.correct}/{stat.attempts} correct"
            for stat in store.load_tracker().weak_topics()
        ]
    finally:
        store.close()


def pending_plan(db_path: Path = DEFAULT_DB_PATH) -> str | None:
    """The schedule text when review is paused, otherwise None."""
    saver = open_checkpointer(db_path)
    try:
        app = build_qa_graph(checkpointer=saver)
        snapshot = app.get_state(thread_config())
        if not snapshot.next:
            return None
        return str(snapshot.values.get("answer", ""))
    finally:
        saver.conn.close()


def run_turn(
    question: str,
    *,
    use_supervisor: bool = False,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """Run one question. The index is built first when chroma_db is empty."""
    ensure_sample_index()
    store = SessionStore(db_path)
    saver = open_checkpointer(db_path)
    try:
        weak = [stat.topic for stat in store.load_tracker().weak_topics()]
        app = build_qa_graph(checkpointer=saver, use_supervisor=use_supervisor)
        state = app.invoke(initial_state(question, weak), thread_config())
    finally:
        store.close()
        saver.conn.close()
    return public_turn(state)


def resume_plan(decision: str, db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Continue a paused review node and store an approved or declined plan."""
    store = SessionStore(db_path)
    saver = open_checkpointer(db_path)
    try:
        app = build_qa_graph(checkpointer=saver)
        state = app.invoke(Command(resume=decision), thread_config())
        status = state.get("plan_status", "")
        if status in {"approved", "declined"}:
            store.save_plan(
                status,
                state.get("revision_plan") or {},
                state.get("answer", ""),
            )
    finally:
        store.close()
        saver.conn.close()
    return public_turn(state)


def record_quiz_choice(quiz: dict, choice: int, db_path: Path = DEFAULT_DB_PATH):
    """Grade one choice and add it to the saved weak-topic counts."""
    question = QuizQuestion.model_validate(quiz)
    grade = grade_quiz_answer(question, choice)
    store = SessionStore(db_path)
    try:
        tracker = store.load_tracker()
        tracker.record(grade)
        store.save_tracker(tracker)
    finally:
        store.close()
    return grade


def save_upload(name: str, data: bytes, upload_dir: Path = UPLOAD_DIR) -> Path:
    """Write one uploaded file and rebuild the index with the sample notes."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(name).name
    path.write_bytes(data)
    ingest_sample_and_uploads(upload_dir)
    return path


def public_turn(state: dict) -> dict:
    """Plain values the page can keep. The interrupt object is not included."""
    quiz = state.get("quiz_question") or {}
    if not isinstance(quiz, dict):
        quiz = {}
    return {
        "question": state.get("question", ""),
        "route": state.get("route", ""),
        "route_reason": state.get("route_reason", ""),
        "answer": state.get("answer", ""),
        "grade_action": state.get("grade_action", ""),
        "grade_reason": state.get("grade_reason", ""),
        "retries": state.get("retries", 0),
        "plan_status": state.get("plan_status", ""),
        "quiz_question": quiz,
        "waiting_for_approval": bool(state.get("__interrupt__")),
        "chunks": [
            {"source": chunk.get("source", ""), "page": chunk.get("page", 0)}
            for chunk in (state.get("chunks") or [])
        ],
    }
