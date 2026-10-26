"""The study page.

    streamlit run studypilot/ui/app.py

Ask a question, answer a quiz, or approve a revision plan. Weak topics and
a paused plan come from the Phase 7 SQLite file. The correct quiz choice is
not rendered; grading uses the stored index after you submit.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from studypilot.llm import tracing_status
from studypilot.ui.service import (
    pending_plan,
    record_quiz_choice,
    resume_plan,
    run_turn,
    save_upload,
    weak_topic_lines,
)


def _show_turn(turn: dict) -> None:
    route = turn.get("route") or ""
    reason = turn.get("route_reason") or ""
    st.caption(f"Route: {route}" + (f" ({reason})" if reason else ""))
    if route == "qa":
        st.caption(
            f"Retries: {turn.get('retries', 0)}  Grade: {turn.get('grade_action', '')}"
        )
        if turn.get("grade_reason"):
            st.caption(turn["grade_reason"])
    st.markdown(turn.get("answer") or "")
    quiz = turn.get("quiz_question") or {}
    if quiz and not st.session_state.get("quiz_feedback"):
        labels = [f"{index}. {choice}" for index, choice in enumerate(quiz["choices"])]
        picked = st.radio("Your answer", labels, index=None)
        if st.button("Submit answer") and picked is not None:
            choice = labels.index(picked)
            grade = record_quiz_choice(quiz, choice)
            st.session_state["quiz_feedback"] = grade.feedback
            st.rerun()
    if st.session_state.get("quiz_feedback"):
        st.info(st.session_state["quiz_feedback"])


def main() -> None:
    st.set_page_config(page_title="StudyPilot", layout="centered")
    st.title("StudyPilot")
    st.caption("Answers, quizzes, and revision plans from your notes.")

    with st.sidebar:
        st.subheader("Weak topics")
        lines = weak_topic_lines()
        st.text("\n".join(lines) if lines else "(none)")
        tracing = tracing_status()
        if tracing == "on":
            st.caption("LangSmith tracing is on.")
        elif tracing == "missing-key":
            st.caption("LANGSMITH_TRACING is set, but LANGSMITH_API_KEY is empty.")
        else:
            st.caption("LangSmith tracing is off.")
        upload = st.file_uploader("Add notes", type=["pdf", "txt"])
        if upload is not None and st.button("Index this file"):
            with st.spinner("Indexing"):
                path = save_upload(upload.name, upload.getvalue())
            st.success(f"Indexed {path.name} with the sample notes.")

    waiting = pending_plan()
    if waiting:
        st.subheader("A revision plan is waiting")
        st.markdown(waiting)
        approve, decline = st.columns(2)
        if approve.button("Approve"):
            st.session_state["turn"] = resume_plan("approve")
            st.rerun()
        if decline.button("Decline"):
            st.session_state["turn"] = resume_plan("decline")
            st.rerun()
        return

    with st.form("ask"):
        question = st.text_input(
            "Question",
            placeholder="What are the Coffman conditions for deadlock?",
        )
        use_supervisor = st.checkbox("Let the model choose the route")
        submitted = st.form_submit_button("Ask")
    if submitted:
        st.session_state["quiz_feedback"] = ""
        if not question.strip():
            st.warning("Type a question first.")
        else:
            try:
                with st.spinner("Working"):
                    st.session_state["turn"] = run_turn(
                        question.strip(),
                        use_supervisor=use_supervisor,
                    )
            except RuntimeError as exc:
                st.error(str(exc))
            except FileNotFoundError as exc:
                st.error(str(exc))

    turn = st.session_state.get("turn")
    if turn:
        if turn.get("waiting_for_approval"):
            st.subheader("Approve this plan")
            st.markdown(turn.get("answer") or "")
            approve, decline = st.columns(2)
            if approve.button("Approve", key="approve-turn"):
                st.session_state["turn"] = resume_plan("approve")
                st.rerun()
            if decline.button("Decline", key="decline-turn"):
                st.session_state["turn"] = resume_plan("decline")
                st.rerun()
        else:
            _show_turn(turn)


if __name__ == "__main__":
    main()
