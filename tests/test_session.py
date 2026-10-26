"""Session rows in SQLite. No API key and no vector index."""

from pathlib import Path

from studypilot.models.schemas import AnswerGrade
from studypilot.quiz import WeakTopicTracker
from studypilot.session import SessionStore


def test_tracker_and_plan_survive_a_new_connection(tmp_path: Path) -> None:
    db = tmp_path / "session.sqlite"
    tracker = WeakTopicTracker()
    tracker.record(
        AnswerGrade(topic="Deadlock", correct=False, score=0.0, feedback="Missed.")
    )
    store = SessionStore(db)
    store.save_tracker(tracker, session_id="local")
    store.save_plan("approved", {"exam_date": "2026-12-01"}, "Revision plan for 2026-12-01", "local")
    store.close()

    reopened = SessionStore(db)
    try:
        loaded = reopened.load_tracker("local")
        saved = reopened.load_plan("local")
    finally:
        reopened.close()

    assert loaded.get("deadlock") is not None
    assert loaded.get("deadlock").attempts == 1
    assert loaded.weak_topics()[0].topic == "Deadlock"
    assert saved is not None
    status, plan, answer = saved
    assert status == "approved"
    assert plan["exam_date"] == "2026-12-01"
    assert "2026-12-01" in answer


def test_sessions_do_not_share_topics(tmp_path: Path) -> None:
    db = tmp_path / "session.sqlite"
    first = WeakTopicTracker()
    first.record(AnswerGrade(topic="deadlock", correct=False, score=0.0, feedback="Missed."))
    second = WeakTopicTracker()
    second.record(AnswerGrade(topic="paging", correct=True, score=1.0, feedback="Correct."))

    store = SessionStore(db)
    try:
        store.save_tracker(first, "student-a")
        store.save_tracker(second, "student-b")
        loaded_a = store.load_tracker("student-a")
        loaded_b = store.load_tracker("student-b")
    finally:
        store.close()

    assert [stat.topic for stat in loaded_a.weak_topics()] == ["deadlock"]
    assert loaded_b.get("deadlock") is None
    assert loaded_b.get("paging") is not None
    assert loaded_b.weak_topics() == []


def test_saving_again_replaces_the_old_counts(tmp_path: Path) -> None:
    db = tmp_path / "session.sqlite"
    tracker = WeakTopicTracker()
    tracker.record(AnswerGrade(topic="deadlock", correct=False, score=0.0, feedback="Missed."))
    store = SessionStore(db)
    try:
        store.save_tracker(tracker, "local")
        tracker.record(AnswerGrade(topic="deadlock", correct=True, score=1.0, feedback="Correct."))
        store.save_tracker(tracker, "local")
        loaded = store.load_tracker("local")
    finally:
        store.close()

    assert loaded.get("deadlock").attempts == 2
    assert loaded.get("deadlock").correct == 1
    assert loaded.get("deadlock").last_feedback == "Correct."
