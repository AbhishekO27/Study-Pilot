"""Eval scores and the page's quiz save. No API key and no Streamlit process."""

from pathlib import Path

from studypilot.evals import format_report, load_cases, score_case, summarize
from studypilot.graph.qa_graph import FALLBACK
from studypilot.llm import tracing_status
from studypilot.ui.service import pending_plan, public_turn, record_quiz_choice, weak_topic_lines


def test_cases_cover_a_hit_and_a_refusal() -> None:
    cases = load_cases()
    by_id = {case.id: case for case in cases}
    assert by_id["coffman"].page == 5
    assert "circular wait" in by_id["coffman"].must_include
    assert by_id["france"].expect_refusal
    assert by_id["france"].page is None


def test_supported_answer_needs_the_facts_and_the_page() -> None:
    case = load_cases()[0]
    state = {
        "answer": (
            "The Coffman conditions are mutual exclusion, hold and wait, "
            "no preemption, and circular wait.\n\n"
            "Retrieved sources:\n- os_lecture_notes.pdf, p. 5"
        ),
        "chunks": [{"source": "os_lecture_notes.pdf", "page": 5, "text": "x"}],
    }
    score = score_case(case, state)
    assert score.correct
    assert score.citation_ok is True
    assert score.hallucinated is False
    assert score.retrieval_hit is True


def test_missing_phrase_is_incorrect_not_a_hallucination() -> None:
    case = next(case for case in load_cases() if case.id == "mutex")
    state = {
        "answer": "A mutex is a lock (os_lecture_notes.pdf, p. 6).",
        "chunks": [{"source": "os_lecture_notes.pdf", "page": 6}],
    }
    score = score_case(case, state)
    assert score.correct is False
    assert score.hallucinated is False
    assert score.citation_ok is True


def test_answering_an_off_notes_question_is_a_hallucination() -> None:
    case = next(case for case in load_cases() if case.id == "france")
    refused = score_case(case, {"answer": FALLBACK, "chunks": []})
    invented = score_case(case, {"answer": "Paris.", "chunks": []})
    assert refused.correct
    assert refused.hallucinated is False
    assert refused.citation_ok is None
    assert invented.correct is False
    assert invented.hallucinated is True


def test_report_averages_leave_refusals_out_of_citations() -> None:
    cases = load_cases()
    scores = [
        score_case(
            cases[0],
            {
                "answer": (
                    "mutual exclusion, hold and wait, no preemption, and circular wait. "
                    "os_lecture_notes.pdf, p. 5"
                ),
                "chunks": [{"source": "os_lecture_notes.pdf", "page": 5}],
            },
        ),
        score_case(next(case for case in cases if case.id == "france"), {"answer": "Paris."}),
        score_case(next(case for case in cases if case.id == "hamlet"), {"answer": FALLBACK}),
    ]
    report = summarize(scores, cases)
    assert report.correct == 2
    assert report.total == 3
    assert report.citation_hit == 1
    assert report.citation_total == 1
    assert report.hallucinations == 1
    assert report.refusal_total == 2
    text = format_report(scores, report)
    assert "hallucination rate 1/2 = 50%" in text
    assert "citation accuracy  1/1 = 100%" in text


def test_public_turn_hides_the_interrupt_object() -> None:
    turn = public_turn(
        {
            "question": "Quiz me on deadlock",
            "route": "quiz",
            "answer": "How many?\n\n0. Two\n1. Four",
            "quiz_question": {"correct_index": 1, "choices": ["Two", "Four"]},
            "__interrupt__": ["pause"],
        }
    )
    assert turn["waiting_for_approval"] is True
    assert "correct_index" not in turn["answer"]
    assert "__interrupt__" not in turn


def test_quiz_choice_is_saved_on_the_session(tmp_path: Path) -> None:
    db = tmp_path / "session.sqlite"
    grade = record_quiz_choice(
        {
            "topic": "deadlock",
            "question": "How many Coffman conditions are there?",
            "choices": ["Two", "Four"],
            "correct_index": 1,
            "source": "os_lecture_notes.pdf",
            "page": 5,
        },
        0,
        db_path=db,
    )
    assert grade.correct is False
    assert weak_topic_lines(db) == ["deadlock: 0/1 correct"]


def test_new_session_has_no_pending_plan(tmp_path: Path) -> None:
    assert pending_plan(tmp_path / "fresh.sqlite") is None


def test_tracing_stays_off_when_the_flag_is_empty(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    assert tracing_status() == "off"
