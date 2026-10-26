"""Pydantic schemas reject shapes the later phases must not accept."""

import pytest
from pydantic import ValidationError

from studypilot.models.schemas import (
    AnswerGrade,
    GradeResult,
    PlanItem,
    QuizQuestion,
    RevisionPlan,
    RouteDecision,
)


def test_grade_result_rewritten_query_defaults_to_empty() -> None:
    grade = GradeResult(supported=True, reason="The excerpt says so.")
    assert grade.rewritten_query == ""


def test_quiz_question_rejects_an_index_past_the_choices() -> None:
    with pytest.raises(ValidationError):
        QuizQuestion(
            topic="deadlock",
            question="How many Coffman conditions are there?",
            choices=["Two", "Four"],
            correct_index=2,
            source="os.pdf",
            page=5,
        )


def test_answer_grade_score_stays_between_zero_and_one() -> None:
    with pytest.raises(ValidationError):
        AnswerGrade(topic="deadlock", correct=False, score=1.5, feedback="Too high.")


def test_route_decision_rejects_an_unknown_branch() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(route="chat", reason="Not a study task.")


def test_revision_plan_rejects_an_impossible_date() -> None:
    with pytest.raises(ValidationError):
        RevisionPlan(
            exam_date="2026-02-31",
            items=[PlanItem(topic="deadlock", why="Missed on the quiz.", minutes=25)],
        )


def test_revision_plan_needs_at_least_one_item() -> None:
    with pytest.raises(ValidationError):
        RevisionPlan(exam_date="2026-12-01", items=[])

    plan = RevisionPlan(
        exam_date="2026-12-01",
        items=[PlanItem(topic="deadlock", why="Missed on the quiz.", minutes=25)],
    )
    assert plan.items[0].minutes == 25
