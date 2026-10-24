"""Revision-plan filtering. No API key and no vector index."""

from studypilot.models.schemas import PlanItem, RevisionPlan
from studypilot.plan import decision_approves, exam_date_from_question, prepare_plan


def _plan() -> RevisionPlan:
    return RevisionPlan(
        exam_date="2026-01-01",
        items=[
            PlanItem(topic="Deadlock", why="Missed on the quiz.", minutes=25),
            PlanItem(topic="deadlock", why="Duplicate spelling.", minutes=10),
            PlanItem(topic="quantum physics", why="Not a weak topic.", minutes=40),
        ],
    )


def test_exam_date_from_question_ignores_impossible_dates() -> None:
    assert exam_date_from_question("Exam on 2026-12-01 please") == "2026-12-01"
    assert exam_date_from_question("Exam on 2026-02-31") is None
    assert exam_date_from_question("No date here") is None


def test_prepare_plan_uses_the_question_date_and_weak_topics() -> None:
    plan = prepare_plan(_plan(), "before my exam date 2026-12-01", ["deadlock"])

    assert plan is not None
    assert plan.exam_date == "2026-12-01"
    assert [item.topic for item in plan.items] == ["Deadlock"]
    assert plan.items[0].minutes == 25


def test_prepare_plan_rejects_a_schedule_with_no_allowed_topic() -> None:
    assert prepare_plan(_plan(), "before 2026-12-01", ["paging"]) is None
    assert prepare_plan(_plan(), "no date in this request", ["deadlock"]) is None


def test_decision_approves_only_the_yes_words() -> None:
    assert decision_approves(" Approve ")
    assert decision_approves("y")
    assert not decision_approves("no")
    assert not decision_approves("later")
