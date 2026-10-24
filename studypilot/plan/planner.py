"""Turn a model-written schedule into a plan the student is allowed to see.

The model chooses minutes and a reason. Python chooses the exam date and
which topics are allowed. A topic that is not on the weak-topic list is
dropped, the same way a quiz question that cites a missing page is dropped.
"""

from __future__ import annotations

import re
from datetime import date

from studypilot.models.schemas import PlanItem, RevisionPlan

# A date in the question wins over whatever date the model filled in.
_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")

_APPROVE = {"approve", "approved", "yes", "y"}


def exam_date_from_question(question: str) -> str | None:
    """Return the first real YYYY-MM-DD in the question, if there is one."""
    match = _ISO_DATE.search(question)
    if match is None:
        return None
    found = match.group(1)
    try:
        date.fromisoformat(found)
    except ValueError:
        return None
    return found


def decision_approves(decision: object) -> bool:
    """True when the human asked to keep the plan."""
    return str(decision).strip().lower() in _APPROVE


def format_plan(plan: RevisionPlan) -> str:
    """The schedule text. Approval is a separate step, so this stays factual."""
    lines = [f"Revision plan for {plan.exam_date}", ""]
    for item in plan.items:
        lines.append(f"- {item.topic} ({item.minutes} min): {item.why}")
    return "\n".join(lines)


def prepare_plan(
    plan: RevisionPlan | None,
    question: str,
    weak_topics: list[str],
) -> RevisionPlan | None:
    """Keep items whose topic is a weak topic, and use the date from the question.

    Returns None when nothing schedulable is left. The graph turns that into
    a fallback sentence instead of showing an invented timetable.
    """
    if plan is None:
        return None
    exam_date = exam_date_from_question(question)
    allowed = [topic.strip() for topic in weak_topics if topic.strip()]
    if exam_date is None or not allowed:
        return None

    kept: list[PlanItem] = []
    seen: set[str] = set()
    for item in plan.items:
        if not _topic_allowed(item.topic, allowed):
            continue
        key = item.topic.casefold().strip()
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    if not kept:
        return None
    return plan.model_copy(update={"exam_date": exam_date, "items": kept})


def _topic_allowed(topic: str, allowed: list[str]) -> bool:
    """Match 'Deadlock' to 'deadlock', and 'deadlock conditions' to 'deadlock'."""
    key = topic.casefold().strip()
    if not key:
        return False
    for name in allowed:
        other = name.casefold().strip()
        if other and (key == other or other in key):
            return True
    return False
