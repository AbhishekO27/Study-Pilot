"""Structured outputs. The model must fill these fields; it cannot add new ones.

Phase 4 uses GradeResult in the graph. Phase 5 asks the model to fill
QuizQuestion. AnswerGrade is filled in code from correct_index, not by the
model. Phase 6 asks the model to fill RevisionPlan, then Python keeps only
the weak topics and the exam date from the question. Phase 7 asks the
model to fill RouteDecision when the supervisor is turned on.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RouteDecision(BaseModel):
    """Which branch should run. The supervisor must fill this; keywords are the fallback."""

    route: Literal["qa", "quiz", "plan"]
    reason: str = Field(description="One short sentence a student could read.")


class GradeResult(BaseModel):
    """Whether an answer is allowed to stand, given the retrieved excerpts."""

    supported: bool = Field(
        description="True only when every factual claim in the answer appears in the excerpts."
    )
    reason: str = Field(description="One or two sentences a student could read.")
    rewritten_query: str = Field(
        default="",
        description="A better search query when supported is false. Empty when supported is true.",
    )


class QuizQuestion(BaseModel):
    """One multiple-choice question grounded in a cited page."""

    topic: str
    question: str
    choices: list[str] = Field(min_length=2, max_length=6)
    correct_index: int = Field(ge=0)
    source: str
    page: int = Field(ge=1)

    @model_validator(mode="after")
    def index_points_at_a_choice(self) -> QuizQuestion:
        if self.correct_index >= len(self.choices):
            raise ValueError("correct_index must point at one of the choices")
        return self


class AnswerGrade(BaseModel):
    """A grade for the student's reply to a quiz question. Not the retrieval grader."""

    topic: str
    correct: bool
    score: float = Field(ge=0, le=1)
    feedback: str


class PlanItem(BaseModel):
    """One block of revision time."""

    topic: str
    why: str
    minutes: int = Field(gt=0, le=180)


class RevisionPlan(BaseModel):
    """A schedule the student approves in Phase 6. Approval is not written to disk."""

    exam_date: str = Field(description="ISO date, YYYY-MM-DD.")
    items: list[PlanItem] = Field(min_length=1, max_length=14)

    @field_validator("exam_date")
    @classmethod
    def exam_date_is_a_real_day(cls, value: str) -> str:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("exam_date must be a real date in YYYY-MM-DD form") from exc
        return value
