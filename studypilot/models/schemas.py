"""Structured outputs. The model must fill these fields; it cannot add new ones.

Phase 4 uses GradeResult in the graph. QuizQuestion, AnswerGrade, and
RevisionPlan are the contracts Phase 5 and Phase 6 will ask Gemini to fill.
Defining them here, before those features, means the shape is already tested.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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
    """A schedule to approve later, in Phase 6. Not saved anywhere yet."""

    exam_date: str = Field(description="ISO date, YYYY-MM-DD.")
    items: list[PlanItem] = Field(min_length=1, max_length=14)
