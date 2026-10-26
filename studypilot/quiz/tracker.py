"""Grade a quiz question and remember which topics are weak.

Generating the question needs the chat model, because the wording has to come from
the retrieved pages. Grading it does not. QuizQuestion already stores
correct_index, so the score is a comparison, not another model call. A
second call could disagree with the question.

The counts live on this object. studypilot.session.SessionStore is what
writes them to SQLite. This class does not open a database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from studypilot.models.schemas import AnswerGrade, QuizQuestion

# Longer phrases first, so "quiz me on " wins over "quiz me ".
_QUIZ_PREFIXES = tuple(
    sorted(
        (
            "please quiz me about ",
            "please quiz me on ",
            "give me a quiz about ",
            "give me a quiz on ",
            "practice question about ",
            "practice question on ",
            "flashcard about ",
            "flashcard on ",
            "quiz me about ",
            "quiz me on ",
            "test me about ",
            "test me on ",
            "quiz about ",
            "quiz on ",
            "quiz me ",
            "test me ",
        ),
        key=len,
        reverse=True,
    )
)


@dataclass
class TopicStat:
    """Running score for one topic. `topic` keeps the first spelling we saw."""

    topic: str
    attempts: int = 0
    correct: int = 0
    last_feedback: str = ""

    @property
    def wrong(self) -> int:
        return self.attempts - self.correct

    @property
    def weak(self) -> bool:
        """Weak when misses are at least as common as correct answers.

        One miss puts the topic on the list. One later correct answer ties
        the count, so it stays there. It drops off only after correct
        answers outnumber misses.
        """
        return self.attempts > 0 and self.wrong >= self.correct


class WeakTopicTracker:
    """In-memory miss counts, keyed so 'Deadlock' and 'deadlock' are one topic."""

    def __init__(self) -> None:
        self._stats: dict[str, TopicStat] = {}

    def record(self, grade: AnswerGrade) -> TopicStat:
        key = grade.topic.casefold()
        stat = self._stats.get(key)
        if stat is None:
            stat = TopicStat(topic=grade.topic.strip() or grade.topic)
            self._stats[key] = stat
        stat.attempts += 1
        if grade.correct:
            stat.correct += 1
        stat.last_feedback = grade.feedback
        return stat

    def get(self, topic: str) -> TopicStat | None:
        return self._stats.get(topic.casefold())

    def weak_topics(self) -> list[TopicStat]:
        weak = [stat for stat in self._stats.values() if stat.weak]
        return sorted(weak, key=lambda stat: stat.topic.casefold())

    def stats(self) -> list[TopicStat]:
        """Every topic, weak or not. SessionStore writes this list."""
        return list(self._stats.values())

    def restore(self, stats: list[TopicStat]) -> None:
        """Replace the in-memory counts. Used when a session file is loaded."""
        self._stats = {stat.topic.casefold(): stat for stat in stats}


def quiz_search_query(question: str) -> str:
    """Drop the 'quiz me on' wording so retrieval searches the topic.

    'Quiz me on deadlock' contains 'quiz', which is not in the notes. The
    useful query is 'deadlock'. If nothing is left after the prefix, keep
    the original sentence.
    """
    text = question.strip()
    lowered = text.lower()
    for prefix in _QUIZ_PREFIXES:
        if lowered.startswith(prefix):
            rest = text[len(prefix) :].strip(" ?.!")
            return rest or text
    return text


def question_is_grounded(question: QuizQuestion, chunks: list[dict]) -> bool:
    """True when source and page were copied from a chunk we actually retrieved."""
    for chunk in chunks:
        if int(chunk["page"]) != question.page:
            continue
        if _same_source(question.source, str(chunk["source"])):
            return True
    return False


def format_quiz(question: QuizQuestion) -> str:
    """The text the student sees. correct_index stays out of this string."""
    lines = [question.question, ""]
    for index, choice in enumerate(question.choices):
        lines.append(f"{index}. {choice}")
    lines.append("")
    lines.append(f"Topic: {question.topic}")
    lines.append(f"Source: {question.source}, p. {question.page}")
    return "\n".join(lines)


def format_weak_topics(tracker: WeakTopicTracker) -> str:
    weak = tracker.weak_topics()
    if not weak:
        return "(none)"
    return "\n".join(
        f"- {stat.topic}: {stat.correct}/{stat.attempts} correct" for stat in weak
    )


def grade_quiz_answer(question: QuizQuestion, choice: int) -> AnswerGrade:
    """Score one reply. choice is the index the student picked, starting at 0."""
    in_range = 0 <= choice < len(question.choices)
    correct = in_range and choice == question.correct_index
    if not in_range:
        feedback = (
            f"Choice {choice} is not one of the options "
            f"(0 to {len(question.choices) - 1})."
        )
    elif correct:
        feedback = (
            f"Correct: {question.choices[choice]} "
            f"({question.source}, p. {question.page})."
        )
    else:
        feedback = (
            f"You chose {question.choices[choice]}. "
            f"The notes say {question.choices[question.correct_index]} "
            f"({question.source}, p. {question.page})."
        )
    return AnswerGrade(
        topic=question.topic,
        correct=correct,
        score=1.0 if correct else 0.0,
        feedback=feedback,
    )


def _same_source(cited: str, stored: str) -> bool:
    cited_name = Path(cited.strip()).name
    stored_name = Path(stored.strip()).name
    return bool(cited_name) and cited_name == stored_name
