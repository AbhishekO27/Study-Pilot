"""Run the Phase 5 quiz: write one question, grade two demo replies, print weak topics.

    python scripts/run_phase5.py
    python scripts/run_phase5.py "Quiz me on round robin"

Writing the question calls the model. Grading does not: the correct choice is
already on the question, so the score is a comparison.

The demo submits a wrong choice, then the right one. After that tie the
topic is still weak. A topic leaves the list only when correct answers
outnumber misses. The counts are in memory for this process; they are not
saved.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_phase3  # noqa: E402
from studypilot.graph.qa_graph import run_qa  # noqa: E402
from studypilot.models.schemas import QuizQuestion  # noqa: E402
from studypilot.quiz import WeakTopicTracker, format_weak_topics, grade_quiz_answer  # noqa: E402

DEMO_QUESTIONS = [
    "Quiz me on deadlock",
]


def _wrong_choice(question: QuizQuestion) -> int:
    for index in range(len(question.choices)):
        if index != question.correct_index:
            return index
    return 0


def _print_grade(tracker: WeakTopicTracker, question: QuizQuestion, choice: int) -> None:
    grade = grade_quiz_answer(question, choice)
    tracker.record(grade)
    label = "correct" if grade.correct else "incorrect"
    print(f"YOUR CHOICE: {choice}. {question.choices[choice]}")
    print(f"GRADE: {label}  score {grade.score}")
    print(f"FEEDBACK: {grade.feedback}")
    print(f"WEAK TOPICS:\n{format_weak_topics(tracker)}\n")


def main() -> int:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    tracker = WeakTopicTracker()
    try:
        run_phase3.ensure_index()
    except FileNotFoundError as exc:
        print(exc)
        return 1

    for question in questions:
        print("=" * 70)
        print(f"QUESTION: {question}")
        try:
            state = run_qa(question)
        except RuntimeError as exc:
            print(f"\n{exc}")
            return 1
        print(f"ROUTE: {state['route']}")
        if state.get("search_query"):
            print(f"SEARCH: {state['search_query']}")
        print(f"\n{state['answer']}\n")

        payload = state.get("quiz_question") or {}
        if state["route"] != "quiz" or not payload:
            continue

        quiz = QuizQuestion.model_validate(payload)
        print("Demo replies (the correct choice was not printed above):")
        _print_grade(tracker, quiz, _wrong_choice(quiz))
        _print_grade(tracker, quiz, quiz.correct_index)
        print(
            "One miss and one correct answer are a tie, so the topic stays weak.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
