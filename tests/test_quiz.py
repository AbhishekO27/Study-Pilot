"""Quiz grading and weak-topic counts. No Gemini and no vector index."""

from studypilot.models.schemas import AnswerGrade, QuizQuestion
from studypilot.quiz import (
    WeakTopicTracker,
    format_quiz,
    grade_quiz_answer,
    question_is_grounded,
    quiz_search_query,
)


def _question() -> QuizQuestion:
    return QuizQuestion(
        topic="deadlock",
        question="How many Coffman conditions are there?",
        choices=["Two", "Four", "Six"],
        correct_index=1,
        source="os.pdf",
        page=5,
    )


def test_quiz_search_query_keeps_the_topic() -> None:
    assert quiz_search_query("Quiz me on deadlock") == "deadlock"
    assert quiz_search_query("Please quiz me on the Banker's algorithm?") == "the Banker's algorithm"
    assert quiz_search_query("Give me a quiz on mutex") == "mutex"
    assert quiz_search_query("What is a deadlock?") == "What is a deadlock?"
    assert quiz_search_query("Quiz me") == "Quiz me"


def test_format_quiz_does_not_reveal_the_correct_index() -> None:
    text = format_quiz(_question())
    assert "1. Four" in text
    assert "correct_index" not in text
    assert "Source: os.pdf, p. 5" in text


def test_question_is_grounded_only_when_the_page_was_retrieved() -> None:
    chunks = [{"text": "notes", "source": "os.pdf", "page": 5}]
    assert question_is_grounded(_question(), chunks)
    assert question_is_grounded(_question(), [{"text": "notes", "source": "notes/os.pdf", "page": 5}])
    assert not question_is_grounded(_question(), [{"text": "notes", "source": "os.pdf", "page": 4}])
    assert not question_is_grounded(_question(), [])


def test_grade_quiz_answer_scores_the_index() -> None:
    question = _question()

    right = grade_quiz_answer(question, 1)
    wrong = grade_quiz_answer(question, 0)
    missing = grade_quiz_answer(question, 9)

    assert right.correct is True
    assert right.score == 1.0
    assert "Four" in right.feedback
    assert wrong.correct is False
    assert wrong.score == 0.0
    assert "You chose Two" in wrong.feedback
    assert "The notes say Four" in wrong.feedback
    assert "(os.pdf, p. 5)" in wrong.feedback
    assert missing.correct is False
    assert missing.score == 0.0


def test_one_miss_is_weak_and_a_tie_stays_weak() -> None:
    tracker = WeakTopicTracker()
    question = _question()

    tracker.record(grade_quiz_answer(question, 0))
    assert [stat.topic for stat in tracker.weak_topics()] == ["deadlock"]

    tracker.record(grade_quiz_answer(question, 1))
    stat = tracker.get("Deadlock")
    assert stat is not None
    assert stat.topic == "deadlock"
    assert stat.attempts == 2
    assert stat.correct == 1
    assert stat.weak is True
    assert [item.topic for item in tracker.weak_topics()] == ["deadlock"]


def test_correct_answers_have_to_outnumber_misses() -> None:
    tracker = WeakTopicTracker()
    miss = AnswerGrade(topic="scheduling", correct=False, score=0.0, feedback="miss")
    hit = AnswerGrade(topic="Scheduling", correct=True, score=1.0, feedback="hit")

    tracker.record(hit)
    assert tracker.weak_topics() == []

    tracker.record(miss)
    assert tracker.get("scheduling").weak is True

    tracker.record(hit)
    # 2 correct, 1 miss. Correct answers outnumber misses, so it leaves the list.
    assert tracker.weak_topics() == []
    assert tracker.get("SCHEDULING").topic == "Scheduling"
    assert tracker.get("scheduling").last_feedback == "hit"


def test_weak_topics_are_sorted_by_name() -> None:
    tracker = WeakTopicTracker()
    tracker.record(AnswerGrade(topic="threads", correct=False, score=0.0, feedback=""))
    tracker.record(AnswerGrade(topic="deadlock", correct=False, score=0.0, feedback=""))

    assert [stat.topic for stat in tracker.weak_topics()] == ["deadlock", "threads"]
