"""Tests for the Phase 1 tools. No API key or network needed."""

from pathlib import Path

from studypilot.tools.calculator import calculator
from studypilot.tools.notes_search import load_paragraphs, search_notes, search_paragraphs

# --- calculator ---------------------------------------------------------------


def test_calculator_basic_arithmetic() -> None:
    assert calculator.invoke({"expression": "17 * 23"}) == "391"
    assert calculator.invoke({"expression": "(1 + 2) ** 2 / 3"}) == "3"
    assert calculator.invoke({"expression": "-4 + 10 % 3"}) == "-3"


def test_calculator_keeps_decimals() -> None:
    assert calculator.invoke({"expression": "1 / 4"}) == "0.25"


def test_calculator_returns_errors_as_text() -> None:
    # Errors must come back as strings so the agent loop never crashes.
    assert calculator.invoke({"expression": "1 / 0"}).startswith("Error")
    assert calculator.invoke({"expression": "2 +"}).startswith("Error")
    assert calculator.invoke({"expression": "9 ** 9 ** 9"}).startswith("Error")


def test_calculator_rejects_code() -> None:
    # This is why we walk the AST instead of calling eval().
    result = calculator.invoke({"expression": "__import__('os').system('echo hi')"})
    assert result.startswith("Error")
    assert calculator.invoke({"expression": "abs(-1)"}).startswith("Error")


# --- notes search -------------------------------------------------------------


def test_notes_file_has_paragraphs() -> None:
    paragraphs = load_paragraphs()
    assert len(paragraphs) >= 10


def test_search_finds_deadlock_paragraph() -> None:
    result = search_notes.invoke({"query": "What are the four conditions for a deadlock?"})
    assert "Coffman" in result
    assert "[paragraph" in result


def test_search_ranks_by_overlap() -> None:
    paragraphs = ["cats like fish", "dogs like bones", "fish swim in water"]
    hits = search_paragraphs("fish water", paragraphs, top_k=2)
    # "fish swim in water" matches both words; "cats like fish" matches one.
    assert [number for number, _ in hits] == [3, 1]


def test_search_returns_message_when_nothing_matches() -> None:
    hits = search_paragraphs("zebra quantum", ["cats like fish"])
    assert hits == []
    result = search_notes.invoke({"query": "zebra giraffe photosynthesis"})
    assert result == "No matching paragraphs found in the notes."
