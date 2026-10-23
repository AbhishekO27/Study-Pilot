"""The Q&A graph, without Gemini and without downloading an embedding model."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from studypilot.graph.qa_graph import (
    FALLBACK,
    MAX_GRAPH_STEPS,
    QUIZ_FALLBACK,
    build_qa_graph,
    classify_question,
    write_quiz_with_model,
)
from studypilot.models.schemas import GradeResult, QuizQuestion
from studypilot.rag.store import ingest_documents

_KEYWORDS = ("deadlock", "coffman", "quantum", "amdahl", "thread")


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower().replace("'", "")
        return [1.0 if word in lowered else 0.0 for word in _KEYWORDS]


class FakeModel:
    """Stands in for Gemini. Records whether the answer node called it."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_prompt = ""

    def invoke(self, messages):
        self.calls += 1
        self.last_prompt = messages[-1].content
        return AIMessage(content="Deadlock needs four Coffman conditions (os.pdf, p. 5).")


def accept_grade(_state: dict) -> GradeResult:
    return GradeResult(supported=True, reason="The draft matches the excerpts.", rewritten_query="")


class ScriptedGrader:
    """Returns a fixed sequence of grades, one per grader visit."""

    def __init__(self, results: list[GradeResult]) -> None:
        self.results = results
        self.calls = 0
        self.queries: list[str] = []

    def __call__(self, state: dict) -> GradeResult:
        self.queries.append(state.get("search_query") or state["question"])
        result = self.results[self.calls]
        self.calls += 1
        return result


def _index(tmp_path: Path) -> tuple[Path, KeywordEmbeddings]:
    documents = [
        Document(
            page_content="A deadlock needs the four Coffman conditions.",
            metadata={"source": "os.pdf", "page": 5},
        ),
        Document(
            page_content="Round Robin uses a time quantum.",
            metadata={"source": "os.pdf", "page": 4},
        ),
    ]
    persist = tmp_path / "chroma"
    embeddings = KeywordEmbeddings()
    ingest_documents(documents, persist_directory=persist, embeddings=embeddings)
    return persist, embeddings


def _run(graph, question: str) -> dict:
    return graph.invoke(
        {
            "question": question,
            "search_query": "",
            "route": "qa",
            "chunks": [],
            "draft": "",
            "answer": "",
            "retries": 0,
            "grade_action": "accept",
            "grade_reason": "",
        },
        config={"recursion_limit": MAX_GRAPH_STEPS},
    )


def test_classify_question() -> None:
    assert classify_question("What is a deadlock?") == "qa"
    assert classify_question("How does CPU scheduling work?") == "qa"
    assert classify_question("Quiz me on deadlock") == "quiz"
    assert classify_question("Build a revision schedule before my exam date") == "plan"


def test_qa_path_retrieves_and_cites(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=embeddings,
        chat_model=model,
        grader=accept_grade,
    )

    state = _run(graph, "What are the Coffman conditions for deadlock?")

    assert state["route"] == "qa"
    assert state["grade_action"] == "accept"
    assert state["retries"] == 0
    assert model.calls == 1
    assert state["chunks"][0]["page"] == 5
    assert "os.pdf, page 5" in model.last_prompt
    assert "(os.pdf, p. 5)" in state["answer"]
    assert "Retrieved sources:" in state["answer"]
    assert "os.pdf, p. 5" in state["answer"]


def test_empty_index_skips_the_model(tmp_path: Path) -> None:
    # retrieve() always returns the nearest chunks when the index has any text,
    # so "unrelated question" is not the empty case. An empty index is.
    persist = tmp_path / "empty"
    persist.mkdir()
    model = FakeModel()
    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=KeywordEmbeddings(),
        chat_model=model,
    )

    state = _run(graph, "What is a deadlock?")

    assert state["route"] == "qa"
    assert state["chunks"] == []
    assert model.calls == 0
    assert state["grade_action"] == "give_up"
    assert state["answer"] == FALLBACK


def _quiz_question(**overrides) -> QuizQuestion:
    fields = {
        "topic": "deadlock",
        "question": "How many Coffman conditions are there?",
        "choices": ["Two", "Four"],
        "correct_index": 1,
        "source": "os.pdf",
        "page": 5,
    }
    fields.update(overrides)
    return QuizQuestion(**fields)


def test_plan_does_not_retrieve_or_call_the_model(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    graph = build_qa_graph(persist_directory=persist, embeddings=embeddings, chat_model=model)

    plan = _run(graph, "What should I study before my exam date?")

    assert plan["route"] == "plan"
    assert "Phase 6" in plan["answer"]
    assert plan["chunks"] == []
    assert model.calls == 0


def test_quiz_retrieves_and_hides_the_correct_index(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    written = _quiz_question()
    seen: dict = {}

    def writer(state: dict) -> QuizQuestion:
        seen["chunks"] = state["chunks"]
        return written

    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=embeddings,
        chat_model=model,
        quiz_writer=writer,
    )

    state = _run(graph, "Quiz me on deadlock")

    assert state["route"] == "quiz"
    assert state["search_query"] == "deadlock"
    assert seen["chunks"][0]["page"] == 5
    assert model.calls == 0
    assert "0. Two" in state["answer"]
    assert "1. Four" in state["answer"]
    assert "correct_index" not in state["answer"]
    assert "Topic: deadlock" in state["answer"]
    assert state["quiz_question"]["correct_index"] == 1
    assert state["quiz_question"]["page"] == 5


def test_ungrounded_quiz_is_dropped(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=embeddings,
        chat_model=FakeModel(),
        quiz_writer=lambda _state: _quiz_question(page=9),
    )

    state = _run(graph, "Quiz me on deadlock")

    assert state["chunks"]
    assert state["answer"] == QUIZ_FALLBACK
    assert state["quiz_question"] == {}


def test_quiz_on_an_empty_index_skips_the_writer(tmp_path: Path) -> None:
    persist = tmp_path / "empty"
    persist.mkdir()
    calls: list[dict] = []

    def writer(state: dict) -> QuizQuestion:
        calls.append(state)
        return _quiz_question()

    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=KeywordEmbeddings(),
        chat_model=FakeModel(),
        quiz_writer=writer,
    )

    state = _run(graph, "Quiz me on deadlock")

    assert calls == []
    assert state["answer"] == FALLBACK
    assert state["quiz_question"] == {}


class FakeStructuredModel:
    """Stands in for with_structured_output. invoke returns whatever we stored."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.schema = None
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_write_quiz_with_model_asks_for_the_schema() -> None:
    question = _quiz_question()
    model = FakeStructuredModel(question)
    state = {
        "question": "Quiz me on deadlock",
        "chunks": [
            {
                "text": "A deadlock needs the four Coffman conditions.",
                "source": "os.pdf",
                "page": 5,
            }
        ],
    }

    got = write_quiz_with_model(state, chat_model=model)

    assert got == question
    assert model.schema is QuizQuestion
    assert "os.pdf, page 5" in model.prompts[0]
    assert "A deadlock needs the four Coffman conditions." in model.prompts[0]


def test_write_quiz_with_model_drops_a_bad_payload() -> None:
    state = {"question": "Quiz me", "chunks": [{"text": "x", "source": "os.pdf", "page": 5}]}

    assert write_quiz_with_model(state, chat_model=FakeStructuredModel({"topic": "nope"})) is None
    assert (
        write_quiz_with_model(state, chat_model=FakeStructuredModel(RuntimeError("bad json")))
        is None
    )


def test_grader_retries_once_then_accepts(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    grader = ScriptedGrader(
        [
            GradeResult(
                supported=False,
                reason="The draft does not name the conditions.",
                rewritten_query="Coffman deadlock conditions",
            ),
            GradeResult(supported=True, reason="The conditions are in the excerpt.", rewritten_query=""),
        ]
    )
    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=embeddings,
        chat_model=model,
        grader=grader,
    )

    state = _run(graph, "What is deadlock?")

    assert grader.calls == 2
    assert model.calls == 2
    assert state["retries"] == 1
    assert state["grade_action"] == "accept"
    assert state["search_query"] == "Coffman deadlock conditions"
    assert "Retrieved sources:" in state["answer"]
    assert "os.pdf, p. 5" in state["answer"]


def test_grader_gives_up_after_two_retries(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    reject = GradeResult(
        supported=False,
        reason="The excerpts never mention this.",
        rewritten_query="still missing",
    )
    grader = ScriptedGrader([reject, reject, reject])
    graph = build_qa_graph(
        persist_directory=persist,
        embeddings=embeddings,
        chat_model=model,
        grader=grader,
    )

    state = _run(graph, "What is deadlock?")

    # First answer, then two retries: three drafts and three grades.
    assert model.calls == 3
    assert grader.calls == 3
    assert state["retries"] == 2
    assert state["grade_action"] == "give_up"
    assert state["answer"] == FALLBACK
    assert "Retrieved sources:" not in state["answer"]
