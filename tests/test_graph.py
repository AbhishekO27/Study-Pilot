"""The Phase 3 graph, without Gemini and without downloading an embedding model."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from studypilot.graph.qa_graph import MAX_GRAPH_STEPS, build_qa_graph, classify_question
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
        {"question": question, "route": "qa", "chunks": [], "answer": ""},
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
    graph = build_qa_graph(persist_directory=persist, embeddings=embeddings, chat_model=model)

    state = _run(graph, "What are the Coffman conditions for deadlock?")

    assert state["route"] == "qa"
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
    assert state["answer"] == "I couldn't find this in your material."


def test_quiz_and_plan_do_not_retrieve_or_call_the_model(tmp_path: Path) -> None:
    persist, embeddings = _index(tmp_path)
    model = FakeModel()
    graph = build_qa_graph(persist_directory=persist, embeddings=embeddings, chat_model=model)

    quiz = _run(graph, "Quiz me on deadlock")
    plan = _run(graph, "What should I study before my exam date?")

    assert quiz["route"] == "quiz"
    assert "Phase 5" in quiz["answer"]
    assert quiz["chunks"] == []
    assert plan["route"] == "plan"
    assert "Phase 6" in plan["answer"]
    assert model.calls == 0
