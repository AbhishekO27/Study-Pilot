"""Q&A as a LangGraph, instead of the hand-written loop in simple_agent.py.

Phase 1 loop, in order:

    call model -> if it asked for a tool, run it and call the model again

This file draws the course-question path as boxes and arrows:

    START -> route -> retrieve -> answer -> END

route is the only branch. A quiz-shaped question goes to a stub, and a
planning-shaped question goes to a different stub. Those features are later
phases; the branch exists now so the conditional edge is real.

The graph has no edge back to an earlier node, so it cannot loop. The
recursion limit below is the same idea as Phase 1's max_iterations, ready
for Phase 4 when a grader edge will point backward.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from studypilot.llm import get_chat_model
from studypilot.rag.store import DEFAULT_PERSIST_DIR, retrieve

Route = Literal["qa", "quiz", "plan"]

# LangGraph stops the run if a graph walks more nodes than this. Ours walks
# four (route, retrieve, answer, end). The cap matters once Phase 4 adds a
# retry edge back to retrieve.
MAX_GRAPH_STEPS = 10

# Checked in this order. "quiz me on my revision schedule" is a quiz.
_QUIZ_HINTS = ("quiz", "test me", "practice question", "flashcard")
_PLAN_HINTS = ("study plan", "revision", "exam date", "what should i study")


class StudyState(TypedDict, total=False):
    """The only data nodes pass to each other.

    A node returns a dict with the keys it wants to change. LangGraph merges
    that dict into this state. It does not see variables inside the node.
    """

    question: str
    route: Route
    chunks: list[dict]
    answer: str


def classify_question(question: str) -> Route:
    """Pick a route from the wording. No model call, so the branch is testable.

    An LLM router is the usual next step (Phase 4 structured output, or the
    Phase 7 supervisor). Rules are enough while quiz and plan are stubs, and
    they cannot drift between test runs.
    """
    text = question.lower()
    if any(hint in text for hint in _QUIZ_HINTS):
        return "quiz"
    if any(hint in text for hint in _PLAN_HINTS):
        return "plan"
    return "qa"


def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into the text the answer model is allowed to use."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] {chunk['source']}, page {chunk['page']}\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def format_sources(chunks: list[dict]) -> str:
    """One line per source page, even if several chunks came from that page."""
    seen: list[str] = []
    for chunk in chunks:
        label = f"{chunk['source']}, p. {chunk['page']}"
        if label not in seen:
            seen.append(label)
    lines = "\n".join(f"- {label}" for label in seen)
    return f"Retrieved sources:\n{lines}"


def _message_text(message: object) -> str:
    """Gemini 3 puts text in blocks; a normal AIMessage stores a string.

    .text flattens the blocks. Tests build a plain AIMessage, whose .text
    is the same string as .content.
    """
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def build_qa_graph(
    persist_directory=DEFAULT_PERSIST_DIR,
    embeddings=None,
    chat_model=None,
):
    """Compile the graph. Tests pass a fake model and a temp Chroma directory."""

    def route_node(state: StudyState) -> dict:
        return {"route": classify_question(state["question"])}

    def retrieve_node(state: StudyState) -> dict:
        # Same retrieve() as Phase 2. The graph does not embed anything itself.
        hits = retrieve(
            state["question"],
            persist_directory=persist_directory,
            embeddings=embeddings,
        )
        chunks = [
            {"text": hit.text, "source": hit.source, "page": hit.page, "score": hit.score}
            for hit in hits
        ]
        return {"chunks": chunks}

    def answer_node(state: StudyState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            # Nothing was retrieved, so there is nothing honest to cite.
            # Phase 4's grader will also catch answers that ignore the chunks.
            return {"answer": "I couldn't find this in your material."}

        prompt = (
            "Answer the question using only the excerpts below. "
            "Cite the source like (filename, p. N) after each claim. "
            "If the excerpts do not contain the answer, say you could not find it "
            "in the material. Do not add facts from outside the excerpts.\n\n"
            f"Excerpts:\n{format_context(chunks)}\n\n"
            f"Question: {state['question']}"
        )
        model = chat_model if chat_model is not None else get_chat_model()
        reply = model.invoke([HumanMessage(content=prompt)])
        # The footer is built from the chunks, not from the model's wording,
        # so a citation is present even when the model forgets to write one.
        answer = f"{_message_text(reply)}\n\n{format_sources(chunks)}"
        return {"answer": answer}

    def quiz_node(state: StudyState) -> dict:
        return {
            "answer": (
                "Quiz mode is not built yet (Phase 5). "
                "The router sent this question here because it looks like a quiz request: "
                f"{state['question']}"
            )
        }

    def plan_node(state: StudyState) -> dict:
        return {
            "answer": (
                "Revision planning is not built yet (Phase 6). "
                "The router sent this question here because it looks like a planning request: "
                f"{state['question']}"
            )
        }

    def choose_next(state: StudyState) -> Route:
        # The conditional edge calls this. The return value must be a key
        # in the path map below.
        return state["route"]

    graph = StateGraph(StudyState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("quiz", quiz_node)
    graph.add_node("plan", plan_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        choose_next,
        {"qa": "retrieve", "quiz": "quiz", "plan": "plan"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("quiz", END)
    graph.add_edge("plan", END)
    return graph.compile()


def run_qa(question: str) -> StudyState:
    """Run one question through the graph. Returns the final state."""
    app = build_qa_graph()
    return app.invoke(
        {"question": question, "route": "qa", "chunks": [], "answer": ""},
        config={"recursion_limit": MAX_GRAPH_STEPS},
    )
