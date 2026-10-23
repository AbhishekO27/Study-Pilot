"""Q&A as a LangGraph, with a grader that can retry retrieval, plus a quiz branch.

    START -> route --plan--> plan --> END
                   --qa or quiz--> retrieve
                         retrieve --qa--> answer -> grade --accept--> END
                                                          --retry--> retrieve
                                                          --give_up--> END
                         retrieve --quiz--> make_quiz --> END

Phase 1 decided the next step inside a Python loop. Phase 3 drew the
known steps as edges. The backward edge is grade returning "retry".

"Max 2 retries" means two extra passes after the first answer. The grader
can run at most three times. recursion_limit is the backstop if a bug
adds another loop; the retry counter is the real limit.

Phase 5 uses the same retrieve node for a quiz, then make_quiz instead of
answer. make_quiz writes one multiple-choice question. It does not grade
the student. Grading happens afterwards, in studypilot.quiz, so the
correct choice is never sent back to the model.
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from studypilot.llm import get_chat_model
from studypilot.models.schemas import GradeResult, QuizQuestion
from studypilot.quiz import format_quiz, question_is_grounded, quiz_search_query
from studypilot.rag.store import DEFAULT_PERSIST_DIR, retrieve

Route = Literal["qa", "quiz", "plan"]
GradeAction = Literal["accept", "retry", "give_up"]
AfterRetrieve = Literal["answer", "make_quiz"]

# First answer, plus two retries, is three trips through retrieve/answer/grade,
# after the route node. 20 leaves room so a bug fails on the retry counter
# instead of on LangGraph's recursion error.
MAX_GRAPH_STEPS = 20

# Extra retrievals after the first answer. Not three total attempts.
MAX_RETRIES = 2

FALLBACK = "I couldn't find this in your material."

# Chunks were found, but the model did not return a question that cites one of them.
QUIZ_FALLBACK = "I couldn't write a quiz question from your material."

# Checked in this order. "quiz me on my revision schedule" is a quiz.
_QUIZ_HINTS = ("quiz", "test me", "practice question", "flashcard")
_PLAN_HINTS = ("study plan", "revision", "exam date", "what should i study")

# Tests pass functions with these shapes so the graph needs no API key.
Grader = Callable[["StudyState"], GradeResult]
QuizWriter = Callable[["StudyState"], QuizQuestion | None]


class StudyState(TypedDict, total=False):
    """The only data nodes pass to each other.

    A node returns a dict with the keys it wants to change. LangGraph merges
    that dict into this state. It does not see variables inside the node.
    """

    question: str
    search_query: str
    route: Route
    chunks: list[dict]
    draft: str
    answer: str
    retries: int
    grade_action: GradeAction
    grade_reason: str
    # model_dump() of a QuizQuestion, or {} when this turn has no question.
    # The correct index lives here. The answer string does not include it.
    quiz_question: dict


def classify_question(question: str) -> Route:
    """Pick a route from the wording. No model call, so the branch is testable.

    Plan is still a stub until Phase 6. An LLM router belongs with the
    Phase 7 supervisor.
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


def grade_with_model(state: StudyState, chat_model=None) -> GradeResult:
    """Ask Gemini for a GradeResult. The schema rejects a free-form paragraph."""
    chunks = state.get("chunks") or []
    prompt = (
        "You grade a study assistant. The answer may only use the excerpts.\n"
        "Set supported to true only if every factual claim in the answer is in the excerpts.\n"
        "If it is not supported, set rewritten_query to a short search query that would "
        "find the missing fact. If it is supported, leave rewritten_query empty.\n\n"
        f"Question: {state['question']}\n\n"
        f"Excerpts:\n{format_context(chunks)}\n\n"
        f"Answer:\n{state.get('draft', '')}"
    )
    model = chat_model if chat_model is not None else get_chat_model()
    structured = model.with_structured_output(GradeResult)
    try:
        result = structured.invoke([HumanMessage(content=prompt)])
    except Exception as exc:  # noqa: BLE001 - a bad grade must not kill the graph
        return GradeResult(
            supported=False,
            reason=f"Grader could not return a grade: {exc}",
            rewritten_query=state["question"],
        )
    if not isinstance(result, GradeResult):
        return GradeResult(
            supported=False,
            reason="Grader returned no grade.",
            rewritten_query=state["question"],
        )
    return result


def write_quiz_with_model(state: StudyState, chat_model=None) -> QuizQuestion | None:
    """Ask Gemini for one QuizQuestion. A bad payload becomes None, not a crash."""
    chunks = state.get("chunks") or []
    prompt = (
        "Write one multiple-choice study question that can be answered from the excerpts only.\n"
        "Use exactly four choices. correct_index is the position of the right choice, starting at 0.\n"
        "topic is a short name for the idea being tested, a few words.\n"
        "source is the filename copied from the excerpt header, such as os_lecture_notes.pdf.\n"
        "page is the integer page from that same header. Cite only a page listed below.\n"
        "The right choice must be a fact stated in the cited excerpt.\n\n"
        f"Student request: {state['question']}\n\n"
        f"Excerpts:\n{format_context(chunks)}"
    )
    model = chat_model if chat_model is not None else get_chat_model()
    structured = model.with_structured_output(QuizQuestion)
    try:
        result = structured.invoke([HumanMessage(content=prompt)])
    except Exception:  # noqa: BLE001 - a bad question must not kill the graph
        return None
    if isinstance(result, QuizQuestion):
        return result
    try:
        return QuizQuestion.model_validate(result)
    except (ValidationError, TypeError, ValueError):
        return None


def build_qa_graph(
    persist_directory=DEFAULT_PERSIST_DIR,
    embeddings=None,
    chat_model=None,
    grader: Grader | None = None,
    quiz_writer: QuizWriter | None = None,
):
    """Compile the graph. Tests pass a fake model, a fake grader, and a temp index."""

    def route_node(state: StudyState) -> dict:
        return {"route": classify_question(state["question"])}

    def retrieve_node(state: StudyState) -> dict:
        # search_query is set when the grader asks for another pass.
        # A quiz request is rewritten first: "Quiz me on deadlock" searches "deadlock".
        # The student's sentence stays in state["question"].
        if state.get("route") == "quiz":
            query = quiz_search_query(state["question"])
        else:
            query = state.get("search_query") or state["question"]
        hits = retrieve(
            query,
            persist_directory=persist_directory,
            embeddings=embeddings,
        )
        chunks = [
            {"text": hit.text, "source": hit.source, "page": hit.page, "score": hit.score}
            for hit in hits
        ]
        update: dict = {"chunks": chunks}
        if state.get("route") == "quiz":
            update["search_query"] = query
        return update

    def answer_node(state: StudyState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            # The grader will turn this into the fallback. No model call,
            # because an empty prompt is how unsupported answers get invented.
            return {"draft": ""}

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
        # Sources are attached later, only if the grader accepts the draft.
        # Attaching them first would let a footer citation pass a bad answer.
        return {"draft": _message_text(reply)}

    def grade_node(state: StudyState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            return {
                "grade_action": "give_up",
                "grade_reason": "Nothing was retrieved.",
                "answer": FALLBACK,
            }

        if grader is not None:
            result = grader(state)
        else:
            result = grade_with_model(state, chat_model)

        if result.supported:
            return {
                "grade_action": "accept",
                "grade_reason": result.reason,
                "answer": f"{state.get('draft', '')}\n\n{format_sources(chunks)}",
            }

        retries = state.get("retries", 0)
        if retries >= MAX_RETRIES:
            return {
                "grade_action": "give_up",
                "grade_reason": result.reason,
                "answer": FALLBACK,
            }

        rewritten = result.rewritten_query.strip() or state["question"]
        return {
            "grade_action": "retry",
            "grade_reason": result.reason,
            "retries": retries + 1,
            "search_query": rewritten,
        }

    def make_quiz_node(state: StudyState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            return {"answer": FALLBACK, "quiz_question": {}}

        if quiz_writer is not None:
            question = quiz_writer(state)
        else:
            question = write_quiz_with_model(state, chat_model)

        # A question that cites a page we did not retrieve is dropped.
        # Showing it would let the model invent a fact and point at a real page.
        if question is None or not question_is_grounded(question, chunks):
            return {"answer": QUIZ_FALLBACK, "quiz_question": {}}

        return {"answer": format_quiz(question), "quiz_question": question.model_dump()}

    def plan_node(state: StudyState) -> dict:
        return {
            "answer": (
                "Revision planning is not built yet (Phase 6). "
                "The router sent this question here because it looks like a planning request: "
                f"{state['question']}"
            )
        }

    def choose_next(state: StudyState) -> Route:
        return state["route"]

    def choose_after_grade(state: StudyState) -> GradeAction:
        # accept and give_up both end the graph. retry is the backward edge.
        return state["grade_action"]

    def choose_after_retrieve(state: StudyState) -> AfterRetrieve:
        # qa and quiz share retrieve. The route decides which node consumes the chunks.
        if state.get("route") == "quiz":
            return "make_quiz"
        return "answer"

    graph = StateGraph(StudyState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("grade", grade_node)
    graph.add_node("make_quiz", make_quiz_node)
    graph.add_node("plan", plan_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        choose_next,
        {"qa": "retrieve", "quiz": "retrieve", "plan": "plan"},
    )
    graph.add_conditional_edges(
        "retrieve",
        choose_after_retrieve,
        {"answer": "answer", "make_quiz": "make_quiz"},
    )
    graph.add_edge("answer", "grade")
    graph.add_conditional_edges(
        "grade",
        choose_after_grade,
        {"accept": END, "give_up": END, "retry": "retrieve"},
    )
    graph.add_edge("make_quiz", END)
    graph.add_edge("plan", END)
    return graph.compile()


def run_qa(question: str) -> StudyState:
    """Run one question through the graph. Returns the final state."""
    app = build_qa_graph()
    return app.invoke(
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
            "quiz_question": {},
        },
        config={"recursion_limit": MAX_GRAPH_STEPS},
    )
