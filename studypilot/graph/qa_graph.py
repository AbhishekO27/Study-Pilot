"""Q&A as a LangGraph, with a grader that can retry retrieval, plus quiz and plan branches.

    START -> route --plan--> draft_plan --ready--> review --> END
                   |                    --missing--> END
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

Phase 6 does not retrieve. A revision plan is scheduled from weak topics
the quiz tracker already recorded, and from an exam date in the question.
draft_plan writes the schedule. review calls interrupt() and waits.
The model call stays in draft_plan: resume restarts review, and a second
model call would write a different plan from the one the student just saw.

Phase 7 can replace the keyword router with a supervisor. The default is
still the keywords, so a test and the earlier scripts do not spend a model
call on routing. Pass use_supervisor=True, or pass a router, to turn it on.
A supervisor failure falls back to the keywords.
"""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from studypilot.llm import get_chat_model
from studypilot.models.schemas import GradeResult, QuizQuestion, RevisionPlan, RouteDecision
from studypilot.plan import (
    decision_approves,
    exam_date_from_question,
    format_plan,
    prepare_plan,
)
from studypilot.quiz import format_quiz, question_is_grounded, quiz_search_query
from studypilot.rag.store import DEFAULT_PERSIST_DIR, retrieve

Route = Literal["qa", "quiz", "plan"]
GradeAction = Literal["accept", "retry", "give_up"]
AfterRetrieve = Literal["answer", "make_quiz"]
AfterPlan = Literal["review", "end"]
PlanStatus = Literal["", "pending", "approved", "declined", "unavailable"]

# First answer, plus two retries, is three trips through retrieve/answer/grade,
# after the route node. 20 leaves room so a bug fails on the retry counter
# instead of on LangGraph's recursion error.
MAX_GRAPH_STEPS = 20

# Extra retrievals after the first answer. Not three total attempts.
MAX_RETRIES = 2

FALLBACK = "I couldn't find this in your material."

# Chunks were found, but the model did not return a question that cites one of them.
QUIZ_FALLBACK = "I couldn't write a quiz question from your material."

PLAN_NEEDS_TOPICS = "Take a quiz first. I build the revision plan from topics you missed."
PLAN_NEEDS_DATE = "Include the exam date as YYYY-MM-DD."
PLAN_FALLBACK = "I couldn't make a revision plan from your weak topics."
PLAN_DECLINED = "Plan declined. Nothing was saved."

# Checked in this order. "quiz me on my revision schedule" is a quiz.
_QUIZ_HINTS = ("quiz", "test me", "practice question", "flashcard")
_PLAN_HINTS = ("study plan", "revision", "exam date", "what should i study")

# Tests pass functions with these shapes so the graph needs no API key.
Grader = Callable[["StudyState"], GradeResult]
QuizWriter = Callable[["StudyState"], QuizQuestion | None]
PlanWriter = Callable[["StudyState"], RevisionPlan | None]
Router = Callable[["StudyState"], RouteDecision | None]

_ROUTES = ("qa", "quiz", "plan")


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
    # Topics the quiz tracker currently calls weak. The plan may not add others.
    weak_topics: list[str]
    # model_dump() of a RevisionPlan, or {} when this turn has no schedule.
    revision_plan: dict
    plan_status: PlanStatus
    # Why this route was chosen. "keyword rules" means the supervisor was not used.
    route_reason: str


def classify_question(question: str) -> Route:
    """Pick a route from the wording. No model call, so the branch is testable.

    This stays the fallback when the Phase 7 supervisor fails or is left off.
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
    """Some replies put text in blocks; a normal AIMessage stores a string.

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
    """Ask the model for a GradeResult. The schema rejects a free-form paragraph."""
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
    """Ask the model for one QuizQuestion. A bad payload becomes None, not a crash."""
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


def write_plan_with_model(state: StudyState, chat_model=None) -> RevisionPlan | None:
    """Ask the model for one RevisionPlan. A bad payload becomes None, not a crash."""
    weak = [topic.strip() for topic in (state.get("weak_topics") or []) if topic.strip()]
    exam_date = exam_date_from_question(state["question"])
    topics = "\n".join(f"- {topic}" for topic in weak)
    prompt = (
        "Write a revision plan for a student.\n"
        "Use only the topics listed under Weak topics. Do not add topics.\n"
        "Each item needs a topic, a one-sentence why, and minutes from 1 to 180.\n"
        "exam_date must be copied exactly from the exam date line.\n\n"
        f"Student request: {state['question']}\n"
        f"Exam date: {exam_date}\n"
        f"Weak topics:\n{topics}"
    )
    model = chat_model if chat_model is not None else get_chat_model()
    structured = model.with_structured_output(RevisionPlan)
    try:
        result = structured.invoke([HumanMessage(content=prompt)])
    except Exception:  # noqa: BLE001 - a bad plan must not kill the graph
        return None
    if isinstance(result, RevisionPlan):
        return result
    try:
        return RevisionPlan.model_validate(result)
    except (ValidationError, TypeError, ValueError):
        return None


def route_with_model(state: StudyState, chat_model=None) -> RouteDecision | None:
    """Ask the model which branch to run. A bad payload becomes None.

    None tells the route node to use classify_question. The graph does not
    stop because the supervisor failed.
    """
    prompt = (
        "You route one request for a study assistant. Pick exactly one route.\n"
        "qa: the student wants an explanation from their notes.\n"
        "quiz: the student wants a practice question, even if they do not say 'quiz'.\n"
        "plan: the student wants a revision schedule or names an exam date.\n"
        "reason is one short sentence.\n\n"
        f"Student request: {state['question']}"
    )
    try:
        model = chat_model if chat_model is not None else get_chat_model()
        structured = model.with_structured_output(RouteDecision)
        result = structured.invoke([HumanMessage(content=prompt)])
    except Exception:  # noqa: BLE001 - routing must fall back to the keywords
        return None
    if isinstance(result, RouteDecision):
        return result
    try:
        return RouteDecision.model_validate(result)
    except (ValidationError, TypeError, ValueError):
        return None


def _accepted_route(decision: object) -> Route | None:
    route = getattr(decision, "route", None)
    if route in _ROUTES:
        return route
    return None


def build_qa_graph(
    persist_directory=DEFAULT_PERSIST_DIR,
    embeddings=None,
    chat_model=None,
    grader: Grader | None = None,
    quiz_writer: QuizWriter | None = None,
    plan_writer: PlanWriter | None = None,
    checkpointer=None,
    router: Router | None = None,
    use_supervisor: bool = False,
):
    """Compile the graph. Tests pass a fake model, a fake grader, and a temp index.

    use_supervisor asks the model to pick qa, quiz, or plan. The default is
    the keyword router, which is what Phases 3 to 6 call.
    """

    def route_node(state: StudyState) -> dict:
        decision = None
        if router is not None:
            try:
                decision = router(state)
            except Exception:  # noqa: BLE001 - a broken router falls back
                decision = None
        elif use_supervisor:
            decision = route_with_model(state, chat_model)
        route = _accepted_route(decision)
        if route is None:
            return {
                "route": classify_question(state["question"]),
                "route_reason": "keyword rules",
            }
        return {"route": route, "route_reason": getattr(decision, "reason", "") or ""}

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

    def draft_plan_node(state: StudyState) -> dict:
        # Topics come from the quiz tracker, and the date comes from the
        # question. Calling the model without either would invent a timetable.
        weak = [topic.strip() for topic in (state.get("weak_topics") or []) if topic.strip()]
        if not weak:
            return {
                "answer": PLAN_NEEDS_TOPICS,
                "revision_plan": {},
                "plan_status": "unavailable",
            }
        if exam_date_from_question(state["question"]) is None:
            return {
                "answer": PLAN_NEEDS_DATE,
                "revision_plan": {},
                "plan_status": "unavailable",
            }

        if plan_writer is not None:
            drafted = plan_writer(state)
        else:
            drafted = write_plan_with_model(state, chat_model)
        plan = prepare_plan(drafted, state["question"], weak)
        if plan is None:
            return {
                "answer": PLAN_FALLBACK,
                "revision_plan": {},
                "plan_status": "unavailable",
            }
        return {
            "answer": format_plan(plan),
            "revision_plan": plan.model_dump(),
            "plan_status": "pending",
        }

    def review_node(state: StudyState) -> dict:
        # interrupt() pauses the graph. On resume this node runs again from
        # the top, and interrupt() returns the human's reply instead of pausing.
        # The schedule was already written by draft_plan, so this restart is cheap.
        decision = interrupt(
            {
                "kind": "revision_plan",
                "question": "Approve this revision plan? Reply approve or decline.",
                "plan": state.get("revision_plan") or {},
                "preview": state.get("answer", ""),
            }
        )
        if decision_approves(decision):
            return {"plan_status": "approved"}
        return {
            "plan_status": "declined",
            "answer": PLAN_DECLINED,
            "revision_plan": {},
        }

    def choose_next(state: StudyState) -> Route:
        return state["route"]

    def choose_after_grade(state: StudyState) -> GradeAction:
        # accept and give_up both end the graph. retry is the backward edge.
        return state["grade_action"]

    def choose_after_plan(state: StudyState) -> AfterPlan:
        # An empty plan is a finished turn. A pending plan waits for the human.
        if state.get("revision_plan"):
            return "review"
        return "end"

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
    graph.add_node("draft_plan", draft_plan_node)
    graph.add_node("review", review_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        choose_next,
        {"qa": "retrieve", "quiz": "retrieve", "plan": "draft_plan"},
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
    graph.add_conditional_edges(
        "draft_plan",
        choose_after_plan,
        {"review": "review", "end": END},
    )
    graph.add_edge("review", END)
    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)


def initial_state(question: str, weak_topics: list[str] | None = None) -> StudyState:
    """The state for one new question. route is overwritten by the route node."""
    return {
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
        "weak_topics": list(weak_topics or []),
        "revision_plan": {},
        "plan_status": "",
        "route_reason": "",
    }


def run_qa(question: str, weak_topics: list[str] | None = None) -> StudyState:
    """Run one question through the graph. Returns the final state.

    A planning question with weak topics and an exam date pauses inside review.
    The returned state then contains __interrupt__. Resume that pause with the
    same compiled graph; this helper builds a new one, so Phase 6 does not use it.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    app = build_qa_graph(checkpointer=InMemorySaver())
    return app.invoke(
        initial_state(question, weak_topics),
        config={
            "recursion_limit": MAX_GRAPH_STEPS,
            "configurable": {"thread_id": "studypilot"},
        },
    )
