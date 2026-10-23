"""A tool-calling agent loop written by hand, with no framework.

This is the whole idea of an agent:

    1. send the conversation to the model
    2. if the model answered in plain text, we are done
    3. otherwise it asked for one or more tool calls: run them
    4. append the results to the conversation and go back to step 1

LangGraph (Phase 3) draws a fixed version of this loop in
studypilot/graph/qa_graph.py: route, then retrieve, then answer. Read this
file first so the graph is the same idea with the branches written out.
"""

from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from studypilot.llm import get_chat_model
from studypilot.tools.calculator import calculator
from studypilot.tools.notes_search import search_notes

TOOLS: list[BaseTool] = [calculator, search_notes]

# Lookup by name, because the model refers to tools by their name string.
TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in TOOLS}

SYSTEM_PROMPT = """You are StudyPilot, a study assistant.
Use the search_notes tool to answer questions about course material, and
cite paragraph numbers from the results. Use the calculator tool for any
arithmetic. If the notes do not contain the answer, say so plainly instead
of guessing. When you have everything you need, reply with the final answer."""

# Hard cap on model calls per question. Without this, a confused model could
# call tools forever. On the free tier, this also caps requests per question.
DEFAULT_MAX_ITERATIONS = 5


def run_tool(tool_call: dict) -> ToolMessage:
    """Execute one tool call requested by the model and wrap the result.

    tool_call looks like: {"name": "calculator", "args": {"expression": "17*23"}, "id": "..."}

    The model expects a ToolMessage for every tool call it made, so even
    failures must come back as a ToolMessage (with the error text), never as
    a raised exception.
    """
    name = tool_call["name"]
    args = tool_call["args"]
    call_id = tool_call["id"]

    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        content = f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS_BY_NAME)}"
    else:
        try:
            content = str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
            content = f"Error while running {name}: {exc}"

    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def run_agent(
    question: str,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    verbose: bool = True,
) -> str:
    """Answer a question, calling tools as needed. Returns the final text."""
    # bind_tools tells the model which functions exist and what arguments they
    # take (from the @tool docstrings and type hints). It does not run them.
    model = get_chat_model().bind_tools(TOOLS)

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for step in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- step {step}: calling model ---")

        ai_msg: AIMessage = model.invoke(messages)

        # Append the original AIMessage. It carries the tool-call ids.
        # A rebuilt message without those ids makes the following
        # ToolMessage invalid, and the next request fails.
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # Plain text, no tool requests: this is the final answer.
            return ai_msg.text

        for tool_call in ai_msg.tool_calls:
            if verbose:
                print(f"model wants {tool_call['name']}({json.dumps(tool_call['args'])})")

            result = run_tool(tool_call)
            messages.append(result)

            if verbose:
                preview = result.content if len(result.content) <= 300 else result.content[:300] + "..."
                print(f"tool returned: {preview}")

    return (
        f"I stopped after {max_iterations} steps without reaching a final answer. "
        "Try asking a narrower question."
    )
