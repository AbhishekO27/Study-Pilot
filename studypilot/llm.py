"""Build the OpenAI chat model from .env settings.

Every phase gets its model from here, so the key and model id are read in
exactly one place. LangSmith tracing stays off unless LANGSMITH_TRACING is
true in .env. The model call then also sets the older LANGCHAIN_* names,
which current LangChain builds still read.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# The repo root is two levels up from this file: studypilot/llm.py -> repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gpt-4o-mini"


def _flag_on(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def tracing_status() -> str:
    """Return off, missing-key, or on. Does not contact LangSmith."""
    load_dotenv(REPO_ROOT / ".env")
    if not _flag_on("LANGSMITH_TRACING"):
        return "off"
    if not os.getenv("LANGSMITH_API_KEY", "").strip():
        return "missing-key"
    return "on"


def _apply_tracing() -> None:
    """Copy LangSmith settings into the names LangChain reads, when enabled."""
    if not _flag_on("LANGSMITH_TRACING"):
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    project = os.getenv("LANGSMITH_PROJECT", "").strip()
    key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project
    if key:
        os.environ["LANGCHAIN_API_KEY"] = key


def get_chat_model(temperature: float = 0.0) -> ChatOpenAI:
    """Return an OpenAI chat model, or raise a clear error if the key is missing.

    temperature 0 makes answers repeatable, which helps when you are reading
    the agent's steps and later when we measure accuracy.
    """
    # Point at the repo's .env explicitly so this works no matter which
    # folder you run a script from.
    load_dotenv(REPO_ROOT / ".env")
    _apply_tracing()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and paste a key "
            "from https://platform.openai.com/api-keys"
        )

    model_name = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
