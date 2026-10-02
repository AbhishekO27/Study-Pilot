"""Build the Gemini chat model from .env settings.

Every phase gets its model from here, so the key and model id are read in
exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# The repo root is two levels up from this file: studypilot/llm.py -> repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "gemini-3.8-flash"


def get_chat_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model, or raise a clear error if the key is missing.

    temperature 0 makes answers repeatable, which helps when you are reading
    the agent's steps and later when we measure accuracy.
    """
    # Point at the repo's .env explicitly so this works no matter which
    # folder you run a script from.
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and paste a key "
            "from https://aistudio.google.com/apikey"
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
