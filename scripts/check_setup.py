"""Check that this machine can run StudyPilot. Does not call the OpenAI API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Running `python scripts/check_setup.py` puts scripts/ on sys.path, not the
# repo root. Insert the root so `import studypilot` works without a pip install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_PYTHON = (3, 11)
# Strings people leave in .env by mistake. Blank counts as missing too.
PLACEHOLDER_KEYS = {
    "",
    "your-key-here",
    "paste-your-key-here",
    "changeme",
}


def main() -> int:
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version < MIN_PYTHON:
        print("Need Python 3.11 or newer.")
        print("Create the venv with: py -3.11 -m venv .venv")
        return 1
    print("Python version: ok")

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("python-dotenv is not installed.")
        print("Activate the venv, then run: pip install -r requirements.txt")
        return 1

    import studypilot

    print(f"Package: studypilot {studypilot.__version__}")

    env_path = ROOT / ".env"
    if not env_path.is_file():
        print("No .env file. From the repo root, run: copy .env.example .env")
        return 1

    # load_dotenv reads the file into the process environment. We never print the key.
    load_dotenv(env_path)
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key.lower() in PLACEHOLDER_KEYS:
        print("OPENAI_API_KEY: missing")
        print("Paste a key from https://platform.openai.com/api-keys into .env")
        print("No API call was made.")
        return 1

    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        print("OPENAI_MODEL: missing")
        print("Set OPENAI_MODEL in .env. The example uses gpt-4o-mini.")
        return 1

    print("OPENAI_API_KEY: set (value hidden)")
    print(f"OPENAI_MODEL: {model}")
    print("Setup check passed. No API call was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
