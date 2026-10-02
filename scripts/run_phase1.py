"""Run the Phase 1 agent on demo questions, or on a question you pass in.

    python scripts/run_phase1.py
    python scripts/run_phase1.py "What is a deadlock, and what is 17 * 23?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import studypilot` work when running this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studypilot.agents.simple_agent import run_agent  # noqa: E402

DEMO_QUESTIONS = [
    # Needs only the calculator.
    "What is 17 * 23 + 5?",
    # Needs only the notes.
    "What are the four conditions required for a deadlock?",
    # Needs both: look up Amdahl's law, then compute with it.
    "Using Amdahl's law from my notes, what is the maximum speedup with 16 cores "
    "if 20 percent of the program is serial?",
]


def main() -> int:
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS

    for question in questions:
        print("=" * 70)
        print(f"QUESTION: {question}")
        try:
            answer = run_agent(question)
        except RuntimeError as exc:
            # Missing key, or similar setup problem. Stop with the message.
            print(f"\n{exc}")
            return 1
        print(f"\nANSWER:\n{answer}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
