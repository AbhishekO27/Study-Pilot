"""The fixed question set Phase 8 scores. Adding a case is a JSON edit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "qa_cases.json"


@dataclass(frozen=True)
class EvalCase:
    """One question and the facts a grounded answer must contain.

    expect_refusal is for questions that are not in the notes. must_include
    is empty there. page is where a supported answer should be cited.
    """

    id: str
    question: str
    must_include: tuple[str, ...]
    expect_refusal: bool
    source: str
    page: int | None


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in raw:
        page = item.get("page")
        cases.append(
            EvalCase(
                id=item["id"],
                question=item["question"],
                must_include=tuple(item.get("must_include") or ()),
                expect_refusal=bool(item.get("expect_refusal", False)),
                source=str(item.get("source") or ""),
                page=None if page is None else int(page),
            )
        )
    return cases
