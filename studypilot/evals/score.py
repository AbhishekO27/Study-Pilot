"""Score a finished graph state. No model call.

Three numbers, each with a different job:

- correctness: the answer has the required phrases, or it is the refusal
  sentence when the notes do not contain the fact.
- citation accuracy: a supported answer names the expected file and page.
  Refusals are left out of this average. A refusal should not cite a page.
- hallucination rate: among questions that should be refused, how many
  still produced some other answer.

Retrieval hit rate is printed too. It is not one of the three scores. It
tells you whether a miss started in Chroma or in the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from studypilot.evals.cases import EvalCase
from studypilot.graph.qa_graph import FALLBACK


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    correct: bool
    citation_ok: bool | None
    hallucinated: bool
    retrieval_hit: bool | None


@dataclass(frozen=True)
class EvalReport:
    correct: int
    total: int
    citation_hit: int
    citation_total: int
    hallucinations: int
    refusal_total: int
    retrieval_hit: int
    retrieval_total: int

    @property
    def correctness(self) -> float:
        return _ratio(self.correct, self.total)

    @property
    def citation_accuracy(self) -> float:
        return _ratio(self.citation_hit, self.citation_total)

    @property
    def hallucination_rate(self) -> float:
        return _ratio(self.hallucinations, self.refusal_total)

    @property
    def retrieval_hit_rate(self) -> float:
        return _ratio(self.retrieval_hit, self.retrieval_total)


def score_case(case: EvalCase, state: dict) -> CaseScore:
    """Score one finished state. citation_ok is None when no citation is required."""
    answer = str(state.get("answer") or "").strip()
    refused = answer == FALLBACK
    if case.expect_refusal:
        return CaseScore(
            case_id=case.id,
            correct=refused,
            citation_ok=None,
            hallucinated=not refused and bool(answer),
            retrieval_hit=None,
        )

    lowered = answer.casefold()
    has_facts = all(phrase.casefold() in lowered for phrase in case.must_include)
    citation_ok = None
    retrieval_hit = None
    if case.page is not None and case.source:
        citation_ok = f"{case.source}, p. {case.page}".casefold() in lowered
        retrieval_hit = _retrieved(state.get("chunks") or [], case.source, case.page)
    return CaseScore(
        case_id=case.id,
        correct=has_facts and not refused,
        citation_ok=citation_ok,
        hallucinated=False,
        retrieval_hit=retrieval_hit,
    )


def summarize(scores: list[CaseScore], cases: list[EvalCase]) -> EvalReport:
    by_id = {case.id: case for case in cases}
    citation = [score.citation_ok for score in scores if score.citation_ok is not None]
    retrieval = [score.retrieval_hit for score in scores if score.retrieval_hit is not None]
    refusals = [score for score in scores if by_id[score.case_id].expect_refusal]
    return EvalReport(
        correct=sum(score.correct for score in scores),
        total=len(scores),
        citation_hit=sum(bool(hit) for hit in citation),
        citation_total=len(citation),
        hallucinations=sum(score.hallucinated for score in refusals),
        refusal_total=len(refusals),
        retrieval_hit=sum(bool(hit) for hit in retrieval),
        retrieval_total=len(retrieval),
    )


def format_report(scores: list[CaseScore], report: EvalReport) -> str:
    lines = ["id            correct  citation  hallucinated  retrieval"]
    for score in scores:
        lines.append(
            f"{score.case_id:<13} "
            f"{_mark(score.correct):<8} "
            f"{_mark(score.citation_ok):<9} "
            f"{_mark(score.hallucinated):<13} "
            f"{_mark(score.retrieval_hit)}"
        )
    lines.append("")
    lines.append(
        f"correctness        {report.correct}/{report.total} = {report.correctness:.0%}"
    )
    lines.append(
        "citation accuracy  "
        f"{report.citation_hit}/{report.citation_total} = {report.citation_accuracy:.0%}"
    )
    lines.append(
        "hallucination rate "
        f"{report.hallucinations}/{report.refusal_total} = {report.hallucination_rate:.0%}"
    )
    lines.append(
        "retrieval hit      "
        f"{report.retrieval_hit}/{report.retrieval_total} = {report.retrieval_hit_rate:.0%} "
        "(diagnostic, not one of the three scores)"
    )
    return "\n".join(lines)


def _retrieved(chunks: list[dict], source: str, page: int) -> bool:
    wanted = Path(source).name
    for chunk in chunks:
        if int(chunk.get("page", -1)) != page:
            continue
        if Path(str(chunk.get("source", ""))).name == wanted:
            return True
    return False


def _ratio(hit: int, total: int) -> float:
    if total == 0:
        return 0.0
    return hit / total


def _mark(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"
