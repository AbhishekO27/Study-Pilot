# StudyPilot

StudyPilot turns a student's lecture PDFs and notes into a study assistant: answers grounded in the material (with citations), quizzes that expose weak topics, and a revision schedule aimed at an exam date.

This repository is built one phase at a time so each piece can be explained in an interview.

**Current phase: 6 — a planning request drafts a revision plan from weak topics and pauses until you approve or decline it.** The weak-topic list and the approved plan last for one process run. The UI is not built.

## Learning goals

- What an agent loop actually is (call a tool, observe the result, repeat)
- When a fixed workflow is enough, and when a LangGraph agent is worth it
- RAG: chunking, embeddings, retrieval, citations
- Reliability: structured outputs, a grader loop, retries, and a hard iteration limit
- Evaluation with numbers (correctness, citation accuracy, hallucination rate)
- Optional tracing with LangSmith

## Tech stack

| Piece | Choice |
| --- | --- |
| Language | Python 3.11 |
| Agent framework | LangGraph + LangChain (from Phase 1 onward) |
| LLM | OpenAI, via `.env` (`gpt-4o-mini` by default) |
| Vector store | Chroma (Phase 2) |
| Persistence | SQLite (later phases) |
| UI | Streamlit (Phase 9) |
| Tracing | LangSmith, optional (Phase 9) |

## Setup (Windows, PowerShell)

Install [Python 3.11](https://www.python.org/downloads/) if `py -3.11 --version` fails. During install, enable the py launcher.

```powershell
cd C:\Users\abhis\Study-Pilot
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and paste a key from [OpenAI API keys](https://platform.openai.com/api-keys). Leave `OPENAI_MODEL` as `gpt-4o-mini` unless you have a reason to change it.

```powershell
python scripts/check_setup.py
```

A passing run prints the Python version, the package version, and `OPENAI_API_KEY: set (value hidden)`. It does not call the API.

## Run the Phase 1 agent

```powershell
pytest                                    # tool tests, no API key needed
python scripts/run_phase1.py              # three demo questions
python scripts/run_phase1.py "What is a deadlock, and what is 17 * 23?"
```

The agent has two tools: a calculator and a keyword search over `data/sample/study_notes.txt` (operating systems notes). Each step is printed, so you can watch the model ask for a tool, get the result, and either ask again or answer. It stops after 5 model calls no matter what.

The loop lives in `studypilot/agents/simple_agent.py` and is about 60 lines. Read it before Phase 3; LangGraph is that loop drawn as a graph.

## Run the Phase 2 RAG pipeline

No API key is required. First ingest downloads a small local embedding model (~80MB) into the Chroma cache.

```powershell
pip install -r requirements.txt
python scripts/run_phase2.py
python scripts/run_phase2.py --query "What is a deadlock?"
pytest
```

`run_phase2.py` indexes `data/sample/os_lecture_notes.pdf` into `chroma_db/` (gitignored) and prints the nearest chunks with filename, page, and distance. Smaller distance means a closer match.

If you edit the sample PDF, rebuild it with `python scripts/make_sample_pdf.py`.

## Run the Phase 3 graph

Q&A calls OpenAI. A quiz-shaped question does too, as of Phase 5. A planning question calls the model only when you pass weak topics and an exam date; approval is Phase 6.

```powershell
python scripts/run_phase3.py
python scripts/run_phase3.py "What are the Coffman conditions for deadlock?"
pytest
```

If `chroma_db/` is missing, the script indexes `os_lecture_notes.pdf` first. The graph is `studypilot/graph/qa_graph.py`. A normal question goes `route`, `retrieve`, `answer`, `grade`. A quiz-shaped question goes `route`, `retrieve`, `make_quiz`. A plan-shaped question stops at the stub.

The Phase 1 loop is still there (`python scripts/run_phase1.py`). Phase 3 does not replace it. It is the same idea with the branch written as an edge.

## Run the Phase 4 grader

Q&A now calls the model twice per attempt: once to draft, once to grade. A grade of "not supported" searches again, up to two extra times. If the notes still do not support an answer, you get `I couldn't find this in your material.`

```powershell
python scripts/run_phase4.py
pytest
```

The demo asks a deadlock question, then asks for the capital of France. The second one should give up, because that fact is not in the sample PDF. `RETRIES` and `GRADE` print above the answer.

Quiz questions, answer grades, and revision plans are Pydantic models in `studypilot/models/schemas.py`. Phase 5 fills `QuizQuestion` from the model. `AnswerGrade` is filled in code. Phase 6 fills `RevisionPlan`, then drops any topic that is not on the weak-topic list.

## Run the Phase 5 quiz

One model call writes the question. The two demo replies are graded locally, from `correct_index`.

```powershell
python scripts/run_phase5.py
python scripts/run_phase5.py "Quiz me on round robin"
pytest
```

The question is printed without the correct choice. The script then submits a wrong choice and the right one. A topic is weak while misses are at least as common as correct answers, so that tie stays on the list. Counts live on a `WeakTopicTracker` in memory. They are not written to disk.

`make_quiz` drops a question whose source and page are not in the retrieved chunks, and the answer string never includes `correct_index`. Grading is `studypilot/quiz/tracker.py`.

## Run the Phase 6 revision plan

One model call writes the schedule. The demo marks `deadlock` as a miss in memory, then pauses for your reply. `approve` keeps the plan. Anything else discards it. The exam date is taken from the question, not from the model.

```powershell
python scripts/run_phase6.py
python scripts/run_phase6.py "Build a revision schedule before my exam date 2026-12-01."
pytest
```

The plan route does not search the PDF. Topics come from the weak-topic list. A topic the model adds that you have not missed is removed before you see the schedule. The pause needs an in-memory checkpointer; the plan is still gone when the process exits.

`docs/LEARNING_NOTES.md` and `docs/DESIGN_DECISIONS.md` are local only (listed in `.gitignore`). They are not part of the repo.

If PowerShell refuses to run `Activate.ps1`, allow local scripts for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Layout

```
studypilot/     importable package (agents, graph, tools, rag, models, quiz, ui)
  llm.py        builds the OpenAI chat model from .env
studypilot/rag/ load, chunk, embed, retrieve
studypilot/graph/  LangGraph Q&A, grader retry, quiz generation, and plan approval
studypilot/models/ Pydantic schemas (grade, quiz, plan)
studypilot/quiz/   grade a choice and track weak topics in memory
studypilot/plan/   keep weak topics and the exam date, then format the schedule
data/sample/    study_notes.txt (Phase 1) and os_lecture_notes.pdf (Phase 2)
evals/          accuracy scripts, added in Phase 8
docs/           local learning notes (not tracked)
tests/          pytest, offline
scripts/        check_setup.py, run_phase1.py through run_phase6.py, make_sample_pdf.py
```

Uploaded notes will go in `data/uploads/`, which git ignores. Do not commit `.env`.

## Status

| Phase | What it adds | State |
| --- | --- | --- |
| 0 | Layout, venv, `.env`, setup check | Done |
| 1 | Hand-written tool-calling loop | Done |
| 2 | RAG pipeline | Done |
| 3 | LangGraph Q&A graph | Done |
| 4 | Grader, structured outputs, fallback | Done |
| 5 | Quiz and weak-topic tracking | Done |
| 6 | Revision plan with human approval | Done |
| 7 | Supervisor and persisted sessions | Not started |
| 8 | Evaluation numbers | Not started |
| 9 | Streamlit UI and README polish | Not started |
