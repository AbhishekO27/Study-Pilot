# StudyPilot

StudyPilot turns a student's lecture PDFs and notes into a study assistant: answers grounded in the material (with citations), quizzes that expose weak topics, and a revision schedule aimed at an exam date.

This repository is built one phase at a time so each piece can be explained in an interview.

**Current phase: 4 — the Q&A graph grades its own answer and retries retrieval at most twice.** Quiz and revision planning have Pydantic schemas, but those features are not built yet. The UI is not built.

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
| LLM | Gemini free tier, via `.env` |
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

Open `.env` and paste a key from [Google AI Studio](https://aistudio.google.com/apikey). Leave `GEMINI_MODEL` as `gemini-3.8-flash` unless you have a reason to change it.

```powershell
python scripts/check_setup.py
```

A passing run prints the Python version, the package version, and `GEMINI_API_KEY: set (value hidden)`. It does not call the API.

## Run the Phase 1 agent

```powershell
pytest                                    # tool tests, no API key needed
python scripts/run_phase1.py              # three demo questions
python scripts/run_phase1.py "What is a deadlock, and what is 17 * 23?"
```

The agent has two tools: a calculator and a keyword search over `data/sample/study_notes.txt` (operating systems notes). Each step is printed, so you can watch the model ask for a tool, get the result, and either ask again or answer. It stops after 5 model calls no matter what.

The loop lives in `studypilot/agents/simple_agent.py` and is about 60 lines. Read it before Phase 3; LangGraph is that loop drawn as a graph.

## Run the Phase 2 RAG pipeline

No Gemini key is required. First ingest downloads a small local embedding model (~80MB) into the Chroma cache.

```powershell
pip install -r requirements.txt
python scripts/run_phase2.py
python scripts/run_phase2.py --query "What is a deadlock?"
pytest
```

`run_phase2.py` indexes `data/sample/os_lecture_notes.pdf` into `chroma_db/` (gitignored) and prints the nearest chunks with filename, page, and distance. Smaller distance means a closer match.

If you edit the sample PDF, rebuild it with `python scripts/make_sample_pdf.py`.

## Run the Phase 3 graph

Q&A calls Gemini. Quiz and planning questions do not: the router sends them to a stub that names the later phase.

```powershell
python scripts/run_phase3.py
python scripts/run_phase3.py "What are the Coffman conditions for deadlock?"
pytest
```

If `chroma_db/` is missing, the script indexes `os_lecture_notes.pdf` first. The graph is `studypilot/graph/qa_graph.py`: `route`, then `retrieve`, then `answer`. A quiz-shaped question skips retrieval.

The Phase 1 loop is still there (`python scripts/run_phase1.py`). Phase 3 does not replace it. It is the same idea with the branch written as an edge.

## Run the Phase 4 grader

Q&A now calls Gemini twice per attempt: once to draft, once to grade. A grade of "not supported" searches again, up to two extra times. If the notes still do not support an answer, you get `I couldn't find this in your material.`

```powershell
python scripts/run_phase4.py
pytest
```

The demo asks a deadlock question, then asks for the capital of France. The second one should give up, because that fact is not in the sample PDF. `RETRIES` and `GRADE` print above the answer.

Quiz questions, answer grades, and revision plans are Pydantic models in `studypilot/models/schemas.py`. The graph does not generate them yet.

`docs/LEARNING_NOTES.md` and `docs/DESIGN_DECISIONS.md` are local only (listed in `.gitignore`). They are not part of the repo.

If PowerShell refuses to run `Activate.ps1`, allow local scripts for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Layout

```
studypilot/     importable package (agents, graph, tools, rag, models, ui)
  llm.py        builds the Gemini chat model from .env
studypilot/rag/ load, chunk, embed, retrieve
data/sample/    study_notes.txt (Phase 1) and os_lecture_notes.pdf (Phase 2)
evals/          accuracy scripts, added in Phase 8
docs/           local learning notes (not tracked)
tests/          pytest, offline
studypilot/graph/  LangGraph Q&A with a grader retry
studypilot/models/ Pydantic schemas (grade, quiz, plan)
scripts/        check_setup.py, run_phase1.py, run_phase2.py, run_phase3.py, run_phase4.py, make_sample_pdf.py
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
| 5 | Quiz and weak-topic tracking | Not started |
| 6 | Revision plan with human approval | Not started |
| 7 | Supervisor and persisted sessions | Not started |
| 8 | Evaluation numbers | Not started |
| 9 | Streamlit UI and README polish | Not started |
