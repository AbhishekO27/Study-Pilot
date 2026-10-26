"""One SQLite file for a study session.

Two different kinds of rows share the file:

- topic_stats and revision_plans are application data. We query them.
- LangGraph's own tables remember a paused graph so approval can resume
  after this process exits. We do not read those tables ourselves.

A session_id is the student key. This demo uses one id, "local".
A second student is another id in the same file, not a new database.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from studypilot.quiz.tracker import TopicStat, WeakTopicTracker

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "sessions" / "studypilot.sqlite"
DEFAULT_SESSION_ID = "local"


class SessionStore:
    """Weak topics and the latest revision plan. Not the graph checkpointer."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS topic_stats (
                session_id TEXT NOT NULL,
                topic_key TEXT NOT NULL,
                topic TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                correct INTEGER NOT NULL,
                last_feedback TEXT NOT NULL,
                PRIMARY KEY (session_id, topic_key)
            );
            CREATE TABLE IF NOT EXISTS revision_plans (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                answer TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def load_tracker(self, session_id: str = DEFAULT_SESSION_ID) -> WeakTopicTracker:
        rows = self._conn.execute(
            """
            SELECT topic, attempts, correct, last_feedback
            FROM topic_stats
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchall()
        tracker = WeakTopicTracker()
        tracker.restore(
            [
                TopicStat(
                    topic=topic,
                    attempts=attempts,
                    correct=correct,
                    last_feedback=feedback,
                )
                for topic, attempts, correct, feedback in rows
            ]
        )
        return tracker

    def save_tracker(
        self,
        tracker: WeakTopicTracker,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        self._conn.execute("DELETE FROM topic_stats WHERE session_id = ?", (session_id,))
        self._conn.executemany(
            """
            INSERT INTO topic_stats
                (session_id, topic_key, topic, attempts, correct, last_feedback)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    stat.topic.casefold(),
                    stat.topic,
                    stat.attempts,
                    stat.correct,
                    stat.last_feedback,
                )
                for stat in tracker.stats()
            ],
        )
        self._conn.commit()

    def save_plan(
        self,
        status: str,
        plan: dict,
        answer: str,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO revision_plans (session_id, status, plan_json, answer)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                plan_json = excluded.plan_json,
                answer = excluded.answer
            """,
            (session_id, status, json.dumps(plan), answer),
        )
        self._conn.commit()

    def load_plan(
        self,
        session_id: str = DEFAULT_SESSION_ID,
    ) -> tuple[str, dict, str] | None:
        row = self._conn.execute(
            """
            SELECT status, plan_json, answer
            FROM revision_plans
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        status, plan_json, answer = row
        return status, json.loads(plan_json), answer


def open_checkpointer(path: Path = DEFAULT_DB_PATH) -> SqliteSaver:
    """LangGraph saver for the same file. Call conn.close() when the run is done.

    This is a second connection. SessionStore does not read its tables.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
