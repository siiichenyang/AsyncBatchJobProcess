import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Self


class EvalRunStore:
    """SQLite-backed storage for eval run records.

    The store intentionally works with plain JSON-compatible dicts/lists so it
    is not coupled to API response models. This keeps the storage schema stable
    while response models evolve, and makes a later migration to Postgres more
    straightforward.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_connection: sqlite3.Connection | None = None

        if db_path == ":memory:":
            self._memory_connection = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._memory_connection.row_factory = sqlite3.Row
            self._initialize(self._memory_connection)
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            connection = self._connect()
            try:
                self._initialize(connection)
            finally:
                connection.close()

    @classmethod
    def from_env(cls) -> Self:
        return cls(os.getenv("EVAL_RUN_DB_PATH") or "data/eval_runs.db")

    def close(self) -> None:
        with self._lock:
            if self._memory_connection is not None:
                self._memory_connection.close()
                self._memory_connection = None

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        connection = sqlite3.connect(
            self.db_path, timeout=30, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_runs (
                run_id TEXT PRIMARY KEY,
                results_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()

    def save_run(
        self,
        run_id: str,
        results: list[dict],
        summary: dict,
    ) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute(
                    """
                    INSERT INTO eval_runs (run_id, results_json, summary_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        run_id,
                        json.dumps(results, ensure_ascii=False),
                        json.dumps(summary, ensure_ascii=False),
                    ),
                )
                connection.commit()
            finally:
                if self._memory_connection is None:
                    connection.close()

    def get_summary(self, run_id: str) -> dict | None:
        with self._lock:
            connection = self._connect()
            try:
                row = connection.execute(
                    """
                    SELECT summary_json
                    FROM eval_runs
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
            finally:
                if self._memory_connection is None:
                    connection.close()

        if row is None:
            return None
        return json.loads(row["summary_json"])
