"""SQLite experiment logger."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                params TEXT,
                result TEXT,
                metrics TEXT
            );
            CREATE TABLE IF NOT EXISTS arc_puzzles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                puzzle_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                program TEXT,
                success INTEGER NOT NULL,
                training_pairs_passed INTEGER,
                total_training_pairs INTEGER,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS loop_iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER NOT NULL,
                problem TEXT NOT NULL,
                solution TEXT,
                verified INTEGER NOT NULL,
                reasoning_trace TEXT,
                timestamp TEXT NOT NULL
            );
        """)
        self.conn.commit()

    def log_experiment(self, project: str, run_id: str, status: str, params: dict | None = None, result: dict | None = None, metrics: dict | None = None):
        self.conn.execute(
            "INSERT INTO experiments (project, run_id, timestamp, status, params, result, metrics) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (project, run_id, datetime.now(timezone.utc).isoformat(), status, json.dumps(params), json.dumps(result), json.dumps(metrics)),
        )
        self.conn.commit()

    def log_arc_attempt(self, puzzle_id: str, attempt: int, program: str, success: bool, pairs_passed: int, total_pairs: int):
        self.conn.execute(
            "INSERT INTO arc_puzzles (puzzle_id, attempt, program, success, training_pairs_passed, total_training_pairs, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (puzzle_id, attempt, program, int(success), pairs_passed, total_pairs, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def log_loop_iteration(self, generation: int, problem: str, solution: str | None, verified: bool, reasoning_trace: str | None = None):
        self.conn.execute(
            "INSERT INTO loop_iterations (generation, problem, solution, verified, reasoning_trace, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (generation, problem, solution, int(verified), reasoning_trace, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def get_stats(self, project: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) FROM experiments WHERE project=?",
            (project,),
        ).fetchone()
        return {"total": row[0], "successes": row[1] or 0}

    def close(self):
        self.conn.close()
