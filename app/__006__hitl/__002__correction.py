"""人工介入记录（SQLite）：支撑「介入率」与「修正类型分布」两个统计口径。

只记三件事：任务总数、哪些任务进入复核、每次复核人做了什么类型的修正。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logging_utils import get_logger  # noqa: E402

log = get_logger("corrections")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    ctype TEXT NOT NULL,
    detail_json TEXT,
    created_at REAL NOT NULL
);
"""


class CorrectionStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            from config import settings

            db_path = settings.session_db.parent / "hitl.sqlite3"
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def log_task(self, task_id: str, status: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO tasks(task_id, status, created_at) VALUES(?,?,?) "
                "ON CONFLICT(task_id) DO UPDATE SET status=excluded.status",
                (task_id, status, time.time()),
            )

    def log_correction(self, task_id: str, ctype: str, detail: dict[str, Any] | None = None) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO corrections(task_id, ctype, detail_json, created_at) VALUES(?,?,?,?)",
                (task_id, ctype, json.dumps(detail, ensure_ascii=False) if detail else None, time.time()),
            )

    def stats(self, n_reviewed: int) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            rows = conn.execute("SELECT ctype FROM corrections").fetchall()
        types = Counter(r[0] for r in rows)
        return {
            "total_tasks": total,
            "reviewed_tasks": n_reviewed,
            "intervention_rate": round(n_reviewed / max(total, 1), 4),
            "correction_types": dict(types),
            "corrections": sum(types.values()),
        }


def main() -> None:
    print(json.dumps(CorrectionStore().stats(0), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
