from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Any, Mapping


class DecisionLedger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    forecast_id TEXT NOT NULL,
                    scenario_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def record(self, action: str, scenario: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
        forecast_id = str(summary["forecast_id"])
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO decisions(created_at, action, forecast_id, scenario_json, summary_json) VALUES (?, ?, ?, ?, ?)",
                (created_at, action, forecast_id, json.dumps(scenario, sort_keys=True), json.dumps(summary, sort_keys=True)),
            )
        return {"id": cursor.lastrowid, "created_at": created_at, "action": action, "forecast_id": forecast_id}

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, created_at, action, forecast_id, summary_json FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row[0], "created_at": row[1], "action": row[2], "forecast_id": row[3], "summary": json.loads(row[4])} for row in rows]
