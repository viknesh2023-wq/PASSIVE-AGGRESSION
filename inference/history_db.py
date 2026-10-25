"""
inference/history_db.py
SQLite history persistence for Streamlit — standalone (no FastAPI).
"""
from __future__ import annotations
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_HERE    = Path(__file__).resolve().parent.parent
DB_PATH  = _HERE / "backend" / "pa_detector.db"

_initialized = False


def _conn() -> sqlite3.Connection:
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    if not _initialized:
        _init_schema(c)
        _initialized = True
    return c


def _init_schema(c: sqlite3.Connection) -> None:
    c.execute("""
        CREATE TABLE IF NOT EXISTS pa_analyses_v2 (
            id            TEXT PRIMARY KEY,
            timestamp     TEXT NOT NULL,
            text_snippet  TEXT NOT NULL,
            context       TEXT,
            sensitivity   TEXT,
            ml_label      TEXT,
            rule_label    TEXT,
            threat_label  TEXT,
            pa_score      REAL,
            model_version TEXT,
            result_json   TEXT NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_v2_ts ON pa_analyses_v2(timestamp)")
    c.commit()


def save(analysis_id: str, text: str, context: str, sensitivity: str, result: dict) -> None:
    snippet = text[:100].replace("\n", " ")
    combined = result.get("combined", {})
    ml       = result.get("ml_result", {})
    threat   = result.get("threat_result", {})
    rule     = result.get("rule_result", {})
    model_v  = result.get("model_info", {}).get("pa_model_version", "unknown")

    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO pa_analyses_v2
               (id, timestamp, text_snippet, context, sensitivity,
                ml_label, rule_label, threat_label, pa_score, model_version, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis_id,
                datetime.now(timezone.utc).isoformat(),
                snippet,
                context,
                sensitivity,
                ml.get("label", ""),
                rule.get("pa_label", ""),
                combined.get("threat", threat.get("label", "")),
                ml.get("pa_score", 0),
                model_v,
                json.dumps(result, default=str),
            ),
        )
        c.commit()


def get_all(limit: int = 200) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, timestamp, text_snippet, context, sensitivity, ml_label, rule_label, "
            "threat_label, pa_score, model_version, result_json "
            "FROM pa_analyses_v2 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_one(analysis_id: str) -> Optional[dict]:
    with _conn() as c:
        row = c.execute(
            "SELECT result_json FROM pa_analyses_v2 WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def delete_all() -> int:
    with _conn() as c:
        count = c.execute("SELECT COUNT(*) FROM pa_analyses_v2").fetchone()[0]
        c.execute("DELETE FROM pa_analyses_v2")
        c.commit()
    return count


def get_stats() -> dict:
    """Return aggregate statistics from history."""
    with _conn() as c:
        total  = c.execute("SELECT COUNT(*) FROM pa_analyses_v2").fetchone()[0]
        labels = c.execute(
            "SELECT ml_label, COUNT(*) as cnt FROM pa_analyses_v2 GROUP BY ml_label"
        ).fetchall()
        threats = c.execute(
            "SELECT threat_label, COUNT(*) as cnt FROM pa_analyses_v2 GROUP BY threat_label"
        ).fetchall()
    return {
        "total":   total,
        "by_label":  {r["ml_label"]: r["cnt"] for r in labels},
        "by_threat": {r["threat_label"]: r["cnt"] for r in threats},
    }


def delete(analysis_id: str) -> bool:
    """Delete a single analysis by ID."""
    with _conn() as c:
        cursor = c.execute("DELETE FROM pa_analyses_v2 WHERE id = ?", (analysis_id,))
        c.commit()
        return cursor.rowcount > 0


def init_db() -> None:
    """Initialize database and schema."""
    with _conn():
        pass

