"""
SQLite database service for history and feedback persistence.
Uses aiosqlite for async SQLite access.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import DB_PATH

logger = logging.getLogger(__name__)


_initialized = False


def _get_conn() -> sqlite3.Connection:
    global _initialized
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if not _initialized:
        init_db_conn(conn)
        _initialized = True
    return conn


def init_db_conn(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id            TEXT PRIMARY KEY,
            timestamp     TEXT NOT NULL,
            text_snippet  TEXT NOT NULL,
            context       TEXT,
            sensitivity   TEXT,
            label         TEXT NOT NULL,
            score         INTEGER NOT NULL,
            model_version TEXT,
            result_json   TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id               TEXT PRIMARY KEY,
            analysis_id      TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            helpful          INTEGER NOT NULL,
            corrected_label  TEXT,
            note             TEXT,
            model_version    TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_ts ON analyses(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_aid ON feedback(analysis_id)")
    conn.commit()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        init_db_conn(conn)
    logger.info("Database initialized at %s", DB_PATH)


def save_analysis(analysis_id: str, text: str, context: str, sensitivity: str, result_dict: dict) -> None:
    """Persist an analysis result. Stores only a text snippet (not full text)."""
    snippet = text[:100].replace('\n', ' ')
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses
            (id, timestamp, text_snippet, context, sensitivity, label, score, model_version, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                datetime.now(timezone.utc).isoformat(),
                snippet,
                context,
                sensitivity,
                result_dict.get("label", ""),
                result_dict.get("score", 0),
                result_dict.get("model_version", ""),
                json.dumps(result_dict),
            ),
        )
        conn.commit()


def get_all_analyses(limit: int = 200) -> list[dict]:
    """Fetch all analyses, most recent first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT result_json FROM analyses ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [json.loads(r["result_json"]) for r in rows]


def get_analysis(analysis_id: str) -> Optional[dict]:
    """Fetch a single analysis by ID."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def delete_all_analyses() -> int:
    """Delete all analyses and return count deleted."""
    with _get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        conn.execute("DELETE FROM analyses")
        conn.commit()
    return count


def save_feedback(
    analysis_id: str,
    helpful: bool,
    corrected_label: Optional[str],
    note: Optional[str],
    model_version: str,
) -> str:
    """Persist user feedback. Returns a new feedback ID."""
    import uuid
    fb_id = str(uuid.uuid4())[:8]
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO feedback
            (id, analysis_id, timestamp, helpful, corrected_label, note, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fb_id,
                analysis_id,
                datetime.now(timezone.utc).isoformat(),
                int(helpful),
                corrected_label,
                note,
                model_version,
            ),
        )
        conn.commit()
    return fb_id
