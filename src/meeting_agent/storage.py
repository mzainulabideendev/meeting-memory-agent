"""SQLite schema and persistence helpers.

SQLite is the structured source of truth: meetings, turns, chunks, extracted
memory records, ingestion manifest, citation-attempt audit log and query logs.
FTS5 provides deterministic keyword retrieval over the same chunks.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meeting_agent.config import get_settings
from meeting_agent.models import (
    ActionItem,
    Chunk,
    Citation,
    DecisionRecord,
    MeetingMeta,
    OpenQuestion,
    Turn,
)
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    letter TEXT NOT NULL,
    source_file TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    turn_count INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ingested',
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_meetings_team ON meetings(team_id, sequence);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    team_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    start_time TEXT NOT NULL,
    text TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_meeting ON turns(meeting_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_speaker ON turns(team_id, speaker);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    team_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    speakers TEXT NOT NULL,
    original_text TEXT NOT NULL,
    embed_text TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_start_offset INTEGER NOT NULL,
    source_end_offset INTEGER NOT NULL,
    embedding_id TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    turn_indices TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_chunks_team ON chunks(team_id, sequence);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    embed_text,
    team_id,
    meeting_id,
    speaker,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    source_file TEXT PRIMARY KEY,
    meeting_id TEXT,
    file_hash TEXT,
    size_bytes INTEGER,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    duration_seconds REAL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_roles (
    team_id TEXT NOT NULL,
    speaker TEXT NOT NULL,
    role TEXT NOT NULL,
    meeting_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    quote TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY (team_id, speaker, role)
);

CREATE TABLE IF NOT EXISTS memory_records (
    record_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    topic TEXT NOT NULL,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    meeting_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    turn_index INTEGER NOT NULL DEFAULT 0,
    speaker TEXT,
    valid_from TEXT,
    valid_to TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL,
    extraction_version TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_team ON memory_records(team_id, sequence);
CREATE INDEX IF NOT EXISTS idx_memory_topic ON memory_records(team_id, topic);

CREATE TABLE IF NOT EXISTS actions (
    action_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    description TEXT NOT NULL,
    owner_speaker TEXT,
    owner_role TEXT,
    deadline TEXT,
    status TEXT NOT NULL,
    meeting_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    evidence TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_team ON actions(team_id, sequence);

CREATE TABLE IF NOT EXISTS open_questions (
    question_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    text TEXT NOT NULL,
    meeting_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    status TEXT NOT NULL,
    resolution TEXT,
    evidence TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_runs (
    meeting_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    extractor TEXT NOT NULL,
    records INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citation_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,
    meeting_id TEXT,
    quote TEXT,
    valid INTEGER NOT NULL,
    reason TEXT,
    source TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_logs (
    query_id TEXT PRIMARY KEY,
    question_id TEXT,
    team TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    metric TEXT PRIMARY KEY,
    value REAL NOT NULL,
    unit TEXT,
    detail TEXT,
    measured_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path | str | None = None) -> None:
        settings = get_settings()
        self.path = Path(path) if path else settings.database_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5 = True
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_schema(self) -> None:
        try:
            with self.connect() as connection:
                connection.executescript(SCHEMA)
                self._migrate(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
                    (str(SCHEMA_VERSION),),
                )
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                self._fts5 = False
                log.warning("SQLite FTS5 unavailable (%s); keyword search will fall back to LIKE", exc)
                with self.connect() as connection:
                    connection.executescript(SCHEMA.replace("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5", "CREATE TABLE IF NOT EXISTS chunks_fts_disabled"))
            else:
                raise

    def _migrate(self, connection: sqlite3.Connection) -> None:
        """Apply additive migrations for databases created by older versions."""
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(memory_records)").fetchall()
        }
        if "turn_index" not in columns:
            connection.execute(
                "ALTER TABLE memory_records ADD COLUMN turn_index INTEGER NOT NULL DEFAULT 0"
            )

    # ------------------------------------------------------------------ reads
    def meeting_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0])

    def chunk_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_meeting(self, meeting_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute("SELECT * FROM meetings WHERE meeting_id = ?", (meeting_id,)).fetchone()

    def get_meeting_raw_text(self, meeting_id: str) -> str | None:
        row = self.get_meeting(meeting_id)
        return row["raw_text"] if row else None

    def get_turn(self, meeting_id: str, turn_index: int) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM turns WHERE meeting_id = ? AND turn_index = ?",
                (meeting_id, turn_index),
            ).fetchone()

    def get_turns(self, meeting_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM turns WHERE meeting_id = ? ORDER BY turn_index", (meeting_id,)
            ).fetchall()

    def get_chunks_for_meeting(self, meeting_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM chunks WHERE meeting_id = ? ORDER BY sequence, chunk_id", (meeting_id,)
            ).fetchall()

    def get_chunks_for_team(self, team_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM chunks WHERE team_id = ? ORDER BY sequence, chunk_id", (team_id,)
            ).fetchall()

    def get_chunks(self, chunk_ids: Iterable[str]) -> list[sqlite3.Row]:
        ids = list(chunk_ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            return connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", ids
            ).fetchall()

    def list_teams(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT team_id, COUNT(*) AS meetings, MIN(sequence) AS first_seq, MAX(sequence) AS last_seq "
                "FROM meetings GROUP BY team_id ORDER BY team_id"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_meetings(self, team_id: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if team_id:
                return connection.execute(
                    "SELECT * FROM meetings WHERE team_id = ? ORDER BY sequence", (team_id,)
                ).fetchall()
            return connection.execute("SELECT * FROM meetings ORDER BY team_id, sequence").fetchall()

    def team_roles(self, team_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM team_roles WHERE team_id = ? ORDER BY speaker, confidence DESC", (team_id,)
            ).fetchall()

    # ----------------------------------------------------------------- writes
    def manifest_status(self, source_file: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM ingestion_manifest WHERE source_file = ?", (source_file,)
            ).fetchone()

    def record_manifest(
        self,
        *,
        source_file: str,
        meeting_id: str | None,
        file_hash: str | None,
        size_bytes: int | None,
        status: str,
        chunk_count: int = 0,
        error: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO ingestion_manifest
                   (source_file, meeting_id, file_hash, size_bytes, status, chunk_count, error, duration_seconds, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_file) DO UPDATE SET
                     meeting_id=excluded.meeting_id, file_hash=excluded.file_hash,
                     size_bytes=excluded.size_bytes, status=excluded.status,
                     chunk_count=excluded.chunk_count, error=excluded.error,
                     duration_seconds=excluded.duration_seconds, ingested_at=excluded.ingested_at""",
                (source_file, meeting_id, file_hash, size_bytes, status, chunk_count, error, duration_seconds, _utcnow()),
            )

    def delete_meeting(self, meeting_id: str) -> list[str]:
        """Remove a meeting and return its chunk ids (for vector deletion)."""
        with self.connect() as connection:
            chunk_ids = [
                r["chunk_id"] for r in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE meeting_id = ?", (meeting_id,)
                ).fetchall()
            ]
            if self._fts5:
                connection.execute("DELETE FROM chunks_fts WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM turns WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM chunks WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM meetings WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM memory_records WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM actions WHERE meeting_id = ?", (meeting_id,))
            connection.execute("DELETE FROM open_questions WHERE meeting_id = ?", (meeting_id,))
            return chunk_ids

    def insert_meeting(self, meta: MeetingMeta) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO meetings
                   (meeting_id, team_id, sequence, letter, source_file, file_hash, word_count,
                    turn_count, raw_text, status, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ingested', ?)""",
                (
                    meta.meeting_id, meta.team_id, meta.sequence, meta.letter, meta.source_file,
                    meta.file_hash, meta.word_count, meta.turn_count, meta.raw_text, _utcnow(),
                ),
            )

    def insert_turns(self, turns: list[Turn]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO turns
                   (turn_id, meeting_id, team_id, sequence, turn_index, speaker, start_time, text,
                    start_offset, end_offset)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (t.turn_id, t.meeting_id, t.team_id, t.sequence, t.turn_index, t.speaker,
                     t.start_time, t.text, t.start_offset, t.end_offset)
                    for t in turns
                ],
            )

    def insert_chunks(self, chunks: list[Chunk]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, meeting_id, team_id, sequence, speaker, speakers, original_text,
                    embed_text, source_file, source_start_offset, source_end_offset,
                    embedding_id, token_count, turn_indices)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (c.chunk_id, c.meeting_id, c.team_id, c.meeting_sequence, c.speaker,
                     json.dumps(c.speakers), c.original_text, c.embed_text, c.source_file,
                     c.source_start_offset, c.source_end_offset, c.embedding_id, c.token_count,
                     json.dumps(c.turn_indices))
                    for c in chunks
                ],
            )
            if self._fts5:
                connection.executemany(
                    "INSERT INTO chunks_fts(chunk_id, embed_text, team_id, meeting_id, speaker) VALUES (?, ?, ?, ?, ?)",
                    [(c.chunk_id, c.embed_text, c.team_id, c.meeting_id, c.speaker) for c in chunks],
                )

    def record_roles(self, team_id: str, roles: list[Any]) -> None:
        with self.connect() as connection:
            for role in roles:
                connection.execute(
                    """INSERT OR REPLACE INTO team_roles
                       (team_id, speaker, role, meeting_id, turn_index, quote, confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (team_id, role.speaker, role.role, role.meeting_id, role.turn_index, role.quote, role.confidence),
                )

    def replace_memory_records(self, team_id: str, records: list[DecisionRecord]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM memory_records WHERE team_id = ?", (team_id,))
            connection.executemany(
                """INSERT OR REPLACE INTO memory_records
                   (record_id, team_id, kind, topic, statement, status, meeting_id, sequence,
                    turn_index, speaker, valid_from, valid_to, supersedes, superseded_by, evidence,
                    confidence, extraction_version, reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (r.record_id, r.team_id, r.kind, r.topic, r.statement, r.status,
                     r.meeting_id, r.meeting_sequence, r.turn_index, r.speaker, r.valid_from_meeting,
                     r.valid_to_meeting, r.supersedes, r.superseded_by,
                     json.dumps([c.model_dump() for c in r.evidence]), r.confidence,
                     r.extraction_version, r.reason, _utcnow())
                    for r in records
                ],
            )

    def get_memory_records(self, team_id: str, topic: str | None = None) -> list[sqlite3.Row]:
        with self.connect() as connection:
            if topic:
                return connection.execute(
                    "SELECT * FROM memory_records WHERE team_id = ? AND topic = ? ORDER BY sequence",
                    (team_id, topic),
                ).fetchall()
            return connection.execute(
                "SELECT * FROM memory_records WHERE team_id = ? ORDER BY sequence, record_id", (team_id,)
            ).fetchall()

    def replace_actions(self, team_id: str, actions: list[ActionItem]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM actions WHERE team_id = ?", (team_id,))
            connection.executemany(
                """INSERT OR REPLACE INTO actions
                   (action_id, team_id, description, owner_speaker, owner_role, deadline, status,
                    meeting_id, sequence, evidence, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (a.action_id, a.team_id, a.description, a.owner_speaker, a.owner_role,
                     a.deadline, a.status, a.meeting_id, a.meeting_sequence,
                     json.dumps([c.model_dump() for c in a.evidence]), a.confidence, _utcnow())
                    for a in actions
                ],
            )

    def get_actions(self, team_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM actions WHERE team_id = ? ORDER BY sequence, action_id", (team_id,)
            ).fetchall()

    def replace_open_questions(self, team_id: str, questions: list[OpenQuestion]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM open_questions WHERE team_id = ?", (team_id,))
            connection.executemany(
                """INSERT OR REPLACE INTO open_questions
                   (question_id, team_id, text, meeting_id, sequence, status, resolution, evidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (q.question_id, q.team_id, q.text, q.meeting_id, q.meeting_sequence,
                     q.status, q.resolution, json.dumps([c.model_dump() for c in q.evidence]), _utcnow())
                    for q in questions
                ],
            )

    def get_open_questions(self, team_id: str) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM open_questions WHERE team_id = ? ORDER BY sequence", (team_id,)
            ).fetchall()

    def log_citation_attempt(
        self, question_id: str | None, citation: Citation, valid: bool, reason: str, source: str
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO citation_attempts(question_id, meeting_id, quote, valid, reason, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (question_id, citation.meeting_id, citation.quote, int(valid), reason, source, _utcnow()),
            )

    def citation_stats(self) -> dict[str, int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT valid, COUNT(*) AS n FROM citation_attempts GROUP BY valid"
            ).fetchall()
            return {("valid" if r["valid"] else "invalid"): int(r["n"]) for r in rows}

    def log_query(self, payload: dict[str, Any], question_id: str | None = None, team: str | None = None) -> None:
        import uuid

        with self.connect() as connection:
            connection.execute(
                "INSERT INTO query_logs(query_id, question_id, team, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (uuid.uuid4().hex, question_id, team, json.dumps(payload, default=str), _utcnow()),
            )

    def record_metric(self, metric: str, value: float, unit: str = "", detail: str = "") -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO performance_metrics(metric, value, unit, detail, measured_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (metric, value, unit, detail, _utcnow()),
            )

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM performance_metrics").fetchall()
            return {r["metric"]: dict(r) for r in rows}
