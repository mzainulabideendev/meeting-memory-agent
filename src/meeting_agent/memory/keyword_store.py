"""SQLite FTS5 keyword retrieval."""

from __future__ import annotations

import re
from typing import Any

from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import STOPWORDS, tokenize_words

log = get_logger(__name__)

_FTS_SPECIAL = re.compile(r'["*():^\-]')


def _sanitize_term(term: str) -> str:
    return _FTS_SPECIAL.sub(" ", term).strip()


def build_match_query(query: str, *, max_terms: int = 16, require_all: bool = False) -> str:
    terms = []
    for term in tokenize_words(query):
        if term in STOPWORDS or len(term) < 2:
            continue
        cleaned = _sanitize_term(term)
        if not cleaned:
            continue
        terms.append(cleaned)
        if len(terms) >= max_terms:
            break
    if not terms:
        return ""
    unique = list(dict.fromkeys(terms))
    joiner = " AND " if require_all else " OR "
    return joiner.join(f'"{t}"' for t in unique)


class KeywordStore:
    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()

    def search(
        self,
        query: str,
        *,
        top_k: int = 12,
        team_id: str | None = None,
        meeting_ids: list[str] | None = None,
        speaker: str | None = None,
        require_all: bool = False,
    ) -> list[dict[str, Any]]:
        match = build_match_query(query, require_all=require_all)
        if not match:
            return []
        filters = []
        params: list[Any] = []
        if team_id:
            filters.append("c.team_id = ?")
            params.append(team_id)
        if meeting_ids:
            filters.append(f"c.meeting_id IN ({','.join('?' for _ in meeting_ids)})")
            params.extend(meeting_ids)
        if speaker:
            filters.append("c.speaker = ?")
            params.append(speaker)
        where_clause = (" AND " + " AND ".join(filters)) if filters else ""

        sql = f"""
            SELECT c.chunk_id, c.meeting_id, c.team_id, c.sequence, c.speaker, c.original_text,
                   c.embed_text, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?{where_clause}
            ORDER BY rank
            LIMIT ?
        """
        try:
            with self.db.connect() as connection:
                rows = connection.execute(sql, [match, *params, top_k]).fetchall()
            results = []
            for row in rows:
                rank = float(row["rank"])
                results.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "meeting_id": row["meeting_id"],
                        "team_id": row["team_id"],
                        "meeting_sequence": int(row["sequence"]),
                        "speaker": row["speaker"],
                        "text": row["original_text"],
                        "rank": rank,
                        # bm25 in SQLite is negative; map to a positive score
                        "score": 1.0 / (1.0 + abs(rank)),
                        "source": "keyword",
                    }
                )
            return results
        except Exception as exc:  # pragma: no cover - fallback path
            log.warning("FTS query failed (%s); falling back to LIKE search", exc)
            return self._like_search(query, top_k=top_k, team_id=team_id, meeting_ids=meeting_ids)

    def _like_search(
        self,
        query: str,
        *,
        top_k: int,
        team_id: str | None,
        meeting_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        terms = [t for t in tokenize_words(query) if t not in STOPWORDS and len(t) > 1][:6]
        if not terms:
            return []
        filters = []
        params: list[Any] = []
        for term in terms:
            filters.append("(lower(original_text) LIKE ?)")
            params.append(f"%{term}%")
        sql = f"SELECT * FROM chunks WHERE {' AND '.join(filters)}"
        if team_id:
            sql += " AND team_id = ?"
            params.append(team_id)
        if meeting_ids:
            sql += f" AND meeting_id IN ({','.join('?' for _ in meeting_ids)})"
            params.extend(meeting_ids)
        sql += " LIMIT ?"
        params.append(top_k)
        with self.db.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "meeting_id": row["meeting_id"],
                "team_id": row["team_id"],
                "meeting_sequence": int(row["sequence"]),
                "speaker": row["speaker"],
                "text": row["original_text"],
                "rank": 0.0,
                "score": 0.5,
                "source": "keyword_like",
            }
            for row in rows
        ]
