"""End-to-end ingestion pipeline: parse -> chunk -> embed -> persist.

The pipeline is incremental and restartable.  A file whose SHA-256 matches the
last successful ingestion is skipped; a changed file has its old meeting, turns,
chunks and vector entries removed atomically before re-ingestion.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from meeting_agent.config import Settings, get_settings
from meeting_agent.ingestion.chunker import chunk_meeting, count_tokens
from meeting_agent.ingestion.metadata import build_team_structures, infer_roles, speaker_distribution
from meeting_agent.ingestion.parser import file_sha256, iter_transcript_files, parse_transcript_text
from meeting_agent.memory.vector_store import VectorStore
from meeting_agent.models import IngestionStats, MeetingMeta, Turn
from meeting_agent.storage import Database
from meeting_agent.utils.embeddings import EmbedderProtocol, get_embedder
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        database: Database | None = None,
        vector_store: VectorStore | None = None,
        embedder: EmbedderProtocol | None = None,
        *,
        reset: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database()
        self.vector_store = vector_store or VectorStore()
        self._embedder = embedder
        self.reset = reset

    @property
    def embedder(self) -> EmbedderProtocol:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def run(self, transcripts_dir: Path | None = None, limit: int | None = None, reset: bool | None = None) -> IngestionStats:
        directory = Path(transcripts_dir) if transcripts_dir else self.settings.resolved_transcripts_dir
        files = iter_transcript_files(directory)
        if limit:
            files = files[:limit]

        stats = IngestionStats(files_seen=len(files), started_at=datetime.now(timezone.utc))
        started = time.perf_counter()

        if reset if reset is not None else self.reset:
            log.warning("Reset requested: clearing existing meetings, chunks and vectors")
            with self.db.connect() as connection:
                connection.execute("DELETE FROM chunks_fts")
                connection.execute("DELETE FROM chunks")
                connection.execute("DELETE FROM turns")
                connection.execute("DELETE FROM meetings")
                connection.execute("DELETE FROM ingestion_manifest")
                connection.execute("DELETE FROM memory_records")
                connection.execute("DELETE FROM actions")
                connection.execute("DELETE FROM open_questions")
            self.vector_store.reset()

        parsed_all: list[tuple[MeetingMeta, list[Turn]]] = []
        seen_meeting_ids: dict[str, Path] = {}
        hash_cache: dict[Path, str] = {}

        for path in files:
            key = str(path.resolve())
            try:
                hash_cache[path] = file_sha256(path)
            except OSError as exc:
                stats.files_failed += 1
                stats.errors.append({"file": key, "error": f"read failed: {exc}"})
                self.db.record_manifest(
                    source_file=key, meeting_id=None, file_hash=None, size_bytes=None,
                    status="failed", error=str(exc),
                )
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                meta, turns = parse_transcript_text(raw, source_file=key)
                meta.file_hash = hash_cache[path]
                meta.word_count = sum(len(t.text.split()) for t in turns)
                meta.turn_count = len(turns)
            except Exception as exc:
                stats.files_failed += 1
                stats.errors.append({"file": key, "error": str(exc)})
                self.db.record_manifest(
                    source_file=key, meeting_id=None, file_hash=hash_cache[path],
                    size_bytes=path.stat().st_size if path.exists() else None,
                    status="failed", error=str(exc),
                )
                log.error("Failed to parse %s: %s", path.name, exc)
                continue

            if meta.meeting_id in seen_meeting_ids:
                stats.files_failed += 1
                message = f"duplicate meeting id {meta.meeting_id} also in {seen_meeting_ids[meta.meeting_id].name}"
                stats.errors.append({"file": key, "error": message})
                self.db.record_manifest(
                    source_file=key, meeting_id=meta.meeting_id, file_hash=meta.file_hash,
                    size_bytes=path.stat().st_size, status="failed", error=message,
                )
                continue
            seen_meeting_ids[meta.meeting_id] = path
            parsed_all.append((meta, turns))

            manifest = self.db.manifest_status(key)
            unchanged = (
                manifest is not None
                and manifest["status"] == "success"
                and manifest["file_hash"] == meta.file_hash
            )
            if unchanged:
                stats.files_skipped += 1
                log.info("Skipping unchanged %s", path.name)
                continue

            chunk_started = time.perf_counter()
            try:
                chunks = chunk_meeting(
                    meta,
                    turns,
                    target_tokens=self.settings.chunk_target_tokens,
                    max_tokens=self.settings.chunk_max_tokens,
                    overlap_tokens=self.settings.chunk_overlap_tokens,
                    encoding=self.settings.tokenizer_encoding,
                )
                # Remove prior artefacts before writing the new version.
                old_chunk_ids = self.db.delete_meeting(meta.meeting_id)
                if old_chunk_ids:
                    self.vector_store.delete(old_chunk_ids)

                self.db.insert_meeting(meta)
                self.db.insert_turns(turns)
                self.db.insert_chunks(chunks)

                embeddings_started = time.perf_counter()
                embeddings: list[list[float]] = []
                batch = max(1, self.settings.embedding_batch_size)
                for i in range(0, len(chunks), batch):
                    embeddings.extend(self.embedder.encode([c.embed_text for c in chunks[i : i + batch]]))
                stats.embedding_seconds += time.perf_counter() - embeddings_started
                self.vector_store.upsert(chunks, embeddings)

                elapsed = time.perf_counter() - chunk_started
                self.db.record_manifest(
                    source_file=key, meeting_id=meta.meeting_id, file_hash=meta.file_hash,
                    size_bytes=path.stat().st_size, status="success", chunk_count=len(chunks),
                    duration_seconds=elapsed,
                )
                self.db.record_metric("chunk_count", float(len(chunks)), "chunks", meta.meeting_id)
                stats.files_ingested += 1
                stats.meetings += 1
                stats.turns += len(turns)
                stats.chunks += len(chunks)
                stats.words += meta.word_count
                stats.tokens += sum(c.token_count for c in chunks)
                log.info(
                    "Ingested %s: %d turns, %d chunks, %d words",
                    meta.meeting_id, len(turns), len(chunks), meta.word_count,
                )
            except Exception as exc:
                stats.files_failed += 1
                stats.errors.append({"file": key, "error": str(exc)})
                self.db.record_manifest(
                    source_file=key, meeting_id=meta.meeting_id, file_hash=meta.file_hash,
                    size_bytes=path.stat().st_size if path.exists() else None,
                    status="failed", error=str(exc),
                )
                log.exception("Ingestion failed for %s", path.name)

        # Role inference is team-wide and cheap; refresh it for every team.
        self._store_roles(parsed_all)

        stats.total_seconds = time.perf_counter() - started
        stats.finished_at = datetime.now(timezone.utc)
        self.db.record_metric("ingestion_total_seconds", stats.total_seconds, "seconds")
        self.db.record_metric("embedding_total_seconds", stats.embedding_seconds, "seconds")
        log.info(
            "Ingestion done: %d ingested, %d skipped, %d failed, %d chunks in %.1fs",
            stats.files_ingested, stats.files_skipped, stats.files_failed, stats.chunks, stats.total_seconds,
        )
        return stats

    def _store_roles(self, parsed_all: list[tuple[MeetingMeta, list[Turn]]]) -> None:
        roles = infer_roles(parsed_all)
        by_team: dict[str, list] = {}
        for role in roles:
            team_id = next(
                (meta.team_id for meta, _ in parsed_all if meta.meeting_id == role.meeting_id), ""
            )
            if team_id:
                by_team.setdefault(team_id, []).append(role)
        for team_id, team_roles in by_team.items():
            self.db.record_roles(team_id, team_roles)
            log.info("Inferred %d role statements for %s", len(team_roles), team_id)


def build_dataset_report(
    settings: Settings | None = None,
    database: Database | None = None,
    stats: IngestionStats | None = None,
) -> dict:
    """Inspect the dataset from actual files (not from assumptions)."""
    settings = settings or get_settings()
    directory = settings.resolved_transcripts_dir
    files = iter_transcript_files(directory)
    parsed: list[tuple[MeetingMeta, list[Turn]]] = []
    malformed: list[dict] = []
    empty: list[str] = []
    for path in files:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            if not raw.strip():
                empty.append(str(path))
                continue
            meta, turns = parse_transcript_text(raw, source_file=str(path))
            meta.file_hash = file_sha256(path)
            parsed.append((meta, turns))
        except Exception as exc:
            malformed.append({"file": str(path), "error": str(exc)})

    structures = build_team_structures(files, parsed)
    total_words = sum(len(t.text.split()) for _, turns in parsed for t in turns)
    total_tokens = sum(count_tokens(t.text) for _, turns in parsed for t in turns)
    speakers: dict[str, int] = {}
    for _, turns in parsed:
        for turn in turns:
            speakers[turn.speaker] = speakers.get(turn.speaker, 0) + 1

    questions_count = 0
    questions_path = settings.resolved_questions_path
    if questions_path.exists():
        try:
            questions_count = len(json.loads(questions_path.read_text(encoding="utf-8")).get("questions", []))
        except Exception:
            questions_count = 0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transcripts_dir": str(directory),
        "total_transcript_files": len(files),
        "total_teams": len(structures),
        "total_meetings": len(parsed),
        "total_words_body_only": total_words,
        "total_estimated_tokens": total_tokens,
        "meetings_per_team": {team: len(s.meetings) for team, s in sorted(structures.items())},
        "team_details": {
            team: {
                "meetings": s.meetings,
                "missing_letters": s.missing_letters,
                "roles": [asdict(r) for r in s.roles],
            }
            for team, s in sorted(structures.items())
        },
        "teams_with_missing_meetings": {
            team: s.missing_letters for team, s in sorted(structures.items()) if s.missing_letters
        },
        "malformed_transcripts": malformed,
        "empty_transcripts": empty,
        "speaker_distribution": dict(sorted(speakers.items())),
        "speaker_distribution_by_meeting": speaker_distribution(parsed),
        "questions_count": questions_count,
        "ingestion_stats": stats.model_dump(mode="json") if stats else None,
        "database_meetings": database.meeting_count() if database else None,
        "database_chunks": database.chunk_count() if database else None,
    }
    return report
