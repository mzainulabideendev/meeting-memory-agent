"""Ingestion workflow: transcripts -> episodic + semantic + temporal memory."""

from __future__ import annotations

import json
from pathlib import Path

from meeting_agent.config import Settings, get_settings
from meeting_agent.ingestion.metadata import infer_roles
from meeting_agent.ingestion.parser import iter_transcript_files, parse_transcript_text
from meeting_agent.ingestion.pipeline import IngestionPipeline, build_dataset_report
from meeting_agent.memory.action_store import ActionStore
from meeting_agent.memory.decision_store import DecisionStore
from meeting_agent.memory.semantic_memory import extract_all_meetings
from meeting_agent.models import IngestionStats, MeetingMeta, Turn
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class IngestionWorkflow:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database()
        self.pipeline = IngestionPipeline(self.settings, self.db)

    def run(
        self,
        *,
        reset: bool = False,
        limit: int | None = None,
        extract: bool = True,
        transcripts_dir: Path | None = None,
    ) -> tuple[IngestionStats, dict]:
        import time

        stats = self.pipeline.run(transcripts_dir=transcripts_dir, reset=reset, limit=limit)

        # Parse all files again from disk to extract semantics from the full
        # corpus (fast; parsing is not the bottleneck).  Skipped files remain in
        # SQLite, but memory extraction is idempotent and refreshable.
        parsed = self._parse_all()
        extraction_started = time.perf_counter()
        if extract and parsed:
            self._extract_memory(parsed)
        extraction_seconds = time.perf_counter() - extraction_started

        report = build_dataset_report(self.settings, self.db, stats)
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.outputs_dir / "dataset_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.db.record_metric("corpus_meetings", float(report["database_meetings"] or 0), "meetings")
        self.db.record_metric("corpus_chunks", float(report["database_chunks"] or 0), "chunks")
        self.db.record_metric("corpus_words", float(report["total_words_body_only"]), "words")
        self.db.record_metric("corpus_tokens", float(report["total_estimated_tokens"]), "tokens")
        self.db.record_metric("semantic_extraction_seconds", extraction_seconds, "seconds")
        return stats, report

    def _parse_all(self) -> list[tuple[MeetingMeta, list[Turn]]]:
        parsed: list[tuple[MeetingMeta, list[Turn]]] = []
        for path in iter_transcript_files(self.settings.resolved_transcripts_dir):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                meta, turns = parse_transcript_text(raw, source_file=str(path))
                parsed.append((meta, turns))
            except Exception as exc:
                log.warning("Skipping %s during memory extraction: %s", path.name, exc)
        return parsed

    def _extract_memory(self, parsed: list[tuple[MeetingMeta, list[Turn]]]) -> None:
        roles = infer_roles(parsed)
        roles_by_team: dict[str, dict[str, str]] = {}
        for role in roles:
            team = next((m.team_id for m, _ in parsed if m.meeting_id == role.meeting_id), "")
            if team:
                roles_by_team.setdefault(team, {}).setdefault(role.speaker, role.role)

        bundles = extract_all_meetings(parsed, roles_by_team)
        decision_store = DecisionStore(self.db)
        action_store = ActionStore(self.db)

        by_team: dict[str, list] = {}
        actions_by_team: dict[str, list] = {}
        questions_by_team: dict[str, list] = {}
        for meta, turns in parsed:
            bundle = bundles[meta.meeting_id]
            by_team.setdefault(meta.team_id, []).extend(bundle.decisions)
            actions_by_team.setdefault(meta.team_id, []).extend(bundle.actions)
            questions_by_team.setdefault(meta.team_id, []).extend(bundle.open_questions)

        for team_id, records in by_team.items():
            decision_store.save(team_id, records)
            log.info("Persisted %d semantic records for %s", len(records), team_id)

        # Conservative completion tracking across each team's meetings.
        for team_id in sorted(by_team):
            team_meetings = [
                (meta.meeting_id, meta.sequence, turns)
                for meta, turns in parsed if meta.team_id == team_id
            ]
            actions = actions_by_team.get(team_id, [])
            questions = questions_by_team.get(team_id, [])
            actions, questions = action_store.update_completion(team_id, team_meetings, actions, questions)
            action_store.save(team_id, actions)
            self.db.replace_open_questions(team_id, questions)
            log.info(
                "Persisted %d actions, %d open questions for %s",
                len(actions), len(questions), team_id,
            )
