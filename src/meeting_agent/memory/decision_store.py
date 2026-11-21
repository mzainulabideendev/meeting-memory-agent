"""Persistence facade for decision and semantic memory."""

from __future__ import annotations

import json

from meeting_agent.memory.temporal_memory import TemporalMemory
from meeting_agent.models import Citation, DecisionRecord
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class DecisionStore:
    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()

    def save(self, team_id: str, records: list[DecisionRecord]) -> list[DecisionRecord]:
        reasoned = TemporalMemory.for_team(team_id, records).reason()
        self.db.replace_memory_records(team_id, reasoned)
        return reasoned

    def load(self, team_id: str) -> list[DecisionRecord]:
        rows = self.db.get_memory_records(team_id)
        records: list[DecisionRecord] = []
        for row in rows:
            evidence = [Citation(**item) for item in json.loads(row["evidence"] or "[]")]
            records.append(
                DecisionRecord(
                    record_id=row["record_id"],
                    team_id=row["team_id"],
                    kind=row["kind"],
                    topic=row["topic"],
                    statement=row["statement"],
                    status=row["status"],
                    meeting_id=row["meeting_id"],
                    meeting_sequence=row["sequence"],
                    turn_index=row["turn_index"] if "turn_index" in row.keys() else 0,
                    speaker=row["speaker"],
                    valid_from_meeting=row["valid_from"],
                    valid_to_meeting=row["valid_to"],
                    supersedes=row["supersedes"],
                    superseded_by=row["superseded_by"],
                    evidence=evidence,
                    confidence=row["confidence"],
                    extraction_version=row["extraction_version"],
                    reason=row["reason"],
                )
            )
        return records

    def timeline(self, team_id: str) -> TemporalMemory:
        return TemporalMemory.for_team(team_id, self.load(team_id))
