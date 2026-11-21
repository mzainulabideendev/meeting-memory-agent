"""Episodic memory: the actual meeting events and conversation turns.

Episodes are never summarised away.  Every higher-level record points back to a
specific meeting turn, and every citation is checked against this layer.
"""

from __future__ import annotations

from meeting_agent.models import MeetingMeta, Turn
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class EpisodicMemory:
    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()

    def teams(self) -> list[dict]:
        return self.db.list_teams()

    def meetings(self, team_id: str | None = None) -> list[MeetingMeta]:
        rows = self.db.list_meetings(team_id)
        return [
            MeetingMeta(
                meeting_id=row["meeting_id"],
                team_id=row["team_id"],
                sequence=row["sequence"],
                letter=row["letter"],
                source_file=row["source_file"],
                file_hash=row["file_hash"],
                word_count=row["word_count"],
                turn_count=row["turn_count"],
                raw_text=row["raw_text"],
            )
            for row in rows
        ]

    def turns(self, meeting_id: str) -> list[Turn]:
        rows = self.db.get_turns(meeting_id)
        return [
            Turn(
                turn_id=r["turn_id"], meeting_id=r["meeting_id"], team_id=r["team_id"],
                sequence=r["sequence"], turn_index=r["turn_index"], speaker=r["speaker"],
                start_time=r["start_time"], text=r["text"], start_offset=r["start_offset"],
                end_offset=r["end_offset"],
            )
            for r in rows
        ]

    def meeting_turn_map(self, meeting_id: str) -> dict[int, Turn]:
        return {t.turn_index: t for t in self.turns(meeting_id)}

    def timeline(self, team_id: str) -> list[dict]:
        events: list[dict] = []
        for meeting in self.meetings(team_id):
            for turn in self.turns(meeting.meeting_id):
                events.append(
                    {
                        "meeting_id": meeting.meeting_id,
                        "sequence": meeting.sequence,
                        "turn_index": turn.turn_index,
                        "speaker": turn.speaker,
                        "start_time": turn.start_time,
                        "text": turn.text,
                    }
                )
        return events

    def context_window(self, meeting_id: str, turn_index: int, radius: int = 2) -> list[Turn]:
        turns = self.turns(meeting_id)
        lo = max(0, turn_index - radius)
        hi = min(len(turns) - 1, turn_index + radius)
        return [t for t in turns if lo <= t.turn_index <= hi]

    def meetings_summary(self) -> dict[str, dict]:
        summary: dict[str, dict] = {}
        for meeting in self.db.list_meetings():
            summary[meeting["meeting_id"]] = {
                "team_id": meeting["team_id"],
                "sequence": meeting["sequence"],
                "turns": meeting["turn_count"],
                "words": meeting["word_count"],
            }
        return summary
