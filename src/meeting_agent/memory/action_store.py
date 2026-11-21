"""Action and commitment memory with conservative completion tracking.

An action is marked ``done`` only when a later turn explicitly says the work was
completed.  Otherwise it stays ``open``.  When ownership is not explicit in the
transcript, ``owner_speaker`` stays ``None`` instead of guessing.
"""

from __future__ import annotations

import json
import re

from meeting_agent.models import ActionItem, Citation, OpenQuestion, Turn
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms, normalize_for_match

log = get_logger(__name__)

COMPLETION_CUES = re.compile(
    r"\b(?:done|finished|sorted|completed|checked|found out|got the (?:price|prices|cost|costs)|"
    r"i have (?:checked|found|asked|done)|i've (?:checked|found|asked|done)|already (?:checked|done|asked))\b",
    re.I,
)

RESOLUTION_CUES = re.compile(
    r"\b(?:it(?:'s| is)|that(?:'s| is)|the answer is|they said|it costs|it will cost|"
    r"the price is|we found out|i found out|turns out)\b",
    re.I,
)


class ActionStore:
    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()

    def save(self, team_id: str, actions: list[ActionItem]) -> None:
        self.db.replace_actions(team_id, actions)

    def load(self, team_id: str) -> list[ActionItem]:
        rows = self.db.get_actions(team_id)
        out: list[ActionItem] = []
        for row in rows:
            out.append(
                ActionItem(
                    action_id=row["action_id"],
                    team_id=row["team_id"],
                    description=row["description"],
                    owner_speaker=row["owner_speaker"],
                    owner_role=row["owner_role"],
                    deadline=row["deadline"],
                    status=row["status"],
                    meeting_id=row["meeting_id"],
                    meeting_sequence=row["sequence"],
                    evidence=[Citation(**x) for x in json.loads(row["evidence"] or "[]")],
                    confidence=row["confidence"],
                )
            )
        return out

    def load_open_questions(self, team_id: str) -> list[OpenQuestion]:
        rows = self.db.get_open_questions(team_id)
        out: list[OpenQuestion] = []
        for row in rows:
            out.append(
                OpenQuestion(
                    question_id=row["question_id"],
                    team_id=row["team_id"],
                    text=row["text"],
                    meeting_id=row["meeting_id"],
                    meeting_sequence=row["sequence"],
                    status=row["status"],
                    resolution=row["resolution"],
                    evidence=[Citation(**x) for x in json.loads(row["evidence"] or "[]")],
                )
            )
        return out

    def update_completion(
        self,
        team_id: str,
        meetings: list[tuple[str, int, list[Turn]]],
        actions: list[ActionItem],
        open_questions: list[OpenQuestion],
    ) -> tuple[list[ActionItem], list[OpenQuestion]]:
        """Conservatively mark actions/questions resolved by later explicit turns."""
        meeting_lookup = {meeting_id: (sequence, turns) for meeting_id, sequence, turns in meetings}

        for action in actions:
            terms = set(content_terms(action.description))
            for meeting_id, (sequence, turns) in sorted(meeting_lookup.items(), key=lambda kv: kv[1][0]):
                if sequence < action.meeting_sequence:
                    continue
                for turn in turns:
                    if sequence == action.meeting_sequence and turn.turn_index <= 0:
                        continue
                    text_norm = normalize_for_match(turn.text)
                    overlap = len(terms & set(content_terms(turn.text))) / max(1, len(terms))
                    if overlap >= 0.5 and COMPLETION_CUES.search(turn.text):
                        action.status = "done"
                        action.evidence.append(Citation(meeting_id=meeting_id, quote=turn.text[:400]))
                        log.info("Action %s marked done by %s", action.action_id, meeting_id)
                        break
                if action.status == "done":
                    break

        for question in open_questions:
            terms = set(content_terms(question.text))
            for meeting_id, (sequence, turns) in sorted(meeting_lookup.items(), key=lambda kv: kv[1][0]):
                if sequence < question.meeting_sequence:
                    continue
                for turn in turns:
                    if sequence == question.meeting_sequence:
                        continue  # require a later meeting
                    overlap = len(terms & set(content_terms(turn.text))) / max(1, len(terms))
                    if overlap >= 0.5 and RESOLUTION_CUES.search(turn.text):
                        question.status = "resolved"
                        question.resolution = turn.text[:500]
                        question.evidence.append(Citation(meeting_id=meeting_id, quote=turn.text[:400]))
                        break
                if question.status == "resolved":
                    break
        return actions, open_questions
