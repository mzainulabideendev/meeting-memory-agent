"""Metadata enrichment: team structure, missing meetings and role inference."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from meeting_agent.models import MeetingMeta, Turn

EXPECTED_SEQUENCES = {"a": 1, "b": 2, "c": 3, "d": 4}
LETTERS = ("a", "b", "c", "d")

ROLE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Project Manager", re.compile(r"\b(project manager|project leader|team leader)\b", re.I)),
    ("Industrial Designer", re.compile(r"\b(industrial designer|industrial design)\b", re.I)),
    ("User Interface Designer", re.compile(r"\b(user interface designer|user interface design|ui designer)\b", re.I)),
    ("Marketing", re.compile(r"\b(marketing expert|marketing manager|marketing)\b", re.I)),
]


@dataclass
class RoleEvidence:
    speaker: str
    role: str
    turn_index: int
    quote: str
    confidence: float
    meeting_id: str = ""


@dataclass
class TeamStructure:
    team_id: str
    meetings: list[str] = field(default_factory=list)
    missing_letters: list[str] = field(default_factory=list)
    extra_letters: list[str] = field(default_factory=list)
    roles: list[RoleEvidence] = field(default_factory=list)


def infer_roles(meetings: list[tuple[MeetingMeta, list[Turn]]]) -> list[RoleEvidence]:
    """Infer roles only from explicit self-introduction statements.

    We deliberately do not guess a role from a name or from generic behaviour.
    A role is accepted only when the speaker's own turn contains a role phrase
    and the turn looks like an introduction (first meeting, short earlier part).
    """
    evidence: list[RoleEvidence] = []
    ordered = sorted(meetings, key=lambda item: item[0].sequence)

    for meta, turns in ordered:
        for turn in turns:
            # Introductions cluster at the start; allow the whole first meeting
            # but only the first 80 turns of later meetings.
            if meta.sequence > 1 and turn.turn_index > 80:
                continue
            text = turn.text
            if not re.search(r"\b(i am|i'm|my name is|i will be|i'll be)\b", text, re.I):
                continue
            for role, pattern in ROLE_PATTERNS:
                match = pattern.search(text)
                if not match:
                    continue
                sentence = _sentence_around(text, match.start(), match.end())
                confidence = 0.85 if meta.sequence == 1 else 0.6
                # Avoid double counting identical role evidence for the same speaker
                if any(e.speaker == turn.speaker and e.role == role for e in evidence):
                    continue
                evidence.append(
                    RoleEvidence(
                        speaker=turn.speaker,
                        role=role,
                        turn_index=turn.turn_index,
                        quote=sentence,
                        confidence=confidence,
                        meeting_id=meta.meeting_id,
                    )
                )
    return evidence


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("?", 0, start), text.rfind("!", 0, start))
    right_candidates = [i for i in (text.find(".", end), text.find("?", end), text.find("!", end)) if i != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    sentence = text[left + 1 : right].strip()
    return sentence or text[max(0, start - 80) : min(len(text), end + 80)].strip()


def build_team_structures(files: list[Path], parsed: list[tuple[MeetingMeta, list[Turn]]]) -> dict[str, TeamStructure]:
    by_team: dict[str, list[str]] = defaultdict(list)
    for meta, _ in parsed:
        by_team[meta.team_id].append(meta.meeting_id)

    structures: dict[str, TeamStructure] = {}
    for team_id, meeting_ids in sorted(by_team.items()):
        letters = sorted({mid[-1] for mid in meeting_ids if mid and mid[-1] in LETTERS})
        missing = [letter for letter in LETTERS if letter not in letters]
        extra = sorted({mid[-1] for mid in meeting_ids} - set(LETTERS))
        structures[team_id] = TeamStructure(
            team_id=team_id,
            meetings=sorted(meeting_ids),
            missing_letters=missing,
            extra_letters=extra,
        )

    roles = infer_roles(parsed)
    for role in roles:
        # attach to the team of the meeting that contains the introduction
        for meta, turns in parsed:
            if any(t.turn_index == role.turn_index and t.speaker == role.speaker for t in turns):
                structures[meta.team_id].roles.append(role)
                break
    return structures


def speaker_distribution(parsed: list[tuple[MeetingMeta, list[Turn]]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for meta, turns in parsed:
        counts = Counter(t.speaker for t in turns)
        result[meta.meeting_id] = dict(sorted(counts.items()))
    return result
