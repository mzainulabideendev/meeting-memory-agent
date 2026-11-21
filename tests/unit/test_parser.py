"""Unit tests for parsing, offsets and metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_agent.ingestion.metadata import build_team_structures, infer_roles
from meeting_agent.ingestion.parser import TranscriptParseError, meeting_id_from_filename, parse_transcript_file, parse_transcript_text
from tests.unit.conftest import TRANSCRIPT_A, TRANSCRIPT_B


def test_meeting_id_from_filename():
    assert meeting_id_from_filename(Path("TS3012a.txt")) == "TS3012a"


def test_parse_metadata_and_turns(parsed_synthetic):
    meta, turns = parsed_synthetic
    assert meta.meeting_id == "ZZ0001a"
    assert meta.team_id == "ZZ0001"
    assert meta.sequence == 1
    assert meta.letter == "a"
    assert meta.header_speakers == ["A", "B"]
    assert len(turns) == 4
    assert [t.speaker for t in turns] == ["A", "B", "A", "B"]
    assert turns[0].start_time == "00:00:01"


def test_original_text_preserved_by_offsets(parsed_synthetic):
    meta, turns = parsed_synthetic
    for turn in turns:
        assert meta.raw_text[turn.start_offset : turn.end_offset] == turn.text


def test_speaker_turn_text_is_verbatim(parsed_synthetic):
    _, turns = parsed_synthetic
    assert "titanium case for the remote control" in turns[0].text
    assert turns[0].text.startswith("Hello team, I am the project manager.")


def test_empty_transcript_rejected():
    with pytest.raises(TranscriptParseError):
        parse_transcript_text("   \n\n", "empty.txt")


def test_transcript_without_turns_rejected():
    with pytest.raises(TranscriptParseError):
        parse_transcript_text("Meeting: ZZ0002a\nTeam: ZZ0002\n\nnothing here", "bad.txt")


def test_duplicate_turns_are_not_merged(parsed_synthetic):
    _, turns = parsed_synthetic
    assert len({t.turn_id for t in turns}) == len(turns)


def test_missing_meeting_detection(tmp_path):
    a = tmp_path / "ZZ0001a.txt"
    b = tmp_path / "ZZ0001b.txt"
    a.write_text(TRANSCRIPT_A, encoding="utf-8")
    b.write_text(TRANSCRIPT_B, encoding="utf-8")
    parsed = [parse_transcript_file(a), parse_transcript_file(b)]
    structures = build_team_structures([a, b], parsed)
    assert structures["ZZ0001"].meetings == ["ZZ0001a", "ZZ0001b"]
    assert structures["ZZ0001"].missing_letters == ["c", "d"]


def test_role_inference_requires_explicit_introduction(parsed_synthetic):
    roles = infer_roles([parsed_synthetic])
    assert any(r.speaker == "A" and r.role == "Project Manager" for r in roles)
    assert (parsed_synthetic[0].raw_text[0]) == "M"
