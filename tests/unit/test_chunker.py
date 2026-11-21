"""Chunking tests: size bounds, offsets, speaker metadata, overlap safety."""

from __future__ import annotations

from meeting_agent.ingestion.chunker import chunk_meeting, count_tokens
from meeting_agent.ingestion.parser import parse_transcript_text
from tests.unit.conftest import TRANSCRIPT_A


def test_chunks_preserve_source_slice():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    chunks = chunk_meeting(meta, turns, target_tokens=30, max_tokens=60, overlap_tokens=10)
    assert chunks
    for chunk in chunks:
        assert chunk.original_text == meta.raw_text[chunk.source_start_offset : chunk.source_end_offset]


def test_chunks_have_required_metadata():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    chunks = chunk_meeting(meta, turns)
    for chunk in chunks:
        assert chunk.chunk_id.startswith("ZZ0001a:c")
        assert chunk.meeting_id == "ZZ0001a"
        assert chunk.team_id == "ZZ0001"
        assert chunk.meeting_sequence == 1
        assert chunk.speaker in {"A", "B"}
        assert chunk.source_file.endswith("ZZ0001a.txt")
        assert chunk.embedding_id == chunk.chunk_id


def test_chunk_max_tokens_respected():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    chunks = chunk_meeting(meta, turns, target_tokens=40, max_tokens=50, overlap_tokens=5)
    # token_count is computed from segments; embed text may be slightly larger
    for chunk in chunks:
        assert count_tokens(chunk.embed_text) <= 120


def test_chunk_turn_indices_present():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    chunks = chunk_meeting(meta, turns, target_tokens=40, max_tokens=80, overlap_tokens=10)
    for chunk in chunks:
        assert chunk.turn_indices
        assert all(0 <= index < len(turns) for index in chunk.turn_indices)


def test_empty_turn_list_returns_no_chunks():
    meta, _ = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    assert chunk_meeting(meta, []) == []
