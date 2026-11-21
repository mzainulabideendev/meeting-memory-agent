"""Semantic extraction and action/open-question behaviour."""

from __future__ import annotations

from meeting_agent.ingestion.parser import parse_transcript_text
from meeting_agent.memory.semantic_memory import detect_topic, extract_semantic_memory
from meeting_agent.memory.temporal_memory import TemporalMemory
from tests.unit.conftest import TRANSCRIPT_A, TRANSCRIPT_B


def test_topic_detection():
    assert detect_topic("we should use a titanium case") == "material"
    assert detect_topic("the battery is too expensive") in {"power_source", "cost"}
    assert detect_topic("hello there how are you") is None


def test_confirmed_decision_extracted_with_evidence():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    bundle = extract_semantic_memory(meta, turns)
    confirmed = [r for r in bundle.decisions if r.status == "confirmed"]
    assert confirmed
    record = confirmed[0]
    assert record.meeting_id == "ZZ0001a"
    assert record.evidence
    assert record.evidence[0].quote
    assert record.confidence >= 0.8


def test_proposal_is_not_confirmed():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    bundle = extract_semantic_memory(meta, turns)
    proposals = [r for r in bundle.decisions if r.status == "proposed"]
    assert any("titanium" in r.statement for r in proposals)
    assert all(r.status != "confirmed" for r in proposals if "titanium" in r.statement)


def test_action_owner_is_speaker():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    bundle = extract_semantic_memory(meta, turns)
    assert any(a.owner_speaker == "A" for a in bundle.actions)


def test_open_question_extracted():
    meta, turns = parse_transcript_text(TRANSCRIPT_A, "ZZ0001a.txt")
    bundle = extract_semantic_memory(meta, turns)
    assert any("LCD" in q.text or "lcd" in q.text.lower() for q in bundle.open_questions)


def test_obligation_is_not_a_confirmed_decision():
    text = (
        "Meeting: Obligation test\nTeam: ZZ0002\nMeeting number: a\n\n"
        "00:00:01 Speaker A\nGuess we'll have to do this after the presentations.\n\n"
        "00:00:05 Speaker B\nYes, okay.\n"
    )
    meta, turns = parse_transcript_text(text, "ZZ0002a.txt")
    bundle = extract_semantic_memory(meta, turns)
    assert all("have to do this" not in r.statement.lower() for r in bundle.decisions if r.status == "confirmed")


def test_temporal_reasoning_across_two_meetings():
    records = []
    for text, name in ((TRANSCRIPT_A, "ZZ0001a.txt"), (TRANSCRIPT_B, "ZZ0001b.txt")):
        meta, turns = parse_transcript_text(text, name)
        records.extend(extract_semantic_memory(meta, turns).decisions)
    reasoned = TemporalMemory(records).reason()
    assert reasoned
    # Every record keeps its evidence and extraction version.
    assert all(r.evidence for r in reasoned)
    assert all(r.extraction_version for r in reasoned)
