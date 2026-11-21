"""Temporal memory: proposals never silently supersede confirmed decisions."""

from __future__ import annotations

from meeting_agent.memory.temporal_memory import TemporalMemory, statement_similarity
from meeting_agent.models import Citation, DecisionRecord


def record(record_id, topic, statement, status, sequence, meeting_id=None):
    return DecisionRecord(
        record_id=record_id,
        team_id="ZZ0001",
        kind="decision",
        topic=topic,
        statement=statement,
        status=status,
        meeting_id=meeting_id or f"ZZ0001{'abcd'[sequence - 1]}",
        meeting_sequence=sequence,
        evidence=[Citation(meeting_id=meeting_id or f"ZZ0001{'abcd'[sequence - 1]}", quote=statement)],
    )


def test_later_confirmed_decision_supersedes_earlier():
    records = [
        record("r1", "material", "we will use titanium for the case", "confirmed", 1),
        record("r2", "material", "we will use plastic instead for the case", "confirmed", 2),
    ]
    reasoned = TemporalMemory(records).reason()
    first = next(r for r in reasoned if r.record_id == "r1")
    second = next(r for r in reasoned if r.record_id == "r2")
    assert first.status == "superseded"
    assert first.superseded_by == "r2"
    assert second.supersedes == "r1"
    assert first.valid_to_meeting == "ZZ0001b"


def test_later_proposal_does_not_supersede_confirmed_decision():
    records = [
        record("r1", "material", "we will use plastic for the case", "confirmed", 1),
        record("r2", "material", "maybe we should use titanium for the case", "proposed", 2),
    ]
    reasoned = TemporalMemory(records).reason()
    first = next(r for r in reasoned if r.record_id == "r1")
    second = next(r for r in reasoned if r.record_id == "r2")
    assert first.status == "confirmed"
    assert second.status == "proposed"
    assert first.superseded_by is None


def test_restatement_does_not_supersede():
    records = [
        record("r1", "material", "we will use plastic for the case", "confirmed", 1),
        record("r2", "material", "we will use plastic for the case", "confirmed", 2),
    ]
    reasoned = TemporalMemory(records).reason()
    assert all(r.status == "confirmed" for r in reasoned)


def test_rejection_closes_proposal():
    records = [
        record("r1", "display", "maybe we should use an LCD screen", "proposed", 1),
        record("r2", "display", "we cannot use an LCD screen it is too expensive", "rejected", 2),
    ]
    reasoned = TemporalMemory(records).reason()
    proposal = next(r for r in reasoned if r.record_id == "r1")
    assert proposal.status == "rejected"


def test_current_decisions_use_latest_confirmed():
    records = [
        record("r1", "material", "we will use titanium for the case", "confirmed", 1),
        record("r2", "material", "we will use plastic instead for the case", "confirmed", 2),
    ]
    memory = TemporalMemory(records)
    memory.reason()
    current = memory.current_decisions()
    assert len(current) == 1
    assert current[0].record_id == "r2"


def test_similarity_is_bounded():
    assert statement_similarity("plastic case for the remote", "plastic case for the remote") == 1.0
    assert statement_similarity("", "anything") == 0.0
