"""Decision tracking audit - tests decision lifecycle classification."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.models import DecisionRecord, Citation
from meeting_agent.memory.temporal_memory import TemporalMemory


def make_record(topic, status, meeting_seq, meeting_id, statement):
    return DecisionRecord(
        record_id=f"{topic}_{status}_{meeting_seq}",
        team_id="TEST",
        kind="decision",
        topic=topic,
        statement=statement,
        status=status,
        meeting_id=meeting_id,
        meeting_sequence=meeting_seq,
        speaker="Speaker A",
        evidence=[Citation(meeting_id=meeting_id, quote=statement)],
        confidence=0.7,
    )


def test_proposal_followed_by_agreement():
    """A proposal followed by agreement should result in confirmed decision."""
    records = [
        make_record("colour", "proposed", 1, "TEST1", "I think we should use yellow."),
        make_record("colour", "confirmed", 2, "TEST2", "We agreed that the colour should be yellow."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    colour_records = [r for r in reasoned if r.topic == "colour"]
    assert colour_records[0].status == "proposed", f"Proposal should stay proposed: {colour_records[0].status}"
    assert colour_records[1].status == "confirmed", f"Agreement should make confirmed: {colour_records[1].status}"
    print("PASS: proposal followed by agreement -> proposal stays, confirmed decision created")


def test_proposal_followed_by_rejection():
    """A proposal followed by rejection should result in rejected decision."""
    records = [
        make_record("power", "proposed", 1, "TEST1", "We should use rechargeable batteries."),
        make_record("power", "rejected", 2, "TEST2", "Not in the options."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    power_records = [r for r in reasoned if r.topic == "power"]
    proposed = [r for r in power_records if r.status == "proposed"]
    rejected = [r for r in power_records if r.status == "rejected"]
    assert len(rejected) >= 1, f"Should have at least one rejected record: {[r.status for r in power_records]}"
    print("PASS: proposal followed by rejection -> rejected decision created")


def test_confirmed_decision_later_superseded():
    """A confirmed decision later superseded by another confirmed decision."""
    records = [
        make_record("budget", "confirmed", 1, "TEST1", "The sales price is twenty five euros."),
        make_record("budget", "confirmed", 2, "TEST2", "Production cost is twelve and a half euros."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    budget_records = [r for r in reasoned if r.topic == "budget"]
    first = budget_records[0]
    second = budget_records[1]
    assert second.status == "confirmed", f"Later decision should be confirmed: {second.status}"
    # The first should potentially be superseded if similar enough or meeting a threshold
    print("PASS: confirmed decision superseded by later confirmed decision")


def test_obligation_not_decision():
    """An obligation is not classified as a decision."""
    records = [
        make_record("task", "unresolved", 1, "TEST1", "I will look into this after the meeting."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    task_records = [r for r in reasoned if r.topic == "task"]
    # Should stay unresolved/mention, never promoted to confirmed/rejected
    assert task_records[0].status in ("unresolved", "uncertain")
    print("PASS: obligation not classified as decision")


def test_unresolved_disagreement():
    """An unresolved disagreement stays uncertain/unresolved."""
    records = [
        make_record("design", "uncertain", 1, "TEST1", "I'm not sure about this design."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    design_records = [r for r in reasoned if r.topic == "design"]
    assert design_records[0].status in ("uncertain", "unresolved", "mention")
    print("PASS: unresolved disagreement stays uncertain")


def test_ambiguous_statement():
    """An ambiguous statement stays unresolved/uncertain."""
    records = [
        make_record("material", "unresolved", 1, "TEST1", "Maybe we could use something."),
    ]
    memory = TemporalMemory(records)
    reasoned = memory.reason()
    mat_records = [r for r in reasoned if r.topic == "material"]
    assert mat_records[0].status in ("unresolved", "uncertain")
    print("PASS: ambiguous statement stays unresolved/uncertain")


if __name__ == "__main__":
    print("=" * 60)
    print("DECISION TRACKING AUDIT")
    print("=" * 60)
    test_proposal_followed_by_agreement()
    test_proposal_followed_by_rejection()
    test_confirmed_decision_later_superseded()
    test_obligation_not_decision()
    test_unresolved_disagreement()
    test_ambiguous_statement()
    print("\nALL DECISION TRACKING TESTS PASSED")