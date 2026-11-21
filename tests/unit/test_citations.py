"""Citation verification is the core trust boundary; these tests are strict."""

from __future__ import annotations

from meeting_agent.agents.citation_agent import CitationValidator
from meeting_agent.models import Citation


def test_exact_quote_is_valid(seeded_db):
    validator = CitationValidator(seeded_db)
    check = validator.validate(
        Citation(
            meeting_id="ZZ0001a",
            quote="We should use a titanium case for the remote control",
        )
    )
    assert check.valid, check.reason
    assert check.speaker == "A"


def test_case_and_whitespace_insensitive(seeded_db):
    validator = CitationValidator(seeded_db)
    check = validator.validate(
        Citation(
            meeting_id="ZZ0001a",
            quote="we   should USE a titanium case\nfor the remote control",
        )
    )
    assert check.valid, check.reason


def test_paraphrase_is_rejected(seeded_db):
    validator = CitationValidator(seeded_db)
    check = validator.validate(
        Citation(
            meeting_id="ZZ0001a",
            quote="We ought to use a titanium shell for the remote controller",
        )
    )
    assert not check.valid
    assert "does not occur verbatim" in check.reason


def test_short_quote_is_rejected(seeded_db):
    validator = CitationValidator(seeded_db)
    check = validator.validate(Citation(meeting_id="ZZ0001a", quote="titanium case"))
    assert not check.valid
    assert "at least 6" in check.reason


def test_unknown_meeting_is_rejected(seeded_db):
    validator = CitationValidator(seeded_db)
    check = validator.validate(
        Citation(meeting_id="ZZ9999z", quote="We should use a titanium case for the remote control")
    )
    assert not check.valid
    assert "does not exist" in check.reason


def test_cross_turn_quote_is_rejected(seeded_db):
    validator = CitationValidator(seeded_db)
    # This stitched quotation exists in the raw text only across two turns.
    check = validator.validate(
        Citation(
            meeting_id="ZZ0001a",
            quote="looks premium. I think we should use plastic instead",
        )
    )
    assert not check.valid
    assert "spans multiple speaker turns" in check.reason


def test_valid_only_filters_rejected(seeded_db):
    validator = CitationValidator(seeded_db)
    checks = validator.validate_many(
        [
            Citation(meeting_id="ZZ0001a", quote="we have decided to use plastic for the case"),
            Citation(meeting_id="ZZ0001a", quote="a made up quotation that is long enough"),
        ],
        question_id="T1",
    )
    assert len(validator.valid_only(checks)) == 1


def test_attempts_are_audited(seeded_db):
    validator = CitationValidator(seeded_db)
    validator.validate_many(
        [Citation(meeting_id="ZZ0001a", quote="this quote does not exist anywhere at all")],
        question_id="T2",
    )
    stats = seeded_db.citation_stats()
    assert stats.get("invalid", 0) >= 1
