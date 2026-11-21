"""Temporal memory: how decisions and proposals evolve across meetings.

Rules encoded here
------------------
* Proposals never automatically supersede a confirmed decision.
* Only a later *confirmed* decision on the same topic supersedes an earlier one.
* Rejections close proposals, not confirmed decisions.
* Contradictory records that cannot be resolved stay ``proposed``/``uncertain``
  and are surfaced as open items rather than being silently flattened.
* All records are preserved; supersession is expressed with ``valid_to``,
  ``superseded_by`` and ``status`` rather than deletion.
"""

from __future__ import annotations

from collections import defaultdict
import re

from meeting_agent.models import DecisionRecord
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms

log = get_logger(__name__)

RESTATEMENT_RE = re.compile(
    r"\b(?:as agreed|we (?:have )?agreed that we|we already (?:agreed|decided)|"
    r"we decided that we|like we (?:agreed|said)|as we (?:agreed|said))\b",
    re.I,
)

# Only explicit revision language can supersede within the same meeting.  Two
# independent confirmations in one meeting (e.g. one about batteries, one about
# material) are different decisions, not a revision of each other.
REVISION_RE = re.compile(
    r"\b(?:actually|instead|rather than|on second thought|scrap|dropped?|lose|lost|"
    r"take out|took out|swap|switch(?:ed)?|changed?|no,|wait)\b",
    re.I,
)


def statement_similarity(a: str, b: str) -> float:
    terms_a = set(content_terms(a))
    terms_b = set(content_terms(b))
    if not terms_a or not terms_b:
        return 0.0
    return len(terms_a & terms_b) / len(terms_a | terms_b)


class TemporalMemory:
    def __init__(self, records: list[DecisionRecord]) -> None:
        self.records = records

    @classmethod
    def for_team(cls, team_id: str, records: list[DecisionRecord]) -> "TemporalMemory":
        return cls([r for r in records if r.team_id == team_id])

    def reason(self) -> list[DecisionRecord]:
        by_topic: dict[str, list[DecisionRecord]] = defaultdict(list)
        for record in self.records:
            by_topic[record.topic].append(record)

        reasoned: list[DecisionRecord] = []
        for topic, group in by_topic.items():
            group.sort(key=lambda r: (r.meeting_sequence, r.turn_index, r.record_id))
            
            # Track the evolution of decisions for this topic
            confirmed_decisions = [r for r in group if r.status == "confirmed"]
            rejected_decisions = [r for r in group if r.status == "rejected"]
            proposed_decisions = [r for r in group if r.status == "proposed"]
            uncertain_decisions = [r for r in group if r.status == "uncertain"]
            
            # Supersession chain among confirmed decisions only.
            for earlier, later in zip(confirmed_decisions, confirmed_decisions[1:]):
                if earlier.record_id == later.record_id:
                    continue
                similarity = statement_similarity(earlier.statement, later.statement)
                same_meeting = later.meeting_sequence == earlier.meeting_sequence
                later_in_order = (later.meeting_sequence, later.turn_index) > (
                    earlier.meeting_sequence, earlier.turn_index
                )
                revision_allowed = (not same_meeting) or bool(
                    REVISION_RE.search(later.statement)
                )
                if similarity < 0.55 and later_in_order and revision_allowed \
                        and not RESTATEMENT_RE.search(later.statement):
                    earlier.status = "superseded"
                    earlier.valid_to_meeting = later.meeting_id
                    earlier.superseded_by = later.record_id
                    earlier.confidence = min(earlier.confidence, 0.8)
                    earlier.reason = (
                        f"superseded by a later confirmed decision ({later.meeting_id}); "
                        "both records are retained"
                    )
                    later.supersedes = earlier.record_id

            # A rejection closes a proposal on the same topic when it is later
            # and sufficiently related.
            for record in proposed_decisions:
                later_rejections = [
                    r for r in group
                    if r.status == "rejected" and r.meeting_sequence >= record.meeting_sequence
                    and r.record_id != record.record_id
                ]
                if any(statement_similarity(record.statement, rej.statement) >= 0.2 for rej in later_rejections):
                    record.status = "rejected"
                    record.valid_to_meeting = max(
                        later_rejections, key=lambda r: r.meeting_sequence
                    ).meeting_id
                    record.reason = "closed by a later rejection on the same topic"
            
            # Also check if a confirmed decision is explicitly rejected later
            for record in confirmed_decisions:
                later_rejections = [
                    r for r in group
                    if r.status == "rejected" and r.meeting_sequence > record.meeting_sequence
                    and r.record_id != record.record_id
                ]
                if any(statement_similarity(record.statement, rej.statement) >= 0.2 for rej in later_rejections):
                    record.status = "superseded"
                    record.valid_to_meeting = max(
                        later_rejections, key=lambda r: r.meeting_sequence
                    ).meeting_id
                    record.superseded_by = max(later_rejections, key=lambda r: r.meeting_sequence).record_id
                    record.reason = "superseded by a later rejection on the same topic"
                    max(later_rejections, key=lambda r: r.meeting_sequence).supersedes = record.record_id

            reasoned.extend(group)
        return sorted(reasoned, key=lambda r: (r.meeting_sequence, r.topic, r.record_id))

    # ------------------------------------------------------------- queries
    def current_decisions(self) -> list[DecisionRecord]:
        current: dict[str, DecisionRecord] = {}
        for record in self.records:
            if record.status != "confirmed":
                continue
            existing = current.get(record.topic)
            if existing is None or (record.meeting_sequence, record.turn_index, record.record_id) > (
                existing.meeting_sequence, existing.turn_index, existing.record_id
            ):
                current[record.topic] = record
        return sorted(current.values(), key=lambda r: r.topic)

    def open_topics(self) -> list[dict]:
        by_topic: dict[str, list[DecisionRecord]] = defaultdict(list)
        for record in self.records:
            by_topic[record.topic].append(record)
        open_items = []
        for topic, group in by_topic.items():
            group.sort(key=lambda r: (r.meeting_sequence, r.turn_index, r.record_id))
            resolutions = [
                r for r in group
                if r.status in {"confirmed", "rejected", "superseded"}
            ]
            last_resolution = max(
                ((r.meeting_sequence, r.turn_index) for r in resolutions), default=(-1, -1)
            )
            outstanding = [
                r for r in group
                if r.status in {"proposed", "unresolved", "uncertain"}
                and (r.meeting_sequence, r.turn_index) > last_resolution
            ]
            if outstanding:
                status = (
                    "uncertain" if any(r.status == "uncertain" for r in outstanding)
                    else "unresolved"
                )
                open_items.append(
                    {
                        "topic": topic,
                        "status": status,
                        "proposals": outstanding,
                        "reason": (
                            "the latest evidence contains an unresolved or uncertain candidate "
                            "with no later resolution"
                        ),
                    }
                )
        return open_items

    def timeline(self) -> dict[str, list[DecisionRecord]]:
        by_topic: dict[str, list[DecisionRecord]] = defaultdict(list)
        for record in sorted(self.records, key=lambda r: (r.meeting_sequence, r.turn_index, r.record_id)):
            by_topic[record.topic].append(record)
        return dict(by_topic)
