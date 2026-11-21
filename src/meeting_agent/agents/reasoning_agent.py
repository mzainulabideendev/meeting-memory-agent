"""Temporal reasoning agent.

This node compares evidence chronologically and consults persisted decision
memory.  It is deliberately deterministic: chronological comparison and
supersession logic do not need an LLM, and doing them in code makes the result
reproducible and auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from meeting_agent.memory.decision_store import DecisionStore
from meeting_agent.memory.semantic_memory import TOPIC_LEXICON, detect_topics
from meeting_agent.models import DecisionRecord, EvidenceChunk
from meeting_agent.retrieval.evidence_collector import EvidenceBundle
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import (
    content_terms,
    normalize_for_match,
    split_sentences,
    term_overlap,
)

log = get_logger(__name__)

CONFIRM = re.compile(
    r"\b(?:agreed|decided|settled|we'll go with|we will go with|let's go with|"
    r"we chose|we choose|we chose|will just take|we choose)\b",
    re.I,
)
PROPOSE = re.compile(r"\b(?:propose|suggest|recommend|what about|how about|we could|maybe we should|i think we should)\b", re.I)
REJECT = re.compile(
    r"\b(?:too expensive|don't think|do not think|we can't|we cannot|not necessary|"
    r"scrap|drop it|no, i|rules out|not in the options|take out|took out)\b",
    re.I,
)
CHANGE = re.compile(r"\b(?:instead|changed|switch|replaced|rather than|superseded|drop|dropped|keep|kept)\b", re.I)
UNRESOLVED = re.compile(
    r"\b(?:still don't have|still do not have|haven't decided|have not decided|"
    r"need to find out|need to decide|not in the options|don't know|do not know)\b",
    re.I,
)
UNCERTAIN = re.compile(r"\b(?:not sure|uncertain|unclear|perhaps|maybe|i guess)\b", re.I)
OBLIGATION = re.compile(r"\b(?:have|has|had|need|needs|needed) to\b|\bmust\b|\brequired to\b|\bsupposed to\b", re.I)
COMMITMENT = re.compile(
    r"\b(?:i'll|i will|i'm going to|i am going to|we're going to|we are going to)\s+"
    r"(?:try|check|find|look|prepare|make|send|write|bring|get|ask|talk|sort|update|calculate|work|implement)\b",
    re.I,
)
SUGGESTION = re.compile(
    r"\b(?:i|we) (?:propose|suggest|recommend)\b|what about\b|how about\b|"
    r"maybe we (?:should|could|can)\b|we could\b|i think we should\b|"
    r"would (?:it|that) be (?:better|possible|an idea)\b|"
    r"i(?:'m| am) in favou?r of\b|i would (?:say|suggest|prefer)\b",
    re.I,
)


@dataclass
class TimelineEvent:
    meeting_id: str
    sequence: int
    speaker: str
    quote: str
    kind: str
    speech_act: str
    score: float
    turn_index: int = 0


@dataclass
class TemporalAnalysis:
    timeline: list[TimelineEvent] = field(default_factory=list)
    decision_memory: list[DecisionRecord] = field(default_factory=list)
    current_decisions: list[DecisionRecord] = field(default_factory=list)
    changed: bool = False
    first_event: TimelineEvent | None = None
    latest_event: TimelineEvent | None = None
    open_topics: list[dict] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def prompt_block(self, max_events: int = 24) -> str:
        lines: list[str] = []
        if self.roles:
            lines.append("Speaker roles (explicitly self-identified in transcripts):")
            lines.extend(f"- Speaker {speaker}: {role}" for speaker, role in sorted(self.roles.items()))
        if self.decision_memory:
            lines.append("Persisted decision memory for this team (rule-extracted candidates):")
            for record in self.decision_memory[:20]:
                lines.append(
                    f"- [{record.status}] {record.meeting_id} ({record.topic}): {record.statement[:260]}"
                )
        if self.current_decisions:
            lines.append("Current confirmed decisions per topic:")
            for record in self.current_decisions[:12]:
                lines.append(f"- {record.topic}: {record.statement[:260]} ({record.meeting_id})")
        if self.timeline:
            lines.append("Chronological evidence events:")
            for event in self.timeline[:max_events]:
                lines.append(
                    f"- {event.meeting_id} ({event.kind}; {event.speech_act}) "
                    f"Speaker {event.speaker}: {event.quote[:300]}"
                )
        if self.notes:
            lines.append("Reasoning notes:")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


class ReasoningAgent:
    def __init__(self, decision_store: DecisionStore | None = None) -> None:
        self.decision_store = decision_store or DecisionStore()

    def reason(self, bundle: EvidenceBundle, team: str | None, question: str) -> TemporalAnalysis:
        analysis = TemporalAnalysis()
        if not team:
            return analysis

        try:
            records = self.decision_store.load(team)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Decision memory unavailable: %s", exc)
            records = []

        query_terms = set(content_terms(question))
        expanded_terms = set(content_terms(" ".join(bundle.analysis.expansion_terms)))
        relevant_topics = set(bundle.analysis.topics or detect_topics(question))
        analysis.decision_memory = [
            r for r in records
            if r.kind in {
                "decision", "statement", "disagreement", "proposal", "suggestion",
                "obligation", "commitment",
            }
            and (
                term_overlap(query_terms | expanded_terms, f"{r.topic} {r.statement}") >= 0.08
                or r.topic in relevant_topics
            )
        ][:30]
        temporal = self.decision_store.timeline(team)
        analysis.current_decisions = [
            r for r in temporal.current_decisions()
            if (
                term_overlap(query_terms | expanded_terms, f"{r.topic} {r.statement}") >= 0.08
                or r.topic in relevant_topics
            )
        ]
        analysis.open_topics = [
            item for item in temporal.open_topics()
            if (
                term_overlap(
                    query_terms | expanded_terms,
                    item["topic"] + " " + " ".join(p.statement for p in item["proposals"]),
                ) >= 0.06
                or item["topic"] in relevant_topics
            )
        ]
        record_status = {
            (r.meeting_id, normalize_for_match(r.statement)): r.status
            for r in temporal.records
        }

        roles = {}
        for row in self._roles(team):
            roles.setdefault(row["speaker"], row["role"])
        analysis.roles = roles

        seen: set[str] = set()
        for chunk in bundle.chunks:
            for turn in chunk.turns:
                sentences = split_sentences(turn.text)
                for sentence_index, sentence in enumerate(sentences):
                    if len(sentence.split()) < 2:
                        continue
                    context_parts = sentences[max(0, sentence_index - 1):sentence_index + 2]
                    contextual_quote = " ".join(context_parts).strip()
                    quote = (
                        contextual_quote
                        if len(contextual_quote.split()) <= 100
                        else sentence.strip()
                    )
                    if len(quote.split()) < 6:
                        continue
                    key = quote.casefold()[:180]
                    if key in seen:
                        continue
                    direct_overlap = term_overlap(query_terms, contextual_quote)
                    expanded_overlap = term_overlap(expanded_terms, contextual_quote)
                    sentence_topics = set(detect_topics(contextual_quote))
                    topic_match = bool(relevant_topics & sentence_topics)
                    has_numeric = bool(
                        re.search(
                            r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                            r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                            r"hundred|thousand)\b",
                            contextual_quote,
                            re.I,
                        )
                    )
                    numeric_match = bundle.analysis.requires_numbers and has_numeric
                    revision_match = bool(
                        CHANGE.search(contextual_quote)
                        and (
                            bundle.analysis.requires_numbers
                            or bundle.analysis.mentions_change
                            or re.search(
                                r"\b(?:drop|dropped|cut|remove|removed|scrap|lose|lost|cost|budget)\b",
                                question,
                                re.I,
                            )
                        )
                        and (
                            not bundle.analysis.meeting_ids
                            or turn.meeting_id in bundle.analysis.meeting_ids
                        )
                    )
                    if not (
                        direct_overlap >= 0.05 or expanded_overlap >= 0.05
                        or topic_match or numeric_match or revision_match
                    ):
                        continue

                    speech_act = self._speech_act(sentence)
                    lexical_status = self._event_status(sentence)
                    stored_status = record_status.get(
                        (turn.meeting_id, normalize_for_match(sentence))
                    )
                    kind = (
                        lexical_status
                        if lexical_status in {"proposed", "rejected", "unresolved", "uncertain"}
                        else stored_status or lexical_status
                    )
                    score = (
                        direct_overlap * 2.0
                        + expanded_overlap * 0.2
                        + (0.15 if topic_match else 0.0)
                        + (0.3 if numeric_match else 0.12 if has_numeric and topic_match else 0.0)
                        + (0.1 if revision_match else 0.0)
                    )
                    if kind == "confirmed":
                        score += 0.3
                    elif kind == "rejected":
                        score += 0.25
                    elif kind == "superseded":
                        score += 0.2
                    elif kind == "proposed":
                        score += 0.12
                    elif kind in {"unresolved", "uncertain"}:
                        score += 0.08
                        if kind == "unresolved" and re.search(
                            r"\b(?:know|missing|lack|still|unresolved)\b", question, re.I
                        ):
                            score += 0.25
                    elif CHANGE.search(contextual_quote):
                        score += 0.15
                    if speech_act == "decision":
                        score += 0.08
                    elif speech_act == "commitment":
                        score += 0.05
                    if score < 0.12:
                        continue

                    seen.add(key)
                    analysis.timeline.append(
                        TimelineEvent(
                            meeting_id=turn.meeting_id,
                            sequence=turn.sequence,
                            speaker=turn.speaker,
                            quote=quote,
                            kind=kind,
                            speech_act=speech_act,
                            score=score,
                            turn_index=turn.turn_index,
                        )
                    )

        analysis.timeline.sort(key=lambda e: (e.sequence, e.score), reverse=False)
        # Keep a generous number of events per meeting, preserving chronology.
        # The cap exists only to bound prompt size; a tight cap used to evict a
        # single decisive sentence (a figure buried mid-meeting) before the
        # answer selector could consider it.
        per_meeting: dict[str, int] = {}
        filtered: list[TimelineEvent] = []
        for event in sorted(analysis.timeline, key=lambda e: (-e.score, e.sequence)):
            if per_meeting.get(event.meeting_id, 0) >= 60:
                continue
            per_meeting[event.meeting_id] = per_meeting.get(event.meeting_id, 0) + 1
            filtered.append(event)
        analysis.timeline = sorted(filtered, key=lambda e: (e.sequence, -e.score))

        if analysis.timeline:
            analysis.first_event = analysis.timeline[0]
            analysis.latest_event = analysis.timeline[-1]
        sequences = sorted({e.sequence for e in analysis.timeline})
        analysis.changed = len(sequences) >= 2 and any(
            e.kind in {"change", "rejected", "confirmed", "superseded"} for e in analysis.timeline
        )
        if len(sequences) < 2:
            analysis.notes.append("Evidence timeline covers fewer than two meetings.")
        proposals = [e for e in analysis.timeline if e.kind == "proposed"]
        confirmed = [e for e in analysis.timeline if e.kind == "confirmed"]
        if proposals and not confirmed:
            analysis.notes.append(
                "Only proposals were found; no explicit confirmation language. Do not treat them as agreed."
            )
        if analysis.open_topics:
            analysis.notes.append(
                "Persisted memory flags unresolved topics: " + ", ".join(item["topic"] for item in analysis.open_topics)
            )
        return analysis

    @staticmethod
    def _speech_act(quote: str) -> str:
        if OBLIGATION.search(quote):
            return "obligation"
        if COMMITMENT.search(quote):
            return "commitment"
        if SUGGESTION.search(quote):
            return "suggestion"
        if CONFIRM.search(quote):
            return "confirmation"
        if REJECT.search(quote):
            return "rejection"
        if PROPOSE.search(quote):
            return "proposal"
        return "statement"

    @staticmethod
    def _event_status(quote: str) -> str:
        if CONFIRM.search(quote):
            return "confirmed"
        if REJECT.search(quote):
            return "rejected"
        if PROPOSE.search(quote) or SUGGESTION.search(quote):
            return "proposed"
        if UNRESOLVED.search(quote):
            return "unresolved"
        if UNCERTAIN.search(quote):
            return "uncertain"
        return "mention"

    def _roles(self, team: str):
        try:
            return self.decision_store.db.team_roles(team)
        except Exception:
            return []
