"""Semantic memory: structured extraction of decisions and conversational facts.

The default extractor is deterministic and evidence-first.  Each record quotes
one speaker turn verbatim, attaches a confidence, and records the extraction
version.  An optional LLM extractor can add higher-recall records; it is never
required to run the pipeline.

Crucially, proposals are not decisions.  A statement is only ``confirmed`` when
the transcript contains explicit agreement language, either inside the turn or
in a short following agreement turn.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from meeting_agent.models import (
    ActionItem,
    Citation,
    DecisionRecord,
    Disagreement,
    MeetingMeta,
    OpenQuestion,
    Turn,
)
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import (
    clean_quote_for_prompt,
    collapse_whitespace,
    content_terms,
    split_sentences,
)

log = get_logger(__name__)

EXTRACTION_VERSION = "rule-v1.2"

# ---------------------------------------------------------------------------
# Topic lexicon.  A topic is a *retrieval/grouping* label, never a claim.
# ---------------------------------------------------------------------------
TOPIC_LEXICON: dict[str, tuple[str, ...]] = {
    "display": ("lcd", "l_c_d", "screen", "display", "touch screen", "touchscreen"),
    "speech_recognition": (
        "speech recognition", "voice recognition", "voice control", "speech control",
        "voice command", "speech", "microphone",
    ),
    "power_source": (
        "battery", "batteries", "rechargeable", "solar", "mains", "energy source",
        "power source", "power supply", "charger", "charging", "aaa", "aa battery",
        "regular battery", "regular batteries", "normal battery", "normal batteries",
        "disposable battery", "disposable batteries",
    ),
    "material": (
        "titanium", "aluminium", "aluminum", "plastic", "rubber", "steel", "wood",
        "material", "casing", "case", "finish", "carbon fibre", "carbon fiber",
    ),
    "colour": ("colour", "color", "yellow", "black", "white", "silver", "red", "blue", "green"),
    "cost": (
        "price", "cost", "costs", "costing", "budget", "expensive", "cheap", "euro", "euros",
        "pound", "pounds", "dollar", "dollars", "price point", "selling price", "finances",
        "financial", "money", "spend", "spent", "production cost",
    ),
    "lost_remote": (
        "lost", "lose", "finder", "locator", "beep", "beeper", "pager", "flash", "flashing",
        "find the remote", "find it", "locate", "recover", "where is the remote",
    ),
    "controls": (
        "button", "buttons", "keypad", "touch pad", "touchpad", "joystick", "scroll",
        "wheel", "navigation", "navigate", "arrow keys", "cursor", "trackball",
    ),
    "target_user": (
        "target group", "age group", "elderly", "older people", "young people",
        "teenager", "children", "target user", "target audience", "demographic",
    ),
    "size": ("size", "small", "smaller", "big", "bigger", "large", "dimension", "handle"),
    "shape": ("shape", "rounded", "round", "square", "ergonomic", "corners", "curved"),
    "market": ("market", "marketing", "advert", "advertising", "selling", "sell", "brand"),
    "prototype": ("prototype", "model", "mock-up", "mock up", "presentation", "demo"),
    "features": ("feature", "features", "function", "functions", "fancy", "advanced"),
}
_TOPIC_PATTERNS = {
    topic: re.compile("|".join(re.escape(term) for term in terms), re.I)
    for topic, terms in TOPIC_LEXICON.items()
}

CONFIRMED_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\bwe (?:have )?(?:agreed|decided|settled|chose|choose)\b",
        r"\bwe(?:'ll| will| are going to| are gonna) (?:go with|use|keep|have(?! to\b)|drop|make|implement|include|add|introduce|take|scrap|remove)\b",
        r"\b(?:okay|ok|right|so|then)[ ,]*will (?:just )?(?:take|use|go with|keep|have)\b",
        r"\bwe(?:'ll| will) just (?:take|use|go with|keep|deliver|do)\b",
        r"\b(?:okay|ok|right|so|then) (?:we'll|we will|let's|lets) (?:go with|use|keep|have|drop|do)\b",
        r"\b(?:let's|lets) (?:go with|use|keep|drop|do|have)\b",
        r"\bthe (?:final|current) (?:decision|choice|design|version) (?:is|will be|was)\b",
        r"\bwe (?:are|'re) (?:definitely|going to|gonna) (?:use|have|go with|keep)\b",
        r"\bthat(?:'s| is) (?:agreed|decided|settled|final)\b",
        r"\bi(?:'ll| will) (?:go with|use|keep)\b",
    )
]

PROPOSAL_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:i|we) (?:propose|suggest|recommend)\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bmaybe we (?:should|could|can)\b",
        r"\bwe should\b",
        r"\bwe could\b",
        r"\bi think we should\b",
        r"\bwould (?:it|that) be (?:better|possible|an idea)\b",
        r"\b(?:one|another) (?:idea|option|possibility) (?:is|would be)\b",
        r"\bi(?:'m| am) in favou?r of\b",
        r"\bi would (?:say|suggest|prefer)\b",
    )
]

REJECTION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:i )?(?:don't|do not) (?:think|like|agree|want)\b",
        r"\bthat(?:'s| is) (?:too|not)\b",
        r"\btoo (?:expensive|big|small|complicated|difficult|heavy)\b",
        r"\bwe (?:can't|cannot|won't|shouldn't)\b",
        r"\bnot (?:a )?(?:good|necessary|needed|possible)\b",
        r"\b(?:drop|scrap|remove|cut|forget|take out|took out|give up|gave up|lose|lost) (?:the|it|that)\b",
        r"\b(?:scrap|drop|remove|cut|lose|take out|took out|give up|gave up)\s+(?:the\s+|a\s+|an\s+)?[a-z]+\b",
        r"\b(?:was|is|were)\s+(?:scrapped|dropped|removed|cut)\b",
        r"\bno,? (?:i|we) (?:think|don't|do not)\b",
        r"\bwe (?:don't|do not) have (?:the|that|those|any)\b",
    )
]

OBLIGATION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:have|has|had|need|needs|needed) to\b",
        r"\bmust\b|\brequired to\b|\bsupposed to\b",
        r"\bwe(?:'ll| will) have to\b",
    )
]

AGREEMENT_CUES = [
    re.compile(p, re.I)
    for p in (
        r"^(?:yes|yeah|yep|okay|ok|right|sure|exactly|agreed|good|fine|absolutely)\b",
        r"\b(?:that sounds|sounds) (?:good|fine|reasonable)\b",
        r"\bi agree\b",
        r"\bthat(?:'s| is) (?:right|true|a good idea)\b",
    )
]

COMMITMENT_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\bi(?:'ll| will| am going to|'m going to)\s+(?:try|check|find|look|prepare|make|send|write|bring|get|ask|talk|sort|update|calculate|work|implement)\b",
        r"\bwe(?:'ll| will| are going to|'re going to)\s+(?:try|check|find|look|prepare|make|send|write|bring|get|ask|talk|sort|update|calculate|work|implement)\b",
    )
]

OBLIGATION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:have|has|had|need|needs|needed) to\b",
        r"\bmust\b|\brequired to\b|\bsupposed to\b",
        r"\bwe(?:'ll| will) have to\b",
    )
]

SUGGESTION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:i|we) (?:propose|suggest|recommend)\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bmaybe we (?:should|could|can)\b",
        r"\bwe should\b",
        r"\bwe could\b",
        r"\bi think we should\b",
        r"\bwould (?:it|that) be (?:better|possible|an idea)\b",
        r"\b(?:one|another) (?:idea|option|possibility) (?:is|would be)\b",
        r"\bi(?:'m| am) in favou?r of\b",
        r"\bi would (?:say|suggest|prefer)\b",
    )
]

DISAGREEMENT_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\bi (?:don't|do not) agree\b",
        r"\b(?:but|however) i (?:think|would|prefer)\b",
        r"\bthat(?:'s| is) not (?:right|true|what i)\b",
        r"\bi(?:'m| am) not sure (?:about|that)\b",
        r"\bno,? (?:but|i think|i'd|i would)\b",
    )
]

ACTION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\bi(?:'ll| will) (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update|calculate|work)\b",
        r"\bi(?:'m| am) going to (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update|calculate|work)\b",
        r"\b(?:can|could) you (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update|calculate|work)\b",
        r"\byou (?:should|need to|will|'ll) (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update)\b",
        r"\bwe (?:need|have) to (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update|calculate)\b",
        r"\bwe(?:'ll| will) (?:check|find|look|prepare|make|do|send|write|bring|get|ask|talk|sort|update|calculate)\b",
        r"\b(?:i'll|i will) (?:have|get) (?:a look|it done|that done)\b",
    )
]

OPEN_QUESTION_CUES = [
    re.compile(p, re.I)
    for p in (
        r"\b(?:we|i) (?:(?:still|really|also|probably|first|quickly) )?"
        r"(?:need|have|want|got) to (?:find out|check|ask|know|look into|see)\b",
        r"\b(?:we|i) (?:don't|do not) know\b",
        r"\b(?:we|i) haven't (?:decided|checked|found out)\b",
        r"\bnot sure (?:yet|about|whether|if)\b",
        r"\b(?:we|i) (?:still )?(?:need|have) to (?:decide|confirm|clarify|research)\b",
        r"\bopen question\b",
        r"\bwe(?:'ll| will) (?:have to )?(?:find out|check|see)\b",
    )
]

# References to a *previous* agreement are not new decisions ("like we decided
# before", "as agreed", "we already agreed").
REFERENCE_TO_PAST_RE = re.compile(
    r"\b(?:like|as) we (?:have )?(?:agreed|decided|said)|"
    r"we (?:already )?(?:agreed|decided) (?:before|earlier|last time)|"
    r"as agreed\b|we said (?:before|earlier)\b",
    re.I,
)

DEADLINE_RE = re.compile(
    r"\b(?:by|before|until|deadline(?: is)?)\s+"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|"
    r"next week|next month|the end of (?:the )?(?:week|month|day)|"
    r"\d{1,2}(?::\d{2})?\s*(?:o'clock|am|pm)?|\d{1,2}(?:st|nd|rd|th)?\s+\w+)",
    re.I,
)

DEADLINE_WORDS = re.compile(r"\b(deadline|by next|by the end|tomorrow|next week)\b", re.I)


def detect_topic(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for topic, pattern in _TOPIC_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        score = len(matches)
        if best is None or score > best[0]:
            best = (score, topic)
    return best[1] if best else None


def detect_topics(text: str, *, max_topics: int = 3) -> list[str]:
    """Return all topics mentioned, ranked by lexical evidence.

    Multi-topic questions are common ("cost of the LCD"), so returning a single
    label loses retrieval signal.  Labels are only used for grouping and
    expansion; they never become answer claims.
    """
    scored: list[tuple[int, str]] = []
    for topic, pattern in _TOPIC_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            scored.append((len(matches), topic))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [topic for _, topic in scored[:max_topics]]


def _record_id(team_id: str, topic: str, meeting_id: str, statement: str) -> str:
    digest = hashlib.sha1(f"{team_id}|{topic}|{meeting_id}|{statement}".encode("utf-8")).hexdigest()
    return f"mem_{digest[:16]}"


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _citation(meeting_id: str, sentence: str) -> Citation:
    return Citation(meeting_id=meeting_id, quote=clean_quote_for_prompt(sentence, max_words=60))


@dataclass
class ExtractionBundle:
    decisions: list[DecisionRecord] = field(default_factory=list)
    actions: list[ActionItem] = field(default_factory=list)
    open_questions: list[OpenQuestion] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)


def extract_semantic_memory(
    meta: MeetingMeta,
    turns: list[Turn],
    *,
    known_roles: dict[str, str] | None = None,
) -> ExtractionBundle:
    """Extract candidate semantic records from one meeting.

    Records are candidates: temporal reasoning later decides which confirmed
    decisions supersede which earlier ones.
    """
    bundle = ExtractionBundle()
    known_roles = known_roles or {}
    seen_statements: set[str] = set()

    for position, turn in enumerate(turns):
        sentences = split_sentences(turn.text)
        for sentence in sentences:
            statement = collapse_whitespace(sentence)
            if len(statement.split()) < 5:
                continue
            key = statement.casefold()[:200]
            if key in seen_statements:
                continue
            seen_statements.add(key)

            topic = detect_topic(statement)
            if topic is None:
                continue

            is_confirmed = _matches(CONFIRMED_CUES, statement)
            if is_confirmed and REFERENCE_TO_PAST_RE.search(statement):
                # Only a non-reference cue counts ("we agreed that ..." vs
                # "like we decided before").
                stripped = REFERENCE_TO_PAST_RE.sub(" ", statement)
                is_confirmed = _matches(CONFIRMED_CUES, stripped)
            is_proposal = _matches(PROPOSAL_CUES, statement) or _matches(SUGGESTION_CUES, statement)
            is_rejection = _matches(REJECTION_CUES, statement)
            is_disagreement = _matches(DISAGREEMENT_CUES, statement)
            is_action = _matches(ACTION_CUES, statement) or _matches(COMMITMENT_CUES, statement)
            is_open = _matches(OPEN_QUESTION_CUES, statement)
            is_obligation = _matches(OBLIGATION_CUES, statement)

            # Agreement in the next turn confirms a proposal from this turn.
            agreement_after = False
            if is_proposal and not is_confirmed and position + 1 < len(turns):
                following = turns[position + 1]
                if following.turn_index == turn.turn_index + 1:
                    following_text = collapse_whitespace(following.text)
                    if len(following_text.split()) <= 14 and _matches(AGREEMENT_CUES, following_text):
                        agreement_after = True

            if is_action:
                # Questions and fragments are not commitments ("are we gonna ...?").
                if "?" in statement or re.search(r"\b(?:are|is|was|were|do|does|did) we\b", statement, re.I):
                    is_action = False
            if is_action:
                owner = turn.speaker if re.search(r"\bi(?:'ll| will| am going to|'m going to)\b", statement, re.I) else None
                deadline_match = DEADLINE_RE.search(statement)
                # Classify as commitment or obligation based on language
                if _matches(COMMITMENT_CUES, statement):
                    kind = "commitment"
                elif is_obligation:
                    kind = "obligation"
                else:
                    kind = "action"
                bundle.actions.append(
                    ActionItem(
                        action_id=f"act_{hashlib.sha1(f'{meta.meeting_id}|{turn.turn_index}|{statement}'.encode()).hexdigest()[:16]}",
                        team_id=meta.team_id,
                        description=statement,
                        owner_speaker=owner,
                        owner_role=known_roles.get(owner) if owner else None,
                        deadline=deadline_match.group(0) if deadline_match else None,
                        status="open",
                        meeting_id=meta.meeting_id,
                        meeting_sequence=meta.sequence,
                        evidence=[_citation(meta.meeting_id, statement)],
                        confidence=0.7 if owner else 0.5,
                        kind=kind,
                    )
                )

            if is_open:
                bundle.open_questions.append(
                    OpenQuestion(
                        question_id=f"oq_{hashlib.sha1(f'{meta.meeting_id}|{turn.turn_index}|{statement}'.encode()).hexdigest()[:16]}",
                        team_id=meta.team_id,
                        text=statement,
                        meeting_id=meta.meeting_id,
                        meeting_sequence=meta.sequence,
                        status="unresolved",
                        evidence=[_citation(meta.meeting_id, statement)],
                    )
                )

            if not (is_confirmed or is_proposal or is_rejection or is_disagreement):
                continue

            if is_disagreement:
                bundle.disagreements.append(
                    Disagreement(
                        disagreement_id=f"dis_{hashlib.sha1(f'{meta.meeting_id}|{turn.turn_index}|{statement}'.encode()).hexdigest()[:16]}",
                        team_id=meta.team_id,
                        topic=topic,
                        positions=[{"speaker": turn.speaker, "position": statement}],
                        resolved=None,
                        meeting_ids=[meta.meeting_id],
                        evidence=[_citation(meta.meeting_id, statement)],
                    )
                )

            # Determine status with better lifecycle classification
            if is_confirmed or agreement_after:
                status = "confirmed"
            elif is_rejection:
                status = "rejected"
            elif is_proposal:
                status = "proposed"
            elif is_disagreement:
                status = "uncertain"  # Disagreements are uncertain until resolved
            else:
                status = "unresolved"

            kind = "decision" if status in {"confirmed", "rejected"} else "statement"
            if is_disagreement and status == "uncertain":
                kind = "disagreement"
            elif is_proposal and status == "proposed":
                kind = "suggestion" if _matches(SUGGESTION_CUES, statement) else "proposal"
            elif is_obligation and status == "unresolved":
                kind = "obligation"

            bundle.decisions.append(
                DecisionRecord(
                    record_id=_record_id(meta.team_id, topic, meta.meeting_id, statement),
                    team_id=meta.team_id,
                    kind=kind,
                    topic=topic,
                    statement=statement,
                    status=status,  # type: ignore[arg-type]
                    meeting_id=meta.meeting_id,
                    meeting_sequence=meta.sequence,
                    turn_index=turn.turn_index,
                    speaker=turn.speaker,
                    valid_from_meeting=meta.meeting_id if status == "confirmed" else None,
                    evidence=[_citation(meta.meeting_id, statement)],
                    confidence=0.85 if is_confirmed else (0.6 if agreement_after else 0.45),
                    extraction_version=EXTRACTION_VERSION,
                    reason="explicit agreement language" if is_confirmed else (
                        "proposal followed by short agreement turn" if agreement_after else "candidate mention"
                    ),
                )
            )

    return bundle


def extract_all_meetings(
    meetings: list[tuple[MeetingMeta, list[Turn]]],
    roles_by_team: dict[str, dict[str, str]] | None = None,
) -> dict[str, ExtractionBundle]:
    roles_by_team = roles_by_team or {}
    results: dict[str, ExtractionBundle] = {}
    for meta, turns in meetings:
        bundle = extract_semantic_memory(meta, turns, known_roles=roles_by_team.get(meta.team_id))
        results[meta.meeting_id] = bundle
        log.info(
            "Extracted %s: %d semantic records, %d actions, %d open questions",
            meta.meeting_id, len(bundle.decisions), len(bundle.actions), len(bundle.open_questions),
        )
    return results
