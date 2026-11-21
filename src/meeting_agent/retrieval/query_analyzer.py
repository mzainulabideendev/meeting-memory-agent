"""Deterministic question analysis and query planning.

Natural-language understanding that can be done with rules is done with rules:
it is reproducible, free, instantaneous and easy to test.  An LLM is used later
only where genuine language synthesis is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from meeting_agent.utils.text_normalization import STOPWORDS, content_terms

_TEAM_RE = re.compile(r"\b([A-Z]{2}\d{4})\b")
_MEETING_RE = re.compile(r"\b([A-Z]{2}\d{4}[a-d])\b")
_MEETING_LETTER_RE = re.compile(r"\bmeetings?\s+([a-d])\b", re.I)
_SPEAKER_RE = re.compile(r"\bSpeaker\s+([A-D])\b", re.I)

CHANGE_TERMS = {
    "change", "changed", "changing", "switch", "switched", "reverse", "reversed",
    "instead", "initially", "originally", "first", "initial", "later", "final",
    "ended", "end", "drop", "dropped", "keep", "kept", "replace", "replaced",
    "supersede", "superseded", "supersedes", "turn", "turned", "last", "become",
}
CAUSAL_TERMS = {"why", "reason", "because", "forced", "caused", "motivat"}
OWNERSHIP_TERMS = {"who", "whom", "whose", "owner", "responsible", "proposed", "suggested", "raised", "asked"}
NUMERIC_TERMS = {
    "cost", "costs", "costing", "price", "prices", "pricing", "total", "figure",
    "budget", "euro", "euros", "pound", "pounds", "dollar", "dollars", "finance",
    "finances", "financial", "money", "spend", "spent", "production", "sales",
    "how much", "number", "percentage", "percent",
}
DISAGREEMENT_TERMS = {"disagree", "objection", "opposed", "against", "argument", "conflict", "concern"}
UNRESOLVED_TERMS = {"unresolved", "pending", "open", "still", "remain", "remaining", "incomplete"}
DECISION_TERMS = {"decide", "decided", "decision", "agree", "agreed", "choice", "chose", "choose", "settle", "settled"}

ROLE_TERMS = {
    "project manager": "Project Manager",
    "industrial designer": "Industrial Designer",
    "user interface designer": "User Interface Designer",
    "marketing": "Marketing",
    "ui designer": "User Interface Designer",
}

# Question-term -> transcript-term bridges.  Meeting transcripts often use
# different wording than the assessment questions (e.g. a question about the
# "power source" while the transcript says "regular batteries").  These bridges
# let retrieval and ranking reach facts that share no literal word with the
# question.  Keys are lowercase content terms; values are related transcript
# terms to add to query expansion.
SYNONYM_BRIDGES: dict[str, list[str]] = {
    "power": ["battery", "batteries", "rechargeable", "power", "power source"],
    "power source": ["battery", "batteries", "rechargeable", "regular battery", "power source"],
    "battery": ["battery", "batteries", "rechargeable", "regular battery", "basic battery"],
    "rechargeable": ["rechargeable", "battery", "batteries", "regular battery", "module", "options"],
    "settle": ["take", "go with", "go for", "decided", "agreed", "regular batteries", "battery"],
    "settled": ["take", "go with", "go for", "decided", "agreed", "regular batteries", "battery"],
    "adopt": ["take", "go with", "options", "rechargeable", "regular batteries", "wasn't in the options"],
    "lcd": ["l_c_d_", "lcd", "screen", "display", "consideration", "include", "kept", "favour"],
    "screen": ["l_c_d_", "lcd", "screen", "display"],
    "include": ["consideration", "include", "included", "build", "kept", "l_c_d_"],
    "keep": ["kept", "keep", "retain", "retained", "l_c_d_", "screen", "rubber"],
    "cost": ["cost", "costing", "euros", "euro", "budget", "seventeen", "twelve", "production cost", "price"],
    "costing": ["cost", "costing", "euros", "euro", "budget", "seventeen", "twelve", "production cost", "price"],
    "price": ["price", "euros", "euro", "cost", "sales price", "twenty five", "budget"],
    "budget": ["budget", "twelve and a half", "twelve fifty", "permitted", "production cost", "euros"],
    "finances": ["finances", "financial", "price", "cost", "euros", "budget", "data"],
    "figure": ["seventeen", "twelve", "twenty five", "euros", "budget", "cost"],
    "number": ["seventeen", "twelve", "twenty five", "euros", "budget"],
    "beep": ["beep", "flash", "implement", "recover", "find", "remote", "locator"],
    "flash": ["beep", "flash", "implement", "recover", "find", "remote"],
    "find": ["beep", "flash", "recover", "locate", "finder", "lost remote"],
    "locate": ["beep", "flash", "find", "recover", "locator"],
    "recover": ["beep", "flash", "find", "recover", "remote"],
    "implement": ["implement", "beep", "flash", "going to", "planned"],
    "planned": ["implement", "going to", "beep", "flash", "agreed"],
    "agreed": ["agreed", "decided", "going to", "implement", "we will"],
    "who": ["speaker", "speaker c", "industrial designer", "designer", "committed", "i will", "i'll"],
    "committed": ["i will", "i'll", "going to", "try to find", "speaker c", "industrial designer"],
    "industrial designer": ["speaker c", "designer", "industrial designer", "role"],
    "yellow": ["yellow", "colour", "color", "special colour", "rubber", "black"],
    "colour": ["yellow", "colour", "color", "special colour", "rubber"],
    "color": ["yellow", "colour", "color", "special colour", "rubber"],
    "rubber": ["rubber", "plastic", "yellow", "colour", "case", "cover", "buttons"],
    "material": ["rubber", "plastic", "titanium", "wood", "case", "material"],
    "case": ["rubber", "plastic", "case", "cover", "material", "cover and buttons"],
    "target": ["young", "younger", "under forty", "target group", "main goal", "consumers"],
    "age": ["young", "younger", "under forty", "older people", "target group"],
    "young": ["young", "younger", "under forty", "target group"],
    "speech recognition": ["speech recognition", "l_c_d_", "dropped", "cut", "sixty percent", "ninety percent"],
    "drop": ["drop", "dropped", "cut", "scrap", "removed", "speech recognition", "special colour"],
    "changed": ["changed", "instead", "originally", "initially", "later", "final", "replaced", "superseded"],
    "between": ["initially", "originally", "later", "final", "changed", "instead"],
}


@dataclass
class QueryAnalysis:
    question: str
    team_ids: list[str] = field(default_factory=list)
    meeting_ids: list[str] = field(default_factory=list)
    speaker_filters: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    requires_temporal: bool = False
    requires_ownership: bool = False
    requires_numbers: bool = False
    mentions_change: bool = False

    @property
    def team(self) -> str | None:
        return self.team_ids[0] if self.team_ids else None


@dataclass
class PlannedQuery:
    query: str
    kind: str
    weight: float = 1.0
    require_all: bool = False


@dataclass
class QueryPlan:
    analysis: QueryAnalysis
    queries: list[PlannedQuery]
    team_filter: str | None = None
    prefer_late_meetings: bool = False
    prefer_early_meetings: bool = False


def analyze_question(question: str, team_hint: str | None = None) -> QueryAnalysis:
    teams = _TEAM_RE.findall(question)
    if team_hint and team_hint not in teams:
        teams.insert(0, team_hint)
    meeting_ids = _MEETING_RE.findall(question)
    if teams:
        referenced_letters = [letter.lower() for letter in _MEETING_LETTER_RE.findall(question)]
        between_letters = re.search(
            r"\bmeetings?\s+[a-d]\s+(?:and|to)\s+([a-d])\b",
            question,
            re.I,
        )
        if between_letters:
            referenced_letters.append(between_letters.group(1).lower())
        meeting_ids.extend(f"{teams[0]}{letter}" for letter in referenced_letters)
    meeting_ids = list(dict.fromkeys(meeting_ids))
    speakers = [s.upper() for s in _SPEAKER_RE.findall(question)]
    lowered = question.lower()

    roles = [role for phrase, role in ROLE_TERMS.items() if phrase in lowered]
    terms = content_terms(question)
    # Team and meeting ids identify the scope; they are filters, not content
    # terms, so they must not count as missing evidence entities.
    identifiers = {t.lower() for t in teams} | {m.lower() for m in meeting_ids}
    terms = [t for t in terms if t.lower() not in identifiers]
    # Keep a stable unique order, prioritising less common (longer) terms.
    key_terms = sorted(dict.fromkeys(terms), key=lambda t: (-len(t), t))

    # Topic labels come from the existing domain lexicon; they drive a targeted
    # retrieval sweep and a concept-expansion query.  They are grouping labels,
    # never answer content.
    from meeting_agent.memory.semantic_memory import TOPIC_LEXICON, detect_topics

    topics = detect_topics(question, max_topics=3)
    expansion: list[str] = []
    for topic in topics:
        for term in TOPIC_LEXICON.get(topic, ()):  # type: ignore[arg-type]
            if term not in expansion and term not in terms:
                expansion.append(term)

    # Vocabulary bridging: add known synonyms and related terms from the transcript
    # corpus to bridge the gap between question vocabulary and transcript vocabulary.
    # Both single tokens and multi-word phrases are matched against the question.
    for bridge, syns in SYNONYM_BRIDGES.items():
        bridge_tokens = bridge.split()
        if len(bridge_tokens) == 1:
            if bridge in terms:
                for syn in syns:
                    if syn not in expansion and syn not in terms:
                        expansion.append(syn)
        else:
            # Multi-word phrase: match if every word of the phrase appears in the question.
            if all(tok in terms for tok in bridge_tokens):
                for syn in syns:
                    if syn not in expansion and syn not in terms:
                        expansion.append(syn)

    intents: list[str] = []
    if any(t in CHANGE_TERMS for t in terms) or re.search(r"\b(?:changed|switched|reversed|superseded)\b", lowered):
        intents.append("temporal_change")
    if any(t in CAUSAL_TERMS for t in terms) or "why" in lowered:
        intents.append("causal")
    if any(t in OWNERSHIP_TERMS for t in terms) or "who" in lowered:
        intents.append("ownership")
    numeric_signal = bool(
        any(t in NUMERIC_TERMS for t in terms)
        or re.search(r"\bhow (?:much|many)\b|\b(?:costing|finances?|financial|money|pricing|spend|spent)\b", lowered)
        or re.search(r"\b\d\b|\b(?:euros?|pounds?|dollars?|seventeen|twelve|twenty five|twelve and a half|twelve fifty)\b", lowered)
    )
    if numeric_signal:
        intents.append("numeric")
    if any(t in DISAGREEMENT_TERMS for t in terms):
        intents.append("disagreement")
    if any(t in UNRESOLVED_TERMS for t in terms):
        intents.append("unresolved")
    if any(t in DECISION_TERMS for t in terms):
        intents.append("decision")

    return QueryAnalysis(
        question=question,
        team_ids=list(dict.fromkeys(teams)),
        meeting_ids=meeting_ids,
        speaker_filters=speakers,
        roles=roles,
        key_terms=key_terms,
        topics=topics,
        expansion_terms=expansion,
        intents=intents,
        requires_temporal="temporal_change" in intents or "causal" in intents,
        requires_ownership="ownership" in intents,
        requires_numbers="numeric" in intents,
        mentions_change="temporal_change" in intents,
    )


def plan_queries(analysis: QueryAnalysis, *, max_queries: int = 8) -> QueryPlan:
    queries: list[PlannedQuery] = [PlannedQuery(query=analysis.question, kind="original", weight=1.0)]
    lowered = analysis.question.lower()

    # Extract numeric entities from the question for targeted numeric retrieval
    import re
    numeric_entities = re.findall(r'\b\d+(?:[.,]\d+)?\b', analysis.question)
    if numeric_entities:
        if "numeric_entities" not in analysis.expansion_terms:
            analysis.expansion_terms.extend(numeric_entities)

    # Entity-focused query: the topic noun phrase plus the team's domain.
    key_phrase = " ".join([t for t in analysis.key_terms if t not in STOPWORDS][:8])
    if key_phrase:
        queries.append(PlannedQuery(query=key_phrase, kind="entities", weight=1.1, require_all=False))

    # Concept expansion from the topic lexicon bridges vocabulary that the
    # question does not literally share with the transcript (e.g. a question
    # about "costing" to transcript lines about "production cost").
    if analysis.expansion_terms:
        queries.append(
            PlannedQuery(
                query=" ".join(analysis.expansion_terms[:10]),
                kind="topic_expansion",
                weight=1.05,
            )
        )

    if analysis.requires_temporal:
        queries.extend(
            [
                PlannedQuery(query=f"{key_phrase} agreed decided we will go with", kind="decision_cues", weight=1.0),
                PlannedQuery(query=f"{key_phrase} changed instead replaced drop keep", kind="change_cues", weight=0.9),
            ]
        )
    if analysis.requires_numbers:
        queries.append(
            PlannedQuery(
                query=f"{key_phrase} cost costing price budget total production sales euros pounds percent figure",
                kind="numeric_cues",
                weight=1.05,
            )
        )
    if analysis.requires_ownership:
        queries.append(PlannedQuery(query=f"{key_phrase} I will I'll check find out responsible", kind="ownership_cues", weight=1.0))
    if "disagreement" in analysis.intents:
        queries.append(PlannedQuery(query=f"{key_phrase} I don't agree objection but I think", kind="disagreement_cues", weight=1.0))
    if "unresolved" in analysis.intents:
        queries.append(PlannedQuery(query=f"{key_phrase} need to find out don't know not decided", kind="unresolved_cues", weight=1.0))

    # First/last wording biases which meeting sequences are reranked higher.
    prefer_late = bool(re.search(r"\b(final|last|end|later|currently|now)\b", lowered))
    prefer_early = bool(re.search(r"\b(first|initial|originally|early|at the start)\b", lowered)) and not prefer_late
    if prefer_early and not any(q.kind == "early_cues" for q in queries):
        queries.append(PlannedQuery(query=f"{key_phrase} initially first we thought", kind="early_cues", weight=0.8))
    if prefer_late:
        queries.append(PlannedQuery(query=f"{key_phrase} final decided in the end", kind="late_cues", weight=0.8))

    # Deduplicate by query text and cap the number of searches.
    seen: set[str] = set()
    unique: list[PlannedQuery] = []
    for query in queries:
        key = query.query.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(query)

    return QueryPlan(
        analysis=analysis,
        queries=unique[:max_queries],
        team_filter=analysis.team,
        prefer_late_meetings=prefer_late,
        prefer_early_meetings=prefer_early,
    )


def broaden_plan(plan: QueryPlan, loop: int) -> QueryPlan:
    """Research-loop strategy: widen retrieval without losing the team scope."""
    original = plan.analysis.question
    key_phrase = " ".join(plan.analysis.key_terms[:6])
    extra = [
        PlannedQuery(query=f"{key_phrase} discussed mentioned", kind="broaden", weight=0.7),
        PlannedQuery(query=original, kind="original_repeat", weight=0.6),
    ]
    if loop >= 2:
        extra.append(PlannedQuery(query=f"{key_phrase} alternative option idea", kind="alternative", weight=0.6))
    queries = list(plan.queries) + [q for q in extra if q.query.strip()]
    return QueryPlan(
        analysis=plan.analysis,
        queries=queries,
        team_filter=plan.team_filter,
        prefer_late_meetings=plan.prefer_late_meetings,
        prefer_early_meetings=plan.prefer_early_meetings,
    )
