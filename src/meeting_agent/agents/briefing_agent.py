"""Team briefing agent.

The brief is assembled from verified memory records and *exact* quotations.
Every bullet is represented as an item with its own citation(s); an item whose
quotation fails deterministic verification is repaired or dropped, never
silently rewritten.

If a generative model is configured it may rephrase the points, but the quotes
still come from the evidence and still pass the same validator.  Without a
generative model the deterministic composer produces the brief.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from meeting_agent.agents.citation_agent import CitationValidator
from meeting_agent.config import Settings, get_settings
from meeting_agent.llm.provider import LLMClient, LLMError, LLMInvalidOutput
from meeting_agent.memory.action_store import ActionStore
from meeting_agent.memory.decision_store import DecisionStore
from meeting_agent.models import (
    ActionItem,
    BriefResult,
    Citation,
    DecisionRecord,
    Disagreement,
    OpenQuestion,
)
from meeting_agent.retrieval.evidence_collector import EvidenceBundle, EvidenceCollector
from meeting_agent.retrieval.hybrid_search import HybridSearch
from meeting_agent.retrieval.query_analyzer import analyze_question, plan_queries
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import collapse_whitespace, content_terms

log = get_logger(__name__)

SECTION_PROMPTS = {
    "decisions": (
        "Decisions Made",
        "List decisions the team explicitly agreed, with their current state. "
        "Do not list proposals as decisions. Each item needs a verbatim quote.",
    ),
    "changes": (
        "What Changed Along the Way",
        "Describe initial proposals, later changes, reasons where stated, and which earlier decisions were superseded. "
        "Each item needs verbatim quote(s).",
    ),
    "actions": (
        "Open Actions and Ownership",
        "List commitments/actions, the responsible speaker if explicitly stated, any explicit deadline, and current status. "
        "Never guess ownership; mark it unclear. Each item needs a verbatim quote.",
    ),
    "disagreements": (
        "Disagreements Still Open",
        "Describe disagreements, participant positions, whether later agreement was reached, and what remains unresolved. "
        "Each item needs a verbatim quote.",
    ),
}

SECTION_QUERIES = {
    "decisions": "we agreed we decided final we will go with chose",
    "changes": "changed instead originally initial later final replaced dropped kept superseded",
    "actions": "I will I'll going to check prepare send find out by deadline responsible",
    "disagreements": "I don't agree but I think objection prefer not sure too expensive",
}


@dataclass
class BriefItem:
    point: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class BriefSection:
    key: str
    title: str
    items: list[BriefItem] = field(default_factory=list)
    note: str | None = None


class BriefingAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        decision_store: DecisionStore | None = None,
        action_store: ActionStore | None = None,
        validator: CitationValidator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(settings=self.settings)
        self.decisions = decision_store or DecisionStore()
        self.actions = action_store or ActionStore()
        self.validator = validator or CitationValidator()
        self.hybrid = HybridSearch(self.settings)
        self.collector = EvidenceCollector(self.settings)

    async def generate(self, team: str) -> BriefResult:
        meetings = self.decisions.db.list_meetings(team)
        if not meetings:
            raise ValueError(f"Team {team} has no ingested meetings")
        meeting_ids = [m["meeting_id"] for m in meetings]

        records = self.decisions.load(team)
        actions = self.actions.load(team)
        open_questions = self.actions.load_open_questions(team)

        sections = [
            self._decisions_section(records),
            self._changes_section(records),
            self._actions_section(actions, team),
            self._disagreements_section(records, team),
        ]

        # Optional LLM enrichment: rewrite the prose while preserving verified
        # quotation items.  The deterministic items remain the fallback.
        if self.llm.provider_name != "heuristic":
            for index, section in enumerate(sections):
                try:
                    sections[index] = await self._enhance_section(section, team)
                except Exception as exc:  # pragma: no cover - depends on provider
                    log.warning("Brief section enhancement failed for %s: %s", section.key, exc)

        markdown = self._render(team, sections, meeting_ids, open_questions)
        citations: list[Citation] = []
        for section in sections:
            for item in section.items:
                for citation in item.citations:
                    if citation not in citations:
                        citations.append(citation)

        return BriefResult(
            team=team,
            markdown=markdown,
            decisions=[r for r in records if r.kind == "decision"],
            actions=actions,
            disagreements=[_as_disagreement(r) for r in records if r.kind == "disagreement"],
            open_questions=open_questions,
            citations=citations,
            meetings_used=meeting_ids,
        )

    # --------------------------------------------------------------- sections
    def _decisions_section(self, records: list[DecisionRecord]) -> BriefSection:
        section = BriefSection(key="decisions", title="A. Decisions Made")
        current = {}
        confirmed = [r for r in records if r.status == "confirmed" and r.kind == "decision"]
        for record in confirmed:
            existing = current.get(record.topic)
            if existing is None or record.meeting_sequence > existing.meeting_sequence:
                current[record.topic] = record
        for record in sorted(current.values(), key=lambda r: (r.meeting_sequence, r.topic)):
            quote = self._first_valid_quote(record)
            if not quote:
                continue
            section.items.append(
                BriefItem(
                    point=(
                        f"{record.topic.replace('_', ' ').title()}: {record.statement} "
                        f"(confirmed in {record.meeting_id}, Speaker {record.speaker or '?'})."
                    ),
                    citations=[quote],
                )
            )
        if not section.items:
            section.note = (
                "No decision with explicit agreement language was extracted for this team. "
                "The transcripts may contain proposals only; they are not reported as decisions."
            )
        return section

    def _changes_section(self, records: list[DecisionRecord]) -> BriefSection:
        section = BriefSection(key="changes", title="B. What Changed Along the Way")
        by_topic: dict[str, list[DecisionRecord]] = {}
        for record in records:
            if record.kind != "decision":
                continue
            by_topic.setdefault(record.topic, []).append(record)

        for topic, group in sorted(by_topic.items()):
            group.sort(key=lambda r: (r.meeting_sequence, r.record_id))
            superseded = [r for r in group if r.status == "superseded"]
            if superseded:
                for old in superseded:
                    new = next((r for r in group if r.record_id == old.superseded_by), None)
                    old_quote = self._first_valid_quote(old)
                    new_quote = self._first_valid_quote(new) if new else None
                    if not old_quote:
                        continue
                    if new and new_quote:
                        point = (
                            f"{topic.replace('_', ' ').title()}: earlier position in {old.meeting_id} "
                            f"(\"{_short(old.statement)}\") was later superseded in {new.meeting_id} "
                            f"by \"{_short(new.statement)}\"."
                        )
                        section.items.append(BriefItem(point=point, citations=[old_quote, new_quote]))
                    else:
                        point = (
                            f"{topic.replace('_', ' ').title()}: the confirmed position in {old.meeting_id} "
                            f"was later marked superseded, but the replacement was not captured with enough evidence."
                        )
                        section.items.append(BriefItem(point=point, citations=[old_quote]))
            else:
                confirmed = [r for r in group if r.status == "confirmed"]
                proposals = [r for r in group if r.status == "proposed"]
                if confirmed and proposals:
                    first_prop = proposals[0]
                    last_conf = confirmed[-1]
                    if first_prop.meeting_sequence < last_conf.meeting_sequence:
                        prop_quote = self._first_valid_quote(first_prop)
                        conf_quote = self._first_valid_quote(last_conf)
                        if prop_quote and conf_quote:
                            section.items.append(
                                BriefItem(
                                    point=(
                                        f"{topic.replace('_', ' ').title()}: discussed as a proposal in "
                                        f"{first_prop.meeting_id} (\"{_short(first_prop.statement)}\") and later "
                                        f"confirmed in {last_conf.meeting_id} (\"{_short(last_conf.statement)}\")."
                                    ),
                                    citations=[prop_quote, conf_quote],
                                )
                            )
        section.items = _dedupe_items(section.items)[:8]
        if not section.items:
            section.note = (
                "No cross-meeting change with both an earlier and a later supported state was found "
                "in the extracted memory and evidence."
            )
        return section

    def _actions_section(self, actions: list[ActionItem], team: str) -> BriefSection:
        section = BriefSection(key="actions", title="C. Open Actions and Ownership")
        roles = {row["speaker"]: row["role"] for row in self.decisions.db.team_roles(team)}
        seen: set[str] = set()
        for action in sorted(actions, key=lambda a: (a.meeting_sequence, a.action_id)):
            key = action.description.casefold()[:120]
            if key in seen:
                continue
            seen.add(key)
            citation = next(
                (c for c in action.evidence if self.validator.validate(c).valid), None
            )
            if citation is None:
                continue
            owner = "ownership unclear"
            if action.owner_speaker:
                role = action.owner_role or roles.get(action.owner_speaker)
                owner = f"Speaker {action.owner_speaker}" + (f" ({role})" if role else "")
            deadline = f"; deadline: {action.deadline}" if action.deadline else ""
            section.items.append(
                BriefItem(
                    point=(
                        f"{action.description} — owner: {owner}; status: {action.status}{deadline} "
                        f"(raised in {action.meeting_id})."
                    ),
                    citations=[citation],
                )
            )
            if len(section.items) >= 10:
                break
        if not section.items:
            section.note = "No explicit commitment with a verifiable quotation was extracted."
        return section

    def _disagreements_section(self, records: list[DecisionRecord], team: str) -> BriefSection:
        section = BriefSection(key="disagreements", title="D. Disagreements Still Open")
        by_topic: dict[str, list[DecisionRecord]] = {}
        for record in records:
            if record.kind != "disagreement":
                continue
            by_topic.setdefault(record.topic, []).append(record)

        confirmed_topics: dict[str, DecisionRecord] = {}
        for record in records:
            if record.status == "confirmed":
                existing = confirmed_topics.get(record.topic)
                if existing is None or record.meeting_sequence > existing.meeting_sequence:
                    confirmed_topics[record.topic] = record

        for topic, group in sorted(by_topic.items()):
            group.sort(key=lambda r: (r.meeting_sequence, r.record_id))
            quotes = []
            positions = []
            for record in group[:3]:
                quote = self._first_valid_quote(record)
                if quote:
                    quotes.append(quote)
                    positions.append(f"Speaker {record.speaker or '?'} in {record.meeting_id}: \"{_short(record.statement)}\"")
            if not positions:
                continue
            later = confirmed_topics.get(topic)
            if later and later.meeting_sequence >= group[-1].meeting_sequence:
                status = f"A later confirmed decision appears in {later.meeting_id}, but the disagreement itself was not explicitly declared resolved."
            else:
                status = "No explicit resolution was found in the available meetings."
            section.items.append(
                BriefItem(
                    point=(
                        f"{topic.replace('_', ' ').title()}: " + " | ".join(positions) + ". " + status
                    ),
                    citations=quotes[:2],
                )
            )
        section.items = _dedupe_items(section.items)[:6]
        if not section.items:
            section.note = (
                "No explicit disagreement statement with a verifiable quotation was found. "
                "This does not prove the team agreed on everything; it means the transcripts reviewed "
                "did not contain an explicit disagreement cue."
            )
        return section

    # ------------------------------------------------------------------ helpers
    def _first_valid_quote(self, record: DecisionRecord | None) -> Citation | None:
        if record is None:
            return None
        for citation in record.evidence:
            if self.validator.validate(citation).valid:
                return citation
        return None

    def _render(
        self,
        team: str,
        sections: list[BriefSection],
        meeting_ids: list[str],
        open_questions: list[OpenQuestion],
    ) -> str:
        lines = [
            f"# Team {team} — Pre-Meeting Brief",
            "",
            f"Based only on the {len(meeting_ids)} ingested meetings: {', '.join(meeting_ids)}.",
            "Every bullet below carries its own verified transcript quotation.",
            "",
        ]
        for section in sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.note:
                lines.append(f"_{section.note}_")
                lines.append("")
            for item in section.items:
                lines.append(f"- {item.point}")
                for citation in item.citations:
                    lines.append(f"  > “{citation.quote}” — {citation.meeting_id}")
                lines.append("")

        unresolved = [q for q in open_questions if q.status == "unresolved"]
        if unresolved:
            lines.append("## Appendix: Unresolved Questions Captured in the Transcripts")
            lines.append("")
            for question in unresolved[:6]:
                citation = next((c for c in question.evidence if self.validator.validate(c).valid), None)
                lines.append(f"- {question.text} (raised in {question.meeting_id}).")
                if citation:
                    lines.append(f"  > “{citation.quote}” — {citation.meeting_id}")
                lines.append("")

        lines.append("---")
        lines.append(
            "Method note: decisions are only labelled confirmed when the transcript contains explicit "
            "agreement language; proposals are not promoted to decisions. Quotations are verbatim and "
            "were checked deterministically against the original transcript text."
        )
        return "\n".join(lines).strip() + "\n"

    def _section_evidence(self, key: str, team: str) -> EvidenceBundle:
        question = SECTION_QUERIES[key]
        plan = plan_queries(analyze_question(question, team))
        semantic = self.hybrid.semantic_search(plan)
        keyword = self.hybrid.keyword_search(plan)
        fused = self.hybrid.fuse(semantic, keyword, limit=8)
        return self.collector.collect(plan, fused)

    async def _enhance_section(self, section: BriefSection, team: str) -> BriefSection:
        """Ask the LLM to improve phrasing without altering the quotation set.

        The deterministic items are supplied, and the model must return the same
        quotation strings.  Any item whose quote fails validation is discarded.
        """
        if not section.items:
            return section
        evidence = self._section_evidence(section.key, team)
        title, instruction = SECTION_PROMPTS[section.key]
        item_payload = [
            {"point": item.point, "citations": [c.model_dump() for c in item.citations]}
            for item in section.items
        ]
        prompt = f"""Rewrite the brief section "{title}" for Team {team}.
{instruction}

Existing verified items (you may improve the prose, merge duplicates, and order them):
{json.dumps(item_payload, ensure_ascii=False, indent=2)}

More evidence:
{evidence.prompt_evidence(max_chars=7000)}

Return JSON: {{"items": [{{"point": "...", "citations": [{{"meeting_id": "...", "quote": "..."}}]}}]}}
Rules: keep every quote character-for-character identical to the source; never invent a quote, owner, number or decision; every item must keep at least one citation; do not add items not supported above."""
        try:
            payload, _ = await self.llm.generate_json(prompt)
        except (LLMError, LLMInvalidOutput) as exc:
            log.warning("Brief enhancement failed: %s", exc)
            return section
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            return section
        enhanced: list[BriefItem] = []
        for raw in raw_items[:10]:
            if not isinstance(raw, dict):
                continue
            citations = []
            for c in raw.get("citations", [])[:2]:
                if not isinstance(c, dict):
                    continue
                try:
                    citations.append(Citation(meeting_id=str(c.get("meeting_id", "")), quote=str(c.get("quote", ""))))
                except ValueError:
                    continue
            checks = self.validator.validate_many(citations, source=f"brief_{section.key}")
            valid = [check.citation for check in checks if check.valid]
            if valid and raw.get("point"):
                enhanced.append(BriefItem(point=str(raw["point"]), citations=valid[:2]))
        if enhanced:
            section.items = enhanced
        return section


def _as_disagreement(record: DecisionRecord) -> Disagreement:
    return Disagreement(
        disagreement_id=record.record_id,
        team_id=record.team_id,
        topic=record.topic,
        positions=[{"speaker": record.speaker or "?", "position": record.statement}],
        resolved=None,
        meeting_ids=[record.meeting_id],
        evidence=list(record.evidence),
    )


def _short(text: str, max_words: int = 28) -> str:
    words = collapse_whitespace(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "…"


def _dedupe_items(items: list[BriefItem]) -> list[BriefItem]:
    seen: set[str] = set()
    out: list[BriefItem] = []
    for item in items:
        key = item.point[:100].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
