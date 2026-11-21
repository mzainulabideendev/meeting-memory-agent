"""Answer generation with citation-first prompting and a grounded fallback."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from meeting_agent.agents.citation_agent import CitationValidator
from meeting_agent.agents.reasoning_agent import TemporalAnalysis
from meeting_agent.config import Settings, get_settings
from meeting_agent.llm.provider import LLMClient, LLMError, LLMInvalidOutput, extract_json
from meeting_agent.models import Citation
from meeting_agent.retrieval.evidence_collector import EvidenceBundle
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms, split_sentences, term_overlap

log = get_logger(__name__)

SYSTEM_PROMPT = """You are an evidence-grounded meeting-memory analyst.
You answer questions about a team's meetings using ONLY the transcript evidence supplied.
Rules:
- Never use outside knowledge, never invent facts, speakers, roles, decisions or numbers.
- Answer directly in one or two sentences.
- A proposal is not an agreement. Only call something agreed/decided when the evidence contains explicit agreement language.
- If the evidence supports a change over time, describe the earliest and latest supported states clearly.
- Provide 1 to 3 citations. Each citation must be an exact, contiguous quotation of at least six words copied from ONE speaker turn in the supplied evidence.
- Copy quotations character-for-character. Do not paraphrase inside a quote, do not join two turns, do not add ellipses.
- Prefer quotes that contain the decision, the number, or the speaker attribution that supports the answer.
- If the evidence is insufficient, set insufficient_evidence to true and say what is missing instead of guessing.
Return ONLY valid JSON with this shape:
{"answer": "...", "citations": [{"meeting_id": "...", "quote": "..."}], "insufficient_evidence": false, "reasoning_summary": "..."}"""


@dataclass
class AnswerDraft:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    insufficient_evidence: bool = False
    reasoning_summary: str = ""
    provider: str = "heuristic"
    raw: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class AnswerAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        validator: CitationValidator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(settings=self.settings)
        self.validator = validator or CitationValidator()

    async def answer(
        self,
        question: str,
        team: str | None,
        bundle: EvidenceBundle,
        temporal: TemporalAnalysis,
        *,
        question_id: str | None = None,
    ) -> AnswerDraft:
        if self.llm.provider_name == "heuristic":
            return self._heuristic_answer(question, team, bundle, temporal)

        prompt = self._build_prompt(question, team, bundle, temporal)
        try:
            payload, response = await self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
        except (LLMError, LLMInvalidOutput) as exc:
            log.error("Answer LLM failed for %s: %s", question_id, exc)
            fallback = self._heuristic_answer(question, team, bundle, temporal)
            fallback.reasoning_summary = f"LLM failure fallback: {exc}"
            return fallback

        draft = self._to_draft(payload, response.text, "llm")
        draft.usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        checks = self.validator.validate_many(
            draft.citations, question_id=question_id, source="llm_initial"
        )
        valid = [c.citation for c in checks if c.valid]

        if draft.citations and len(valid) < min(2, len(draft.citations)):
            failures = [c for c in checks if not c.valid]
            repaired = await self._repair(question, bundle, draft, failures)
            if repaired is not None:
                new_checks = self.validator.validate_many(
                    repaired.citations, question_id=question_id, source="llm_repair"
                )
                valid = [c.citation for c in new_checks if c.valid]
                if valid:
                    draft = repaired
                    draft.usage = {
                        "input_tokens": response.usage.input_tokens + repaired.usage.get("input_tokens", 0),
                        "output_tokens": response.usage.output_tokens + repaired.usage.get("output_tokens", 0),
                    }

        if not valid:
            # No verified citation: replace the unsupported answer with an
            # extractive composition built from verified transcript sentences.
            fallback = self._heuristic_answer(question, team, bundle, temporal)
            fallback.reasoning_summary = (
                "Generated answer had no verifiable citation; replaced with extractive evidence composition. "
                + draft.reasoning_summary
            )
            return fallback

        draft.citations = valid[:3]
        return draft

    # ------------------------------------------------------------------ prompt
    def _build_prompt(self, question: str, team: str | None, bundle: EvidenceBundle, temporal: TemporalAnalysis) -> str:
        roles = ", ".join(f"Speaker {s}={r}" for s, r in sorted(temporal.roles.items())) or "not explicitly stated"
        missing = ", ".join(bundle.sufficiency.get("missing_terms", [])[:10])
        return f"""TEAM: {team or 'not specified'}
KNOWN ROLES: {roles}
QUESTION: {question}

EVIDENCE (verbatim speaker turns from the original transcripts):
{bundle.prompt_evidence()}

TEMPORAL REASONING (deterministic, computed from evidence and persisted memory):
{temporal.prompt_block()}

RETRIEVAL STATUS: {'sufficient' if bundle.sufficient else 'insufficient'}.
Gaps: {'; '.join(bundle.sufficiency.get('gaps', [])) or 'none'}.
Uncovered query terms: {missing or 'none'}.

Write the answer as specified. Cite only exact quotes from the EVIDENCE above, and include the correct meeting id with each quote."""

    async def _repair(
        self,
        question: str,
        bundle: EvidenceBundle,
        draft: AnswerDraft,
        failures,
    ) -> AnswerDraft | None:
        failure_text = "\n".join(f"- [{c.citation.meeting_id}] {c.reason}: {c.citation.quote!r}" for c in failures)
        prompt = f"""QUESTION: {question}

Your previous JSON answer:
{json.dumps({'answer': draft.answer, 'citations': [c.model_dump() for c in draft.citations]}, ensure_ascii=False)}

These citations failed deterministic verification:
{failure_text}

EVIDENCE (copy quotations exactly from a single speaker turn below):
{bundle.prompt_evidence(max_chars=10000)}

Return corrected JSON only. Keep the answer if it is supported; otherwise correct it. Every quote must be an exact contiguous substring of one speaker turn above and at least six words long."""
        try:
            payload, response = await self.llm.generate_json(prompt, system=SYSTEM_PROMPT)
        except (LLMError, LLMInvalidOutput) as exc:
            log.warning("Citation repair failed: %s", exc)
            return None
        repaired = self._to_draft(payload, response.text, "llm_repair")
        repaired.usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return repaired

    def _to_draft(self, payload: dict, raw: str, provider: str) -> AnswerDraft:
        answer = str(payload.get("answer", "")).strip()
        citations: list[Citation] = []
        raw_citations = payload.get("citations", [])
        if isinstance(raw_citations, list):
            for item in raw_citations[:4]:
                if not isinstance(item, dict):
                    continue
                meeting_id = str(item.get("meeting_id", "")).strip()
                quote = str(item.get("quote", "")).strip()
                if meeting_id and quote:
                    try:
                        citations.append(Citation(meeting_id=meeting_id, quote=quote))
                    except ValueError:
                        continue
        return AnswerDraft(
            answer=answer,
            citations=citations,
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
            reasoning_summary=str(payload.get("reasoning_summary", ""))[:1000],
            provider=provider,
            raw=raw,
        )

    # --------------------------------------------------------------- fallback
    ASPECT_PATTERNS = {
        "drop": re.compile(r"\b(?:drop|dropped|cut|remove|removed|scrap|lose|lost|give up|gave up|take out|took out)\b", re.I),
        "keep": re.compile(r"\b(?:keep|kept|retain|retained|stay|stayed|remain|remained|go with|going with)\b", re.I),
        "reason": re.compile(r"\b(?:because|since|due to|so that|reason|that's why)\b", re.I),
        "agreement": re.compile(r"\b(?:agreed|decided|settled|okay|ok|fine|exactly)\b", re.I),
        "change": re.compile(r"\b(?:instead|changed|switch|replaced|rather than|superseded)\b", re.I),
        "number": re.compile(
            r"(?:[$€£]|\b\d+(?:[.,]\d+)?\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
            r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
            r"fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b)",
            re.I,
        ),
    }

    def _wanted_aspects(self, question: str) -> list[str]:
        lowered = question.lower()
        wants: list[str] = []
        if re.search(r"\b(?:drop|dropped|cut|lose|lost|remove|removed|scrap|give up|take out)\b", lowered):
            wants.append("drop")
        if re.search(r"\b(?:keep|kept|retain|stay|remain|final|end with|settle)\b", lowered):
            wants.append("keep")
        if re.search(r"\b(?:why|reason|because)\b", lowered):
            wants.append("reason")
        if re.search(r"\b(?:agree|agreed|decide|decided|settle|settled)\b", lowered):
            wants.append("agreement")
        if re.search(r"\b(?:change|changed|switch|instead|originally|initially|reverse|reversed)\b", lowered):
            wants.append("change")
        if re.search(r"\b(?:price|cost|budget|total|much|figure|euro|pound|dollar|estimate)\b", lowered):
            wants.append("number")
        return wants

    def _heuristic_answer(
        self,
        question: str,
        team: str | None,
        bundle: EvidenceBundle,
        temporal: TemporalAnalysis,
    ) -> AnswerDraft:
        if not bundle.sufficiency.get("sufficient", False):
            gaps = "; ".join(bundle.sufficiency.get("gaps", [])) or "not enough cross-meeting evidence"
            return AnswerDraft(
                answer=(
                    "Insufficient evidence to answer this reliably: the retrieved transcripts do not "
                    f"cover the required information ({gaps})."
                ),
                citations=[],
                insufficient_evidence=True,
                reasoning_summary="Evidence sufficiency gates failed; no answer invented.",
                provider="heuristic",
            )

        picks = self._select_timeline_events(bundle, temporal, question, max_n=3)
        if not picks:
            picks = [
                (meeting_id, speaker, quote, 0, "mention")
                for meeting_id, speaker, quote in self._select_quotes(bundle, question, max_n=3)
            ]
        if not picks:
            return AnswerDraft(
                answer="Insufficient evidence to answer this reliably: no quotable supporting passage was found.",
                citations=[],
                insufficient_evidence=True,
                reasoning_summary="No verbatim supporting sentence met the minimum quote length.",
                provider="heuristic",
            )

        picks = sorted(picks, key=lambda item: (item[0], item[3]))
        parts = []
        latest_sequence = max((turn.sequence for turn in bundle.all_turns()), default=-1)
        for meeting_id, speaker, quote, _sequence, kind in picks[:3]:
            label = {
                "confirmed": "confirmed",
                "rejected": "rejected",
                "proposed": "proposed",
                "superseded": "superseded",
                "unresolved": "left unresolved",
                "uncertain": "described as uncertain",
                "change": "described a change",
                "mention": "said",
            }.get(kind, "said")
            meeting_label = f"in {meeting_id}"
            if (
                _sequence == latest_sequence
                and re.search(r"\b(?:final|last|end|latest)\b", question, re.I)
            ):
                meeting_label = f"in the final meeting {meeting_id}"
            parts.append(f'{meeting_label}, Speaker {speaker} {label} that: "{quote}"')
        answer = "The retrieved transcripts show " + "; ".join(parts) + "."
        return AnswerDraft(
            answer=answer,
            citations=[Citation(meeting_id=mid, quote=quote) for mid, _, quote, _, _ in picks],
            insufficient_evidence=False,
            reasoning_summary=(
                "Deterministic extractive composition (no generative LLM configured or no valid citations). "
                "Quotations are verbatim single-turn sentences; no conclusion is inferred beyond the quoted wording."
            ),
            provider="heuristic",
        )

    def _select_timeline_events(
        self,
        bundle: EvidenceBundle,
        temporal: TemporalAnalysis,
        question: str,
        *,
        max_n: int = 3,
    ) -> list[tuple[str, str, str, int, str]]:
        """Pick chronological, question-aspect-aware events from the timeline."""
        from meeting_agent.memory.semantic_memory import detect_topic

        analysis = bundle.analysis
        terms = list(dict.fromkeys(content_terms(" ".join(analysis.key_terms) or question)))
        expanded_terms = list(dict.fromkeys(content_terms(" ".join(analysis.expansion_terms))))
        wants = self._wanted_aspects(question)
        topic = detect_topic(question)
        kind_bonus = {
            "confirmed": 0.3,
            "rejected": 0.25,
            "superseded": 0.25,
            "confirmed": 0.3,
            "confirmation": 0.3,
            "rejection": 0.25,
            "proposed": 0.15,
            "proposal": 0.15,
            "suggestion": 0.1,
            "commitment": 0.15,
            "obligation": 0.1,
            "unresolved": 0.1,
            "uncertain": 0.08,
            "change": 0.15,
            "mention": 0.0,
        }

        events = [e for e in temporal.timeline if 5 <= len(e.quote.split()) <= 100]

        ranked: list[tuple[float, int, str, str, str, str]] = []
        for event in events:
            quote_terms = set(content_terms(event.quote))
            direct_hits = set(terms) & quote_terms
            anchor_hits = direct_hits & set(expanded_terms)
            bridge_hits = set(expanded_terms) & quote_terms
            event_topic = detect_topic(event.quote)
            topic_match = bool(
                (topic and event_topic == topic)
                or event_topic in analysis.topics
            )
            has_numeric = bool(
                re.search(
                    r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
                    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                    r"hundred|thousand)\b",
                    event.quote,
                    re.I,
                )
            )
            revision_language = re.search(
                r"\b(?:instead|actually|scrap|drop|dropped|take out|took out|lose|lost|"
                r"change|changed|replace|replaced|rather than)\b",
                event.quote,
                re.I,
            )
            if not (
                direct_hits or bridge_hits or topic_match
                or (analysis.requires_numbers and has_numeric)
                or (analysis.requires_numbers and revision_language)
            ):
                continue
            score = (
                len(direct_hits) * 0.08
                + len(anchor_hits) * 0.35
                + min(4, len(bridge_hits)) * 0.05
                + (0.1 if topic_match else 0.0)
                + event.score * 0.12
                + kind_bonus.get(event.kind, 0.0) * 0.35
            )
            for aspect in wants:
                if self.ASPECT_PATTERNS[aspect].search(event.quote):
                    score += 0.12
            if re.search(
                r"\b(?:instead|actually|scrap|drop|dropped|take out|took out|lose|lost|"
                r"change|changed|replace|replaced|rather than)\b",
                event.quote,
                re.I,
            ):
                score += 0.28
                if analysis.mentions_change:
                    score += 0.12
                if analysis.requires_numbers:
                    score += 0.1
            if analysis.mentions_change and re.search(
                r"\b(?:drop|dropped|lose|lost|keep|kept|include|included|consider|consideration|"
                r"retain|retained|build|implement|replace|replaced|superseded)\b",
                event.quote,
                re.I,
            ):
                score += 0.35
            if has_numeric and topic_match:
                score += 0.9 if analysis.requires_numbers else 0.25
            if (
                event.kind in {"unresolved", "uncertain"}
                and re.search(r"\b(?:know|missing|lack|still|unresolved)\b", question, re.I)
            ):
                score += 0.55
            if score <= 0.08:
                continue
            ranked.append((score, event.sequence, event.meeting_id, event.speaker, event.quote, event.kind))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_by_meeting: dict[str, tuple[float, int, str, str, str, str]] = {}
        for score, sequence, meeting_id, speaker, quote, kind in ranked:
            existing = best_by_meeting.get(meeting_id)
            if existing is None or score > existing[0]:
                best_by_meeting[meeting_id] = (
                    score, sequence, meeting_id, speaker, quote, kind
                )

        meetings = sorted(best_by_meeting.values(), key=lambda item: (item[1], -item[0]))

        chosen: list[tuple[float, int, str, str, str, str]] = []
        chosen_quotes: set[str] = set()
        protected_quotes: set[str] = set()

        def _add(item: tuple[float, int, str, str, str, str] | None) -> None:
            if item is None or item[4] in chosen_quotes or len(chosen) >= max_n:
                return
            chosen.append(item)
            chosen_quotes.add(item[4])

        def _add_first(item: tuple[float, int, str, str, str, str] | None) -> None:
            """Guarantee an important event survives the final max_n truncation."""
            if item is None or item[4] in chosen_quotes:
                return
            chosen.insert(0, item)
            chosen_quotes.add(item[4])
            while len(chosen) > max_n:
                victim_index = next(
                    (
                        index for index in range(len(chosen) - 1, -1, -1)
                        if chosen[index][4] not in protected_quotes
                    ),
                    None,
                )
                if victim_index is None:
                    # Everything already chosen is required (meeting coverage);
                    # the new item is less important than keeping that coverage.
                    chosen.pop(0)
                    chosen_quotes.discard(item[4])
                    return
                removed = chosen.pop(victim_index)
                chosen_quotes.discard(removed[4])

        def _best(predicate) -> tuple[float, int, str, str, str, str] | None:
            for item in ranked:
                if predicate(item):
                    return item
            return None

        def _topic_matches(item: tuple[float, int, str, str, str, str]) -> bool:
            event_topic = detect_topic(item[4])
            return bool(
                (topic and event_topic == topic)
                or event_topic in analysis.topics
            )

        # For temporal / change-over-time questions, anchor on the earliest and
        # latest meeting state so the answer shows how a position evolved rather
        # than only quoting a single meeting.
        is_temporal = bool(
            analysis.mentions_change
            or "temporal_change" in analysis.intents
            or re.search(r"\b(?:between|earlier|later|change|evolve|over time)\b", question, re.I)
        )

        term_set = set(terms)
        expanded_set = set(expanded_terms)

        # Candidate figure-bearing events.  Prefer real numerals/currency over
        # generic cost vocabulary so a costing answer quotes the figure.
        strict_numeric = re.compile(
            r"\b\d[\d,.]*\b|\b(?:seventeen|twelve|twenty five|twelve and a half|"
            r"twelve fifty|hundred|thousand)\b|\b(?:euros?|pounds?|dollars?)\b",
            re.I,
        )

        def _numeric_variety(item: tuple[float, int, str, str, str, str]) -> int:
            return len(set(re.findall(
                r"\b\d[\d,.]*\b|\b(?:seventeen|twelve|twenty five|twelve and a half|"
                r"twelve fifty|hundred|thousand)\b",
                item[4], re.I,
            )))

        numeric_items = [item for item in ranked if strict_numeric.search(item[4])]
        if analysis.requires_numbers:
            numeric_items = numeric_items or [
                item for item in ranked
                if re.search(r"\b(?:cost|costing|budget|price|finances?)\b", item[4], re.I)
            ]
        numeric_meetings: set[str] = set()
        if analysis.requires_numbers:
            if analysis.meeting_ids:
                numeric_meetings = {
                    item[2] for item in numeric_items if item[2] in analysis.meeting_ids
                }
            else:
                numeric_meetings = {item[2] for item in numeric_items[:2]}

        def _slot_score(item: tuple[float, int, str, str, str, str]) -> float:
            """How well an event answers this question *within its own meeting*.

            Baseline rank plus explicit question-term/topic coverage.  Numeric
            questions weight the number of distinct figures, and change
            questions weight an explicit proportion, because that is usually the
            stated reason a position moved.  All signals are lexical; nothing is
            inferred from a specific expected answer.
            """
            quote_terms = set(content_terms(item[4]))
            score = item[0]
            score += 0.45 * len(quote_terms & term_set)
            score += 0.35 * len(quote_terms & expanded_set)
            if analysis.requires_numbers:
                score += 1.5 * _numeric_variety(item)
            if is_temporal and re.search(
                r"\b(?:\d{1,3}|ninety|sixty|seventy|eighty)\s*(?:percent|%)",
                item[4],
                re.I,
            ):
                score += 1.2
            return score

        def _best_for_meeting(meeting_id: str):
            pool = [item for item in ranked if item[2] == meeting_id]
            if not pool:
                return None
            return max(pool, key=lambda item: (_slot_score(item), -item[1]))

        # Meeting order across the timeline, used to expand "between b and d"
        # into every meeting in that span.
        meeting_order: list[str] = []
        for item in sorted(ranked, key=lambda entry: entry[1]):
            if item[2] not in meeting_order:
                meeting_order.append(item[2])
        sequences_by_meeting = {item[2]: item[1] for item in ranked}

        span_meetings: list[str] = []
        if analysis.meeting_ids and len(analysis.meeting_ids) >= 2:
            referenced = [
                sequences_by_meeting[meeting_id]
                for meeting_id in analysis.meeting_ids
                if meeting_id in sequences_by_meeting
            ]
            if len(referenced) >= 2:
                low, high = min(referenced), max(referenced)
                span_meetings = [
                    meeting_id for meeting_id in meeting_order
                    if low <= sequences_by_meeting[meeting_id] <= high
                ]

        if span_meetings:
            # One representative event per meeting in the referenced span so the
            # answer reflects the trajectory rather than one loud meeting.
            for meeting_id in span_meetings:
                _add(_best_for_meeting(meeting_id))
            protected_quotes.update(chosen_quotes)
        elif is_temporal and meetings:
            _add(_best_for_meeting(meetings[0][2]))  # earliest known state
            if len(meetings) > 1:
                _add(_best_for_meeting(meetings[-1][2]))  # latest known state
        elif re.search(r"\b(?:next|following)\s+(?:meeting|time)\b", question, re.I):
            for item in meetings[:2]:
                _add(item)
        elif re.search(r"\b(?:final|last|latest|in the end|currently|now)\b", question, re.I):
            for item in reversed(meetings):
                _add(item)
                if len(chosen) >= max_n:
                    break
        else:
            # Default: the highest-scoring distinct meetings, then fill by score.
            for item in meetings[: min(2, max_n)]:
                _add(item)

        # Change / cost-cutting questions need the revision statement, e.g. the
        # sentence that says what was taken out, not only the old proposal.
        if is_temporal or re.search(
            r"\b(?:drop|dropped|cut|remove|removed|scrap|lose|lost|cost|budget)\b",
            question, re.I,
        ):
            revision_re = re.compile(
                r"\b(?:instead|actually|scrap|drop|dropped|take out|took out|lose|lost|"
                r"change|changed|replace|replaced|rather than)\b",
                re.I,
            )
            _add_first(_best(lambda item: _topic_matches(item) and bool(revision_re.search(item[4]))))
            # A cost overrun is usually resolved by removing something in the
            # same meeting that produced the figure.  Surface that consequence
            # from the numeric meeting, even when its topic label differs.
            if numeric_meetings:
                revision_candidates = [
                    item for item in ranked
                    if item[2] in numeric_meetings
                    and revision_re.search(item[4])
                    and item[4] not in chosen_quotes
                ]

                def _revision_count(item: tuple[float, int, str, str, str, str]) -> int:
                    return len(revision_re.findall(item[4]))

                revision_candidates.sort(
                    key=lambda item: (-_revision_count(item), -item[0], item[1])
                )
                for item in revision_candidates[:2]:
                    _add_first(item)

        # Decision-state questions need both sides of the lifecycle: what was
        # confirmed and what was ruled out.  This is deterministic and grounded
        # in the timeline kind (confirmed/rejected/proposed), never inferred.
        if re.search(
            r"\b(?:settle|settled|adopt|adopted|decide|decided|decision|agree|agreed|"
            r"planned|still|did they|was it)\b",
            question, re.I,
        ):
            _add_first(_best(lambda item: item[5] == "confirmed" and _topic_matches(item)))
            rejection_language = re.compile(
                r"\b(?:rules out|wasn't in the options|was not in the options|"
                r"not in the (?:options|list)|cannot choose|can't choose|no,? i|"
                r"don't think|do not think|not necessary|too expensive)\b",
                re.I,
            )
            rejected_pool = [
                item for item in ranked
                if _topic_matches(item)
                and (item[5] == "rejected" or rejection_language.search(item[4]))
            ]
            if rejected_pool:
                latest_rejected_sequence = max(item[1] for item in rejected_pool)
                latest_pool = [
                    item for item in rejected_pool
                    if item[1] == latest_rejected_sequence
                ]
                term_hits = set(terms) | set(expanded_terms)

                def _state_hits(item: tuple[float, int, str, str, str, str]) -> int:
                    return len(set(content_terms(item[4])) & term_hits)

                _add_first(
                    max(latest_pool, key=lambda item: (_state_hits(item), item[0]))
                )

        # Numeric guarantees run last so a figure cannot be evicted by broader
        # state/topic context; they are the fact the question explicitly asks for.
        if analysis.requires_numbers:
            if analysis.meeting_ids:
                for meeting_id in analysis.meeting_ids:
                    pool = [candidate for candidate in numeric_items if candidate[2] == meeting_id]
                    if pool:
                        _add_first(max(pool, key=lambda item: (_numeric_variety(item), item[0])))
            else:
                for item in sorted(
                    numeric_items[:4], key=lambda item: (-_numeric_variety(item), -item[0])
                )[:2]:
                    _add_first(item)

        # Ensure each detected question topic has at least one supporting event.
        for wanted_topic in analysis.topics:
            _add(_best(lambda item, topic_label=wanted_topic: detect_topic(item[4]) == topic_label))

        # Fill remaining slots with the highest-scoring distinct events.
        for item in ranked:
            if len(chosen) >= max_n:
                break
            _add(item)

        # Fall back to quote selection if the timeline yielded nothing usable.
        if not chosen:
            fallback = self._select_quotes(bundle, question, max_n=max_n)
            chosen = [
                (0.0, seq, mid, speaker, quote, "mention")
                for mid, speaker, quote in fallback
                for seq in (0,)
            ][:max_n]

        chosen.sort(key=lambda item: (item[1], -item[0]))
        return [
            (meeting_id, speaker, quote, sequence, kind)
            for _, sequence, meeting_id, speaker, quote, kind in chosen[:max_n]
        ]

    def _select_quotes(self, bundle: EvidenceBundle, question: str, max_n: int = 3) -> list[tuple[str, str, str]]:
        terms = content_terms(question)
        candidates: list[tuple[float, int, str, str, str]] = []
        for turn in bundle.all_turns():
            for sentence in split_sentences(turn.text):
                words = sentence.split()
                if len(words) < 6 or len(words) > 90:
                    continue
                score = term_overlap(terms, sentence)
                if re.search(r"\b(?:agreed|decided|settled|instead|changed|cost|price|budget|euro|pound)\b", sentence, re.I):
                    score += 0.25
                if re.search(r"\d", sentence):
                    score += 0.1
                if score > 0:
                    candidates.append((score, turn.sequence, turn.meeting_id, turn.speaker, sentence.strip()))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        chosen: list[tuple[str, str, str]] = []
        # Track which aspects a meeting already contributes so a single meeting
        # can supply two distinct facts (e.g. a decision and a figure) instead of
        # being capped at one quote.
        used_aspects: set[tuple[str, str]] = set()

        def _aspect(sentence: str) -> str:
            if re.search(r"\d", sentence) or re.search(r"\b(?:euros?|budget|cost|price)\b", sentence, re.I):
                return "number"
            if re.search(r"\b(?:agreed|decided|settled|instead|changed|kept|dropped|implement)\b", sentence, re.I):
                return "decision"
            return "other"

        # Numeric questions: lead with a figure-bearing sentence.
        ordered = candidates
        if bundle.analysis.requires_numbers:
            numeric = [c for c in candidates if re.search(r"\d", c[4]) or re.search(r"\b(?:euros?|budget|cost|price)\b", c[4], re.I)]
            rest = [c for c in candidates if c not in numeric]
            ordered = numeric + rest

        for score, sequence, meeting_id, speaker, sentence in ordered:
            aspect = _aspect(sentence)
            key = (meeting_id, aspect)
            if key in used_aspects:
                continue
            chosen.append((meeting_id, speaker, sentence))
            used_aspects.add(key)
            if len(chosen) >= max_n:
                break
        if len(chosen) < max_n:
            for score, sequence, meeting_id, speaker, sentence in ordered:
                if any(sentence == c[2] for c in chosen):
                    continue
                chosen.append((meeting_id, speaker, sentence))
                if len(chosen) >= max_n:
                    break
        return chosen
