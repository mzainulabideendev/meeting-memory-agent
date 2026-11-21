"""Evidence collection, sufficiency assessment and research-loop expansion."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from meeting_agent.config import Settings, get_settings
from meeting_agent.memory.keyword_store import KeywordStore
from meeting_agent.models import EvidenceChunk, Turn
from meeting_agent.retrieval.query_analyzer import QueryAnalysis, QueryPlan
from meeting_agent.retrieval.reranker import EvidenceReranker
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms, split_sentences, term_overlap

log = get_logger(__name__)

DECISION_LANGUAGE = re.compile(
    r"\b(?:agreed|decided|settled|we'll go with|we will go with|final|instead|changed|"
    r"replaced|drop|dropped|keep|kept|superseded)\b",
    re.I,
)
OWNERSHIP_LANGUAGE = re.compile(
    r"\b(?:i'll|i will|i'm going to|i am going to|can you|could you|responsible|check|find out)\b",
    re.I,
)


@dataclass
class EvidenceBundle:
    analysis: QueryAnalysis
    chunks: list[EvidenceChunk] = field(default_factory=list)
    meetings: list[str] = field(default_factory=list)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    research_loops: int = 0
    queries: list[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return bool(self.sufficiency.get("sufficient", False))

    @property
    def covered_terms(self) -> list[str]:
        return list(self.sufficiency.get("covered_terms", []))

    def prompt_evidence(self, *, max_chars: int = 14000, max_turns: int = 80) -> str:
        """Render evidence as numbered speaker turns for the answer prompt.

        Turns, not chunks, are passed to the model because citations must come
        from within a single speaker turn.
        """
        blocks: list[str] = []
        total = 0
        turn_count = 0
        for chunk in self.chunks:
            header = f"[{chunk.meeting_id} | meeting {chunk.meeting_sequence} | chunk {chunk.chunk_id}]"
            blocks.append(header)
            total += len(header)
            for turn in chunk.turns:
                line = f"{turn.start_time} Speaker {turn.speaker}: {turn.text}"
                if total + len(line) > max_chars or turn_count >= max_turns:
                    break
                blocks.append(line)
                total += len(line)
                turn_count += 1
            if total >= max_chars or turn_count >= max_turns:
                break
        return "\n".join(blocks)

    def all_turns(self) -> list[Turn]:
        seen: set[str] = set()
        turns: list[Turn] = []
        for chunk in self.chunks:
            for turn in chunk.turns:
                if turn.turn_id in seen:
                    continue
                seen.add(turn.turn_id)
                turns.append(turn)
        return sorted(turns, key=lambda t: (t.sequence, t.turn_index))

    def render_digest(self, *, max_sentences: int = 12) -> str:
        """Extractive digest used only when no generative model is configured."""
        turns = self.all_turns()
        scored: list[tuple[float, Turn, str]] = []
        terms = self.analysis.key_terms
        for turn in turns:
            for sentence in split_sentences(turn.text):
                if len(sentence.split()) < 6:
                    continue
                score = term_overlap(terms, sentence)
                if self.analysis.requires_numbers and re.search(r"\d", sentence):
                    score += 0.35
                if self.analysis.requires_temporal and DECISION_LANGUAGE.search(sentence):
                    score += 0.25
                if self.analysis.requires_ownership and OWNERSHIP_LANGUAGE.search(sentence):
                    score += 0.2
                if score > 0:
                    scored.append((score, turn, sentence))
        scored.sort(key=lambda item: (-item[0], item[1].sequence, item[1].turn_index))
        chosen: list[tuple[Turn, str]] = []
        per_meeting: dict[str, int] = {}
        for score, turn, sentence in scored:
            if per_meeting.get(turn.meeting_id, 0) >= 3:
                continue
            chosen.append((turn, sentence))
            per_meeting[turn.meeting_id] = per_meeting.get(turn.meeting_id, 0) + 1
            if len(chosen) >= max_sentences:
                break
        chosen.sort(key=lambda item: (item[0].sequence, item[0].turn_index))
        return "\n".join(
            f"{turn.meeting_id} Speaker {turn.speaker}: \"{sentence}\"" for turn, sentence in chosen
        )


class EvidenceCollector:
    def __init__(
        self,
        settings: Settings | None = None,
        database: Database | None = None,
        reranker: EvidenceReranker | None = None,
        keyword_store: KeywordStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database()
        self.reranker = reranker or EvidenceReranker(self.settings)
        self.keyword_store = keyword_store or KeywordStore(self.db)
        self._turn_cache: dict[str, list[Turn]] = {}

    def _turns(self, meeting_id: str) -> list[Turn]:
        if meeting_id not in self._turn_cache:
            rows = self.db.get_turns(meeting_id)
            self._turn_cache[meeting_id] = [
                Turn(
                    turn_id=r["turn_id"], meeting_id=r["meeting_id"], team_id=r["team_id"],
                    sequence=r["sequence"], turn_index=r["turn_index"], speaker=r["speaker"],
                    start_time=r["start_time"], text=r["text"], start_offset=r["start_offset"],
                    end_offset=r["end_offset"],
                )
                for r in rows
            ]
        return self._turn_cache[meeting_id]

    def collect(self, plan: QueryPlan, candidates: list[dict[str, Any]]) -> EvidenceBundle:
        bundle = EvidenceBundle(analysis=plan.analysis, queries=[q.query for q in plan.queries])
        seen: set[str] = set()
        for candidate in candidates[: self.settings.max_evidence_chunks]:
            if candidate["chunk_id"] in seen or not candidate.get("meeting_id"):
                continue
            seen.add(candidate["chunk_id"])
            row = self.db.get_chunks([candidate["chunk_id"]])
            if not row:
                continue
            record = row[0]
            turn_indices = json.loads(record["turn_indices"] or "[]")
            all_turns = self._turns(record["meeting_id"])
            turns = [t for t in all_turns if t.turn_index in set(turn_indices)]
            bundle.chunks.append(
                EvidenceChunk(
                    chunk_id=record["chunk_id"],
                    meeting_id=record["meeting_id"],
                    team_id=record["team_id"],
                    meeting_sequence=record["sequence"],
                    speaker=record["speaker"],
                    text=record["original_text"],
                    score=float(candidate.get("rerank_score", candidate.get("score", 0.0))),
                    sources=candidate.get("sources", []),
                    turns=turns,
                )
            )
            if record["meeting_id"] not in bundle.meetings:
                bundle.meetings.append(record["meeting_id"])
        bundle.meetings.sort()
        bundle.sufficiency = self.assess(bundle)
        if not bundle.sufficient:
            bundle = self.expand(bundle)
        return bundle

    def assess(self, bundle: EvidenceBundle) -> dict[str, Any]:
        analysis = bundle.analysis
        if not bundle.chunks:
            return {
                "sufficient": False,
                "score": 0.0,
                "gaps": ["no evidence retrieved"],
                "covered_terms": [],
                "distinct_meetings": 0,
            }

        combined = " ".join(chunk.text for chunk in bundle.chunks)
        combined_terms = set(content_terms(combined))
        key_terms = list(dict.fromkeys(analysis.key_terms))[:30]
        covered = [t for t in key_terms if t in combined_terms]
        coverage = len(covered) / max(1, len(key_terms))

        distinct_meetings = len(bundle.meetings)
        sequences = sorted({chunk.meeting_sequence for chunk in bundle.chunks})
        gates: dict[str, bool] = {
            "two_or_more_meetings": distinct_meetings >= 2,
            "entity_coverage": coverage >= 0.35 or len(covered) >= 4,
            "team_scope_ok": (
                analysis.team is None
                or all(chunk.team_id == analysis.team for chunk in bundle.chunks)
            ),
        }
        gaps: list[str] = []
        if not gates["two_or_more_meetings"]:
            gaps.append("evidence comes from fewer than two meetings")
        if not gates["entity_coverage"]:
            gaps.append("query entities are not well covered")
        if not gates["team_scope_ok"]:
            gaps.append("evidence contains other teams")

        if analysis.requires_numbers:
            numeric = bool(
                re.search(r"\b\d[\d,.]*\b", combined)
                or re.search(
                    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b",
                    combined,
                    re.I,
                )
            )
            gates["numeric_evidence"] = numeric
            if not numeric:
                gaps.append("question asks for a number but no numeric evidence was found")
        if analysis.requires_temporal:
            temporal = len(sequences) >= 2 and bool(DECISION_LANGUAGE.search(combined))
            gates["temporal_evidence"] = temporal
            if not temporal:
                gaps.append("change/decision language across meetings is missing")
        if analysis.requires_ownership:
            ownership = bool(OWNERSHIP_LANGUAGE.search(combined))
            gates["ownership_evidence"] = ownership
            if not ownership:
                gaps.append("no explicit ownership language found")
        if "disagreement" in analysis.intents:
            disagreement = bool(
                re.search(r"\b(?:don't agree|do not agree|no, i|but i think|objection)\b", combined, re.I)
            )
            gates["disagreement_evidence"] = disagreement
            if not disagreement:
                gaps.append("no explicit disagreement language found")

        score = sum(1 for value in gates.values() if value) / max(1, len(gates))
        # Stronger requirement: all critical gates must pass.
        critical = ["two_or_more_meetings", "entity_coverage", "team_scope_ok"]
        sufficient = all(gates.get(g, False) for g in critical) and score >= 0.7
        if analysis.requires_temporal:
            sufficient = sufficient and gates.get("temporal_evidence", False)
        if analysis.requires_numbers:
            sufficient = sufficient and gates.get("numeric_evidence", False)

        return {
            "sufficient": sufficient,
            "score": round(score, 3),
            "gaps": gaps,
            "covered_terms": covered,
            "missing_terms": [t for t in key_terms if t not in covered],
            "distinct_meetings": distinct_meetings,
            "distinct_sequences": sequences,
            "gates": gates,
        }

    def expand(self, bundle: EvidenceBundle) -> EvidenceBundle:
        """Add coverage from unrepresented meetings of the same team.

        This is deterministic research: no extra LLM call, and still bounded by
        ``max_evidence_chunks``.
        """
        team = bundle.analysis.team
        if not team:
            return bundle
        rows = self.db.list_meetings(team)
        missing = [row for row in rows if row["meeting_id"] not in bundle.meetings]
        query = " ".join(bundle.analysis.key_terms[:10]) or bundle.analysis.question
        added = 0
        for row in missing:
            if len(bundle.chunks) >= self.settings.max_evidence_chunks:
                break
            hits = self.keyword_store.search(
                query, top_k=2, team_id=team, meeting_ids=[row["meeting_id"]]
            )
            if not hits:
                # fall back to the first chunk of the meeting for context
                chunks = self.db.get_chunks_for_meeting(row["meeting_id"])[:1]
                hits = [{"chunk_id": c["chunk_id"]} for c in chunks]
            for hit in hits:
                if len(bundle.chunks) >= self.settings.max_evidence_chunks:
                    break
                records = self.db.get_chunks([hit["chunk_id"]])
                if not records:
                    continue
                record = records[0]
                if any(c.chunk_id == record["chunk_id"] for c in bundle.chunks):
                    continue
                turn_indices = set(json.loads(record["turn_indices"] or "[]"))
                turns = [t for t in self._turns(record["meeting_id"]) if t.turn_index in turn_indices]
                bundle.chunks.append(
                    EvidenceChunk(
                        chunk_id=record["chunk_id"],
                        meeting_id=record["meeting_id"],
                        team_id=record["team_id"],
                        meeting_sequence=record["sequence"],
                        speaker=record["speaker"],
                        text=record["original_text"],
                        score=0.01,
                        sources=["coverage_expansion"],
                        turns=turns,
                    )
                )
                if record["meeting_id"] not in bundle.meetings:
                    bundle.meetings.append(record["meeting_id"])
                added += 1
        if added:
            bundle.history.append(f"coverage expansion added {added} chunk(s) from unrepresented meetings")
            bundle.meetings.sort()
            bundle.sufficiency = self.assess(bundle)
        return bundle
