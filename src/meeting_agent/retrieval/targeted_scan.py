"""Deterministic targeted retrieval sweep over a team's own chunks.

Hybrid retrieval is good at matching the wording of a question, but assessment
questions often describe a *concept* whose decisive passage uses different
words (a question about "costing" versus a transcript line about "production
cost"), or the decisive passage is a single sentence buried in a long meeting.

This scanner is a bounded, deterministic complement to semantic + FTS search:

* question terms are weighted by inverse document frequency inside the team's
  corpus, so rare words such as ``beep`` or ``finances`` dominate common ones;
* topic-lexicon and query-expansion terms add concept recall without inventing
  facts;
* decision / revision language, numbers and meeting-recency bias are explicit
  bonuses rather than hidden model behaviour.

It only *retrieves* existing verbatim chunks.  It never writes answer text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from meeting_agent.config import Settings, get_settings
from meeting_agent.memory.semantic_memory import TOPIC_LEXICON
from meeting_agent.retrieval.query_analyzer import QueryPlan
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms

log = get_logger(__name__)

DECISION_RE = re.compile(
    r"\b(?:agreed|decided|settled|we'll go with|we will go with|chose|choose|"
    r"take out|took out|scrap|dropped|instead|keep|kept|implement)\b",
    re.I,
)
REVISION_RE = re.compile(
    r"\b(?:instead|actually|changed|scrap|drop|dropped|lose|lost|take out|took out|"
    r"rather than|replace|replaced)\b",
    re.I,
)
NUMERIC_RE = re.compile(
    r"\b\d[\d,.]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b|"
    r"\b(?:euros?|pounds?|dollars?|budget|cost|costing|price)\b",
    re.I,
)


@dataclass
class TargetedHit:
    chunk_id: str
    meeting_id: str
    team_id: str
    meeting_sequence: int
    speaker: str
    text: str
    score: float


class TargetedScanner:
    def __init__(self, settings: Settings | None = None, database: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self.db = database or Database()
        self._cache: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ public
    def scan(self, plan: QueryPlan, *, limit: int = 10) -> list[dict[str, Any]]:
        """Return the strongest team-scoped chunks for this plan.

        Results are shaped like retrieval hits so the existing fusion/rerank
        pipeline can consume them without special cases.
        """
        analysis = plan.analysis
        team = plan.team_filter or analysis.team
        if not team:
            return []
        rows = self._team_rows(team)
        if not rows:
            return []

        key_terms = list(dict.fromkeys(t.lower() for t in analysis.key_terms if t))
        expansion_terms = list(
            dict.fromkeys(t.lower() for t in analysis.expansion_terms if t)
        )
        topic_terms: set[str] = set()
        for topic in analysis.topics:
            for term in TOPIC_LEXICON.get(topic, ()):  # type: ignore[arg-type]
                topic_terms.update(content_terms(term))

        idf = self._idf(rows, key_terms + expansion_terms)
        key_weight = sum(idf.get(t, 1.0) for t in key_terms) or 1.0
        expansion_weight = sum(idf.get(t, 0.5) for t in expansion_terms) or 1.0

        sequences = [row["sequence"] for row in rows]
        min_seq, max_seq = min(sequences), max(sequences)
        span = max(1, max_seq - min_seq)

        scored: list[TargetedHit] = []
        for row in rows:
            text = row["original_text"]
            terms = set(content_terms(text))
            if not terms:
                continue
            direct = sum(idf.get(t, 1.0) for t in key_terms if t in terms)
            bridge = sum(idf.get(t, 0.5) for t in expansion_terms if t in terms)
            topic_hits = len(topic_terms & terms)

            score = (
                1.4 * (direct / key_weight)
                + 0.6 * (bridge / expansion_weight)
                + 0.08 * min(4, topic_hits)
            )
            if analysis.requires_temporal and DECISION_RE.search(text):
                score += 0.18
            if REVISION_RE.search(text):
                score += 0.08
            if analysis.requires_numbers and NUMERIC_RE.search(text):
                score += 0.30
            if analysis.requires_ownership and re.search(
                r"\b(?:i'll|i will|i'm going to|check|find out|responsible)\b", text, re.I
            ):
                score += 0.12
            if plan.prefer_late_meetings:
                score += 0.10 * ((row["sequence"] - min_seq) / span)
            elif plan.prefer_early_meetings:
                score += 0.10 * ((max_seq - row["sequence"]) / span)

            # Require at least one real signal; never return a chunk that only
            # matched a recency bonus.
            if direct <= 0 and bridge <= 0 and topic_hits == 0:
                continue
            scored.append(
                TargetedHit(
                    chunk_id=row["chunk_id"],
                    meeting_id=row["meeting_id"],
                    team_id=row["team_id"],
                    meeting_sequence=int(row["sequence"]),
                    speaker=row["speaker"],
                    text=text,
                    score=score,
                )
            )

        scored.sort(key=lambda hit: (-hit.score, hit.meeting_sequence, hit.chunk_id))
        top = scored[:limit]
        if not top:
            return []
        best = max(hit.score for hit in top) or 1.0
        results: list[dict[str, Any]] = []
        for rank, hit in enumerate(top, start=1):
            results.append(
                {
                    "chunk_id": hit.chunk_id,
                    "meeting_id": hit.meeting_id,
                    "team_id": hit.team_id,
                    "meeting_sequence": hit.meeting_sequence,
                    "speaker": hit.speaker,
                    "text": hit.text,
                    "rank": rank,
                    "score": hit.score / best,
                    "source": "targeted",
                    "query": "targeted_scan",
                    "query_kind": "targeted",
                    "weight": 1.0,
                }
            )
        log.debug(
            "targeted scan: %d chunks scored, %d returned for %s",
            len(scored), len(results), analysis.question[:80],
        )
        return results

    # ----------------------------------------------------------------- helpers
    def _team_rows(self, team_id: str) -> list[dict[str, Any]]:
        if team_id not in self._cache:
            self._cache[team_id] = [
                {
                    "chunk_id": row["chunk_id"],
                    "meeting_id": row["meeting_id"],
                    "team_id": row["team_id"],
                    "sequence": int(row["sequence"]),
                    "speaker": row["speaker"],
                    "original_text": row["original_text"],
                    "terms": set(content_terms(row["original_text"])),
                }
                for row in self.db.get_chunks_for_team(team_id)
            ]
        return self._cache[team_id]

    @staticmethod
    def _idf(rows: list[dict[str, Any]], terms: list[str]) -> dict[str, float]:
        n = len(rows)
        result: dict[str, float] = {}
        for term in set(terms):
            df = sum(1 for row in rows if term in row["terms"])
            result[term] = math.log(1.0 + n / (1.0 + df))
        return result
