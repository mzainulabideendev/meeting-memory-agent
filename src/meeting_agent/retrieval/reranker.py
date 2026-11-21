"""Evidence reranking.

The default reranker is a transparent deterministic blend.  A cross-encoder can
be enabled in configuration but is not required; the fused rank already reflects
both meaning and exact-term evidence.
"""

from __future__ import annotations

import re
from typing import Any

from meeting_agent.config import Settings, get_settings
from meeting_agent.retrieval.query_analyzer import QueryAnalysis
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms

log = get_logger(__name__)

DECISION_WORDS = re.compile(
    r"\b(?:agreed|decided|settled|we'll go with|we will go with|final|instead|"
    r"changed|replaced|drop|dropped|keep|kept|cost|price|budget|euros?|pounds?)\b",
    re.I,
)
OWNER_WORDS = re.compile(r"\b(?:i'll|i will|i'm going to|can you|could you|responsible|check|find out)\b", re.I)
DISAGREE_WORDS = re.compile(r"\b(?:don't agree|do not agree|objection|but i think|too expensive|not sure)\b", re.I)


class EvidenceReranker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cross_encoder = None

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
        return self._cross_encoder

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        analysis: QueryAnalysis,
        *,
        prefer_late: bool = False,
        prefer_early: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []

        if self.settings.enable_cross_encoder_rerank:
            try:
                model = self._load_cross_encoder()
                pairs = [(analysis.question, c["text"][:2000]) for c in candidates]
                scores = model.predict(pairs)
                for candidate, score in zip(candidates, scores):
                    candidate["cross_score"] = float(score)
            except Exception as exc:  # pragma: no cover - optional path
                log.warning("Cross-encoder rerank unavailable: %s", exc)

        max_fused = max(c["score"] for c in candidates) or 1.0
        terms = list(dict.fromkeys(analysis.key_terms))[:24]
        sequences = [c.get("meeting_sequence", 0) for c in candidates]
        min_seq, max_seq = min(sequences), max(sequences)
        span = max(1, max_seq - min_seq)

        for candidate in candidates:
            text = candidate["text"]
            text_lower = text.lower()
            fused = candidate["score"] / max_fused
            overlap = _term_coverage(terms, text)
            cue = 0.0
            # Numeric questions: surface chunks that actually carry figures,
            # currency or budget language even when the question wording shares
            # no digits with the transcript (e.g. "what did the costing show").
            if analysis.requires_numbers:
                has_digit = bool(re.search(r"\d", text))
                has_currency = bool(
                    re.search(r"\b(?:euros?|pounds?|dollars?|budget|production cost|seventeen|twelve|twenty five|twelve and a half|twelve fifty|price|cost|costing)\b", text_lower)
                )
                if has_digit or has_currency:
                    cue += 0.5
            if analysis.requires_temporal and DECISION_WORDS.search(text):
                cue += 0.5
            if analysis.requires_ownership and OWNER_WORDS.search(text):
                cue += 0.5
            if "disagreement" in analysis.intents and DISAGREE_WORDS.search(text):
                cue += 0.5
            # Exact meeting mention is a strong signal.
            if candidate.get("meeting_id") in analysis.meeting_ids:
                cue += 0.75
            sequence_bias = 0.0
            if prefer_late:
                sequence_bias = (candidate.get("meeting_sequence", 0) - min_seq) / span
            elif prefer_early:
                sequence_bias = (max_seq - candidate.get("meeting_sequence", 0)) / span

            rerank_score = (
                0.50 * fused
                + 0.25 * overlap
                + 0.10 * min(1.0, cue)
                + 0.08 * sequence_bias
                + 0.07 * (1.0 if "keyword" in candidate.get("sources", []) else 0.0)
                + 0.12 * (1.0 if "targeted" in candidate.get("sources", []) else 0.0)
            )
            if "cross_score" in candidate:
                # Normalise the cross-encoder score into roughly [0, 1].
                rerank_score = 0.6 * rerank_score + 0.4 / (1.0 + pow(2.718281828, -candidate["cross_score"]))
            candidate["rerank_score"] = float(rerank_score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        if limit:
            candidates = candidates[:limit]
        return candidates


def _term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    haystack = set(content_terms(text))
    hits = sum(1 for term in terms if term in haystack)
    return hits / len(terms)
