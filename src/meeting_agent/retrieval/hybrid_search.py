"""Hybrid semantic + keyword retrieval with reciprocal rank fusion."""

from __future__ import annotations

from typing import Any

from meeting_agent.config import Settings, get_settings
from meeting_agent.memory.keyword_store import KeywordStore
from meeting_agent.memory.vector_store import VectorStore
from meeting_agent.retrieval.query_analyzer import PlannedQuery, QueryPlan
from meeting_agent.utils.embeddings import EmbedderProtocol, get_embedder
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


def _where_for_team(team_id: str | None) -> dict[str, Any] | None:
    return {"team_id": team_id} if team_id else None


class HybridSearch:
    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        keyword_store: KeywordStore | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore()
        self.keyword_store = keyword_store or KeywordStore()
        self._embedder = embedder

    @property
    def embedder(self) -> EmbedderProtocol:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def semantic_search(self, plan: QueryPlan, *, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.vector_top_k
        results: list[dict[str, Any]] = []
        where = _where_for_team(plan.team_filter)
        for planned in plan.queries:
            embedding = self.embedder.encode([planned.query])[0]
            hits = self.vector_store.search(embedding, top_k=k, where=where)
            for rank, hit in enumerate(hits, start=1):
                metadata = hit.get("metadata", {})
                results.append(
                    {
                        "chunk_id": hit["chunk_id"],
                        "meeting_id": metadata.get("meeting_id"),
                        "team_id": metadata.get("team_id"),
                        "meeting_sequence": int(metadata.get("meeting_sequence", 0)),
                        "speaker": metadata.get("speaker", ""),
                        "text": hit.get("document", ""),
                        "rank": rank,
                        "score": float(hit.get("similarity", 0.0)),
                        "query": planned.query,
                        "query_kind": planned.kind,
                        "weight": planned.weight,
                        "source": "semantic",
                    }
                )
        return results

    def keyword_search(self, plan: QueryPlan, *, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.keyword_top_k
        results: list[dict[str, Any]] = []
        for planned in plan.queries:
            hits = self.keyword_store.search(
                planned.query,
                top_k=k,
                team_id=plan.team_filter,
                speaker=plan.analysis.speaker_filters[0] if len(plan.analysis.speaker_filters) == 1 else None,
                require_all=planned.require_all,
            )
            for rank, hit in enumerate(hits, start=1):
                results.append(
                    {
                        **hit,
                        "rank": rank,
                        "query": planned.query,
                        "query_kind": planned.kind,
                        "weight": planned.weight,
                        "source": hit.get("source", "keyword"),
                    }
                )
        return results

    def fuse(self, semantic: list[dict], keyword: list[dict], *, limit: int | None = None, analysis=None) -> list[dict[str, Any]]:
        """Reciprocal rank fusion plus source weighting and deduplication."""
        fused: dict[str, dict[str, Any]] = {}
        rrf_k = self.settings.rrf_k

        # Collect all query terms for boosting
        query_terms = set()
        if analysis:
            query_terms.update(analysis.key_terms)
            query_terms.update(analysis.expansion_terms)

        def add(hits: list[dict], source_weight: float) -> None:
            for hit in hits:
                chunk_id = hit["chunk_id"]
                contribution = source_weight * hit.get("weight", 1.0) / (rrf_k + hit["rank"])
                entry = fused.get(chunk_id)
                if entry is None:
                    entry = {
                        "chunk_id": chunk_id,
                        "meeting_id": hit.get("meeting_id"),
                        "team_id": hit.get("team_id"),
                        "meeting_sequence": hit.get("meeting_sequence", 0),
                        "speaker": hit.get("speaker", ""),
                        "text": hit.get("text", ""),
                        "score": 0.0,
                        "sources": [],
                        "queries": [],
                    }
                    fused[chunk_id] = entry
                entry["score"] += contribution
                source = hit.get("source", "unknown")
                if source not in entry["sources"]:
                    entry["sources"].append(source)
                query_kind = hit.get("query_kind")
                if query_kind and query_kind not in entry["queries"]:
                    entry["queries"].append(query_kind)

        add(semantic, 1.0)
        add(keyword, 1.15)  # exact terms/numbers are disproportionately valuable

        # Boost chunks that contain numeric entities matching the question
        if query_terms:
            numeric_terms = {t for t in query_terms if any(c.isdigit() for c in t)}
            if numeric_terms:
                for entry in fused.values():
                    text_lower = entry["text"].lower()
                    for num_term in numeric_terms:
                        if num_term in text_lower:
                            entry["score"] *= 1.5  # Strong boost for numeric matches
                            break

        results = list(fused.values())
        # Drop overlapping near-duplicates: the highest-scoring chunk of a
        # heavily overlapping pair wins.  Chunks from different meetings never
        # get deduplicated against each other.
        results.sort(key=lambda r: r["score"], reverse=True)
        deduped: list[dict[str, Any]] = []
        for candidate in results:
            candidate_terms = set(candidate["text"].lower().split())
            if any(
                candidate["meeting_id"] == kept["meeting_id"]
                and _jaccard(candidate_terms, set(kept["text"].lower().split())) > 0.75
                for kept in deduped
            ):
                continue
            deduped.append(candidate)
            if limit and len(deduped) >= limit:
                break
        return deduped


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
