"""Retrieval agent: runs semantic and keyword search through the hybrid layer."""

from __future__ import annotations

from meeting_agent.config import Settings, get_settings
from meeting_agent.retrieval.evidence_collector import EvidenceBundle, EvidenceCollector
from meeting_agent.retrieval.hybrid_search import HybridSearch
from meeting_agent.retrieval.query_analyzer import QueryPlan
from meeting_agent.retrieval.reranker import EvidenceReranker
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class RetrievalAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        hybrid: HybridSearch | None = None,
        reranker: EvidenceReranker | None = None,
        collector: EvidenceCollector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.hybrid = hybrid or HybridSearch(self.settings)
        self.reranker = reranker or EvidenceReranker(self.settings)
        self.collector = collector or EvidenceCollector(self.settings)

    def semantic(self, plan: QueryPlan) -> list[dict]:
        return self.hybrid.semantic_search(plan)

    def keyword(self, plan: QueryPlan) -> list[dict]:
        return self.hybrid.keyword_search(plan)

    def merge(self, semantic: list[dict], keyword: list[dict], plan: QueryPlan) -> list[dict]:
        fused = self.hybrid.fuse(semantic, keyword, limit=self.settings.max_evidence_chunks * 2, analysis=plan.analysis)
        return self.reranker.rerank(
            fused,
            plan.analysis,
            prefer_late=plan.prefer_late_meetings,
            prefer_early=plan.prefer_early_meetings,
            limit=self.settings.max_evidence_chunks,
        )

    def collect(self, plan: QueryPlan, candidates: list[dict]) -> EvidenceBundle:
        return self.collector.collect(plan, candidates)
