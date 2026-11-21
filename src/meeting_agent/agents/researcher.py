"""Researcher agent: bounded multi-step evidence gathering."""

from __future__ import annotations

from meeting_agent.agents.retrieval_agent import RetrievalAgent
from meeting_agent.config import Settings, get_settings
from meeting_agent.retrieval.evidence_collector import EvidenceBundle
from meeting_agent.retrieval.query_analyzer import QueryPlan, broaden_plan
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class Researcher:
    def __init__(self, settings: Settings | None = None, retrieval_agent: RetrievalAgent | None = None) -> None:
        self.settings = settings or get_settings()
        self.retrieval_agent = retrieval_agent or RetrievalAgent(self.settings)

    def gather(self, plan: QueryPlan, *, evidence: EvidenceBundle | None = None, loop: int = 0) -> EvidenceBundle:
        """One retrieval round; the workflow decides whether to loop again."""
        active_plan = broaden_plan(plan, loop) if loop > 0 else plan
        semantic = self.retrieval_agent.semantic(active_plan)
        keyword = self.retrieval_agent.keyword(active_plan)
        candidates = self.retrieval_agent.merge(semantic, keyword, active_plan)
        bundle = self.retrieval_agent.collect(active_plan, candidates)
        bundle.research_loops = loop
        if evidence is not None:
            # Preserve previously retrieved evidence while adding new material.
            existing = {chunk.chunk_id for chunk in evidence.chunks}
            merged_chunks = list(evidence.chunks)
            for chunk in bundle.chunks:
                if chunk.chunk_id not in existing:
                    merged_chunks.append(chunk)
            merged_chunks = merged_chunks[: self.settings.max_evidence_chunks]
            bundle.chunks = merged_chunks
            bundle.meetings = sorted({c.meeting_id for c in merged_chunks})
            bundle.sufficiency = self.retrieval_agent.collector.assess(bundle)
            bundle.history = evidence.history + [
                f"research loop {loop}: searched with {len(active_plan.queries)} queries "
                f"({len(bundle.chunks)} evidence chunks)"
            ]
        else:
            bundle.history.append(
                f"initial retrieval: {len(active_plan.queries)} queries, {len(bundle.chunks)} evidence chunks"
            )
        log.info(
            "Research round %d: %d chunks, meetings=%s, sufficient=%s, gaps=%s",
            loop, len(bundle.chunks), bundle.meetings, bundle.sufficient, bundle.sufficiency.get("gaps"),
        )
        return bundle
