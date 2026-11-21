"""LangGraph answer workflow.

Nodes are real execution steps with separate responsibilities.  Retrieval loops
are bounded by ``max_research_loops``.  The graph records a trace of every step
so a reviewer can see exactly how an answer was produced.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from meeting_agent.agents.answer_agent import AnswerAgent
from meeting_agent.agents.citation_agent import CitationValidator
from meeting_agent.agents.reasoning_agent import ReasoningAgent
from meeting_agent.agents.researcher import Researcher
from meeting_agent.config import Settings, get_settings
from meeting_agent.llm.provider import LLMClient
from meeting_agent.models import AnswerRecord, AnswerResult, Citation
from meeting_agent.retrieval.query_analyzer import QueryPlan, analyze_question, plan_queries
from meeting_agent.retrieval.targeted_scan import TargetedScanner
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class AnswerState(TypedDict, total=False):
    question_id: str
    question: str
    team: str | None
    plan: QueryPlan
    semantic: list[dict]
    keyword: list[dict]
    candidates: list[dict]
    evidence: Any
    sufficient: bool
    loop: int
    temporal: Any
    draft: Any
    checks: list
    result: AnswerResult
    trace: list[str]
    errors: list[str]
    started_at: float


class AnswerWorkflow:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(settings=self.settings)
        self.validator = CitationValidator()
        self.researcher = Researcher(self.settings)
        self.scanner = TargetedScanner(self.settings)
        self.reasoner = ReasoningAgent()
        self.answer_agent = AnswerAgent(self.llm, self.validator, self.settings)
        self.graph = self._build_graph()

    # ----------------------------------------------------------------- helpers
    def _trace(self, state: AnswerState, message: str) -> list[str]:
        return list(state.get("trace", [])) + [message]

    def _build_graph(self):
        graph = StateGraph(AnswerState)

        async def analyze_node(state: AnswerState) -> dict:
            analysis = analyze_question(state["question"], state.get("team"))
            plan = plan_queries(analysis)
            return {
                "plan": plan,
                "trace": self._trace(
                    state,
                    f"question_analyzer: team={analysis.team} intents={analysis.intents} "
                    f"terms={analysis.key_terms[:8]}",
                ),
            }

        async def plan_node(state: AnswerState) -> dict:
            plan = state["plan"]
            return {
                "trace": self._trace(
                    state,
                    f"query_planner: {len(plan.queries)} queries "
                    f"({', '.join(q.kind for q in plan.queries)})",
                )
            }

        async def semantic_node(state: AnswerState) -> dict:
            hits = await asyncio.to_thread(self.researcher.retrieval_agent.semantic, state["plan"])
            return {
                "semantic": hits,
                "trace": self._trace(state, f"semantic_retrieval: {len(hits)} hits"),
            }

        async def keyword_node(state: AnswerState) -> dict:
            hits = await asyncio.to_thread(self.researcher.retrieval_agent.keyword, state["plan"])
            return {
                "keyword": hits,
                "trace": self._trace(state, f"keyword_retrieval: {len(hits)} hits"),
            }

        async def collect_node(state: AnswerState) -> dict:
            plan = state["plan"]
            targeted = await asyncio.to_thread(self.scanner.scan, plan)
            candidates = await asyncio.to_thread(
                self.researcher.retrieval_agent.merge,
                state.get("semantic", []),
                state.get("keyword", []) + targeted,
                plan,
            )
            evidence = await asyncio.to_thread(self.researcher.retrieval_agent.collect, plan, candidates)
            if state.get("evidence") is not None:
                # merge with evidence from earlier rounds
                prior = state["evidence"]
                existing = {c.chunk_id for c in prior.chunks}
                for chunk in evidence.chunks:
                    if chunk.chunk_id not in existing:
                        prior.chunks.append(chunk)
                prior.chunks = prior.chunks[: self.settings.max_evidence_chunks]
                prior.meetings = sorted({c.meeting_id for c in prior.chunks})
                prior.sufficiency = self.researcher.retrieval_agent.collector.assess(prior)
                evidence = prior
            return {
                "candidates": candidates,
                "evidence": evidence,
                "sufficient": evidence.sufficient,
                "trace": self._trace(
                    state,
                    f"evidence_collector: {len(evidence.chunks)} chunks, meetings={evidence.meetings}, "
                    f"sufficient={evidence.sufficient}, score={evidence.sufficiency.get('score')}, "
                    f"targeted={len(targeted)}",
                ),
            }

        async def assess_node(state: AnswerState) -> dict:
            evidence = state["evidence"]
            gaps = evidence.sufficiency.get("gaps", [])
            return {
                "sufficient": evidence.sufficient,
                "trace": self._trace(
                    state,
                    f"sufficiency_evaluator: sufficient={evidence.sufficient} gaps={gaps}",
                ),
            }

        async def research_node(state: AnswerState) -> dict:
            loop = int(state.get("loop", 0)) + 1
            return {
                "loop": loop,
                "trace": self._trace(state, f"research_loop: starting bounded loop {loop}/{self.settings.max_research_loops}"),
            }

        async def temporal_node(state: AnswerState) -> dict:
            evidence = state["evidence"]
            temporal = await asyncio.to_thread(
                self.reasoner.reason, evidence, state.get("team"), state["question"]
            )
            return {
                "temporal": temporal,
                "trace": self._trace(
                    state,
                    f"temporal_reasoning: {len(temporal.timeline)} events, changed={temporal.changed}, "
                    f"current_decisions={len(temporal.current_decisions)}",
                ),
            }

        async def answer_node(state: AnswerState) -> dict:
            draft = await self.answer_agent.answer(
                state["question"],
                state.get("team"),
                state["evidence"],
                state["temporal"],
                question_id=state.get("question_id"),
            )
            return {
                "draft": draft,
                "trace": self._trace(
                    state,
                    f"answer_generator: provider={draft.provider} citations={len(draft.citations)} "
                    f"insufficient={draft.insufficient_evidence}",
                ),
            }

        async def citation_node(state: AnswerState) -> dict:
            draft = state["draft"]
            checks = self.validator.validate_many(
                draft.citations, question_id=state.get("question_id"), source="final"
            )
            valid = [check.citation for check in checks if check.valid]
            # Deduplicate identical citations and cap at three.
            unique: list[Citation] = []
            seen: set[tuple[str, str]] = set()
            for citation in valid:
                key = (citation.meeting_id, citation.quote.casefold())
                if key in seen:
                    continue
                seen.add(key)
                unique.append(citation)
            draft.citations = unique[:3]
            if not unique and not draft.insufficient_evidence:
                draft.insufficient_evidence = True
                draft.answer = (
                    "Insufficient verified evidence to support an answer: no quotation passed "
                    "deterministic verification against the original transcripts."
                )
            return {
                "checks": checks,
                "draft": draft,
                "trace": self._trace(
                    state,
                    f"citation_validator: {len(valid)}/{len(checks)} valid; "
                    f"{len(draft.citations)} returned",
                ),
            }

        async def final_node(state: AnswerState) -> dict:
            draft = state["draft"]
            evidence = state["evidence"]
            result = AnswerResult(
                question_id=state.get("question_id", ""),
                team=state.get("team"),
                answer=draft.answer,
                citations=draft.citations,
                evidence_meetings=evidence.meetings,
                insufficient_evidence=draft.insufficient_evidence,
                citation_checks=state.get("checks", []),
                retrieval_trace=state.get("trace", []),
                latency_seconds=time.perf_counter() - state.get("started_at", time.perf_counter()),
                llm_calls=self.llm.calls,
            )
            return {
                "result": result,
                "trace": self._trace(state, "final_response: answer assembled"),
            }

        graph.add_node("analyze", analyze_node)
        graph.add_node("plan", plan_node)
        graph.add_node("semantic", semantic_node)
        graph.add_node("keyword", keyword_node)
        graph.add_node("collect", collect_node)
        graph.add_node("assess", assess_node)
        graph.add_node("research", research_node)
        graph.add_node("temporal", temporal_node)
        graph.add_node("answer", answer_node)
        graph.add_node("citations", citation_node)
        graph.add_node("final", final_node)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "plan")
        graph.add_edge("plan", "semantic")
        graph.add_edge("semantic", "keyword")
        graph.add_edge("keyword", "collect")
        graph.add_edge("collect", "assess")

        def route_after_assessment(state: AnswerState) -> str:
            if state.get("sufficient", False):
                return "temporal"
            if int(state.get("loop", 0)) >= self.settings.max_research_loops:
                return "temporal"
            return "research"

        graph.add_conditional_edges(
            "assess",
            route_after_assessment,
            {"temporal": "temporal", "research": "research"},
        )
        graph.add_edge("research", "semantic")
        graph.add_edge("temporal", "answer")
        graph.add_edge("answer", "citations")
        graph.add_edge("citations", "final")
        graph.add_edge("final", END)
        return graph.compile()

    # ------------------------------------------------------------------- public
    async def run_one(self, question_id: str, question: str, team: str | None = None) -> AnswerResult:
        started = time.perf_counter()
        initial: AnswerState = {
            "question_id": question_id,
            "question": question,
            "team": team,
            "loop": 0,
            "trace": [],
            "errors": [],
            "started_at": started,
        }
        try:
            final_state = await self.graph.ainvoke(initial, {"recursion_limit": 60})
            result = final_state.get("result")
            if result is None:
                raise RuntimeError("workflow returned no result")
        except Exception as exc:
            log.exception("Workflow failed for %s", question_id)
            result = AnswerResult(
                question_id=question_id,
                team=team,
                answer=f"Workflow error: {type(exc).__name__}: {exc}",
                citations=[],
                insufficient_evidence=True,
                retrieval_trace=[f"workflow_error: {exc}"],
                latency_seconds=time.perf_counter() - started,
            )
        try:
            from meeting_agent.storage import Database

            Database().log_query(
                {
                    "answer": result.answer,
                    "citations": [c.model_dump() for c in result.citations],
                    "meetings": result.evidence_meetings,
                    "trace": result.retrieval_trace,
                    "provider": self.llm.provider_name,
                },
                question_id=question_id,
                team=team,
            )
        except Exception as exc:  # pragma: no cover - logging must not fail the run
            log.warning("Could not log query: %s", exc)
        return result

    async def run_questions(self, questions: list[dict], *, concurrency: int | None = None) -> list[AnswerResult]:
        limit = concurrency or max(1, self.settings.llm_max_concurrency)
        semaphore = asyncio.Semaphore(limit)

        async def guarded(item: dict) -> AnswerResult:
            async with semaphore:
                log.info("Answering %s (%s)", item["question_id"], item.get("team"))
                result = await self.run_one(item["question_id"], item["question"], item.get("team"))
                log.info(
                    "%s done in %.1fs: %s",
                    item["question_id"], result.latency_seconds, result.answer[:120],
                )
                return result

        return list(await asyncio.gather(*(guarded(item) for item in questions)))

    def run_sync(self, question_id: str, question: str, team: str | None = None) -> AnswerResult:
        return asyncio.run(self.run_one(question_id, question, team))
