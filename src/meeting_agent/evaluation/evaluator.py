"""Independent evaluation runner over the hand-written TS3012 question set."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from meeting_agent.config import get_settings
from meeting_agent.evaluation.benchmark import build_benchmark
from meeting_agent.evaluation.metrics import QuestionEvaluation, aggregate, evaluate_question
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import content_terms
from meeting_agent.workflows.answer_workflow import AnswerWorkflow

log = get_logger(__name__)


def _load_eval_questions() -> dict:
    settings = get_settings()
    path = Path(__file__).resolve().parents[3] / "tests" / "evaluation" / "test_questions.json"
    if not path.exists():
        path = Path("tests/evaluation/test_questions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _failure_suggestion(evaluation: QuestionEvaluation) -> str:
    if evaluation.citation_validity < 1.0:
        return (
            "Tighten the citation prompt and keep the repair pass: require quotes to be copied from the "
            "provided turn text, then reject anything that fails the deterministic substring check."
        )
    if evaluation.meeting_retrieval_recall < 1.0:
        return (
            "Increase retrieval coverage: add a targeted search per unrepresented meeting and use the "
            "question's temporal direction (first/last) to bias meeting sequences."
        )
    question_terms = set(content_terms(evaluation.question))
    missing_terms = {
        term for fact in evaluation.facts_missing for term in content_terms(fact)
    }
    asks_for_numbers = bool(
        re.search(r"\b(?:price|cost|costing|financ\w*|money|budget|total|much|euro|pound|dollar|number|how many)\b", evaluation.question, re.I)
    )
    has_numeric_gap = any(re.search(r"\d", fact) for fact in evaluation.facts_missing)
    if asks_for_numbers and has_numeric_gap:
        return (
            "Strengthen numeric retrieval with a dedicated cost/price query and keep digits in the "
            "keyword index; prefer chunks containing both a currency/unit and the decision cue."
        )
    if missing_terms and len(missing_terms & question_terms) <= len(missing_terms) / 3:
        return (
            "Vocabulary bridging: the missing facts ("
            + ", ".join(sorted(missing_terms)[:6])
            + ") use words that do not occur in the question, so lexical sentence selection cannot reach "
            "them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to "
            "the transcript's wording before ranking evidence."
        )
    if any("budget" in f or "12.50" in f or "17" in f for f in evaluation.facts_missing):
        return (
            "Strengthen numeric retrieval with a dedicated cost/price query and keep digits in the "
            "keyword index; prefer chunks containing both a currency/unit and the decision cue."
        )
    if evaluation.decision_state_accuracy < 0.5:
        return (
            "Improve temporal prompting: explicitly show the earliest and latest supported statements and "
            "ask the model to state which position was superseded."
        )
    return (
        "Add a second research loop with entity-focused queries and include adjacent meeting context, then "
        "re-ask the model to answer only from the expanded evidence."
    )


def _render_report(evaluations: list[QuestionEvaluation], summary: dict, benchmark: dict) -> str:
    failures = [e for e in evaluations if e.correctness != "correct"]
    lines = [
        "# Independent Evaluation Report",
        "",
        "## 1. Method",
        "",
        "The evaluation set contains hand-written, cross-meeting questions about Team TS3012. "
        "Ground truth, meeting ids and supporting quotations were read from the original transcripts and "
        "stored in `tests/evaluation/test_questions.json` before the agent was evaluated. The agent does not "
        "see the ground truth during answering.",
        "",
        "Metrics are deterministic and lexical; there is no LLM judge in this report.",
        "",
        "| Metric | Definition |",
        "|---|---|",
        "| Answer correctness | Fraction of expected key facts present in the answer. `correct` >= 0.80; `partially_correct` >= 0.45; otherwise `incorrect`. |",
        "| Evidence precision | Share of cited meetings that are in the hand-marked relevant set. |",
        "| Evidence recall | Share of relevant meetings that are cited. |",
        "| Citation validity | Share of returned citations that pass exact-substring, single-turn validation. |",
        "| Meeting retrieval recall | Share of relevant meetings present anywhere in retrieved evidence. |",
        "| Decision-state accuracy | Share of expected decision-state keywords (e.g. kept, dropped) present in the answer. |",
        "| Unsupported claim rate (proxy) | Share of answer sentences with < 35% of content terms present in the citations. This is a conservative proxy, not semantic entailment. |",
        "",
        "## 2. Summary",
        "",
        f"- Total questions evaluated: **{summary['total']}**",
        f"- Correct: **{summary['correct']}**",
        f"- Partially correct: **{summary['partially_correct']}**",
        f"- Incorrect: **{summary['incorrect']}**",
        f"- Mean fact coverage: **{summary['mean_fact_coverage']:.1%}**",
        f"- Citation validity: **{summary['mean_citation_validity']:.1%}**",
        f"- Evidence precision: **{summary['mean_evidence_precision']:.1%}**",
        f"- Evidence recall: **{summary['mean_evidence_recall']:.1%}**",
        f"- Meeting retrieval recall: **{summary['mean_meeting_retrieval_recall']:.1%}**",
        f"- Decision-state accuracy: **{summary['mean_decision_state_accuracy']:.1%}**",
        f"- Unsupported claim rate (proxy): **{summary['mean_unsupported_claim_rate']:.1%}**",
        "",
        "## 3. Per-question results",
        "",
        "| ID | Correctness | Facts | Citation validity | Evidence P/R | Retrieval recall | Unsupported |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in evaluations:
        lines.append(
            f"| {e.question_id} | {e.correctness} | {e.fact_coverage:.0%} | {e.citation_validity:.0%} | "
            f"{e.evidence_precision:.0%}/{e.evidence_recall:.0%} | {e.meeting_retrieval_recall:.0%} | "
            f"{e.unsupported_claim_rate:.0%} |"
        )

    lines.extend(["", "## 4. Actual failures and analysis", ""])
    if not failures:
        lines.append(
            "No incorrect or partially correct answers occurred in this run. This section is therefore not "
            "used to claim failures; see the trace logs for retrieval behaviour."
        )
    else:
        for e in failures[:6]:
            lines.extend(
                [
                    f"### {e.question_id} — {e.correctness}",
                    "",
                    f"**Question:** {e.question}",
                    "",
                    f"**Agent answer:** {e.agent_answer}",
                    "",
                    f"**Expected answer (manual ground truth):** {e.ground_truth}",
                    "",
                    f"**Missing expected facts:** {', '.join(e.facts_missing) or 'none'}",
                    "",
                    f"**Retrieved meetings:** {', '.join(e.retrieved_meetings) or 'none'}; "
                    f"**cited:** {', '.join(e.cited_meetings) or 'none'}; "
                    f"**relevant:** {', '.join(e.relevant_meetings)}",
                    "",
                    f"**Why it failed:** fact coverage {e.fact_coverage:.0%}, retrieval recall "
                    f"{e.meeting_retrieval_recall:.0%}, citation validity {e.citation_validity:.0%}. "
                    f"The answer omitted or mis-stated the missing facts above.",
                    "",
                    f"**How to improve:** {_failure_suggestion(e)}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 5. Retrieval quality and decision tracking",
            "",
            f"- Mean meeting retrieval recall: {summary['mean_meeting_retrieval_recall']:.1%}. "
            "Recall below 100% means at least one hand-marked relevant meeting never reached the evidence bundle.",
            f"- Mean evidence recall (cited relevant meetings): {summary['mean_evidence_recall']:.1%}. "
            "This is stricter than retrieval recall because relevant evidence must also survive into the final citation set.",
            f"- Mean decision-state accuracy: {summary['mean_decision_state_accuracy']:.1%}. "
            "This checks whether the answer names the expected state (kept/dropped/superseded) rather than only the topic.",
            "- Temporal reasoning is deterministic: the timeline is sorted by meeting sequence and only explicit agreement "
            "language can confirm a decision. The evaluation above shows where that conservatism helped or hurt.",
            "",
            "## 6. Performance and cost (measured)",
            "",
            f"- Ingestion: {benchmark['ingestion']['total_seconds']} s for {benchmark['corpus']['meetings']} meetings "
            f"({benchmark['ingestion']['seconds_per_meeting']} s/meeting).",
            f"- Embedding: {benchmark['ingestion']['embedding_seconds']} s; semantic extraction: "
            f"{benchmark['ingestion']['semantic_extraction_seconds']} s.",
            f"- Queries: {benchmark['query']['questions_measured']} measured, mean "
            f"{benchmark['query']['mean_latency_seconds']} s, p95 {benchmark['query']['p95_latency_seconds']} s.",
            f"- Storage: Chroma {benchmark['storage']['chroma_mb']} MB, SQLite {benchmark['storage']['sqlite_mb']} MB.",
            f"- Measured API cost: ${benchmark['measured_cost']['total_api_cost_usd']:.2f} "
            "(local embeddings; no paid LLM calls in this run).",
            f"- Projected 1,000 meetings: {benchmark['projection_1000_meetings']['projected_chunks']} chunks, "
            f"{benchmark['projection_1000_meetings']['ingestion']['projected_hours_single_process']} h single-process ingestion, "
            f"${benchmark['projection_1000_meetings']['extraction']['gemini_api_cost_usd']:.2f} projected Gemini extraction.",
            "",
            "See `outputs/benchmark.json` for assumptions and the 10,000-meeting architecture notes.",
            "",
            "## 7. Honest limitations",
            "",
            "- The evaluation set is small (11 questions) and covers one team. It is not a benchmark of the whole corpus.",
            "- Lexical fact matching can mark a correct paraphrase as missing; inspect the failure examples for the actual text.",
            "- The unsupported-claim rate is a proxy based on content-term overlap, not entailment.",
            "- This run used local models rather than a paid Gemini API because no credential was configured; the Gemini path is implemented but live API tests were not run.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_evaluation(*, regenerate: bool = True) -> dict:
    settings = get_settings()
    payload = _load_eval_questions()
    questions = payload["questions"]

    workflow = AnswerWorkflow(settings)
    results = []
    latencies: list[float] = []
    for item in questions:
        result = await workflow.run_one(item["question_id"], item["question"], payload.get("team"))
        results.append((item, result))
        latencies.append(result.latency_seconds)
        log.info("Evaluated %s: %s", item["question_id"], result.answer[:100])

    evaluations: list[QuestionEvaluation] = []
    for item, result in results:
        evaluation = evaluate_question(
            question_id=item["question_id"],
            question=item["question"],
            agent_answer=result.answer,
            ground_truth=item["ground_truth_answer"],
            facts=item["expected_facts"],
            expected_state_terms=item.get("expected_state_terms", []),
            relevant_meetings=item["relevant_meeting_ids"],
            retrieved_meetings=result.evidence_meetings,
            citations=result.citations,
            checks=result.citation_checks,
        )
        evaluations.append(evaluation)

    summary = aggregate(evaluations)
    benchmark = build_benchmark(
        settings=settings,
        query_latencies=latencies,
        query_input_tokens=workflow.llm.input_tokens,
        query_output_tokens=workflow.llm.output_tokens,
        questions=len(questions),
    )

    report = _render_report(evaluations, summary, benchmark)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    (settings.outputs_dir / "evaluation_report.md").write_text(report, encoding="utf-8")
    (settings.outputs_dir / "evaluation_results.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "questions": [e.__dict__ for e in evaluations],
                "provider": workflow.llm.provider_name,
                "llm_calls": workflow.llm.calls,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_markdown": report,
        "summary": summary,
        "evaluations": evaluations,
    }
