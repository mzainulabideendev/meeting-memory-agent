"""Evaluation metrics with explicit, reproducible definitions.

These are lexical/structural metrics, not an LLM judge.  They are deliberately
conservative and documented in the generated report so readers can interpret
them correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from meeting_agent.models import AnswerRecord, Citation
from meeting_agent.utils.text_normalization import content_terms, normalize_for_match, split_sentences

_FRAME = re.compile(
    r"^(the retrieved evidence shows|in [a-z]{2}\d{4}[a-d], speaker [a-d] said|insufficient evidence)",
    re.I,
)


def fact_coverage(answer: str, facts: list[str]) -> tuple[float, list[str], list[str]]:
    """Return coverage where each fact may list alternatives separated by '|'."""
    normalized = normalize_for_match(answer)
    matched: list[str] = []
    missing: list[str] = []
    for fact in facts:
        alternatives = [normalize_for_match(part.strip()) for part in fact.split("|") if part.strip()]
        if any(alt in normalized for alt in alternatives):
            matched.append(fact)
        else:
            missing.append(fact)
    return (len(matched) / max(1, len(facts)), matched, missing)


def classify_correctness(coverage: float) -> str:
    if coverage >= 0.8:
        return "correct"
    if coverage >= 0.45:
        return "partially_correct"
    return "incorrect"


def citation_validity(checks) -> float:
    if not checks:
        return 0.0
    return sum(1 for check in checks if getattr(check, "valid", False)) / len(checks)


def evidence_precision(cited_meetings: list[str], relevant_meetings: list[str]) -> float:
    if not cited_meetings:
        return 0.0
    relevant = set(relevant_meetings)
    return sum(1 for m in cited_meetings if m in relevant) / len(cited_meetings)


def evidence_recall(cited_meetings: list[str], relevant_meetings: list[str]) -> float:
    if not relevant_meetings:
        return 0.0
    return len(set(cited_meetings) & set(relevant_meetings)) / len(set(relevant_meetings))


def meeting_retrieval_recall(retrieved: list[str], relevant: list[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & set(relevant)) / len(set(relevant))


def decision_state_accuracy(answer: str, expected_terms: list[str]) -> float:
    if not expected_terms:
        return 1.0
    normalized = normalize_for_match(answer)
    matched = 0
    for term in expected_terms:
        # Each entry may list several acceptable state words separated by '|'
        # (e.g. "kept|remained|still|final").  The answer satisfies the term
        # when ANY alternative is present; never require the literal pipe string.
        alternatives = [normalize_for_match(part.strip()) for part in term.split("|") if part.strip()]
        if any(alt in normalized for alt in alternatives):
            matched += 1
    return matched / len(expected_terms)


def unsupported_claim_rate(answer: str, citations: list[Citation]) -> float:
    """Proxy: a sentence is supported when >=35% of its content terms appear in
    any citation quotation.  This over-counts unsupported prose and is reported
    as a proxy, not as a semantic entailment judgement.
    """
    sentences = [s for s in split_sentences(answer) if len(s.split()) >= 4 and not _FRAME.match(s)]
    if not sentences:
        return 0.0
    citation_terms: set[str] = set()
    for citation in citations:
        citation_terms |= set(content_terms(citation.quote))
    unsupported = 0
    for sentence in sentences:
        terms = set(content_terms(sentence))
        if not terms:
            continue
        overlap = len(terms & citation_terms) / len(terms)
        if overlap < 0.35:
            unsupported += 1
    return unsupported / len(sentences)


@dataclass
class QuestionEvaluation:
    question_id: str
    question: str
    agent_answer: str
    ground_truth: str
    facts_matched: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)
    fact_coverage: float = 0.0
    correctness: str = "incorrect"
    citation_validity: float = 0.0
    evidence_precision: float = 0.0
    evidence_recall: float = 0.0
    meeting_retrieval_recall: float = 0.0
    decision_state_accuracy: float = 0.0
    unsupported_claim_rate: float = 0.0
    retrieved_meetings: list[str] = field(default_factory=list)
    cited_meetings: list[str] = field(default_factory=list)
    relevant_meetings: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    notes: str = ""


def evaluate_question(
    *,
    question_id: str,
    question: str,
    agent_answer: str,
    ground_truth: str,
    facts: list[str],
    expected_state_terms: list[str],
    relevant_meetings: list[str],
    retrieved_meetings: list[str],
    citations: list[Citation],
    checks,
) -> QuestionEvaluation:
    coverage, matched, missing = fact_coverage(agent_answer, facts)
    cited = [c.meeting_id for c in citations]
    return QuestionEvaluation(
        question_id=question_id,
        question=question,
        agent_answer=agent_answer,
        ground_truth=ground_truth,
        facts_matched=matched,
        facts_missing=missing,
        fact_coverage=round(coverage, 3),
        correctness=classify_correctness(coverage),
        citation_validity=round(citation_validity(checks), 3),
        evidence_precision=round(evidence_precision(cited, relevant_meetings), 3),
        evidence_recall=round(evidence_recall(cited, relevant_meetings), 3),
        meeting_retrieval_recall=round(meeting_retrieval_recall(retrieved_meetings, relevant_meetings), 3),
        decision_state_accuracy=round(decision_state_accuracy(agent_answer, expected_state_terms), 3),
        unsupported_claim_rate=round(unsupported_claim_rate(agent_answer, citations), 3),
        retrieved_meetings=retrieved_meetings,
        cited_meetings=cited,
        relevant_meetings=relevant_meetings,
        citations=[c.model_dump() for c in citations],
    )


def aggregate(evaluations: list[QuestionEvaluation]) -> dict:
    n = len(evaluations) or 1
    counts = {"correct": 0, "partially_correct": 0, "incorrect": 0}
    for evaluation in evaluations:
        counts[evaluation.correctness] += 1
    return {
        "total": len(evaluations),
        "correct": counts["correct"],
        "partially_correct": counts["partially_correct"],
        "incorrect": counts["incorrect"],
        "mean_fact_coverage": round(sum(e.fact_coverage for e in evaluations) / n, 3),
        "mean_citation_validity": round(sum(e.citation_validity for e in evaluations) / n, 3),
        "mean_evidence_precision": round(sum(e.evidence_precision for e in evaluations) / n, 3),
        "mean_evidence_recall": round(sum(e.evidence_recall for e in evaluations) / n, 3),
        "mean_meeting_retrieval_recall": round(sum(e.meeting_retrieval_recall for e in evaluations) / n, 3),
        "mean_decision_state_accuracy": round(sum(e.decision_state_accuracy for e in evaluations) / n, 3),
        "mean_unsupported_claim_rate": round(sum(e.unsupported_claim_rate for e in evaluations) / n, 3),
    }
