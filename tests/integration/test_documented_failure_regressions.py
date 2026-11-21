"""Regression coverage for the four manually documented answer failures."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meeting_agent.agents.answer_agent import AnswerAgent  # noqa: E402
from meeting_agent.agents.citation_agent import CitationValidator  # noqa: E402
from meeting_agent.agents.reasoning_agent import ReasoningAgent  # noqa: E402
from meeting_agent.config import Settings  # noqa: E402
from meeting_agent.memory.decision_store import DecisionStore  # noqa: E402
from meeting_agent.memory.semantic_memory import extract_semantic_memory  # noqa: E402
from meeting_agent.models import EvidenceChunk  # noqa: E402
from meeting_agent.retrieval.evidence_collector import EvidenceBundle  # noqa: E402
from meeting_agent.retrieval.query_analyzer import analyze_question  # noqa: E402
from meeting_agent.storage import Database  # noqa: E402
from meeting_agent.ingestion.parser import parse_transcript_file  # noqa: E402

pytestmark = pytest.mark.integration

FAILURE_IDS = ("E02", "E03", "E05", "E07")
EXPECTED_EVIDENCE = {
    "E02": ("regular batteries", "rechargeable"),
    "E03": ("lose", "consideration", "ninety percent"),
    "E05": ("finances", "seventeen euros", "twelve and a half"),
    "E07": ("beep", "flash", "implement"),
}


def _questions() -> dict[str, dict]:
    path = ROOT / "tests" / "evaluation" / "test_questions.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["question_id"]: item for item in payload["questions"]}


@pytest.mark.parametrize("question_id", FAILURE_IDS)
async def test_documented_failure_evidence_is_selected(question_id, tmp_path):
    settings = Settings(
        database_path=tmp_path / "meetings.db",
        chroma_path=tmp_path / "chroma",
        outputs_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        llm_provider="heuristic",
    )
    db = Database(settings.database_path)
    item = _questions()[question_id]
    analysis = analyze_question(item["question"], team_hint="TS3012")
    chunks = []
    records = []
    transcript_dir = ROOT / "data" / "Catalist_AI_Intern_Build_Test_Data" / "transcripts"

    for meeting_id in item["relevant_meeting_ids"]:
        meta, turns = parse_transcript_file(transcript_dir / f"{meeting_id}.txt")
        db.insert_meeting(meta)
        db.insert_turns(turns)
        records.extend(extract_semantic_memory(meta, turns).decisions)
        chunks.append(
            EvidenceChunk(
                chunk_id=meeting_id,
                meeting_id=meeting_id,
                team_id=meta.team_id,
                meeting_sequence=meta.sequence,
                speaker="",
                text=" ".join(turn.text for turn in turns),
                turns=turns,
            )
        )

    decision_store = DecisionStore(db)
    decision_store.save("TS3012", records)
    bundle = EvidenceBundle(
        analysis=analysis,
        chunks=chunks,
        meetings=item["relevant_meeting_ids"],
        sufficiency={"sufficient": True},
    )
    temporal = ReasoningAgent(decision_store).reason(bundle, "TS3012", item["question"])
    draft = await AnswerAgent(settings=settings).answer(
        item["question"], "TS3012", bundle, temporal, question_id=question_id
    )

    assert draft.citations
    checks = CitationValidator(db).validate_many(draft.citations)
    assert all(check.valid for check in checks)
    cited_evidence = " ".join(citation.quote.lower() for citation in draft.citations)
    for term in EXPECTED_EVIDENCE[question_id]:
        assert term in cited_evidence, (
            draft.answer + "\n" + "\n".join(
                f"{event.meeting_id} ({event.kind}): {event.quote}"
                for event in temporal.timeline
            )
        )

    cited_meetings = {citation.meeting_id for citation in draft.citations}
    if question_id == "E03":
        assert {"TS3012b", "TS3012c", "TS3012d"} <= cited_meetings
    elif question_id == "E05":
        assert {"TS3012c", "TS3012d"} <= cited_meetings
    elif question_id in {"E02", "E07"}:
        assert {"TS3012b", "TS3012c"} <= cited_meetings

    if question_id == "E05":
        assert re.search(r"\b(?:17|seventeen)\b", draft.answer, re.I)
        assert re.search(r"\b(?:12\.50|twelve and a half)\b", draft.answer, re.I)
