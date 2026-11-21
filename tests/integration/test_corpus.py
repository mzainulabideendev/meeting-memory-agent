"""Integration tests against the ingested corpus and generated artefacts.

These tests are read-only and skip automatically when the corpus has not been
ingested (so a fresh checkout can still run the unit suite).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meeting_agent.agents.citation_agent import CitationValidator  # noqa: E402
from meeting_agent.config import get_settings  # noqa: E402
from meeting_agent.memory.keyword_store import KeywordStore  # noqa: E402
from meeting_agent.models import Citation  # noqa: E402
from meeting_agent.storage import Database  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db() -> Database:
    settings = get_settings()
    if not settings.database_path.exists():
        pytest.skip("corpus not ingested")
    database = Database(settings.database_path)
    if database.meeting_count() == 0:
        pytest.skip("corpus not ingested")
    return database


def test_whole_corpus_ingested(db: Database):
    assert db.meeting_count() == 138
    assert db.chunk_count() >= 1000
    assert len(db.list_teams()) == 35


def test_chunk_offsets_match_raw_transcript(db: Database):
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT c.original_text, c.source_start_offset, c.source_end_offset, m.raw_text "
            "FROM chunks c JOIN meetings m ON m.meeting_id = c.meeting_id LIMIT 200"
        ).fetchall()
    assert rows
    for row in rows:
        assert row["raw_text"][row["source_start_offset"] : row["source_end_offset"]] == row["original_text"]


def test_fts_search_returns_team_scoped_hits(db: Database):
    rows = KeywordStore(db).search("speech recognition", top_k=5, team_id="TS3012")
    assert rows
    assert all(row["team_id"] == "TS3012" for row in rows)


def test_generated_brief_citations_are_verbatim(db: Database):
    path = get_settings().outputs_dir / "brief.md"
    if not path.exists():
        pytest.skip("brief not generated")
    text = path.read_text(encoding="utf-8")
    # Rendered as: > “quote” — meeting
    pairs = re.findall(r"^  > “(.+?)” — (\S+)$", text, re.MULTILINE)
    assert pairs, "brief should contain verified quotations"
    validator = CitationValidator(db)
    for quote, meeting_id in pairs:
        check = validator.validate(Citation(meeting_id=meeting_id, quote=quote))
        assert check.valid, f"{meeting_id}: {check.reason} :: {quote[:80]}"


def test_answers_json_contract(db: Database):
    path = get_settings().outputs_dir / "answers.json"
    if not path.exists():
        pytest.skip("answers not generated")
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers = payload["answers"]
    assert [a["question_id"] for a in answers] == [f"Q{i:02d}" for i in range(1, 26)]
    assert payload["brief"].strip()
    validator = CitationValidator(db)
    for answer in answers:
        assert answer["answer"].strip()
        assert len(answer["citations"]) <= 3
        for citation in answer["citations"]:
            check = validator.validate(Citation(**citation))
            assert check.valid, f"{answer['question_id']}: {check.reason}"
