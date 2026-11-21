"""Shared test fixtures. All paths are temporary; tests never touch the real corpus."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from meeting_agent.config import Settings  # noqa: E402
from meeting_agent.ingestion.parser import parse_transcript_text  # noqa: E402
from meeting_agent.storage import Database  # noqa: E402

TRANSCRIPT_A = """Meeting: ZZ0001a
Team: ZZ0001
Meeting number: 1 (a) of the team's four meetings
Speakers: Speaker A, Speaker B
Times: turn start time from the start of the recording (HH:MM:SS)
Source: synthetic test fixture


00:00:01 Speaker A
Hello team, I am the project manager. We should use a titanium case for the remote control because it looks premium.

00:00:12 Speaker B
I think we should use plastic instead because titanium is far too expensive for our budget.

00:00:20 Speaker A
Okay, we have decided to use plastic for the case. I will check the exact price by Friday.

00:00:31 Speaker B
Good, and we still need to find out whether the LCD screen fits the budget.
"""

TRANSCRIPT_B = """Meeting: ZZ0001b
Team: ZZ0001
Meeting number: 2 (b) of the team's four meetings
Speakers: Speaker A, Speaker B
Times: turn start time from the start of the recording (HH:MM:SS)
Source: synthetic test fixture


00:00:02 Speaker A
Welcome back. We agreed that the plastic case stays, and I have checked the price with the supplier.

00:00:14 Speaker B
Great. I also found out that the LCD screen would cost four euros, so we should drop it.

00:00:25 Speaker A
Yes, we decided to drop the LCD screen and keep the plastic case instead.
"""


@pytest.fixture
def synthetic_text() -> str:
    return TRANSCRIPT_A


@pytest.fixture
def parsed_synthetic():
    return parse_transcript_text(TRANSCRIPT_A, source_file="ZZ0001a.txt")


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture
def seeded_db(db: Database) -> Database:
    from meeting_agent.ingestion.parser import parse_transcript_file
    import tempfile

    for name, text in (("ZZ0001a", TRANSCRIPT_A), ("ZZ0001b", TRANSCRIPT_B)):
        path = Path(tempfile.mkdtemp()) / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        meta, turns = parse_transcript_file(path)
        db.insert_meeting(meta)
        db.insert_turns(turns)
    return db


@pytest.fixture
def test_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=ROOT / "data" / "Catalist_AI_Intern_Build_Test_Data",
        chroma_path=tmp_path / "chroma",
        database_path=tmp_path / "meetings.db",
        outputs_dir=tmp_path / "outputs",
        log_dir=tmp_path / "logs",
        llm_provider="heuristic",
        embedding_batch_size=4,
    )
