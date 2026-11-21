"""Validated domain models shared across the system."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MemoryStatus = Literal[
    "proposed", "confirmed", "rejected", "superseded", "unresolved", "uncertain"
]


class MeetingMeta(BaseModel):
    meeting_id: str
    team_id: str
    sequence: int
    letter: str
    source_file: str
    file_hash: str = ""
    word_count: int = 0
    turn_count: int = 0
    raw_text: str = ""
    header_speakers: list[str] = Field(default_factory=list)


class Turn(BaseModel):
    turn_id: str
    meeting_id: str
    team_id: str
    sequence: int
    turn_index: int
    speaker: str
    start_time: str
    text: str
    start_offset: int
    end_offset: int


class Chunk(BaseModel):
    chunk_id: str
    meeting_id: str
    team_id: str
    meeting_sequence: int
    speaker: str
    speakers: list[str] = Field(default_factory=list)
    original_text: str
    embed_text: str
    source_file: str
    source_start_offset: int
    source_end_offset: int
    embedding_id: str
    token_count: int
    turn_indices: list[int] = Field(default_factory=list)


class Citation(BaseModel):
    meeting_id: str
    quote: str

    @field_validator("quote")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("quote must not be empty")
        return value


class CitationCheck(BaseModel):
    citation: Citation
    valid: bool
    reason: str
    turn_index: int | None = None
    speaker: str | None = None


class AnswerRecord(BaseModel):
    question_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AnswerResult(AnswerRecord):
    team: str | None = None
    evidence_meetings: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    retrieval_trace: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0
    llm_calls: int = 0


class DecisionRecord(BaseModel):
    record_id: str
    team_id: str
    kind: str = "decision"
    topic: str
    statement: str
    status: MemoryStatus
    meeting_id: str
    meeting_sequence: int
    turn_index: int = 0
    speaker: str | None = None
    valid_from_meeting: str | None = None
    valid_to_meeting: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    extraction_version: str = "rule-v1"
    reason: str | None = None


class ActionItem(BaseModel):
    action_id: str
    team_id: str
    description: str
    owner_speaker: str | None = None
    owner_role: str | None = None
    deadline: str | None = None
    status: Literal["open", "done", "unclear"] = "open"
    meeting_id: str
    meeting_sequence: int
    evidence: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    kind: Literal["action", "commitment", "obligation"] = "action"


class OpenQuestion(BaseModel):
    question_id: str
    team_id: str
    text: str
    meeting_id: str
    meeting_sequence: int
    status: Literal["unresolved", "resolved", "uncertain"] = "unresolved"
    resolution: str | None = None
    evidence: list[Citation] = Field(default_factory=list)


class Disagreement(BaseModel):
    disagreement_id: str
    team_id: str
    topic: str
    positions: list[dict[str, str]] = Field(default_factory=list)
    resolved: bool | None = None
    resolution: str | None = None
    meeting_ids: list[str] = Field(default_factory=list)
    evidence: list[Citation] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    chunk_id: str
    meeting_id: str
    team_id: str
    meeting_sequence: int
    speaker: str
    text: str
    score: float = 0.0
    sources: list[str] = Field(default_factory=list)
    turns: list[Turn] = Field(default_factory=list)


class Question(BaseModel):
    question_id: str
    team: str | None = None
    question: str


class IngestionStats(BaseModel):
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    meetings: int = 0
    turns: int = 0
    chunks: int = 0
    words: int = 0
    tokens: int = 0
    embedding_seconds: float = 0.0
    total_seconds: float = 0.0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    errors: list[dict[str, str]] = Field(default_factory=list)


class BriefResult(BaseModel):
    team: str
    markdown: str
    decisions: list[DecisionRecord] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    meetings_used: list[str] = Field(default_factory=list)
