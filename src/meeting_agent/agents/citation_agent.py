"""Deterministic citation verification against the original transcripts.

A citation is valid only when:
1. the meeting id exists in the ingested corpus;
2. the quotation has at least six words (assessment requirement);
3. the quotation occurs verbatim in that meeting's original transcript text,
   ignoring case and whitespace only;
4. the quotation lies inside a single speaker turn (not stitched across turns).

No fuzzy matching is used.  If a quote fails, it is rejected and recorded; the
system never silently edits a quote to make it pass.
"""

from __future__ import annotations

from meeting_agent.models import Citation, CitationCheck
from meeting_agent.storage import Database
from meeting_agent.utils.logging import get_logger
from meeting_agent.utils.text_normalization import (
    contains_quote,
    normalize_for_match,
)

log = get_logger(__name__)

MIN_QUOTE_WORDS = 6


class CitationValidator:
    def __init__(self, database: Database | None = None) -> None:
        self.db = database or Database()
        self._meeting_cache: dict[str, dict] = {}

    def _meeting(self, meeting_id: str) -> dict | None:
        key = meeting_id.strip()
        if key in self._meeting_cache:
            return self._meeting_cache[key]
        row = self.db.get_meeting(key)
        if row is None:
            # Case-insensitive fallback (meeting ids are stored uppercase).
            with self.db.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM meetings WHERE lower(meeting_id) = lower(?)", (key,)
                ).fetchone()
        if row is not None:
            self._meeting_cache[key] = dict(row)
            return self._meeting_cache[key]
        return None

    def meeting_exists(self, meeting_id: str) -> bool:
        return self._meeting(meeting_id) is not None

    def validate(self, citation: Citation) -> CitationCheck:
        meeting = self._meeting(citation.meeting_id)
        if meeting is None:
            return CitationCheck(
                citation=citation, valid=False,
                reason=f"meeting id {citation.meeting_id!r} does not exist in the corpus",
            )

        normalized_quote = normalize_for_match(citation.quote)
        word_count = len(normalized_quote.split())
        if word_count < MIN_QUOTE_WORDS:
            return CitationCheck(
                citation=citation, valid=False,
                reason=f"quotation has {word_count} words; at least {MIN_QUOTE_WORDS} are required",
            )

        raw_text = meeting["raw_text"]
        turns = self.db.get_turns(meeting["meeting_id"])
        if not contains_quote(raw_text, citation.quote):
            if self._spans_adjacent_turns(normalized_quote, turns):
                return CitationCheck(
                    citation=citation,
                    valid=False,
                    reason="quotation spans multiple speaker turns (or crosses turn boundaries)",
                )
            return CitationCheck(
                citation=citation,
                valid=False,
                reason="quotation does not occur verbatim in the meeting transcript",
            )

        # Must be inside one speaker turn.
        for turn in turns:
            if contains_quote(turn["text"], citation.quote):
                return CitationCheck(
                    citation=citation,
                    valid=True,
                    reason="verbatim quotation found inside one speaker turn",
                    turn_index=int(turn["turn_index"]),
                    speaker=turn["speaker"],
                )

        return CitationCheck(
            citation=citation,
            valid=False,
            reason="quotation spans multiple speaker turns (or crosses turn boundaries)",
        )

    @staticmethod
    def _spans_adjacent_turns(normalized_quote: str, turns: list[dict]) -> bool:
        """Diagnostic only: does the quote sit across two consecutive turns?

        The transcript separates turns with blank lines, so a quote stitched
        from the end of one turn and the start of the next is *not* verbatim in
        the raw text.  Detecting that case lets us report the real reason
        instead of a generic mismatch.
        """
        texts = [normalize_for_match(turn["text"]) for turn in turns]
        for first, second in zip(texts, texts[1:]):
            if normalized_quote in f"{first} {second}":
                return True
        return False

    def validate_many(
        self,
        citations: list[Citation],
        *,
        question_id: str | None = None,
        source: str = "answer",
    ) -> list[CitationCheck]:
        checks: list[CitationCheck] = []
        for citation in citations:
            check = self.validate(citation)
            checks.append(check)
            self.db.log_citation_attempt(question_id, citation, check.valid, check.reason, source)
            if not check.valid:
                log.warning("Rejected citation for %s: %s | quote=%r", question_id, check.reason, citation.quote[:120])
        return checks

    def valid_only(self, checks: list[CitationCheck]) -> list[Citation]:
        return [check.citation for check in checks if check.valid]

    def find_turn_for_quote(self, meeting_id: str, quote: str) -> tuple[int, str] | None:
        meeting = self._meeting(meeting_id)
        if meeting is None or not contains_quote(meeting["raw_text"], quote):
            return None
        for turn in self.db.get_turns(meeting["meeting_id"]):
            if contains_quote(turn["text"], quote):
                return int(turn["turn_index"]), turn["speaker"]
        return None
