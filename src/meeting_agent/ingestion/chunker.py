"""Token-aware chunking that preserves exact source offsets and speakers.

Chunks are contiguous slices of the original transcript file.  That makes every
chunk independently verifiable: ``raw_text[start:end] == chunk.original_text``.
Speaker turns are never merged across meeting boundaries and are only split when
a single turn is larger than the configured maximum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from meeting_agent.models import Chunk, MeetingMeta, Turn
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)

_WORD_RE = re.compile(r"\S+\s*")


@lru_cache(maxsize=4)
def _encoder(encoding: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding)
    except Exception:  # pragma: no cover - offline fallback
        return None


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    enc = _encoder(encoding)
    if enc is None:
        # Conservative approximation used only when tiktoken is unavailable.
        return max(1, int(len(text.split()) * 1.35))
    return len(enc.encode(text, disallowed_special=()))


@dataclass
class Segment:
    turn_index: int
    speaker: str
    start_offset: int
    end_offset: int
    text: str
    tokens: int


def _split_large_turn(turn: Turn, max_tokens: int, encoding: str) -> list[Segment]:
    if count_tokens(turn.text, encoding) <= max_tokens:
        return [
            Segment(
                turn_index=turn.turn_index,
                speaker=turn.speaker,
                start_offset=turn.start_offset,
                end_offset=turn.end_offset,
                text=turn.text,
                tokens=count_tokens(turn.text, encoding),
            )
        ]

    pieces: list[Segment] = []
    buffer: list[str] = []
    buffer_start: int | None = None
    buffer_end: int | None = None
    buffer_tokens = 0
    for match in _WORD_RE.finditer(turn.text):
        word = match.group(0)
        word_tokens = count_tokens(word, encoding)
        if buffer and buffer_tokens + word_tokens > max_tokens:
            pieces.append(
                Segment(
                    turn_index=turn.turn_index,
                    speaker=turn.speaker,
                    start_offset=turn.start_offset + (buffer_start or 0),
                    end_offset=turn.start_offset + (buffer_end or 0),
                    text="".join(buffer).rstrip(),
                    tokens=buffer_tokens,
                )
            )
            buffer, buffer_start, buffer_end, buffer_tokens = [], None, None, 0
        if buffer_start is None:
            buffer_start = match.start()
        buffer.append(word)
        buffer_end = match.end()
        buffer_tokens += word_tokens

    if buffer and buffer_start is not None and buffer_end is not None:
        pieces.append(
            Segment(
                turn_index=turn.turn_index,
                speaker=turn.speaker,
                start_offset=turn.start_offset + buffer_start,
                end_offset=turn.start_offset + buffer_end,
                text="".join(buffer).rstrip(),
                tokens=buffer_tokens,
            )
        )
    return pieces


def chunk_meeting(
    meta: MeetingMeta,
    turns: list[Turn],
    *,
    target_tokens: int = 520,
    max_tokens: int = 700,
    overlap_tokens: int = 80,
    encoding: str = "cl100k_base",
) -> list[Chunk]:
    if not turns:
        return []

    segments: list[Segment] = []
    for turn in turns:
        segments.extend(_split_large_turn(turn, max_tokens, encoding))

    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_tokens = 0
    index = 0

    def flush() -> None:
        nonlocal current, current_tokens, index
        if not current:
            return
        start = current[0].start_offset
        end = current[-1].end_offset
        original = meta.raw_text[start:end]
        # The slice must reproduce the chunk exactly.  If a turn was trimmed of
        # trailing blank lines, use the segment text for embed_text only.
        speakers: list[str] = []
        speaker_tokens: dict[str, int] = {}
        for seg in current:
            if seg.speaker not in speakers:
                speakers.append(seg.speaker)
            speaker_tokens[seg.speaker] = speaker_tokens.get(seg.speaker, 0) + seg.tokens
        dominant = max(speaker_tokens.items(), key=lambda kv: kv[1])[0]
        embed_text = "\n".join(f"Speaker {s.speaker}: {s.text}" for s in current)
        chunk = Chunk(
            chunk_id=f"{meta.meeting_id}:c{index:04d}",
            meeting_id=meta.meeting_id,
            team_id=meta.team_id,
            meeting_sequence=meta.sequence,
            speaker=dominant,
            speakers=speakers,
            original_text=original,
            embed_text=embed_text,
            source_file=meta.source_file,
            source_start_offset=start,
            source_end_offset=end,
            embedding_id=f"{meta.meeting_id}:c{index:04d}",
            token_count=current_tokens,
            turn_indices=sorted({s.turn_index for s in current}),
        )
        chunks.append(chunk)
        index += 1

        # Build overlap tail for the next chunk, but never let overlap alone
        # constitute the whole next chunk (prevents duplicate chunks at file end).
        if overlap_tokens <= 0:
            current, current_tokens = [], 0
            return
        tail: list[Segment] = []
        tail_tokens = 0
        for seg in reversed(current):
            if tail_tokens + seg.tokens > overlap_tokens:
                break
            tail.insert(0, seg)
            tail_tokens += seg.tokens
        current, current_tokens = tail, tail_tokens

    for seg in segments:
        if current and current_tokens + seg.tokens > max_tokens:
            flush()
            # If the overlap tail plus the new segment is still too large, drop it.
            if current and current_tokens + seg.tokens > max_tokens:
                current, current_tokens = [], 0
        current.append(seg)
        current_tokens += seg.tokens
        if current_tokens >= target_tokens:
            # Peek: flush only when the next segment would exceed the max, or at
            # a natural turn boundary once we are near the target.
            if current_tokens >= max_tokens * 0.85:
                flush()

    if current:
        # final flush without leaving a duplicate tail
        tail = current
        if chunks and tail and all(
            seg.start_offset >= chunks[-1].source_start_offset
            and seg.end_offset <= chunks[-1].source_end_offset
            for seg in tail
        ):
            return chunks
        current = tail
        flush()

    return chunks
