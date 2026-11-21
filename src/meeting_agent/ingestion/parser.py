"""Parsing of the AMI-derived transcript format.

The parser never rewrites transcript content.  It records character offsets into
the exact file text so that every chunk and every citation can be traced back to
the original bytes on disk.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from meeting_agent.models import MeetingMeta, Turn
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)

_MEETING_RE = re.compile(r"^Meeting:\s*(\S+)\s*$", re.MULTILINE)
_TEAM_RE = re.compile(r"^Team:\s*(\S+)\s*$", re.MULTILINE)
_NUMBER_RE = re.compile(r"^Meeting number:\s*(\d+)\s*\(([a-z])\)", re.MULTILINE)
_SPEAKERS_RE = re.compile(r"^Speakers:\s*(.+)$", re.MULTILINE)
# A turn header is a timestamp followed by "Speaker X" on its own line.
_TURN_HEADER = re.compile(r"^(\d{2}:\d{2}:\d{2})[ \t]+Speaker[ \t]+([A-Za-z0-9]+)[ \t]*$", re.MULTILINE)


class TranscriptParseError(ValueError):
    """Raised when a transcript cannot be parsed into turns."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meeting_id_from_filename(path: Path) -> str:
    """Derive the meeting id from the filename (e.g. TS3012a.txt -> TS3012a)."""
    return path.stem


def parse_transcript_text(raw_text: str, source_file: str = "<memory>") -> tuple[MeetingMeta, list[Turn]]:
    if not raw_text.strip():
        raise TranscriptParseError(f"{source_file}: file is empty")

    meeting_match = _MEETING_RE.search(raw_text)
    team_match = _TEAM_RE.search(raw_text)
    number_match = _NUMBER_RE.search(raw_text)
    speakers_match = _SPEAKERS_RE.search(raw_text)

    filename_id = Path(source_file).stem if source_file != "<memory>" else ""
    meeting_id = meeting_match.group(1) if meeting_match else filename_id
    if not meeting_id:
        raise TranscriptParseError(f"{source_file}: could not determine meeting id")

    team_id = team_match.group(1) if team_match else re.sub(r"[a-z]$", "", meeting_id)
    if number_match:
        sequence = int(number_match.group(1))
        letter = number_match.group(2)
    else:
        suffix = meeting_id[-1].lower()
        sequence = ord(suffix) - ord("a") + 1 if "a" <= suffix <= "z" else 1
        letter = suffix

    header_speakers: list[str] = []
    if speakers_match:
        header_speakers = re.findall(r"Speaker\s+([A-Za-z0-9]+)", speakers_match.group(1))

    headers = list(_TURN_HEADER.finditer(raw_text))
    if not headers:
        raise TranscriptParseError(f"{source_file}: no speaker turns found")

    turns: list[Turn] = []
    for index, match in enumerate(headers):
        body_start = match.end()
        # skip the single newline terminating the header line
        if raw_text.startswith("\r\n", body_start):
            body_start += 2
        elif body_start < len(raw_text) and raw_text[body_start] == "\n":
            body_start += 1
        body_end = headers[index + 1].start() if index + 1 < len(headers) else len(raw_text)
        raw_body = raw_text[body_start:body_end]
        # Trim only trailing blank lines from the stored body; offsets still
        # point at the exact raw span so a raw slice is reproducible.
        body = raw_body.rstrip("\r\n")
        if not body.strip():
            continue
        turns.append(
            Turn(
                turn_id=f"{meeting_id}:t{index:05d}",
                meeting_id=meeting_id,
                team_id=team_id,
                sequence=sequence,
                turn_index=len(turns),
                speaker=match.group(2),
                start_time=match.group(1),
                text=body,
                start_offset=body_start,
                end_offset=body_start + len(body),
            )
        )

    if not turns:
        raise TranscriptParseError(f"{source_file}: all speaker turns were empty")

    meta = MeetingMeta(
        meeting_id=meeting_id,
        team_id=team_id,
        sequence=sequence,
        letter=letter,
        source_file=source_file,
        word_count=sum(len(t.text.split()) for t in turns),
        turn_count=len(turns),
        raw_text=raw_text,
        header_speakers=header_speakers,
    )
    return meta, turns


def parse_transcript_file(path: Path) -> tuple[MeetingMeta, list[Turn]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, turns = parse_transcript_text(raw, source_file=str(path))
    meta.file_hash = file_sha256(path)
    return meta, turns


def iter_transcript_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.txt") if p.is_file())
