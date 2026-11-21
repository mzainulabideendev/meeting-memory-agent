"""Inspect transcript turns by keyword (development/demonstration helper)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.ingestion.parser import parse_transcript_file  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/inspect_transcript.py FILE [keyword ...]")
        raise SystemExit(2)
    path = Path(sys.argv[1])
    keywords = [k.lower() for k in sys.argv[2:]]
    meta, turns = parse_transcript_file(path)
    print(f"{meta.meeting_id} | Team {meta.team_id} | meeting {meta.sequence} | {len(turns)} turns")
    for turn in turns:
        if not keywords or any(k in turn.text.lower() for k in keywords):
            print(f"--- turn {turn.turn_index} | {turn.start_time} | Speaker {turn.speaker}")
            print(turn.text)


if __name__ == "__main__":
    main()
