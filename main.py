"""Convenience entry point.

Usage:
    python main.py ingest [--reset]
    python main.py answer
    python main.py brief --team TS3012
    python main.py evaluate
    python main.py ask "question" --team TS3012
    python main.py inspect
    python main.py status

Equivalent to ``python -m meeting_agent <command>``.  The package lives in
``src/`` so this script inserts that directory on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from meeting_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
