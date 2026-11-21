"""Structured, secret-free logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False


class _RedactFilter(logging.Filter):
    """Defensive redaction of common credential shapes in log records."""

    import re as _re

    _PATTERNS = (
        _re.compile(r"(AIza[0-9A-Za-z_\-]{20,})"),
        _re.compile(r"(sk-[0-9A-Za-z]{16,})"),
        _re.compile(r"(?i)(api[_-]?key\s*[=:]\s*)(\S+)"),
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        redacted = message
        for pattern in self._PATTERNS:
            redacted = pattern.sub(lambda m: m.group(1) + "***" if m.lastindex == 1 else "***", redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def configure_logging(level: str | None = None, log_dir: Path | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    from meeting_agent.config import get_settings

    settings = get_settings()
    numeric = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    directory = log_dir or settings.log_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(directory / "meeting_agent.log", encoding="utf-8"))
    except OSError:
        pass

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(_RedactFilter())

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()
    for handler in handlers:
        root.addHandler(handler)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
