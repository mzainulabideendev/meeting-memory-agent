"""Briefing workflow: memory + evidence -> verified team brief."""

from __future__ import annotations

import asyncio
from pathlib import Path

from meeting_agent.agents.briefing_agent import BriefingAgent
from meeting_agent.config import Settings, get_settings
from meeting_agent.models import BriefResult
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class BriefingWorkflow:
    def __init__(self, settings: Settings | None = None, agent: BriefingAgent | None = None) -> None:
        self.settings = settings or get_settings()
        self.agent = agent or BriefingAgent(self.settings)

    async def run(self, team: str, output_path: Path | None = None) -> BriefResult:
        result = await self.agent.generate(team)
        path = output_path or (self.settings.outputs_dir / "brief.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.markdown, encoding="utf-8")
        log.info("Brief for %s written to %s (%d citations)", team, path, len(result.citations))
        return result

    def run_sync(self, team: str, output_path: Path | None = None) -> BriefResult:
        return asyncio.run(self.run(team, output_path))
