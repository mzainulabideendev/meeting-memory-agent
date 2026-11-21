"""Regression tests for the four documented failures (E02, E03, E05, E07)."""

from __future__ import annotations

import pytest
from meeting_agent.workflows.answer_workflow import AnswerWorkflow
from meeting_agent.config import get_settings


def test_regression_failures():
    settings = get_settings()
    workflow = AnswerWorkflow(settings)

    # Test cases corresponding to E02, E03, E05, E07
    test_cases = [
        {
            "question_id": "E02",
            "question": "What power source did TS3012 settle on, and did they adopt the rechargeable option that was discussed?",
            "team": "TS3012",
        },
        {
            "question_id": "E03",
            "question": "How did TS3012's position on the LCD screen change between meetings b and d?",
            "team": "TS3012",
        },
        {
            "question_id": "E05",
            "question": "What did TS3012 know about finances in meeting c, and what did the final costing in meeting d show?",
            "team": "TS3012",
        },
        {
            "question_id": "E07",
            "question": "What shape and features were initially proposed for the lost remote locator?",
            "team": "TS3012",
        },
    ]

    for tc in test_cases:
        result = workflow.run_sync(tc["question_id"], tc["question"], tc["team"])
        assert result is not None
        assert result.answer is not None
        # Verify citations are valid
        for check in result.citation_checks:
            assert check.valid, f"Citation failed for {tc['question_id']}: {check.reason}"
