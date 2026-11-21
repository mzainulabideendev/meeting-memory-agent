"""JSON output schema and cost model tests."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from meeting_agent.utils.cost_calculator import Measurements, PRICING, measured_costs, project_to_scale

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["answers", "brief"],
    "properties": {
        "answers": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["question_id", "answer", "citations"],
                "properties": {
                    "question_id": {"type": "string", "pattern": "^Q\\d{2}$"},
                    "answer": {"type": "string", "minLength": 1},
                    "citations": {
                        "type": "array",
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "required": ["meeting_id", "quote"],
                            "properties": {
                                "meeting_id": {"type": "string", "minLength": 3},
                                "quote": {"type": "string", "minLength": 10},
                            },
                        },
                    },
                },
            },
        },
        "brief": {"type": "string", "minLength": 1},
    },
}


def test_valid_answer_payload():
    payload = {
        "answers": [
            {
                "question_id": "Q01",
                "answer": "A grounded answer.",
                "citations": [{"meeting_id": "TS3012b", "quote": "a quotation of at least six words"}],
            }
        ],
        "brief": "Brief text.",
    }
    jsonschema.validate(payload, ANSWER_SCHEMA)


def test_invalid_question_id_rejected():
    payload = {"answers": [{"question_id": "X1", "answer": "a", "citations": []}], "brief": "b"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, ANSWER_SCHEMA)


def test_too_many_citations_rejected():
    payload = {
        "answers": [
            {
                "question_id": "Q01",
                "answer": "a",
                "citations": [{"meeting_id": "TS3012b", "quote": "one two three four five six"}] * 4,
            }
        ],
        "brief": "b",
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(payload, ANSWER_SCHEMA)


def test_measured_costs_are_zero_for_local():
    m = Measurements(meetings=10, words=1000, tokens=1500, chunks=20, ingest_seconds=5, embedding_seconds=2)
    costs = measured_costs(m)
    assert costs["total_api_cost_usd"] == 0.0


def test_projection_scales_linearly():
    m = Measurements(
        meetings=100, words=100_000, tokens=140_000, chunks=2000,
        ingest_seconds=1000, embedding_seconds=400, query_input_tokens=10_000,
        query_output_tokens=2_000, questions=10, vector_bytes=10_000_000, sqlite_bytes=50_000_000,
    )
    projection = project_to_scale(m, 1000)
    assert projection["projected_words"] == 1_000_000
    assert projection["projected_chunks"] == 20_000
    assert projection["extraction"]["gemini_api_cost_usd"] > 0
    assert projection["pricing"]["as_of"]


def test_gemini_pricing_math():
    pricing = PRICING["gemini_2_0_flash"]
    cost = pricing.llm_cost(1_000_000, 1_000_000)
    assert cost == pytest.approx(0.50, rel=1e-6)
