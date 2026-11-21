"""LLM provider robustness: JSON parsing, retries, failure handling."""

from __future__ import annotations

import asyncio

import pytest

from meeting_agent.llm.provider import (
    BaseProvider,
    LLMClient,
    LLMError,
    LLMInvalidOutput,
    LLMResponse,
    LLMUsage,
    build_provider,
    extract_json,
)
from meeting_agent.config import Settings


def test_extract_json_plain():
    assert extract_json('{"answer": "x"}') == {"answer": "x"}


def test_extract_json_fenced():
    assert extract_json('```json\n{"answer": "x"}\n```') == {"answer": "x"}


def test_extract_json_surrounded_by_prose():
    assert extract_json('Here is the result: {"answer": "x"} done') == {"answer": "x"}


def test_extract_json_rejects_non_object():
    with pytest.raises(LLMInvalidOutput):
        extract_json("[1, 2, 3]")


def test_extract_json_rejects_malformed():
    with pytest.raises(LLMInvalidOutput):
        extract_json('{"answer": "x"')


class FlakyProvider(BaseProvider):
    name = "flaky"
    model = "fake"

    def __init__(self, settings):
        super().__init__(settings)
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise LLMError("temporary")
        return LLMResponse(text='{"answer":"ok"}', provider=self.name, model=self.model, usage=LLMUsage(5, 2))


def test_retry_succeeds_after_transient_failures(test_settings):
    settings = test_settings.model_copy(update={"llm_max_retries": 3, "llm_retry_backoff_seconds": 0.0})
    provider = FlakyProvider(settings)
    client = LLMClient(provider=provider, settings=settings)
    response = asyncio.run(client.generate("hi"))
    assert response.text
    assert provider.calls == 3
    assert client.calls == 1
    assert client.input_tokens == 5


def test_retry_exhaustion_raises(test_settings):
    class AlwaysFail(BaseProvider):
        name = "fail"
        model = "fake"

        async def generate(self, *args, **kwargs):
            raise LLMError("nope")

    settings = test_settings.model_copy(update={"llm_max_retries": 2, "llm_retry_backoff_seconds": 0.0})
    client = LLMClient(provider=AlwaysFail(settings), settings=settings)
    with pytest.raises(LLMError):
        asyncio.run(client.generate("hi"))


def test_auto_provider_is_honest_without_credentials(test_settings):
    settings = test_settings.model_copy(update={"llm_provider": "auto", "gemini_api_key": None, "local_llm_enabled": False})
    provider = build_provider(settings)
    assert provider.name == "heuristic"


def test_explicit_gemini_without_key_is_unavailable(test_settings):
    from meeting_agent.llm.provider import GeminiProvider

    settings = test_settings.model_copy(update={"gemini_api_key": None})
    provider = GeminiProvider(settings)
    assert provider.available is False
    with pytest.raises(LLMError):
        asyncio.run(provider.generate("hi"))
