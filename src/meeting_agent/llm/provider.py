"""LLM provider abstraction with retries, timeouts and JSON repair.

Providers
---------
gemini             Google Gemini via the official ``google-genai`` SDK.
openai_compatible  Any /chat/completions endpoint (gateway, vLLM, Ollama).
local              A local HuggingFace causal-LM (no network after download).
heuristic          Deterministic non-LLM fallback; the workflow then uses
                   extractive answer composition and labels it as such.

No credential is ever logged or persisted.  A missing key disables the Gemini
provider rather than silently faking a response.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from meeting_agent.config import Settings, get_settings
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMInvalidOutput(LLMError):
    pass


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_seconds: float = 0.0
    attempts: int = 1


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, tolerating fences and surrounding prose."""
    cleaned = strip_code_fences(text)
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
        raise LLMInvalidOutput("model returned JSON that is not an object")
    except json.JSONDecodeError:
        pass

    # Find the first balanced object.
    start = cleaned.find("{")
    if start == -1:
        raise LLMInvalidOutput("no JSON object found in model output")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(cleaned)):
        char = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = cleaned[start : i + 1]
                try:
                    value = json.loads(snippet)
                except json.JSONDecodeError as exc:
                    raise LLMInvalidOutput(f"malformed JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise LLMInvalidOutput("model returned JSON that is not an object")
                return value
    raise LLMInvalidOutput("unbalanced JSON object in model output")


class BaseProvider:
    name = "base"
    supports_json_mode = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = settings.llm_model

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_json_mode = True

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.api_key = settings.gemini_api_key

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not configured")

        def _call() -> LLMResponse:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
                max_output_tokens=max_output_tokens or self.settings.llm_max_output_tokens,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            return LLMResponse(
                text=response.text or "",
                provider=self.name,
                model=self.model,
                usage=LLMUsage(
                    input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                    output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                ),
            )

        started = time.perf_counter()
        try:
            response = await asyncio.to_thread(_call)
        except Exception as exc:  # normalise SDK errors
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                raise LLMRateLimitError(message) from exc
            if "timeout" in message.lower():
                raise LLMTimeoutError(message) from exc
            raise LLMError(message) from exc
        response.latency_seconds = time.perf_counter() - started
        return response


class OpenAICompatibleProvider(BaseProvider):
    name = "openai_compatible"
    supports_json_mode = True

    @property
    def available(self) -> bool:
        return bool(self.settings.openai_base_url)

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        import httpx

        if not self.settings.openai_base_url:
            raise LLMError("OPENAI_BASE_URL is not configured")
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        payload = {
            "model": self.model,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]
            ),
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_output_tokens or self.settings.llm_max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.settings.llm_timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code == 429:
                    raise LLMRateLimitError(response.text[:500])
                response.raise_for_status()
                data = response.json()
        except LLMRateLimitError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise LLMError(str(exc)) from exc

        usage = data.get("usage", {})
        return LLMResponse(
            text=data["choices"][0]["message"]["content"] or "",
            provider=self.name,
            model=data.get("model", self.model),
            usage=LLMUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
            ),
            latency_seconds=time.perf_counter() - started,
        )


class LocalHFProvider(BaseProvider):
    """Local HuggingFace causal LM backend.

    ``transformers`` is an optional dependency.  The model is downloaded once
    and then runs offline on CPU/GPU.  This provider is intentionally explicit:
    it is only selected when ``LLM_PROVIDER=local`` or local generation is
    enabled, because the first call may download several gigabytes.
    """

    name = "local"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = settings.local_model_name
        self._tokenizer = None
        self._model = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local LLM %s on %s (first run downloads weights)", self.model, self.settings.local_device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        dtype = torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.settings.local_device)
        self._model.eval()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync,
                prompt,
                system,
                temperature,
                max_output_tokens,
                json_mode,
            )

    def _generate_sync(
        self,
        prompt: str,
        system: str | None,
        temperature: float | None,
        max_output_tokens: int | None,
        json_mode: bool,
    ) -> LLMResponse:
        import torch

        self._load()
        assert self._tokenizer is not None and self._model is not None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self.settings.local_device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        max_new = max_output_tokens or self.settings.local_max_new_tokens
        started = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=(temperature or 0) > 0,
                temperature=max(temperature or self.settings.llm_temperature, 0.01),
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0][input_tokens:]
        text_out = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return LLMResponse(
            text=text_out,
            provider=self.name,
            model=self.model,
            usage=LLMUsage(input_tokens=input_tokens, output_tokens=int(generated.shape[-1])),
            latency_seconds=time.perf_counter() - started,
        )


class HeuristicProvider(BaseProvider):
    """Marks that no generative model is available.

    The workflow branches to deterministic extractive composition instead of
    calling this class, so it never pretends to be an LLM answer.
    """

    name = "heuristic"

    @property
    def available(self) -> bool:
        return True

    async def generate(self, *args: Any, **kwargs: Any) -> LLMResponse:  # noqa: D401
        raise LLMError("No generative LLM configured; use the deterministic extractive path")


class LLMClient:
    """Retry/timeout wrapper that tracks token usage across a run."""

    def __init__(self, provider: BaseProvider | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latency_seconds = 0.0
        self.failures: list[str] = []

    @property
    def provider_name(self) -> str:
        return self.provider.name

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        json_mode: bool = False,
        max_retries: int | None = None,
    ) -> LLMResponse:
        retries = self.settings.llm_max_retries if max_retries is None else max_retries
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.provider.generate(
                        prompt,
                        system=system,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        json_mode=json_mode,
                    ),
                    timeout=self.settings.llm_timeout_seconds,
                )
                self.calls += 1
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens
                self.latency_seconds += response.latency_seconds
                response.attempts = attempt
                return response
            except (LLMRateLimitError, LLMTimeoutError, LLMError, asyncio.TimeoutError) as exc:
                last_error = exc
                self.failures.append(f"{type(exc).__name__}: {exc}")
                log.warning("LLM attempt %d/%d failed: %s", attempt, retries, exc)
                if attempt < retries:
                    await asyncio.sleep(self.settings.llm_retry_backoff_seconds * attempt)
        raise LLMError(f"LLM failed after {retries} attempts: {last_error}")

    async def generate_json(self, prompt: str, **kwargs: Any) -> tuple[dict[str, Any], LLMResponse]:
        kwargs["json_mode"] = True
        response = await self.generate(prompt, **kwargs)
        return extract_json(response.text), response


def build_provider(settings: Settings | None = None) -> BaseProvider:
    settings = settings or get_settings()
    choice = (settings.llm_provider or "auto").lower()

    if choice == "gemini":
        return GeminiProvider(settings)
    if choice == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    if choice == "local":
        return LocalHFProvider(settings)
    if choice == "heuristic":
        return HeuristicProvider(settings)

    # auto
    gemini = GeminiProvider(settings)
    if gemini.available:
        return gemini
    openai = OpenAICompatibleProvider(settings)
    if openai.available:
        return openai
    if settings.local_llm_enabled:
        local = LocalHFProvider(settings)
        if local.available:
            return local
    return HeuristicProvider(settings)


def provider_status(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    provider = build_provider(settings)
    return {
        "provider": provider.name,
        "model": provider.model,
        "generative": provider.name != "heuristic",
        "gemini_key_configured": bool(settings.gemini_api_key),
        "openai_base_url_configured": bool(settings.openai_base_url),
        "local_enabled": settings.local_llm_enabled,
    }
