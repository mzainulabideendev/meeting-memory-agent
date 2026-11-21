"""Central configuration.

Every path, model and limit is configurable through environment variables or a
`.env` file.  Nothing secret is ever written to logs or persisted in artefacts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file lives at <root>/src/meeting_agent/config.py
ROOT_DIR = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    """Find the unpacked assessment bundle without hardcoding its name."""
    data = ROOT_DIR / "data"
    candidates = [p for p in data.glob("*/transcripts") if p.is_dir()]
    if candidates:
        return candidates[0].parent
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths -----------------------------------------------------------
    data_dir: Path = Field(default_factory=_default_data_dir)
    transcripts_dir: Path | None = None
    questions_path: Path | None = None
    chroma_path: Path = ROOT_DIR / "storage" / "chroma"
    database_path: Path = ROOT_DIR / "storage" / "meetings.db"
    outputs_dir: Path = ROOT_DIR / "outputs"
    log_dir: Path = ROOT_DIR / "storage" / "logs"

    # --- Chunking / embeddings ------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    chunk_target_tokens: int = 520
    chunk_max_tokens: int = 700
    chunk_overlap_tokens: int = 80
    tokenizer_encoding: str = "cl100k_base"

    # --- Retrieval -------------------------------------------------------
    vector_top_k: int = 12
    keyword_top_k: int = 12
    max_evidence_chunks: int = 24
    max_research_loops: int = 2
    rrf_k: int = 60
    enable_cross_encoder_rerank: bool = False
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM -------------------------------------------------------------
    llm_provider: str = "auto"  # auto | gemini | openai_compatible | local | heuristic
    gemini_api_key: str | None = None
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 2.0
    llm_max_concurrency: int = 2

    # OpenAI-compatible endpoint (optional, e.g. a hosted gateway or Ollama)
    openai_base_url: str | None = None
    openai_api_key: str | None = None

    # Local HF model (optional offline generation backend)
    local_llm_enabled: bool = False
    local_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_max_new_tokens: int = 512
    local_device: str = "cpu"
    local_context_tokens: int = 6144

    # --- Behaviour -------------------------------------------------------
    log_level: str = "INFO"
    random_seed: int = 13

    @field_validator("outputs_dir", "chroma_path", "database_path", "log_dir", mode="after")
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        return value if value.is_absolute() else (ROOT_DIR / value)

    @property
    def resolved_transcripts_dir(self) -> Path:
        return self.transcripts_dir or (self.data_dir / "transcripts")

    @property
    def resolved_questions_path(self) -> Path:
        return self.questions_path or (self.data_dir / "questions.json")

    def ensure_directories(self) -> None:
        for path in (
            self.chroma_path,
            self.database_path.parent,
            self.outputs_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
