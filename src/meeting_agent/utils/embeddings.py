"""Configurable sentence-embedding backend."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from meeting_agent.config import get_settings
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


class EmbedderProtocol(Protocol):
    model_name: str
    dimension: int | None

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy singleton wrapper around SentenceTransformers.

    The model is local and free to use; no API credentials are required and no
    embedding API charge is ever claimed in the cost model.
    """

    _instance: "SentenceTransformerEmbedder | None" = None

    def __init__(self, model_name: str | None = None, device: str | None = None, batch_size: int | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self._model = None
        self.dimension: int | None = None

    @classmethod
    def get(cls) -> "SentenceTransformerEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


class HashingEmbedder:
    """Deterministic offline embedder used only when the ML stack is absent.

    It is a feature-hashing bag-of-words vector.  The system records that this
    backend was used so results are never presented as neural embeddings.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.model_name = f"hashing-trick-{dimension}"
        self.dimension = dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        from meeting_agent.utils.text_normalization import tokenize_words

        out: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            for token in tokenize_words(text):
                idx = hash(token) % self.dimension
                sign = 1.0 if hash(token + "#") % 2 == 0 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            out.append(vec.tolist())
        return out


def get_embedder() -> EmbedderProtocol:
    settings = get_settings()
    try:
        return SentenceTransformerEmbedder.get()
    except ImportError:  # pragma: no cover - depends on optional install
        log.warning("sentence-transformers is not installed; using deterministic hashing embeddings")
        return HashingEmbedder()
