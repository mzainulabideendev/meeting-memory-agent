"""Persistent vector memory backed by ChromaDB."""

from __future__ import annotations

from typing import Any

from meeting_agent.config import get_settings
from meeting_agent.models import Chunk
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "meeting_chunks"


class VectorStore:
    def __init__(self, path: str | None = None, collection_name: str = COLLECTION_NAME) -> None:
        settings = get_settings()
        self.path = str(path or settings.chroma_path)
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.path,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        metadatas = [
            {
                "meeting_id": c.meeting_id,
                "team_id": c.team_id,
                "meeting_sequence": int(c.meeting_sequence),
                "speaker": c.speaker,
                "speakers": ",".join(c.speakers),
                "source_file": c.source_file,
                "start_offset": int(c.source_start_offset),
                "end_offset": int(c.source_end_offset),
                "token_count": int(c.token_count),
            }
            for c in chunks
        ]
        # Chroma has practical batch limits; 256 is comfortably safe.
        for i in range(0, len(chunks), 256):
            batch_chunks = chunks[i : i + 256]
            self.collection.upsert(
                ids=[c.embedding_id for c in batch_chunks],
                embeddings=embeddings[i : i + 256],
                documents=[c.embed_text for c in batch_chunks],
                metadatas=metadatas[i : i + 256],
            )

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.collection.delete(ids=ids)

    def delete_meeting(self, meeting_id: str) -> None:
        self.collection.delete(where={"meeting_id": meeting_id})

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 12,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        n = max(1, min(top_k, self.count()))
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "document": document,
                    "metadata": metadata or {},
                    "distance": float(distance),
                    # cosine distance -> bounded similarity
                    "similarity": max(0.0, 1.0 - float(distance)),
                }
            )
        return rows

    def reset(self) -> None:
        if self._client is not None:
            self._client.delete_collection(self.collection_name)
            self._collection = None
