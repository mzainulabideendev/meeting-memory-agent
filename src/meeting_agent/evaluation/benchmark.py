"""Performance and storage benchmarking from actual measured runs."""

from __future__ import annotations

import json
from pathlib import Path

from meeting_agent.config import Settings, get_settings
from meeting_agent.storage import Database
from meeting_agent.utils.cost_calculator import (
    Measurements,
    measured_costs,
    project_to_scale,
    recommend_10k_architecture,
    PRICING_AS_OF,
)
from meeting_agent.utils.logging import get_logger

log = get_logger(__name__)


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def collect_measurements(
    settings: Settings | None = None,
    database: Database | None = None,
    *,
    query_latencies: list[float] | None = None,
    query_input_tokens: int = 0,
    query_output_tokens: int = 0,
    questions: int = 0,
) -> Measurements:
    settings = settings or get_settings()
    db = database or Database()
    metrics = db.get_metrics()

    meetings = db.meeting_count()
    with db.connect() as connection:
        words = int(connection.execute("SELECT COALESCE(SUM(word_count),0) FROM meetings").fetchone()[0])
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        tokens = int(connection.execute("SELECT COALESCE(SUM(token_count),0) FROM chunks").fetchone()[0])
    ingest_seconds = metrics.get("ingestion_total_seconds", {}).get("value", 0.0)
    embedding_seconds = metrics.get("embedding_total_seconds", {}).get("value", 0.0)
    extraction_seconds = metrics.get("semantic_extraction_seconds", {}).get("value", 0.0)
    query_seconds = sum(query_latencies) if query_latencies else 0.0

    return Measurements(
        meetings=meetings,
        words=words,
        tokens=tokens,
        chunks=chunks,
        ingest_seconds=ingest_seconds,
        embedding_seconds=embedding_seconds,
        extraction_seconds=extraction_seconds,
        query_input_tokens=query_input_tokens,
        query_output_tokens=query_output_tokens,
        query_seconds=query_seconds,
        questions=questions,
        vector_bytes=directory_size(settings.chroma_path),
        sqlite_bytes=settings.database_path.stat().st_size if settings.database_path.exists() else 0,
    )


def build_benchmark(
    *,
    settings: Settings | None = None,
    database: Database | None = None,
    query_latencies: list[float] | None = None,
    query_input_tokens: int = 0,
    query_output_tokens: int = 0,
    questions: int = 0,
) -> dict:
    settings = settings or get_settings()
    m = collect_measurements(
        settings,
        database,
        query_latencies=query_latencies,
        query_input_tokens=query_input_tokens,
        query_output_tokens=query_output_tokens,
        questions=questions,
    )
    latencies = sorted(query_latencies or [])
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    benchmark = {
        "measured_at": PRICING_AS_OF,
        "corpus": {
            "meetings": m.meetings,
            "words": m.words,
            "chunks": m.chunks,
            "chunk_tokens": m.tokens,
            "avg_chunks_per_meeting": round(m.chunks / max(1, m.meetings), 2),
        },
        "ingestion": {
            "total_seconds": round(m.ingest_seconds, 2),
            "embedding_seconds": round(m.embedding_seconds, 2),
            "semantic_extraction_seconds": round(m.extraction_seconds, 2),
            "seconds_per_meeting": round(m.ingest_seconds / max(1, m.meetings), 2),
            "words_per_second": round(m.words / max(0.001, m.ingest_seconds), 1),
        },
        "query": {
            "questions_measured": questions,
            "mean_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p50_latency_seconds": round(p50, 2),
            "p95_latency_seconds": round(p95, 2),
            "input_tokens": query_input_tokens,
            "output_tokens": query_output_tokens,
        },
        "storage": {
            "chroma_bytes": m.vector_bytes,
            "chroma_mb": round(m.vector_bytes / 1e6, 2),
            "sqlite_bytes": m.sqlite_bytes,
            "sqlite_mb": round(m.sqlite_bytes / 1e6, 2),
            "bytes_per_meeting": round((m.vector_bytes + m.sqlite_bytes) / max(1, m.meetings)),
        },
        "measured_cost": measured_costs(m),
        "projection_1000_meetings": project_to_scale(m, 1000),
        "projection_10000_meetings": project_to_scale(m, 10000),
        "scale_recommendations_10000": recommend_10k_architecture(),
        "note": (
            "1000/10000-meeting figures are linear projections from the measured 138-meeting run, "
            "not production benchmarks. See assumptions in each section."
        ),
    }
    path = settings.outputs_dir / "benchmark.json"
    path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    log.info("Benchmark written to %s", path)
    return benchmark
