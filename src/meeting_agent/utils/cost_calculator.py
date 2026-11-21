"""Cost and scale estimation from measured performance.

No API price is invented.  The price table below is configurable and records
the source and retrieval date.  Because this build uses a local embedding model
and (by default) a local/heuristic generator, the measured API spend is $0; the
1,000-meeting estimate shows both the measured-local scenario and a projected
Gemini scenario using the published token prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

PRICING_AS_OF = "2026-11-19"
PRICING_SOURCES = {
    "gemini_2_0_flash": "https://ai.google.dev/gemini-api/docs/pricing (standard tier, accessed 2026-11-19)",
    "openai_text_embedding_3_small": "https://developers.openai.com/api/docs/models/text-embedding-3-small (accessed 2026-11-19)",
}


@dataclass
class Pricing:
    name: str
    input_usd_per_million: float
    output_usd_per_million: float
    embedding_usd_per_million: float = 0.0
    source: str = ""
    as_of: str = PRICING_AS_OF

    def llm_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.input_usd_per_million
            + output_tokens / 1_000_000 * self.output_usd_per_million
        )

    def embedding_cost(self, tokens: int) -> float:
        return tokens / 1_000_000 * self.embedding_usd_per_million


PRICING = {
    "gemini_2_0_flash": Pricing(
        name="gemini-2.0-flash",
        input_usd_per_million=0.10,
        output_usd_per_million=0.40,
        embedding_usd_per_million=0.0,
        source=PRICING_SOURCES["gemini_2_0_flash"],
    ),
    "openai_text_embedding_3_small": Pricing(
        name="text-embedding-3-small",
        input_usd_per_million=0.0,
        output_usd_per_million=0.0,
        embedding_usd_per_million=0.02,
        source=PRICING_SOURCES["openai_text_embedding_3_small"],
    ),
    "local": Pricing(
        name="local models",
        input_usd_per_million=0.0,
        output_usd_per_million=0.0,
        embedding_usd_per_million=0.0,
        source="self-hosted; API cost $0 (electricity/hardware not priced)",
    ),
}


@dataclass
class Measurements:
    meetings: int
    words: int
    tokens: int
    chunks: int
    ingest_seconds: float
    embedding_seconds: float
    extraction_seconds: float = 0.0
    query_input_tokens: int = 0
    query_output_tokens: int = 0
    query_seconds: float = 0.0
    questions: int = 0
    vector_bytes: int = 0
    sqlite_bytes: int = 0


def measured_costs(m: Measurements) -> dict:
    local = PRICING["local"]
    return {
        "embedding_api_cost_usd": local.embedding_cost(m.tokens),
        "extraction_api_cost_usd": 0.0,
        "query_reasoning_api_cost_usd": local.llm_cost(m.query_input_tokens, m.query_output_tokens),
        "total_api_cost_usd": 0.0,
        "note": "This run used local sentence-transformer embeddings and no paid LLM API calls.",
    }


def project_to_scale(m: Measurements, target_meetings: int = 1000, *, assumed_extraction_output_ratio: float = 0.08) -> dict:
    """Linear projection with explicit assumptions.

    Ingest time and tokens are scaled by the ratio of meetings.  Query volume is
    separated from ingestion.  Extraction cost is a *projection* of one LLM pass
    over every transcript, not a measured Gemini extraction run.
    """
    ratio = target_meetings / max(1, m.meetings)
    gemini = PRICING["gemini_2_0_flash"]
    embedding_api = PRICING["openai_text_embedding_3_small"]

    projected_tokens = int(m.tokens * ratio)
    projected_words = int(m.words * ratio)
    projected_chunks = int(m.chunks * ratio)
    projected_ingest_seconds = m.ingest_seconds * ratio
    projected_embedding_seconds = m.embedding_seconds * ratio

    extraction_input = projected_tokens
    extraction_output = int(projected_tokens * assumed_extraction_output_ratio)
    extraction_cost = gemini.llm_cost(extraction_input, extraction_output)

    # Query-time: assume the measured per-question token use if available,
    # otherwise a conservative fixed budget per question.
    per_q_in = (m.query_input_tokens / m.questions) if (m.questions and m.query_input_tokens) else 6000
    per_q_out = (m.query_output_tokens / m.questions) if (m.questions and m.query_output_tokens) else 400
    query_cost = gemini.llm_cost(int(per_q_in), int(per_q_out))

    return {
        "target_meetings": target_meetings,
        "scale_factor": round(ratio, 3),
        "projected_words": projected_words,
        "projected_tokens": projected_tokens,
        "projected_chunks": projected_chunks,
        "embedding": {
            "local_api_cost_usd": 0.0,
            "openai_embedding_3_small_api_cost_usd": embedding_api.embedding_cost(projected_tokens),
            "projected_seconds_cpu": round(projected_embedding_seconds, 1),
            "assumption": "linear scaling of measured CPU embedding throughput; no parallelism assumed",
        },
        "extraction": {
            "projected_input_tokens": extraction_input,
            "projected_output_tokens": extraction_output,
            "gemini_api_cost_usd": round(extraction_cost, 2),
            "assumption": (
                "one extraction pass over every chunk with Gemini 2.0 Flash; output assumed to be "
                f"{int(assumed_extraction_output_ratio * 100)}% of input tokens. This run used deterministic "
                "rule extraction, so this line is a projection, not a measured API bill."
            ),
        },
        "query_time": {
            "assumed_input_tokens_per_question": int(per_q_in),
            "assumed_output_tokens_per_question": int(per_q_out),
            "gemini_api_cost_per_question_usd": round(query_cost, 4),
            "gemini_api_cost_per_1000_questions_usd": round(query_cost * 1000, 2),
        },
        "ingestion": {
            "projected_seconds_single_process": round(projected_ingest_seconds, 1),
            "projected_hours_single_process": round(projected_ingest_seconds / 3600, 2),
            "assumption": "linear from measured local CPU run; no batching across processes, no cache",
        },
        "storage": {
            "measured_bytes_per_meeting_vector": round(m.vector_bytes / max(1, m.meetings)),
            "measured_bytes_per_meeting_sqlite": round(m.sqlite_bytes / max(1, m.meetings)),
            "projected_vector_gb": round(m.vector_bytes * ratio / 1e9, 2),
            "projected_sqlite_gb": round(m.sqlite_bytes * ratio / 1e9, 2),
        },
        "pricing": {
            "as_of": PRICING_AS_OF,
            "gemini_2_0_flash_input_usd_per_million": gemini.input_usd_per_million,
            "gemini_2_0_flash_output_usd_per_million": gemini.output_usd_per_million,
            "text_embedding_3_small_usd_per_million": embedding_api.embedding_usd_per_million,
            "sources": PRICING_SOURCES,
        },
        "caveats": [
            "This is a linear projection from 138 meetings, not a production benchmark.",
            "LLM extraction would be parallelised and retried in production; latency and cost change accordingly.",
            "Model prices change; the table is dated and configurable.",
            "Local embedding cost excludes hardware and electricity.",
            "At 10,000 meetings, sharded vector indexes, FTS partitioning, incremental re-embedding and queue-based ingestion are required.",
        ],
    }


def recommend_10k_architecture() -> list[str]:
    return [
        "Shard Chroma collections by team or by 2,000-meeting buckets; query only relevant shards.",
        "Partition SQLite FTS by corpus slice, or move FTS and metadata to Postgres with per-team indexes.",
        "Ingest through a queue with idempotent, hash-based jobs; embed in parallel worker processes.",
        "Cache query embeddings and frequent question analyses; reuse decision memory instead of re-extracting.",
        "Store only evidence windows in prompts (top-k chunks), never whole meetings.",
        "Add observability for retrieval recall, citation failure rate and extraction drift.",
        "Run nightly temporal reconciliation rather than recomputing all supersession chains per query.",
    ]
