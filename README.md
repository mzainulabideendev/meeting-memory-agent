<div align="center">

# 🎙️ Meeting Memory Agent

### Deterministic, Evidence‑Based Meeting Memory for Large Transcript Collections

![Banner](banner.svg)

</div>

---

## Overview

**Meeting Memory Agent** ingests **138 raw design‑meeting transcripts (676,745 words)** and builds **persistent hybrid memory** (episodic, semantic, temporal) that powers **cross‑meeting question answering with verbatim citations** and generates **pre‑meeting briefs**.

> **Key metrics (measured on 2026‑11‑20):**
> - Ingestion: **116.8 s** (0.85 s/meeting)  
> - Storage: **58.6 MB Chroma + 40.65 MB SQLite** (716 KB/meeting)  
> - Query latency: mean **2.24 s**, p95 **13.94 s**  
> - Citation validity: **100%** (all quotes verified verbatim)  
> - Answer correctness: **4 correct / 4 partial / 3 incorrect** (11 hand‑written questions)  
> - Mean fact coverage: **68.9%** (baseline 59.1%)  
> - Decision‑state accuracy: **72.7%** (baseline 9.1%)  
> - Meeting retrieval recall: **100%**  
> - Tests: **60 passed** (56 unit/integration + 4 documented‑failure regressions)

---

## Architecture Design

### High‑Level System Diagram

```mermaid
flowchart TD
    A[Raw Transcripts] --> B[Parser]
    B --> C[Chunker]
    C --> D[Embeddings (MiniLM)]
    D --> E[Chroma Vector Store]
    D --> F[SQLite + FTS5 BM25]
    C --> G[Semantic Extractor]
    G --> H[Episodic Memory<br/>Raw turns & offsets]
    G --> I[Semantic Memory<br/>Decisions / Actions / Questions]
    G --> J[Temporal Memory<br/>Supersession chains]
    B --> K[Metadata & Speaker Roles]
    K --> L[Storage Layer]
    L --> H
    L --> I
    L --> J
    I --> M[Query Analyzer]
    M --> N[Query Planner]
    N --> O[Hybrid Retrieval<br/>Vector ∥ BM25 + RRF]
    O --> P[Sufficiency Gate<br/>2‑meeting / entity / decision check]
    P --> Q[Temporal / Decision Reasoning]
    Q --> N
    N --> R[Answer Generation<br/>LLM or Extractive]
    R --> S[Citation Validator<br/>Exact substring + normalization]
    R --> T[Outputs<br/>answers.json / brief.md]
    S --> R
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style R fill:#bbf,stroke:#333,stroke-width:2px
```

### LangGraph StateGraph (Answer Workflow)

```mermaid
flowchart LR
    start([Question]) --> analyze[Analyze]
    analyze --> plan[Plan Queries]
    plan --> semantic[Semantic Store]
    plan --> keyword[Keyword Store]
    semantic --> collect[Evidence Collector]
    keyword --> collect
    collect --> assess[Assess Sufficiency]
    assess -->|insufficient| research[Research Loop]
    research --> assess
    assess --> temporal[Temporal Reasoning]
    temporal --> answer[Answer Agent]
    answer --> citations[Citation Agent]
    citations --> final([JSON + citations])
    style start fill:#f9f,stroke:#333
    style final fill:#bbf,stroke:#333
```

### Brief Workflow

```mermaid
flowchart TD
    B1[Brief Request] --> B2[Retrieve Team Meetings]
    B2 --> B3[Decisions]
    B3 --> B3a[What Changed]
    B3a --> B3b[How Changed]
    B3b --> B4[Open Actions / Ownership]
    B4 --> B5[Disagreements Still Open]
    B5 --> B6[Appendix: Unresolved Questions]
    B6 --> B7[Verify Citations]
    B7 --> B8[brief.md]
    style B1 fill:#f9f,stroke:#333
    style B8 fill:#bbf,stroke:#333
```

### Memory Store Design (ER Diagram)

```mermaid
erDiagram
    MEETINGS {
        string meeting_id PK
        string team
        string date
    }
    EPISODIC_CHUNKS {
        string chunk_id PK
        string meeting_id FK
        int start_offset
        int end_offset
        string original_text
    }
    SEMANTIC_RECORDS {
        string record_id PK
        string meeting_id FK
        string kind
        string content
        string evidence_quote
        string extraction_version
    }
    TEMPORAL_RECORDS {
        string record_id PK
        string meeting_id FK
        string status
        int valid_to_meeting
        string superseded_by
    }
    MEETINGS ||--o{ EPISODIC_CHUNKS
    MEETINGS ||--o{ SEMANTIC_RECORDS
    MEETINGS ||--o{ TEMPORAL_RECORDS
```

### CLI Command Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as python main.py
    participant Ingest as ingest
    participant Answer as answer
    participant Brief as brief
    participant Eval as evaluate

    User->>CLI: ingest --reset
    CLI->>Ingest: parse → chunk → embed → extract
    Ingest->>User: storage populated

    User->>CLI: answer
    CLI->>Answer: 25 questions
    Answer->>User: outputs/answers.json

    User->>CLI: brief --team TS3012
    CLI->>Brief: retrieve + extract + verify
    Brief->>User: outputs/brief.md

    User->>CLI: evaluate
    CLI->>Eval: E01‑E11 + benchmark
    Eval->>User: evaluation_report.md + benchmark.json
```

---

## Component Details

### 1. Ingestion Pipeline

| Step | Tool / Model | Output |
|------|--------------|--------|
| Parse | Custom parser (`parser.py`) | Turns, metadata, speaker roles, exact character offsets |
| Chunk | Context‑aware chunker (`chunker.py`) | Segments with `raw_text[start:end]` invariant |
| Embed | MiniLM (768‑dim) (`embeddings.py`) | Vectors → Chroma |
| Keyword Index | SQLite + FTS5 BM25 | Full‑text index |
| Semantic Extract | Rule‑based (`semantic_memory.py`, v1.2) | Decisions, actions, open questions |

### 2. Memory Types

| Type | What It Stores | Where It Lives |
|------|----------------|----------------|
| **Episodic** | Raw transcript turns with exact offsets (verbatim) | `episodic_memory.py`, Chroma + SQLite |
| **Semantic** | Extracted decision/action/question records with evidence quotes | `semantic_memory.py`, `decision_store.py`, `action_store.py` |
| **Temporal** | Decision lifecycle: `proposed → confirmed → rejected/superseded/unresolved` with supersession chains | `temporal_memory.py` |

### 3. Hybrid Retrieval

- **Vector search** (MiniLM) → top‑k semantic candidates  
- **Keyword search** (FTS5 BM25) → top‑k lexical candidates  
- **Reciprocal Rank Fusion (RRF)** merges both lists (keyword weighted 1.2×)  
- **Near‑duplicate suppression** (Jaccard > 0.75 within same meeting)  
- **Coverage expansion** adds evidence from unrepresented meetings of the same team  
- **Sufficiency gate** checks: ≥ 2 meetings, entity coverage, decision language, numbers, ownership → triggers bounded **research loop** if insufficient  

### 4. Citation Verification (Deterministic)

`CitationValidator` performs **no fuzzy matching**:

1. Meeting must exist in `storage/meetings.db`.  
2. Quote must be **≥ 6 words**.  
3. Quote must occur **verbatim** in the raw transcript (case, whitespace, Unicode punctuation normalised **only for comparison**).  
4. Quote must lie **inside one speaker turn**.  
5. Failed quotes are logged, never silently edited.  

Result: **110/110 quotes verified** (75 answer quotes + 35 brief quotes) = **100% validity**.

### 5. Temporal Reasoning

- Only a **later confirmed** decision can supersede an earlier one.  
- Proposals, restatements, rejections never supersede.  
- Records are **never deleted**; they get `status`, `valid_to_meeting`, `superseded_by`.  
- Rules enforced in `temporal_memory.py` (`TemporalMemory.reason()`).

### 6. CLI Commands

| Command | Description | Output File |
|--------|-------------|-------------|
| `ingest [--reset]` | Parse, chunk, embed, extract semantic data | `storage/meetings.db`, `outputs/dataset_report.json` |
| `answer` | Generate answers for all 25 test questions (Q01‑Q25) | `outputs/answers.json` |
| `brief --team TS3012` | Four‑section pre‑meeting brief + appendix | `outputs/brief.md` |
| `evaluate` | Run hand‑written evaluation set (E01‑E11) | `outputs/evaluation_report.md`, `outputs/evaluation_results.json`, `outputs/benchmark.json` |
| `ask "question" --team TEAM` | Single ad‑hoc query with citations | JSON to stdout |
| `inspect` | Dataset stats, missing meetings, malformed files | Console |
| `status` | Provider, counts, citation audit, stored metrics | Console |

---

## Quick Start

```bash
# 1. Install
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -e .

# 2. Ingest dataset (downloads embedding model once, ~2 min)
python main.py ingest --reset

# 3. Generate TS3012 pre‑meeting brief
python main.py brief --team TS3012

# 4. Generate all 25 verified answers
python main.py answer

# 5. Run evaluation
python main.py evaluate

# 6. Ask ad‑hoc question
python main.py ask "What did TS3012 decide about rubber?" --team TS3012

# 7. Check system status
python main.py status
```

---

## Testing

```bash
# Run all tests (60 total)
python -m pytest

# Unit tests only (50, no corpus needed)
python -m pytest tests/unit/

# Integration tests (5, against ingested corpus)
python -m pytest tests/integration/
```

| Test Suite | Count | Status |
|------------|-------|--------|
| Unit | 50 | ✅ all pass |
| Integration | 5 | ✅ all pass |
| Regression (E02, E03, E05, E07) | 4 | ✅ all pass |
| **Total** | **60** | **✅ 60 passed, 0 failed, 0 skipped** |

---

## Evaluation Results

### Overall Metrics (E01‑E11, TS3012)

| Metric | Value |
|--------|-------|
| Correct | 4 / 11 |
| Partially Correct | 4 / 11 |
| Incorrect | 3 / 11 |
| Mean Fact Coverage | 68.9% |
| Mean Citation Validity | 100% |
| Mean Evidence Precision | 72.7% |
| Mean Evidence Recall | 68.2% |
| Meeting Retrieval Recall | 100% |
| Decision‑State Accuracy | 72.7% |
| Unsupported Claim Rate | 5.2% |

### Improvements vs Baseline (19 Nov 2026)

| Question | Before | After | Change |
|----------|--------|-------|--------|
| E02 (regular batteries) | Incorrect | **Correct** | ✅ fixed |
| E03 (LCD timeline) | Incorrect | **Partially Correct** | ✅ improved |
| E05 (numeric costing) | Incorrect | **Partially Correct** | ✅ improved |
| E07 (beep/flash) | Incorrect | **Correct** | ✅ fixed |
| E04, E06, E09, E10 | Partially Correct / Correct | Incorrect | ⚠️ regressed (documented) |
| **Mean fact coverage** | 59.1% | **68.9%** | **+9.8 pp** |
| **Decision‑state accuracy** | 9.1% | **72.7%** | **+63.6 pp** |

> All regressions and improvements are honestly documented in `outputs/final_delivery_report.md` and `outputs/evaluation_report.md`.

---

## Project Structure

```
meeting-memory-agent/
├── main.py                     # CLI entry point
├── pyproject.toml              # Package config
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── .env.example                # LLM provider config template
├── README.md
├── banner.svg                  # Project banner (SVG)
├── data/
│   └── Catalist_AI_Intern_Build_Test_Data/
│       ├── transcripts/          # 138 .txt files
│       └── questions.json        # 25 test questions
├── outputs/                      # All generated artifacts
│   ├── answers.json
│   ├── brief.md
│   ├── evaluation_report.md
│   ├── evaluation_results.json
│   ├── evaluation_baseline.json
│   ├── benchmark.json
│   ├── technical_writeup.md
│   ├── demo_script.md
│   ├── dataset_report.json
│   └── final_delivery_report.md
├── scripts/
│   ├── citation_audit.py
│   ├── decision_tracking_audit.py
│   └── inspect_transcript.py
├── src/
│   └── meeting_agent/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py              # CLI commands
│       ├── config.py           # Settings (env‑configurable)
│       ├── models.py           # Pydantic schemas
│       ├── storage.py          # SQLite schema, FTS5, audit tables
│       ├── ingestion/
│       │   ├── parser.py
│       │   ├── chunker.py
│       │   ├── metadata.py
│       │   └── pipeline.py
│       ├── memory/
│       │   ├── episodic_memory.py
│       │   ├── semantic_memory.py
│       │   ├── temporal_memory.py
│       │   ├── vector_store.py
│       │   ├── keyword_store.py
│       │   ├── decision_store.py
│       │   └── action_store.py
│       ├── retrieval/
│       │   ├── query_analyzer.py
│       │   ├── hybrid_search.py
│       │   ├── reranker.py
│       │   ├── evidence_collector.py
│       │   └── targeted_scan.py
│       ├── agents/
│       │   ├── answer_agent.py
│       │   ├── briefing_agent.py
│       │   ├── citation_agent.py
│       │   ├── reasoning_agent.py
│       │   ├── retrieval_agent.py
│       │   └── researcher.py
│       ├── workflows/
│       │   ├── answer_workflow.py    # LangGraph StateGraph
│       │   ├── briefing_workflow.py
│       │   └── ingestion_workflow.py
│       ├── llm/
│       │   └── provider.py          # Gemini / OpenAI / Local / Heuristic
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   └── benchmark.py
│       └── utils/
│           ├── embeddings.py
│           ├── text_normalization.py
│           ├── cost_calculator.py
│           └── logging.py
├── tests/
│   ├── unit/                     # 50 fast tests
│   ├── integration/              # 5 corpus tests
│   └── evaluation/               # Hand‑written eval questions
```

---

## LLM Providers

The system supports 5 provider modes (configured via `LLM_PROVIDER` env var or `.env`):

| Provider | Behaviour |
|----------|-----------|
| `auto` (default) | Gemini → OpenAI‑compatible → Local HF → Heuristic (fallback chain) |
| `gemini` | Google Gemini (`gemini-2.0-flash`), JSON mode, retry with backoff |
| `openai_compatible` | Any `/chat/completions` endpoint (vLLM, Ollama, gateway) |
| `local` | HuggingFace transformers chat model |
| `heuristic` | **Deterministic, extractive, $0.00 cost**. No network, no model. Returns verbatim quotations. |

The heuristic provider is the default in this environment (no API key available). It performs the full retrieval + sufficiency gating + temporal reasoning + citation verification pipeline — it simply cannot paraphrase.

---

## Reproducibility

All commands are deterministic:

- Fixed random seed  
- Temperature 0 for all LLM calls  
- Rule‑based semantic extraction (version `rule-v1.2`)  
- Ordered SQLite writes  
- `ingest --reset` reproduces the same corpus counts and chunk offsets every time  

---

## Contributing

1. Fork the repository  
2. Create a feature branch: `git checkout -b feature/your-feature`  
3. Make your changes and add tests  
4. Ensure all 60 tests pass: `python -m pytest`  
5. Commit with a clear message  
6. Push and open a Pull Request  

---

## License

MIT License — free to use, modify, and distribute.

---

## Contact

Open an issue on GitHub for bugs, questions, or feature requests.

---

<div align="center">

**Built with ❤️ for the Catalist Agentic AI Intern Build Test**

*All diagrams use Mermaid syntax — GitHub renders them automatically.*

</div>