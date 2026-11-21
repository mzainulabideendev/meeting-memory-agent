# Technical Write-Up — Agentic Meeting Memory

**Author:** implementation for the Catalist Agentic AI Intern build test
**Date:** 19 November 2026
**Corpus:** AMI design-meeting transcripts, 138 files, 35 teams, 676,745 body words

---

## 1. What was built

A memory system that turns raw meeting transcripts into queryable organisational
memory and answers cross-meeting questions with exact, machine-verified
citations. It is not a prompt wrapper: the corpus is never pasted into a single
prompt, and no assessment answer is hard-coded. The pipeline is:

1. **Ingest** — parse every transcript with character-exact offsets, chunk it
   contiguously, embed it, index it in SQLite FTS5 and ChromaDB.
2. **Extract memory** — decision candidates, proposals, rejections, actions,
   open questions and disagreements, each with a verbatim quote.
3. **Reason temporally** — build supersession chains across meetings.
4. **Retrieve** — per-team hybrid (vector + BM25) retrieval with RRF fusion and
   a bounded LangGraph research loop driven by a sufficiency gate.
5. **Answer** — cite-first generation with a deterministic validator; any quote
   that fails verification is discarded, never repaired by editing.
6. **Brief** — a four-section pre-meeting brief for TS3012, each bullet with
   its own verified quotation.
7. **Evaluate** — 11 hand-written, manually verified TS3012 questions, plus a
   cost/performance benchmark extrapolated from measured runs.

All artefacts are in `outputs/`: `answers.json`, `brief.md`,
`dataset_report.json`, `evaluation_report.md`, `evaluation_results.json`,
`benchmark.json`.

---

## 2. Data engineering and provenance

**Parsing.** Each file has a short header (`Meeting:`, `Team:`,
`Meeting number:`, `Speakers:`) followed by turns of the form
`HH:MM:SS Speaker X` plus text. The parser records `start_offset`/`end_offset`
into the original file text. Invariant enforced in tests:
`raw_text[start:end] == turn.text`. Duplicate turn text is not merged; turn ids
are content-addressed.

**Chunking.** Chunks are contiguous raw slices, not re-flowed text.
`raw_text[start:end] == chunk.original_text` is asserted for 200 chunks in the
integration suite. A separate `embed_text` adds speaker labels for the embedding
model; it is never cited. Target 520 tokens, max 700, overlap 80 (cl100k).

**Dataset facts (measured, `outputs/dataset_report.json`).** 138 transcripts,
35 teams, 676,745 body words, 852,106 estimated tokens (raw text), 1,594 chunks,
11.55 chunks/meeting. IS1002 is missing meeting `a`; IS1005 is missing meeting
`d`. No malformed files. Speaker roles are inferred only when a speaker
explicitly introduces themselves (“I'm Laura and I'm the project manager”); the
role record stores the justifying quote.

**Storage.** SQLite holds meetings, turns, chunks, FTS5 index, memory records,
action items, open questions, query log, citation-attempt audit and metrics.
Chroma holds 384-dimensional MiniLM embeddings keyed by chunk id.

---

## 3. Memory model

### 3.1 Episodic
Meetings, turns, chunks. Exact offsets make every quotation auditable against
the source file.

### 3.2 Semantic (`rule-v1.2`)
Deterministic candidate extraction over sentences, using topic lexicons and
cue lists:

* **Confirmed** requires explicit agreement language (“we agreed”, “we decided”,
  “let's go with”, …) or a short following agreement turn.
* **Proposed** covers “maybe we should”, “what about”, “I think we should”.
  A proposal is **not** a decision.
* **Rejected** covers “too expensive”, “we can't”, “drop it”, …
* References to a *past* agreement (“like we decided before”, “as agreed”) are
  explicitly excluded, because they are summaries, not new decisions.
* Actions require an ownership cue and are rejected if the sentence is a
  question. Owner is only set when the speaker says “I'll/I will/I'm going to”.
* Open questions, disagreements and deadlines are extracted separately.

Every record keeps `extraction_version`, `confidence`, and one verbatim
`Citation`.

### 3.3 Temporal
`TemporalMemory` groups records by topic. Only a **later confirmed** decision
can supersede an earlier confirmed one, and only when:

* the meetings differ (`later.meeting_sequence > earlier.meeting_sequence`),
* statement similarity is below 0.55, and
* the later statement is not a restatement (“we agreed that we…”, “as agreed”).

Proposals never supersede. Rejections close proposals but not confirmed
decisions. Records are never deleted: they get `status`, `valid_to_meeting`,
`superseded_by`, `supersedes`.

---

## 4. Retrieval and the agent graph

**Question analysis** is rule-based and reproducible: team/meeting ids,
speaker filters, roles, key terms, and intents (temporal change, causal,
ownership, numeric, disagreement, unresolved, decision). Team and meeting ids
are treated as filters, not content terms, so they do not distort coverage.

**Query planning** produces up to six queries: the original question, an
entity-focused query, and intent-specific cue queries (decision, change,
numeric, ownership, disagreement, unresolved), plus early/late bias.

**Hybrid search** runs vector search (MiniLM) and FTS5 BM25 for every planned
query, scoped to the question's team. Results are fused with reciprocal-rank
fusion (k=60); keyword hits get 1.15× weight because exact numbers and names
matter. Near-duplicate chunks within a meeting (Jaccard > 0.75) are suppressed.

**Evidence collection** keeps up to 24 chunks, reconstructs the underlying
speaker turns, and runs a sufficiency gate:

* at least two meetings,
* entity coverage (≥ 35% of key terms or ≥ 4 terms),
* all evidence belongs to the requested team,
* numbers present when the question asks for a number,
* decision/change language across meetings when the question is temporal,
* ownership language when the question asks who,
* disagreement language when the question asks about conflict.

If the gate fails, the collector deterministically adds keyword-matched chunks
from unrepresented meetings of the same team (bounded by the evidence cap), and
the graph can take up to two additional research loops.

**LangGraph** (`workflows/answer_workflow.py`) implements this as explicit
nodes: `analyze → plan → semantic → keyword → collect → assess →
(research → semantic …) → temporal → answer → citations → final`. The full trace
is returned with each answer and stored in the query log.

---

## 5. Citation verification — the trust boundary

`CitationValidator` is deterministic and has no fuzzy fallback:

1. the meeting id must exist in the ingested corpus;
2. the quotation must contain at least six words;
3. it must occur verbatim in the meeting's raw text, with only case,
   whitespace, Unicode form and typographic quote/dash variants normalised;
4. it must lie inside a **single speaker turn**.

If a quote fails, it is rejected and logged with the reason. The system never
edits a quotation to make it pass. On the LLM path, one repair attempt is made
with the specific failure reasons; if no citation survives, the unsupported
answer is replaced by an extractive composition or by an explicit
“insufficient evidence” response.

Every attempt is written to the `citation_attempts` table, so the rejection rate
is auditable (`python main.py status`).

---

## 6. Evaluation

**Design.** I wrote 11 questions about Team TS3012 after reading the four
meetings. Each has:

* a ground-truth answer,
* the relevant meeting ids,
* one exact supporting quotation per claim, and
* expected key facts (alternatives separated by `|`).

All 23 supporting quotations were re-checked against the raw transcripts before
the run (`contains_quote` returns true for all). The agent never sees this file
during answering.

**Metrics** (all deterministic; no LLM judge):

| Metric | Definition |
|---|---|
| Answer correctness | fraction of expected facts present; correct ≥ 0.80, partial ≥ 0.45 |
| Citation validity | share of returned quotes passing exact validation |
| Evidence precision / recall | cited meetings vs hand-marked relevant meetings |
| Meeting retrieval recall | relevant meetings present anywhere in the evidence bundle |
| Decision-state accuracy | share of expected state words (kept/dropped/…) |
| Unsupported claim rate | **proxy**: share of answer sentences with < 35% of content terms in citations |

**Results (actual run).**

* 11 questions: **3 correct, 4 partially correct, 4 incorrect**
* Mean fact coverage **59.1 %**
* Citation validity **100 %** (every returned quotation verified)
* Evidence precision **71.2 %**, evidence recall **63.6 %**
* Meeting retrieval recall **100 %** (all relevant meetings reached the bundle)
* Decision-state accuracy **9.1 %**
* Unsupported-claim proxy **0 %** (answers are extractive quotations)

The gap between 100 % retrieval recall and 9.1 % decision-state accuracy is the
central finding: retrieval finds the right meetings, but the deterministic
provider cannot convert them into the required analytical statement.

**Four real failures** (full text in `evaluation_report.md`):

1. **E02 (power source).** The answer quoted three sentences about
   *rechargeable* batteries but never “regular batteries”. The question does not
   contain the words “regular” or “normal”, so lexical selection has no bridge
   to the answer vocabulary.
2. **E03 (LCD timeline).** The answer quoted a rejection in b, a rejection in c
   and a statement in d, but did not say the final state kept the LCD and cut
   speech recognition. It captures evidence, not the conclusion.
3. **E05 (finances).** The answer selected sentences containing “final design”
   but no numbers, despite the question being numeric. The 17-euro and 12.50-euro
   statements were not in the retrieved timeline, so the numeric boost had
   nothing to boost.
4. **E07 (lost-remote feature).** The answer found the agreement to add a
   locator but never surfaced “beep and flash”, again because the question does
   not contain those words.

These are honest, reproducible failures, not artefact of the metric: the
ground-truth statements exist in the corpus and were verified by hand.

### Why this run is extractive

No `GEMINI_API_KEY` was present and outbound model download throughput measured
~26 KB/s (13 MB in 513 s), making the 3 GB local model impractical. The
heuristic provider was therefore used. The Gemini, OpenAI-compatible and local
providers are fully implemented (JSON mode, retries with exponential backoff,
timeout, token accounting, output validation) and covered by unit tests; they
were **not** exercised against a live endpoint. Any claim of LLM-level answer
quality would be unsupported.

---

## 7. Performance and cost

**Measured, 138 meetings, CPU-only laptop-class machine:**

| Stage | Time |
|---|---|
| Full ingestion (parse, chunk, embed, extract, persist) | 79.8 s |
| Embedding | 59.5 s |
| Rule-based semantic extraction | 12.1 s |
| Ingestion throughput | 0.58 s/meeting, 8,477 words/s |
| Query p50 / mean / p95 (11 questions) | 0.33 s / 1.27 s / 11.0 s |
| Storage | Chroma 50.2 MB, SQLite 37.7 MB (637 KB/meeting) |
| Paid API cost | $0.00 |

The p95 is dominated by the one-off model load; steady-state queries do not call
a generative model in this run.

### 1,000-meeting estimate (linear projection from the measurement above)

* ~4.9 M words, ~6.7 M tokens, ~11,550 chunks
* ingestion ~579 s single-process (~0.16 h); embedding ~431 s CPU
* storage ~0.36 GB vector + ~0.27 GB SQLite
* projected Gemini 2.0 Flash extraction (input 6.74 M, output assumed 8 %):
  **~$0.89**
* projected query cost at 6,000 input / 400 output tokens per question:
  **~$0.0008/question, ~$0.76 per 1,000 questions**
* OpenAI `text-embedding-3-small` alternative: **~$0.13** for the corpus

### 10,000-meeting production plan

Linear numbers: ~115k chunks, ~1.6 h single-process ingestion, ~$8.89
extraction, ~6.4 GB storage. That is small in absolute terms, so the real
problems are operational, not cost:

1. **Shard** Chroma by team or by 2,000-meeting bucket; query only relevant
   shards, and keep a global “current decision” projection.
2. **Partition** FTS/metadata: SQLite is fine to a few million rows, but move to
   Postgres with per-team indexes when concurrency grows.
3. **Queue-based ingestion** with idempotent content-hash jobs and parallel
   embedding workers; re-embed only changed chunks.
4. **Cache** query embeddings, question analyses and decision-memory
   projections; reuse them across questions.
5. **Nightly temporal reconciliation** instead of recomputing supersession
   chains at query time.
6. **Monitor** retrieval recall, citation-failure rate, extraction drift and
   provider error rates; alert on citation-failure spikes.
7. **Prompt windows only** — top-k evidence, never whole meetings.

Prices are dated 2026-11-19 and linked in `utils/cost_calculator.py`
(Gemini API pricing page; OpenAI embeddings pricing page). They are projections,
not bills; model prices change.

---

## 8. Limitations and what I would do next

**Limitations**

* The evaluation set is one team and 11 questions. It is not a corpus-wide
  benchmark.
* No live LLM was available. Extractive answers are faithful but do not
  synthesise; multi-part “what/why/who” questions suffer most.
* Rule extraction has false positives (anchored agreement language) and false
  negatives (implicit agreement). Decision memory is a candidate index, not
  ground truth.
* Lexical metrics under-credit correct paraphrases and the unsupported-claim
  rate is a term-overlap proxy, not entailment.
* Within-meeting supersession is deliberately disabled. An earlier and later
  statement in the same meeting is reported as a proposal/change pair rather
  than a supersession, and a revision that is filed under a different topic
  (for example, a cost plan that scraps a previously agreed feature) appears as
  two separate records. The brief shows both and does not delete the earlier
  one; a reader can see the revision, but the record is not re-labelled
  automatically.
* Speaker roles are only known when self-introduced; otherwise the system says
  “Speaker X” and marks role unknown.

**Next steps, in priority order**

1. Run the same evaluation with Gemini 2.0 Flash and publish the delta. The
   prompt already enforces citation-first answering and a repair pass.
2. Add a semantic entailment check (NLI) for the unsupported-claim metric, and
   human-grade a 50-question corpus-wide set.
3. Replace rule extraction with an LLM extractor plus the deterministic
   validator as a guardrail; keep rule output as the fallback and diff the two.
4. Add a cross-encoder reranker over the top 50 fused chunks (the hook exists;
   it is disabled because it adds ~100 ms and a 90 MB model).
5. Add a Streamlit UI over the FastAPI-less CLI: question box, answer, per-bullet
   citations, and a timeline view of superseded decisions.

---

## 9. Reproduction

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .
python main.py ingest --reset
python main.py brief --team TS3012
python main.py answer
python main.py evaluate
python -m pytest -q
```

Measured outputs are checked in under `outputs/`. Integration tests skip
automatically when the corpus has not been ingested; they never write to it.
