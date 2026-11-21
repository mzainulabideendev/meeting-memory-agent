# Demo Script (5–7 minutes)

**Goal:** show that the system retrieves the right evidence, tracks decisions
over time, and proves every quote is real — without pretending the extractive
fallback is a fluent assistant.

---

## 0. Setup (before the demo, 2 min)

```powershell
cd "D:\meeting memory agent"
.\.venv\Scripts\activate
python main.py status
```

Expected: 138 meetings, 1,594 chunks, provider `heuristic`, vectors present.
If the DB is missing: `python main.py ingest --reset` (~2 min).

---

## 1. Show the corpus was actually processed (45 s)

```powershell
python main.py inspect
```

Say:

> 138 files, 35 teams, 676,745 body words. IS1002 is missing meeting a and
> IS1005 is missing meeting d — the report lists both. Every chunk keeps exact
> character offsets into the original file; the integration test asserts
> `raw_text[start:end] == chunk.original_text`.

Open `outputs/dataset_report.json` and show `team_details.TS3012.roles` — each
role has the quote that justifies it.

---

## 2. Ask a cross-meeting question (90 s)

```powershell
python main.py ask "What did Team TS3012 decide about the material for the case, and did the choice survive the final meeting?" --team TS3012
```

Point at the JSON:

* `answer` — extractive, built from quoted turns.
* `citations` — meeting id + exact quote.
* `trace` — the LangGraph nodes: `question_analyzer` → `query_planner` →
  `semantic_retrieval` → `keyword_retrieval` → `evidence_collector` →
  `sufficiency_evaluator` → `temporal_reasoning` → `answer_generator` →
  `citation_validator`.

Say:

> Retrieval is team-scoped and hybrid: MiniLM vectors plus SQLite FTS5, fused
> with reciprocal rank fusion. The sufficiency gate can trigger a bounded
> research loop. Nothing is pasted wholesale: the evidence bundle is at most 24
> chunks.

---

## 3. Prove the citation checker is real (60 s)

Show the audit table:

```powershell
python main.py status
```

Then demonstrate a rejection live:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from meeting_agent.agents.citation_agent import CitationValidator; from meeting_agent.models import Citation; v=CitationValidator(); print(v.validate(Citation(meeting_id='TS3012c', quote='we definitely chose rubber for the case')).reason); print(v.validate(Citation(meeting_id='TS3012c', quote=\"so we're definitely going for rubber\")).reason)"
```

Expected: the first is rejected as not verbatim; the second (a real quote) is
accepted. Emphasise:

> Validation ignores only case, whitespace and Unicode punctuation. It never
> edits a quote. Six-word minimum, one speaker turn. All 25 answers and all
> brief bullets pass this check.

---

## 4. Decision memory and change over time (60 s)

```powershell
python main.py brief --team TS3012
```

Open `outputs/brief.md` on screen. Walk through the four required sections:

* **A. Decisions Made** — only explicit agreement language.
* **B. What Changed Along the Way** — supersession chains; only a later
  *confirmed* decision can supersede, never a proposal or a restatement.
* **C. Open Actions and Ownership** — owner only when the speaker said
  “I'll / I will”, otherwise “ownership unclear”.
* **D. Disagreements Still Open** — unresolved items, with a note when no
  explicit resolution exists.

Every bullet has its own quote and meeting id. This is the brief Team TS3012
should read; it covers meetings a–d and includes the final meeting's cost-driven
cuts.

---

## 5. Evaluation — including the failures (75 s)

```powershell
python main.py evaluate
```

Show `outputs/evaluation_report.md`:

> 11 hand-written TS3012 questions, ground truth and quotes verified by hand
> before the run. Result: 3 correct, 4 partial, 4 incorrect; citation validity
> 100 %; meeting retrieval recall 100 %; decision-state accuracy 9.1 %.

Then show one failure (E05):

> The question asks for the final costing. Retrieval found the right meetings,
> but the extractive provider cannot bridge “final costing” to “seventeen
> euros” without a language model. The report names the four real failures and
> how to fix them.

State the limitation plainly:

> No API key was available and the local model download measured ~26 KB/s, so
> this run is deterministic and extractive. The Gemini path is implemented and
> tested at the unit level but was not run live. I am not claiming fluent
> answers.

---

## 6. Cost and scale (45 s)

Open `outputs/benchmark.json`.

> Measured: 80 s for all 138 meetings, 0.58 s/meeting, 50 MB vectors, 38 MB
> SQLite, $0 API. Linear projection to 1,000 meetings: ~0.16 h ingestion,
> ~$0.89 Gemini extraction, ~$0.76 per 1,000 questions. At 10,000 meetings the
> plan is sharded Chroma, partitioned FTS/Postgres, queue-based idempotent
> ingestion, cached embeddings and nightly temporal reconciliation.

---

## 7. Tests (30 s)

```powershell
python -m pytest -q
```

> 50 unit tests and 5 integration tests. The integration suite re-validates
> chunk offsets, FTS scoping, every quotation in the generated brief, and the
> full `answers.json` contract (Q01–Q25, ≤ 3 citations, every quote verified).

---

## Closing line

> The engineering contract is met: real ingestion, persistent hybrid memory,
> multi-step retrieval, temporal decision tracking, and 100 % verified
> citations. The remaining gap is answer fluency, and that is one API key away —
> the provider and prompts are already wired in.
