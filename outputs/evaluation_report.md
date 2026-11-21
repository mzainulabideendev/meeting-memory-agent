# Independent Evaluation Report

## 1. Method

The evaluation set contains hand-written, cross-meeting questions about Team TS3012. Ground truth, meeting ids and supporting quotations were read from the original transcripts and stored in `tests/evaluation/test_questions.json` before the agent was evaluated. The agent does not see the ground truth during answering.

Metrics are deterministic and lexical; there is no LLM judge in this report.

| Metric | Definition |
|---|---|
| Answer correctness | Fraction of expected key facts present in the answer. `correct` >= 0.80; `partially_correct` >= 0.45; otherwise `incorrect`. |
| Evidence precision | Share of cited meetings that are in the hand-marked relevant set. |
| Evidence recall | Share of relevant meetings that are cited. |
| Citation validity | Share of returned citations that pass exact-substring, single-turn validation. |
| Meeting retrieval recall | Share of relevant meetings present anywhere in retrieved evidence. |
| Decision-state accuracy | Share of expected decision-state keywords (e.g. kept, dropped) present in the answer. |
| Unsupported claim rate (proxy) | Share of answer sentences with < 35% of content terms present in the citations. This is a conservative proxy, not semantic entailment. |

## 2. Summary

- Total questions evaluated: **11**
- Correct: **4**
- Partially correct: **4**
- Incorrect: **3**
- Mean fact coverage: **68.9%**
- Citation validity: **100.0%**
- Evidence precision: **72.7%**
- Evidence recall: **68.2%**
- Meeting retrieval recall: **100.0%**
- Decision-state accuracy: **72.7%**
- Unsupported claim rate (proxy): **5.2%**

## 3. Per-question results

| ID | Correctness | Facts | Citation validity | Evidence P/R | Retrieval recall | Unsupported |
|---|---|---|---|---|---|---|
| E01 | partially_correct | 50% | 100% | 67%/50% | 100% | 0% |
| E02 | correct | 100% | 100% | 100%/100% | 100% | 0% |
| E03 | partially_correct | 67% | 100% | 100%/100% | 100% | 17% |
| E04 | incorrect | 33% | 100% | 67%/50% | 100% | 0% |
| E05 | partially_correct | 75% | 100% | 100%/100% | 100% | 0% |
| E06 | incorrect | 33% | 100% | 67%/50% | 100% | 0% |
| E07 | correct | 100% | 100% | 33%/50% | 100% | 20% |
| E08 | correct | 100% | 100% | 33%/50% | 100% | 20% |
| E09 | incorrect | 33% | 100% | 100%/50% | 100% | 0% |
| E10 | partially_correct | 67% | 100% | 67%/50% | 100% | 0% |
| E11 | correct | 100% | 100% | 67%/100% | 100% | 0% |

## 4. Actual failures and analysis

### E01 — partially_correct

**Question:** Which material did Team TS3012 confirm for the remote's case, and did that choice survive the final budget cuts?

**Agent answer:** The retrieved transcripts show in TS3012b, Speaker A described as uncertain that: "Maybe yeah, or maybe you can find out i what people are willing to pay. We are going for twenty five Euros sales price, but just try to find out what they're willing to pay for it, because if they're willing to pay more, we could lose a little profit and maybe attract more customers, uh we just have to see what it looks like."; in the final meeting TS3012d, Speaker A said that: "You have the total of seventeen Euros in production cost, which is higher than the twelve and a half that we are permitted to use. So, easy. What do we scrap."; in the final meeting TS3012d, Speaker A rejected that: "So and so it's still stuck with thirteen, so I had to take out the special colour I suppose. And, yeah, I didn't see anything else I could take out. Yeah, I could take out the push-buttons, but we need those.".

**Expected answer (manual ground truth):** The team confirmed rubber in TS3012c ('so we're definitely going for rubber') and the final design in TS3012d still used rubber for the cover and buttons.

**Missing expected facts:** rubber

**Retrieved meetings:** TS3012a, TS3012b, TS3012c, TS3012d; **cited:** TS3012b, TS3012d, TS3012d; **relevant:** TS3012c, TS3012d

**Why it failed:** fact coverage 50%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (rubber) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

### E03 — partially_correct

**Question:** How did TS3012's position on the LCD screen change between meetings b and d?

**Agent answer:** The retrieved transcripts show in TS3012b, Speaker A proposed that: "Oh yeah. I think we should lose the L_C_D_ screen, like you said. I think for example it's it's huge I think the L_C_D_ is huge, it consumes batteries like hell."; in TS3012c, Speaker A rejected that: "Don't think it's an issue, okay. Oh let's go for let's say because it is uh very hot in n our target group I think, we should take it take it t to consideration to build an L_C_D_ screen in it."; in TS3012d, Speaker A said that: "Because the L_C_D_ has more support on customer side. There are ninety one percent of uh the people, or something like that. But ninety percent who favour an L_C_D_ display, and only sixty percent that favour speech recognition.".

**Expected answer (manual ground truth):** In TS3012b Speaker A wanted to lose the LCD because it was huge and consumed batteries; by TS3012c they considered including it, and in TS3012d it was kept while speech recognition was cut because about ninety percent of customers favoured an LCD.

**Missing expected facts:** include|included|keep|kept|retain|retained

**Retrieved meetings:** TS3012b, TS3012c, TS3012d; **cited:** TS3012b, TS3012c, TS3012d; **relevant:** TS3012b, TS3012c, TS3012d

**Why it failed:** fact coverage 67%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (include, included, keep, kept, retain, retained) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

### E04 — incorrect

**Question:** When and why was speech recognition dropped from TS3012's final design?

**Agent answer:** The retrieved transcripts show in TS3012b, Speaker D said that: "Um on my report, I didn't uh ish I didn't show it in my uh presentation, because my computer crashed. Um they want to pay for an L_C_D_ screen and speech recognition."; in the final meeting TS3012d, Speaker A rejected that: "So, easy. What do we scrap. Well think I had the best solution that I came up with is just to s take out the speech recognition."; in the final meeting TS3012d, Speaker A said that: "Because the L_C_D_ has more support on customer side. There are ninety one percent of uh the people, or something like that. But ninety percent who favour an L_C_D_ display, and only sixty percent that favour speech recognition.".

**Expected answer (manual ground truth):** It was dropped in the final meeting TS3012d because the full design cost seventeen euros against a permitted twelve and a half, and the LCD had stronger customer support (about ninety percent versus sixty percent).

**Missing expected facts:** drop|dropped|removed|scrapped|cut, budget|cost|12.50|twelve and a half|17|seventeen

**Retrieved meetings:** TS3012b, TS3012c, TS3012d; **cited:** TS3012b, TS3012d, TS3012d; **relevant:** TS3012c, TS3012d

**Why it failed:** fact coverage 33%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (12, 17, 50, budget, cost, cut) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

### E05 — partially_correct

**Question:** What did TS3012 know about finances in meeting c, and what did the final costing in meeting d show?

**Agent answer:** The retrieved transcripts show in TS3012c, Speaker A left unresolved that: "And the consumer, yeah, we aim for young consumers, so that's fine. And well actually we yeah, we did need some more data on finances, but w yeah we still don't have that apparently. That's too bad."; in the final meeting TS3012d, Speaker A said that: "You have the total of seventeen Euros in production cost, which is higher than the twelve and a half that we are permitted to use. So, easy. What do we scrap."; in the final meeting TS3012d, Speaker A rejected that: "So and so it's still stuck with thirteen, so I had to take out the special colour I suppose. And, yeah, I didn't see anything else I could take out. Yeah, I could take out the push-buttons, but we need those.".

**Expected answer (manual ground truth):** In TS3012c they still lacked financial data; in TS3012d the costing showed seventeen euros against a twelve and a half euro budget, forcing them to drop speech recognition and the special colour.

**Missing expected facts:** speech recognition

**Retrieved meetings:** TS3012a, TS3012b, TS3012c, TS3012d; **cited:** TS3012c, TS3012d, TS3012d; **relevant:** TS3012c, TS3012d

**Why it failed:** fact coverage 75%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (recognition, speech) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

### E06 — incorrect

**Question:** What happened to TS3012's yellow colour decision when the final budget was applied?

**Agent answer:** The retrieved transcripts show in TS3012b, Speaker A described as uncertain that: "Maybe yeah, or maybe you can find out i what people are willing to pay. We are going for twenty five Euros sales price, but just try to find out what they're willing to pay for it, because if they're willing to pay more, we could lose a little profit and maybe attract more customers, uh we just have to see what it looks like."; in the final meeting TS3012d, Speaker A confirmed that: "So, generally what I came up with, in order to be cou to to have production cost of twelve and a half Euros, spe scrap speech recognition and the separate covers can account for the if people want it, we'll just then we'll do it in black. We'll just deliver it in black, have the it has all the function that it's supposed to have, and if you want it if you want the custom design, then you can buy the separate covers. You make it d orange or whatever you want."; in the final meeting TS3012d, Speaker A rejected that: "And, yeah, I didn't see anything else I could take out. Yeah, I could take out the push-buttons, but we need those. So, generally what I came up with, in order to be cou to to have production cost of twelve and a half Euros, spe scrap speech recognition and the separate covers can account for the if people want it, we'll just then we'll do it in black.".

**Expected answer (manual ground truth):** Yellow had been agreed and Speaker C wanted a yellow remote, but in TS3012d the special yellow colour was removed for cost reasons; the base unit would be black with custom covers offered separately, while rubber was kept over yellow.

**Missing expected facts:** yellow, special colour|special color|dropped|removed

**Retrieved meetings:** TS3012a, TS3012b, TS3012c, TS3012d; **cited:** TS3012b, TS3012d, TS3012d; **relevant:** TS3012c, TS3012d

**Why it failed:** fact coverage 33%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (color, colour, dropped, removed, special, yellow) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

### E09 — incorrect

**Question:** When TS3012 had to cut costs in the final meeting, what disagreement arose about rubber versus yellow, and how was it resolved?

**Agent answer:** The retrieved transcripts show in the final meeting TS3012d, Speaker A confirmed that: "So, generally what I came up with, in order to be cou to to have production cost of twelve and a half Euros, spe scrap speech recognition and the separate covers can account for the if people want it, we'll just then we'll do it in black. We'll just deliver it in black, have the it has all the function that it's supposed to have, and if you want it if you want the custom design, then you can buy the separate covers. You make it d orange or whatever you want."; in the final meeting TS3012d, Speaker A rejected that: "So and so it's still stuck with thirteen, so I had to take out the special colour I suppose. And, yeah, I didn't see anything else I could take out. Yeah, I could take out the push-buttons, but we need those."; in the final meeting TS3012d, Speaker A said that: "You have the total of seventeen Euros in production cost, which is higher than the twelve and a half that we are permitted to use. So, easy. What do we scrap.".

**Expected answer (manual ground truth):** Speaker C suggested losing rubber and taking plastic, but Speaker A favoured keeping rubber because it had more advantages than the yellow colour; the outcome was to keep rubber and drop the special yellow colour.

**Missing expected facts:** rubber, keep|kept|drop|dropped|plastic

**Retrieved meetings:** TS3012a, TS3012b, TS3012c, TS3012d; **cited:** TS3012d, TS3012d, TS3012d; **relevant:** TS3012c, TS3012d

**Why it failed:** fact coverage 33%, retrieval recall 100%, citation validity 100%. The answer omitted or mis-stated the missing facts above.

**How to improve:** Vocabulary bridging: the missing facts (drop, dropped, keep, kept, plastic, rubber) use words that do not occur in the question, so lexical sentence selection cannot reach them. Add query expansion / synonym generation (or an LLM) that maps the question's terms to the transcript's wording before ranking evidence.

## 5. Retrieval quality and decision tracking

- Mean meeting retrieval recall: 100.0%. Recall below 100% means at least one hand-marked relevant meeting never reached the evidence bundle.
- Mean evidence recall (cited relevant meetings): 68.2%. This is stricter than retrieval recall because relevant evidence must also survive into the final citation set.
- Mean decision-state accuracy: 72.7%. This checks whether the answer names the expected state (kept/dropped/superseded) rather than only the topic.
- Temporal reasoning is deterministic: the timeline is sorted by meeting sequence and only explicit agreement language can confirm a decision. The evaluation above shows where that conservatism helped or hurt.

## 6. Performance and cost (measured)

- Ingestion: 116.8 s for 138 meetings (0.85 s/meeting).
- Embedding: 87.17 s; semantic extraction: 15.94 s.
- Queries: 11 measured, mean 2.24 s, p95 13.94 s.
- Storage: Chroma 58.6 MB, SQLite 40.65 MB.
- Measured API cost: $0.00 (local embeddings; no paid LLM calls in this run).
- Projected 1,000 meetings: 11550 chunks, 0.24 h single-process ingestion, $0.89 projected Gemini extraction.

See `outputs/benchmark.json` for assumptions and the 10,000-meeting architecture notes.

## 7. Honest limitations

- The evaluation set is small (11 questions) and covers one team. It is not a benchmark of the whole corpus.
- Lexical fact matching can mark a correct paraphrase as missing; inspect the failure examples for the actual text.
- The unsupported-claim rate is a proxy based on content-term overlap, not entailment.
- This run used local models rather than a paid Gemini API because no credential was configured; the Gemini path is implemented but live API tests were not run.
