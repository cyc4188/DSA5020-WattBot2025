# Experiment: Pi Agentic RAG v2 + Safe AST Decimal Sandbox + Thinking Ablation

**Track:** `pi-v2-sandbox`
**Status:** Both runs complete (`--thinking false` and `--thinking true`), 41-question train evaluation.

---

## 1. Executive Summary

v1 of the Pi agentic RAG pipeline scored **0.7628 weighted** on the 41-question
train split (31/41 value accuracy = **75.61%**, **0.7683** citation Jaccard,
**8/39** false abstentions, **413,014** total tokens). Post-mortem analysis
isolated five structural failure modes that cap the agent well below the
full-context ceiling. v2 (`wattbot/pi_agent_v2.py` & `pi_rag/agent_v2.mjs`)
addresses each one directly, leaving the v1 code path untouched. With thinking
disabled, v2 lifts the weighted score to **0.8628** (35/41 = **85.37%** value
accuracy, **0.8659** citation Jaccard, **0.9268** abstention accuracy, only
**3/39** false abstentions, **2/2** true abstentions) at the cost of a larger
token budget driven by richer tool output and more search calls. Enabling
thinking (`--thinking true`) raises the weighted score again to **0.8866**
(36/41 = **87.80%** value accuracy, **0.8862** citation Jaccard, **0.9512**
abstention accuracy, **2/39** false abstentions) for roughly **1.74×** the tokens.

### Diagnosed v1 Failure Modes

**Failure Mode 1 — Rigid 2-operand `calculate` tool.**
The v1 calculator accepted exactly two operands, so any genuine multi-step
expression (e.g. `30000 / 96 / 24`) had to be decomposed by hand across turns,
and standard unit/time/scale constants (`1000000`, `24`, `365`) were rejected as
ungrounded values. Multi-step arithmetic therefore failed or consumed extra turns.

**Failure Mode 2 — Premature abstention on turn 1.**
When the initial BM25 query missed a denominator or a secondary paper, v1 would
sometimes declare the question unanswerable immediately — *before* calling
`search_corpus` at all — converting recoverable questions into false abstentions
(8/39 of the v1 misses).

**Failure Mode 3 — Sequential-only tool calls + tight call budget.**
`parallel_tool_calls: false` combined with `max_calls=5` meant that a single
failed or uninformative tool error burned a large fraction of the turn budget,
leaving no room to recover.

**Failure Mode 4 — Hardcoded `enable_thinking: false`.**
v1 pinned thinking off with no controlled ablation. There was no evidence about
whether `enable_thinking: true` on DeepSeek-V4.1-Flash helps or hurts this task.

**Failure Mode 5 — Raw retrieval over dirty text + unit-scaling bug.**
Single-query BM25 (`top_k=8`, 1250 chars) ran over uncleaned ligatures,
line-wrap hyphens, and decimal-corrupted PyMuPDF `find_tables()` rows.
Separately, `pipeline.py#L104` skipped unit_hint conversion whenever the
question itself mentioned a scaled unit — e.g. a question phrased "in million
liters" with `unit_hint="liters"` was left unscaled, so the submitted value and
the expected unit disagreed.

---

## 2. v2 Architecture

v2 lives in new files (`wattbot/pi_agent_v2.py`, `pi_rag/agent_v2.mjs`) and does
not modify the v1 implementation, so the two pipelines can be evaluated
side-by-side.

### 2.1 Safe AST-Whitelisted Python `Decimal` Evaluator (`eval_math`)

- **Strict AST whitelist.** Only `Expression`, `BinOp`, `UnaryOp`, `Name`,
  `Constant`, `Call` (restricted to `abs`/`min`/`max`), `Load`, `Add`, `Sub`,
  `Mult`, `Div`, `Pow`, `UAdd`, `USub` are permitted. No attribute access, no
  subscripts, no arbitrary calls.
- **Resource limits.** Max expression length 300 chars, max 64 AST nodes, max
  depth 12.
- **Precision.** Evaluation runs inside `localcontext(prec=32)` so full
  `Decimal` precision is preserved without premature float rounding.
- **Named operands.** `named_operands` must be *grounded* against previously seen
  passages via `source_quote_v2` token-span matching. The matcher normalizes
  NFKC ligatures, thousands commas, Markdown table pipes/bold, and PDF
  line-wrap de-hyphenation. Operands may also be chained from a prior
  calculation (`c1`, `c2`, …) instead of re-quoted.
- **Universal constants.** An explicit whitelist of conversion/math constants is
  usable directly *without* a verbatim source quote: `24`, `60`, `3600`,
  `86400`, `365`, `365.25`, `8760`, `100`, `1000`, `1e6`, `1e9`, `1e12`;
  lbs↔kg `0.45359237` / `2.20462262185`; gallons↔liters `3.785411784`;
  MJ↔kWh `3.6`; plus named constants such as `HOURS_PER_DAY`,
  `DAYS_PER_YEAR`, `MILLION`, `BILLION`, `LB_PER_KG`, `KG_PER_LB`,
  `L_PER_GAL`. Every expression still requires **at least one grounded
  operand**, so constants cannot be used to fabricate an ungrounded answer.

### 2.2 Anti-Premature-Abstention Guard

In pi mode, `submit_answer(answerable=false)` is **blocked** whenever
`search_calls < 1`. The guard emits a `premature_abstention_blocked` event and
instructs the agent to search for the missing evidence/denominator before it is
allowed to abstain. This directly closes Failure Mode 2.

### 2.3 Richer Tools, Clean `read_page`, Hybrid Retrieval (`CleanSearchIndex`)

- **Batching & budget.** `parallel_tool_calls: true` (calls are still executed
  sequentially over stdin/stdout IPC to avoid race conditions, but multiple tool
  calls can be batched per LLM turn), `max_calls=8`, `top_k=10` (1800 chars).
- **`list_documents()`** exposes all 32 competition papers (`ref_id`, title,
  page count) so the agent can plan across the corpus.
- **`search_corpus(query, ref_id?, top_k?)`** uses NFKC-normalized +
  de-hyphenated BM25, plus stem sub-tokens (`co2e`→`co2`, `h100s`→`h100`,
  `v100x64`→`v100`), document-title indexing, and dense `BAAI/bge-small-en-v1.5`
  retrieval fused via RRF.
- **`read_page(ref_id, page)`** renders pages with `pymupdf4llm.to_markdown`,
  producing clean Markdown tables with zero displaced decimals. Quotes are
  automatically back-projected onto `wattbot-index.json` page spans, so
  `verify_rag_trace` remains 100% compatible.
- **Unit-scaling fix.** `normalize_submission_v2` normalizes scaled units
  (million liters → liters, million USD → USD) so the submitted value always
  aligns with `unit_hint`, fixing the Failure Mode 5 scaling bug.

### 2.4 Configurable `--thinking` Flag & `reasoning_content` Protocol

- Supports `--thinking false` vs `--thinking true`
  (`chat_template_kwargs: {enable_thinking: bool}`).
- When `thinking=true`, `maxTokens=6144`; `reasoning_content` is preserved
  across assistant tool-call turns, and
  `completion_tokens_details.reasoning_tokens` is tracked for cost accounting.

---

## 3. OpenCode Development Evidence

All code and unit tests were authored via
`opencode run -m hkustgz-aigc/DeepSeek-V4.1-Flash --auto`
(sessions `ses_e0de42066fferlEkG9xSMLpglg`,
`ses_e0ddc96f4ffeVAkCYJSQKixOlq`, and subsequent build sessions).

Test status: **41 unit tests passing** via `unittest discover -s tests`,
including **27 new tests** in `tests/test_pi_agent_v2.py`.

---

## 4. Controlled 41-Question Train Evaluation

All rows evaluated on the same 41-question train split.

| Configuration | Weighted | Value acc | Citation Jaccard | Abstention acc | False abst. | True abst. | API calls | Total tokens | Tok/q |
|---|---|---|---|---|---|---|---|---|---|
| v1 RAG | 0.7604 | — | — | — | — | — | — | — | — |
| v1 Full-Context | 0.7988 | — | — | — | — | — | — | — | — |
| v1 Pi RAG | 0.7628 | 31/41 (75.61%) | 0.7683 | — | 8/39 | — | — | 413,014 | — |
| **v2 Pi RAG `--thinking false`** | **0.8628** | **35/41 (85.37%)** | **0.8659** | **0.9268** | **3/39** | **2/2** | **124** | **951,926** | **23,218** |
| **v2 Pi RAG `--thinking true`** | **0.8866** | **36/41 (87.80%)** | **0.8862** | **0.9512** | **2/39** | **2/2 (100%)** | **131** | **1,656,974** | **40,414** |

**Reading the table.** Turning on the v2 toolchain without thinking yields a
+0.1000 weighted improvement over v1 Pi RAG (0.7628 → 0.8628), +9.76 pp value
accuracy, +0.0976 citation Jaccard, and cuts false abstentions from 8 to 3 while
preserving both true abstentions. The gains are bought with a higher token
budget: 951,926 tokens vs 413,014 (≈2.3×) at 23,218 tok/q, consistent with richer
1800-char page reads and more search calls per question (124 API calls total).

Adding thinking lifts the weighted score further to **0.8866** (0.886585) —
**+0.1238** over v1 Pi RAG (0.7628) and **+0.0238** over `--thinking false`
(0.8628). Value accuracy rises to **36/41 (87.80%)**, citation Jaccard to
**0.8862** (0.886179), and abstention accuracy to **0.9512** (39/41), with false
abstentions down to **2/39** and both true abstentions retained (2/2 = 100%).
The cost is substantial: **1,656,974** total tokens (1,599,526 prompt, 57,448
completion, including 32,789 reasoning tokens) at **40,414.00 tok/q**, a median
latency of **9.013 s/call**, and **131** API calls (94 `pi_agent`, 37 `verify`).

---

## 5. Thinking Ablation Analysis

The two v2 runs are a controlled ablation: identical corpus, index, tools,
prompts and 41-question split, differing only in `chat_template_kwargs:
{"enable_thinking": bool}`.

### 5.1 API Confirmation

HKUST(GZ) DeepSeek-V4.1-Flash natively supports
`chat_template_kwargs={"enable_thinking": true}`, returning
`message.reasoning_content` and
`usage.completion_tokens_details.reasoning_tokens`. This confirms the flag is
honored end-to-end and that reasoning tokens are separately billable/accounted
(32,789 reasoning tokens observed across the true run).

### 5.2 Cost & Tool-Use Comparison

| Metric | `--thinking false` | `--thinking true` | Delta |
|---|---|---|---|
| Total tokens | 951,926 | 1,656,974 | +74% |
| Tokens/question | 23,217.71 | 40,414.00 | +74% |
| Reasoning tokens | 0 | 32,789 | — |
| Median latency/call | 5.598 s | 9.013 s | +61% |
| API calls | 124 | 131 (94 `pi_agent`, 37 `verify`) | +7 |
| `list_documents` | 2 | 4 | +2 |
| `search_corpus` | 30 | 65 | +2.17× |
| `read_page` | 9 | 9 | 0 |
| `eval_math` | 13 | 13 | 0 |
| `submit_answer` | 50 | 48 | −2 |

Thinking roughly doubles search effort (`search_corpus` 30 → 65, **2.17×**) and
total token spend (**1.74×**), while adding only 7 API calls — the extra work is
concentrated in more searches per turn rather than more LLM turns.

### 5.3 Per-Question Ablation

**Gains from `thinking=true` (+2 questions solved).**

- **q124.** With thinking, the agent searched and extracted the exact table value
  **5,439,000 liters** instead of the abstract summary **5,400,000 liters**.
- **q200.** With thinking, the agent issued targeted follow-up searches in
  `patterson2021` to verify the Evolved Transformer comparison, correctly
  answering **False ("0")** with **1.00** citation Jaccard instead of abstaining.

**Trade-off (−1 question, +74% tokens).**

- **q297.** `thinking=false` ran `eval_math` on the exact table figures and got
  **8.720430108×**, whereas `thinking=true` cited the prose summary
  ("seven times less energy" → **7**). The reasoning pass anchored on the more
  salient narrative sentence rather than the precise table.

### 5.4 Union Result

The union of the two v2 configurations (`--thinking false ∪ --thinking true`)
solves **37/41 (90.24%)** of the train questions, versus 35/41 and 36/41
individually — i.e. the two modes fail on partly disjoint questions, making a
routing/ensemble strategy a natural next step.

## 6. Reproduction

Use the original v1 index and the shared Node dependencies. Run both
configurations with separate output directories.

```bash
npm ci --prefix pi_rag
python3 -m wattbot index
python3 -m wattbot.pi_agent_v2 --split train --workers 2 --thinking false \
  --output-dir outputs/pi-rag-v2/thinking-off/train
python3 -m wattbot.pi_agent_v2 --split train --workers 2 --thinking true \
  --output-dir outputs/pi-rag-v2/thinking-on/train
```
