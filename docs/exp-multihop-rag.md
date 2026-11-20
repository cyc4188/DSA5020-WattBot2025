# Multi-Hop Sub-Question Decomposition + Contextual Hybrid Retrieval & Reranking

Experiment report for the `wattbot/multihop_rag.py` track.

## 1. Overview & Diagnosed Root Causes Solved

### Root Cause 1 — Decontextualized tiny PDF blocks, broken table decimals, unnormalized tokens

The v1 pipeline (`BM25 + BGE-small`) regressed Recall@10 from **94.87% to 92.31%** because it indexed
17,207 fragmented PDF text blocks torn from their section context, mangled table decimals during
extraction, and stored raw unnormalized tokens.

**Solution:** `build_contextual_index` rebuilds the corpus with
`pymupdf4llm.to_markdown(..., page_chunks=True)` across all **32 PDFs (641 pages)**, producing
**4,769 contextualized nodes** — **4,570** section-aware overlapping prose chunks plus **199**
full-header Markdown tables — versus the 17,207 fragmented v1 nodes.

Every node carries:
- `search_text` — prefixed with `[ref_id | title | p.{page} | {section}]` so the retriever sees document,
  title, page and section identity.
- `clean_text` — NFKC normalization, line-wrap de-hyphenation, and HTML `<br>` / `<sup>` / `<u>` cleanup.
- `clean_tokens` — decimal/technical compounds preserved (`6.1b`, `5.439`, `gpt-3`, `co2e`) plus Snowball
  English stemming.

### Root Cause 2 — Paraphrase-only `PLAN` failing multi-hop and cross-table questions

The paraphrase-only planner emitted a single query and could not resolve questions requiring multiple
facts or cross-table reconciliation.

**Solution:**
- `PLAN_MULTIHOP` performs fact-slot decomposition: it emits `needs_decomposition` and 2–4 entity-isolated
  atomic `fact_slots`, each with domain synonym expansion.
- `search_multihop` retrieves and cross-encoder reranks each slot independently with guaranteed
  per-slot round-robin quotas.
- Iterative IRCoT follow-up retrieval consumes `missing_facts` plus `adjacent_page_nodes`.
- `safe_eval_decimal` performs AST-based arithmetic verification, and `normalize_submission_fields`
  applies unit-hint scaling.

### Root Cause 3 — Overly pedantic verifier false rejections and citation dilution

The strict verifier falsely rejected valid answers (e.g. `q253`) and citation Jaccard was diluted by
over-citation on `q153` and `q263`.

**Solution:**
- Minimal-sufficient citation rules replace blanket citation requirements.
- IDF-weighted unigram+bigram `lexical_alignment` tie-breaks passages with tied values.
- `VERIFY_MULTIHOP` performs error-typed verification with the categories `none`, `missing_support`,
  `factual_mismatch`, `math_or_unit_error`, `over_citation`, `format_error`, and automatically invokes
  `prune_candidate_citations`.

## 2. Retrieval-Only Benchmark

**39 answerable train questions, original question only, zero LLM calls.**

| Retrieval Configuration | Doc Recall@5 | Doc Recall@10 | Doc Recall@20 | MRR | Missed@10 |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline v1 BM25 + BGE-small (reported) | — | 92.31% | — | — | 3 questions |
| Baseline v1 BM25 (`wattbot-index.json`) | 92.31% | 94.87% | 97.44% | 0.8719 | `q075`, `q153`, `q272` |
| Contextual v2 Dense (`BAAI/bge-small-en-v1.5`) | 80.77% | 88.46% | 92.31% | 0.7618 | 5 questions |
| Contextual v2 Hybrid RRF (`BM25 + BGE-small`) | 88.46% | 94.87% | **100.00%** | 0.8552 | `q075`, `q272`, `q280` |
| Contextual v2 Hybrid + `Xenova/ms-marco-MiniLM-L-12-v2` | **93.59%** | 96.15% | 98.72% | **0.9100** | `q272`, `q280` |
| Contextual v2 BM25 (`clean_tokens` + stem + prefix) | **93.59%** | **97.44%** | 97.44% | 0.9064 | `q075` (2nd doc), `q272` (2nd doc) |
| Contextual v2 Hybrid + `jinaai/jina-reranker-v1-turbo-en` | 92.31% | **97.44%** | 97.44% | 0.8940 | `q075` (2nd doc), `q272` (2nd doc) |

**Highlights**

- Contextual v2 BM25 and Contextual v2 Hybrid + Jina-Reranker both achieve **100.0% Recall@10 on all 37
  single-document train questions** — **97.44% overall** across all 39 answerable questions, which is
  **+2.57% over baseline BM25** and **+5.13% over baseline BM25+BGE-small**.
- Contextual v2 Hybrid RRF reaches **100.0% Recall@20** across all 39 questions, including both documents
  of every multi-document question.

## 3. Full 41-Question End-to-End Train Evaluation

Results under `outputs/multihop-retrieval/train/`.

| System | Value Accuracy | Citation Jaccard | Abstention Acc | Weighted Score | Correct / False Blank | API Calls | Total Tokens | Mean Tokens / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Closed-Book Direct (`DeepSeek-V4.1-Flash`) | 36.59% (15/41) | 0.0488 | 53.66% (22/41) | 0.3354 | 1/2 / 18/39 | 41 | 10,283 | 250.80 |
| Baseline RAG (v1 BM25, `DeepSeek-V4.1-Flash`) | 73.17% (30/41) | 0.8252 | 87.80% (36/41) | 0.7604 | 2/2 / 5/39 | 153 | 501,292 | 12,226.63 |
| Pi RAG (field-semantics retest) | 75.61% (31/41) | 0.7683 | 80.49% (33/41) | 0.7628 | 2/2 / 8/39 | 114 | 413,014 | 10,073.51 |
| Full-Context Baseline (`DeepSeek-V4.1-Flash`) | 80.49% (33/41) | 0.6829 | 92.68% (38/41) | 0.7988 | 2/2 / 3/39 | 42 | 28,870,040 | 704,147.32 |
| **Multi-Hop Contextual Hybrid RAG (Ours)** | **78.05% (32/41)** | **0.8374** | **87.80% (36/41)** | **0.7988** | **2/2 / 5/39** | **135** | **380,094** | **9,270.59** |

## 4. Per-Question Question-Type & Failure-Mode Analysis

### Fixed vs Baseline RAG

- **`q009` (GShard-600B net CO2e):** `43 -> 4.3 tCO2e` (weighted `0.250 -> 1.000`). Clean
  `pymupdf4llm` Markdown table extraction eliminated displaced decimals.
- **`q253` (Wu et al. 2021 platform-level caching multiplier):** `is_blank -> 6.7 multiplier` (weighted
  `0.000 -> 1.000`). The error-typed verifier plus passage-excerpt context recognized the section heading
  `Platform-Level Caching` paired with `application-level caching` in body text.
- **`q263` (ML inference share of total compute demand `[80,90]` percent):** citation
  `['luccioni2024'] -> ['chung2025']` (Jaccard `0.00 -> 1.00`, weighted `0.850 -> 1.000`).
  `lexical_alignment` tie-breaking matched the question's exact phrasing
  ("ML inference reportedly accounts for 80–90% of the total compute demand").
- **`q316` (Llama-3.1 training health cost in Altoona, Iowa):** `2.51 -> 2510000 USD` (weighted
  `0.250 -> 1.000`). Multi-row table header preservation (`Health Cost | (million $)`) plus unit-hint
  scaling to `USD`.
- **`q153` (Tracking runtime True/False):** reduced over-citation from 3 docs
  (`['cottier2024', 'strubell2019', 'xia2024']`, Jaccard `0.333`) to 2 docs
  (`['xia2024', 'strubell2019']`, Jaccard `0.500`).
- **`q272` (6.1B model training electricity in US household-years):** decomposed into independent
  `training_energy` and `us_household_electricity` fact slots; retrieved `dodge2022` (103,500 kWh) plus the
  household electricity reference instead of abstaining (Jaccard `0.00 -> 0.333`).

### Token Breakdown by Stage

Source: `outputs/multihop-retrieval/train/token-usage.json`.

- **`plan`:** 41 calls, 13,777 prompt + 5,497 completion = **19,274 tokens**
- **`answer`:** 59 calls, 319,735 prompt + 11,563 completion = **331,298 tokens**
- **`verify`:** 35 calls, 27,023 prompt + 2,499 completion = **29,522 tokens**
- **Total:** 135 calls, **380,094 tokens** (9,270.59 tokens/question, median response 6.26s)

## 5. Reproducibility Commands

```bash
# Build the contextualized v2 index
python3 -m wattbot.multihop_rag index

# Retrieval-only benchmark (zero LLM calls, 39 answerable train questions)
python3 -m wattbot.multihop_rag benchmark-retrieval

# Full end-to-end run over the train set
python3 -m wattbot.multihop_rag run --split train --output outputs/multihop-retrieval/train

# Unit tests
python3 -m unittest discover -s tests
```
