# Experiment: PDF Parsing, Text Cleaning & Full-Context Citation Recovery (`parser-fix`)

Track goal: fix corpus-quality and citation-grounding failures in the v1 pipeline
without changing the answer model or spending extra LLM tokens at evaluation time.

## 1. Root Causes Diagnosed & Fixed

| # | Root cause | v1 evidence | Fix |
| --- | --- | --- | --- |
| 1 | Table corruption & missing borderless tables (`wattbot/clean_index.py`) | PyMuPDF `table.extract()` corrupted **445 / 835 (53.3%)** rows: displaced decimals (`4.3 -> 43 .` in `patterson2021`, `5.439 -> 5439 .` in `li2025b`, `2.51 -> 251 .` in `han2024`), stripped multi-level headers/units (`million $`, `million L`), and **0** tables from borderless booktabs papers (`strubell2019`, `luccioni2025b`) | Hybrid extraction: word-center coordinate assignment inside `find_tables()` cell bboxes, plus `pymupdf4llm` Markdown fallback for borderless tables. Preserve multi-level headers and section subheaders, attach nearby captions, emit dual `table_row` + `table_block` nodes |
| 2 | Text ligatures & line-wrap hyphenation (`wattbot/clean_index.py`, `wattbot/retrieval.py`) | **497 / 16,372 (3.04%)** text nodes contained Unicode ligatures (`ﬁ` / `ﬂ`); **1,317 / 16,372 (8.04%)** contained broken line-wrap hyphens (`efﬁ- ciency`, `green- house`) | NFKC normalization and prefix-aware line-wrap de-hyphenation in both index building (`clean_text`) and BM25 tokenization (`tokens`). `SearchIndex` remains backward-compatible with v1 and v2 indexes |
| 3 | Full-context citation drop (`wattbot/citation_recovery.py`, `wattbot/baselines/full_context.py`) | Strict exact-page `compact(quote) in compact(text)` dropped valid citations on 8 correctly-answered train questions (`q009, q102, q139, q207, q221, q253, q282, q304`) | 4-tier quote matching (`exact`, `token_span`, `ellipsis_segments`, `token_lcs` with numeric-token guard) across 4 scopes (`same_page`, `adjacent_page`, `same_doc`, `corpus_unique`). Zero-token offline trace replay |
| 4 | Unit-hint scaling & verifier over-rejection (`wattbot/pipeline.py`) | `'in <unit>'` bypass let prose units override `unit_hint`; verifier rejected valid heading-vs-body modifier paraphrases | Removed bypass so `unit_hint` always wins, with unit alias canonicalization (`million L -> liters`, `million $ -> USD`, `thousands of grams -> grams`). Clarified `ANSWER` unit scaling and refined `VERIFY` to enforce scale conversion without false rejections |

## 2. Corpus Index: v1 vs v2

Source: `data/processed/wattbot-index.json` (v1) vs
`outputs/parser-fix/wattbot-index-v2.json` (v2).

| Metric | v1 Index | v2 Clean Index | Change |
| --- | ---: | ---: | ---: |
| Total nodes | 17,207 | 17,374 | +167 |
| Text nodes | 16,372 | 16,372 | 0 |
| Extracted tables | 131 | 142 | +11 (incl. `strubell2019` Tables 1–4, `luccioni2025b`) |
| Documents with tables | 27 / 32 | 29 / 32 | +2 docs |
| Table row nodes (`table_row`) | 835 | 819 | -16 (filtered bogus bibliography rows) |
| Multi-row table blocks (`table_block`) | 0 | 183 | +183 |
| Displaced-decimal table rows (`\d\s+\.`) | 445 / 835 (53.29%) | 0 / 819 (0.00%) | -100.0% (39 caption-only footnote periods `detail 4 .`) |
| Text nodes with ligatures (`ﬁ`–`ﬆ`) | 497 / 16,372 (3.04%) | 0 / 16,372 (0.00%) | -100.0% |
| Text nodes with broken hyphens | 1,317 / 16,372 (8.04%) | 0 / 16,372 (0.00%) | -100.0% |

## 3. Offline Full-Context Citation Recovery Replay (Zero LLM Tokens)

### Train — 41 questions
Replayed from `outputs/baseline-full-context/train/trace.jsonl`.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Value accuracy | 80.49% (33/41) | 80.49% (33/41) | 0 |
| Citation Jaccard | 0.6829 | **0.8841** | **+0.2012** |
| Abstention accuracy | 92.68% (38/41) | 92.68% (38/41) | 0 |
| Weighted score | 0.7988 | **0.8290** | **+0.0302** |

All 8 previously dropped answered train questions (`q009, q102, q139, q207,
q221, q253, q282, q304`) recovered verbatim: **17** recovered citation events,
**1** ungrounded quote dropped.

### Test — 282 questions
Replayed from `outputs/baseline-full-context/test/trace.jsonl`.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Answered questions with valid grounded citations | 206 / 259 (79.54%) | **255 / 259 (98.46%)** | **+49 questions** |

**99** recovered citation events. On Kaggle post-competition evaluation (2026-11-20), the recovered test predictions (`ref 58605673`) improved Full-Context from **0.858 Public / 0.815 Private** (`ref 58605663`) to **0.883 Public / 0.837 Private** (+0.025 Public, +0.022 Private) with zero additional LLM tokens. See [Kaggle test results](kaggle-test-results.md).

## 4. End-to-End 41-Question Train Evaluation

| System | Value Acc | Citation Jaccard | Abstention Acc | Weighted Score | Total Tokens (41 q) | Mean Tokens / Q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline RAG (v1 index) | 73.17% (30/41) | 0.8252 | 87.80% (36/41) | 0.7604 | 501,292 | 12,226.6 |
| Baseline Full-Context (v1, strict) | 80.49% (33/41) | 0.6829 | 92.68% (38/41) | 0.7988 | 28,870,040 | 704,147.3 |
| **RAG on Clean v2 Index (`parser-fix`)** | **80.49% (33/41)** | **0.8821** | **92.68% (38/41)** | **0.8287** | **377,159** | **9,199.0** |
| **Full-Context + Citation Recovery (`parser-fix`)** | **80.49% (33/41)** | **0.8841** | **92.68% (38/41)** | **0.8290** | **28,870,040** (+0 replay) | **704,147.3** |

### Highlighted per-question RAG fixes

| Question | Before | After | Cause addressed |
| --- | --- | --- | --- |
| `q009` | `43` | `4.3 tCO2e` | Displaced decimal |
| `q253` | `is_blank` | `6.7 multiplier` | De-hyphenation + gentler verifier |
| `q262` | `is_blank` | `1438 lbs` | `strubell2019` borderless Table 3 captured |
| `q316` | `2.51` | `2510000 USD` | Multi-level header `Health Cost (million $)` + `unit_hint` scaling |
| `q153` | Jaccard `0.33` | Jaccard `1.00` | Citation recovery |
| `q263` | Jaccard `0.00` | Jaccard `0.33` | Citation recovery |

Known residual differences (not regressions in value extraction): `q124`
returns `5,400,000` vs gold `5,439,000` because the retrieved prose
`5.4 million liters` precedes Table 1's `5.439`; it is correctly scaled via the
fixed `unit_hint` conversion. `q078` returns `[10.5, 47.2]` from Table 1's last
column rather than prose `[0.02, 0.1]`.

## 5. Reproduction

```bash
# Build v2 clean index and print the v1/v2 audit
python3 -m wattbot.clean_index \
  --compare-v1 data/processed/wattbot-index.json

# RAG on the clean v2 index (41-question train run)
python3 -m wattbot run \
  --questions data/raw/train_QA.csv \
  --index outputs/parser-fix/wattbot-index-v2.json \
  --output outputs/parser-fix/rag-v2-train/predictions.csv \
  --trace outputs/parser-fix/rag-v2-train/trace.jsonl
python3 -m wattbot score \
  --questions data/raw/train_QA.csv \
  --output outputs/parser-fix/rag-v2-train/predictions.csv \
  --metrics outputs/parser-fix/rag-v2-train/metrics.json

# Full-context baseline on v2
python3 -m wattbot.baselines.full_context \
  --split train \
  --index outputs/parser-fix/wattbot-index-v2.json \
  --output-dir outputs/parser-fix/full-context-recovered/train

# Zero-token citation recovery replay
python3 -m wattbot.citation_recovery \
  --trace outputs/baseline-full-context/train/trace.jsonl \
  --index outputs/parser-fix/wattbot-index-v2.json \
  --metadata data/raw/metadata.csv \
  --questions data/raw/train_QA.csv \
  --output-dir outputs/parser-fix/full-context-recovered/train

# Tests
python3 -m unittest discover -s tests
```

**53/53 unit tests pass** (`unittest discover -s tests`).
