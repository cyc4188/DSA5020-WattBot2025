# Experiment: Two-Stage Document-Level Routing + Full-Document Reading

## 1. Overview & Motivation

The WattBot 2025 corpus is small enough that per-document reading is viable: only
32 PDFs (641 pages, 2,700,431 cleaned Markdown chars). **30 of the 32 papers are
6–32 pages (~6k–27k tokens each, mean ~16 pages)**; only `amazon2023` (98p) and
`stone2022` (52p) are long reports.

Chunk RAG (0.7604 weighted, 73.17% value accuracy, 12,226.63 tok/q) loses
multi-level table headers, borderless tables (`strubell2019`), and cross-page
context within a paper (`q124`, `q158`, `q200`, `q253`, `q262`, `q297`, `q316`).

Full-Context (0.7988 weighted, 80.49% value accuracy, 0.6829 citation Jaccard,
704,147.32 tok/q) costs 57.6× more tokens and suffers from attention dilution and
strict single-page quote drops on 8 correctly answered questions.

Two-Stage Document-Level Routing (`wattbot/doc_routing.py`) bridges both:
**0.8951 weighted score**, **87.80% (36/41) value accuracy**, **0.9268 citation
Jaccard**, **97.56% (40/41) abstention accuracy** at **84,975.27 tok/q**
(**12.07% of Full-Context token cost — an 8.29× / 87.93% reduction**).

## 2. Architecture (`wattbot/doc_routing.py`)

### Stage 0 — Offline Clean Per-Document Markdown & Structured Document Cards

- Extracts per-page Markdown across all 32 PDFs (641 pages) via
  `pymupdf4llm.to_markdown(..., page_chunks=True, use_ocr=False)` with a per-page
  `pymupdf` raw-block safety fallback, NFKC normalization, HTML `<br>`/`<u>`/`<sup>`
  cleanup, and non-table line-wrap de-hyphenation (`markdown_corpus.json`).
  Preserves borderless booktabs tables (`strubell2019` Tables 1–3) and exact table
  decimals (`patterson2021` Table 4, `li2025b`, `han2024`, `zschache2025`).
- Automatically extracts `year_aliases` from metadata year, `ref_id`, citation
  string, PDF metadata dates, and page-1 publication/arXiv headers (e.g.
  `chen2024` -> `["2024", "2025"]`, `li2025b` -> `["2023", "2025"]`,
  `samsi2024` -> `["2023", "2024"]`), plus section headings and Table/Figure
  captions.
- Enriches all 32 cards via one offline `DeepSeek-V4.1-Flash` call per document
  (`document_cards.json`, 32 calls, 241,879 total offline tokens) capturing
  `summary`, `models`, `hardware_and_infra`, `datasets_and_benchmarks`,
  `metrics_and_topics`, and `reference_baselines`.

### Stage 1 — Hybrid Document Router + Oversized-Doc Page Windowing

- Indexes all 641 cleaned pages and all 32 document cards in a decimal- and
  hyphen-aware BM25 index (`routing_tokens`), aggregating per-document scores via
  `top1_page + 0.5*top2_page + 0.25*top3_page + 0.35*card_score`.
- Routes via `DeepSeek-V4.1-Flash` over all 32 compact document cards + top-8 BM25
  full-text candidates with centered highest-IDF snippets, fused with a BM25 Top-1
  safeguard into `top_docs=3` selected documents.
- Standard documents (`<= 35` pages and `<= 130,000` chars: 30/32 PDFs) are fed in
  full with `[ref_id, p. N]` markers; oversized documents (`amazon2023` 98p,
  `stone2022` 52p) are windowed to `<= 22` pages (pages 1–2 + top BM25 pages with
  `±1` neighbor pages and `[... pages X-Y omitted ...]` markers).

### Stage 2 — Full-Document Reading, Grounded 5-Tier Citation Verification & Unit Normalization

- Prompts `DeepSeek-V4.1-Flash` to prefer exact primary/appendix table figures over
  rounded prose summaries, retain full calculation precision (`>= 6` significant
  digits), convert values to `unit_hint`, use concise canonical entity names, and
  cite minimal sufficient `ref_id`s.
- 5-tier citation recovery (`verify_and_recover_citations`): (1) exact/compact match
  on cited page, (2) token-span `source_quote` match on cited page, (3) adjacent page
  `page ± 1`, (4) document-wide page search within supplied doc, (5) full-page
  content-token overlap fallback for transposed multi-row tables.
- Always converts scaled units (`million liters` -> `liters`, `million USD` -> `USD`)
  to `unit_hint` in `normalize_doc_routing_fields`, and triggers at most 1 adaptive
  expansion retry onto unexamined candidate docs when Stage 2 abstains.

## 3. Stage-1 Document Routing Recall (39 Answerable Train Questions)

| Strategy | Recall@1 | Exact Cov@1 | Recall@2 | Exact Cov@2 | Recall@3 | Exact Cov@3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25-only (page + card aggregation) | 83.33% | 82.05% (32/39) | 93.59% | 92.31% (36/39) | 93.59% | 92.31% (36/39) |
| LLM Card Router only | 94.87% | 92.31% (36/39) | 94.87% | 92.31% (36/39) | 94.87% | 92.31% (36/39) |
| **Hybrid Router (LLM Cards + BM25)** | **94.87%** | **92.31% (36/39)** | **94.87%** | **92.31% (36/39)** | **94.87%** | **92.31% (36/39)** |

## 4. End-to-End 41-Question Train Evaluation Comparison

| Metric | Closed-book | Chunk RAG | Full-Context | **Two-Stage Doc-Routing** | Delta vs RAG | Delta vs Full-Context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Local Weighted Score** | 0.3354 | 0.7604 | 0.7988 | **0.8951** | **+0.1347** | **+0.0963** |
| **Answer Value Accuracy** | 36.59% (15/41) | 73.17% (30/41) | 80.49% (33/41) | **87.80% (36/41)** | **+14.63% (+6 q)** | **+7.31% (+3 q)** |
| **Ref Doc Jaccard (mean)** | 0.0488 | 0.8252 | 0.6829 | **0.9268** | **+0.1016** | **+0.2439** |
| **Abstention Accuracy** | 53.66% (22/41) | 87.80% (36/41) | 92.68% (38/41) | **97.56% (40/41)** | **+9.76% (+4 q)** | **+4.88% (+2 q)** |
| **Correct Blank (Unanswerable)** | 1/2 | 2/2 | 2/2 | **2/2** | 0 | 0 |
| **False Blank (Answerable)** | 18/39 | 5/39 | 3/39 | **1/39** | **-4** | **-2** |
| **Total Tokens (41 q)** | 10,283 | 501,292 | 28,870,040 | **3,483,986** | 6.95× | **0.1207× (-87.93%)** |
| **Mean Tokens / Question** | 250.80 | 12,226.63 | 704,147.32 | **84,975.27** | 6.95× | **0.1207× (8.29× cheaper)** |

## 5. Key Qualitative Wins & Error Analysis

- **Fixed vs both RAG & Full-Context**: `q124` (`5439000 liters` exact table value +
  unit scaling vs `5.4 million liters`), `q200` (`0` on Evolved Transformer WMT curve
  in `patterson2021`), `q297` (`8.72043010752688` full precision `48.66/5.58` in
  `zschache2025` vs `8.7` premature rounding).
- **Fixed vs Full-Context**: `q158` (`13.2 percent` in `chen2024` via `"2025"` year
  alias), `q202` (`Financial Sentiment Analysis` concise canonical name in
  `khan2025`), plus **all 8 Full-Context dropped-citation questions**
  (`q009, q102, q139, q207, q221, q253, q282, q304`) recovered to `1.00` citation
  Jaccard.
- **Fixed vs Chunk RAG**: `q009` (`4.3 tCO2e` clean table in `patterson2021`),
  `q253` (`6.7 multiplier` in `wu2021a`), `q262` (`1438 lbs` borderless table in
  `strubell2019`), `q263` (`[80,90]` in `chung2025`), `q316` (`2510000 USD` in
  `han2024`).
- **Remaining 5 value misses on 41 train questions**: `q078` (`[10,50]` mL reported
  instead of `[0.02,0.1]` 500 mL bottles), `q170` (`14.5` vs `14.4` days from
  `1438 / (36156/365)`), `q215` (bar chart value inside `schwartz2019` Figure 2,
  properly abstained without VLM), `q272` (`1` vs `1.3` household-years), `q280`
  (`13.0208333` unrounded vs `13` integer days in `shen2024`).

## 5.5 282-Question Test Set & Kaggle Evaluation

Run on `data/raw/test_Q.csv` (282 questions, `top-docs=3`, `workers=4`, reusing `outputs/doc-routing/corpus_cache`, output in `outputs/doc-routing/test/predictions.csv`):

- **Coverage:** 258 answered / 24 abstained (`validate_rows` clean).
- **Token usage:** 24,323,365 total tokens across 590 API calls (**86,253.07 tok/q**; routing 7,392,077 + reading 16,931,288, median response 9.81s) — **8.0× fewer tokens** than Full-Context (689,804 tok/q).
- **Kaggle post-competition score (`ref 58606928`, 2026-11-20):** **0.885 Public / 0.841 Private** — highest single-model score on both Public and Private leaderboards (exceeding Full-Context + Citation Recovery `0.883 / 0.837`). Full table in [Kaggle test results](kaggle-test-results.md).

## 6. Reproduction Commands

```bash
# Stage 0: build the cleaned Markdown corpus and document cards
python3 -m wattbot.doc_routing build

# Stage 1: evaluate document routing recall on the train split
python3 -m wattbot.doc_routing eval-router \
  --questions data/raw/train_QA.csv \
  --output-dir outputs/doc-routing/router-eval \
  --top-docs 3

# Stage 2: full end-to-end run on the train split (add --limit N to smoke-test)
python3 -m wattbot.doc_routing run \
  --questions data/raw/train_QA.csv \
  --output-dir outputs/doc-routing/train \
  --top-docs 3

# Unit tests
python3 -m unittest discover -s tests
```
