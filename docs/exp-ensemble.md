# Experiment: Consensus Ensembles and Difficulty-Aware Cascaded Routing

All numbers below are taken directly from `outputs/cascade-ensemble/pareto-train-41q.md`,
`outputs/cascade-ensemble/test-diagnostics-282q.json` and
`outputs/cascade-ensemble/test-cascade-rag-fc-diag.json`.

## 1. Summary & Architecture

Two complementary systems were added on top of the existing RAG / Full-Context baselines.

**System 1 — Multi-Path Consensus Ensemble** (`wattbot/ensemble.py`). Aggregates any
number of worker prediction CSVs (optionally with traces) into one submission:

- Numeric equivalence at ±0.1% (`rel_tol = 0.001`) with complete-linkage clustering of
  answer values, so near-identical numeric answers merge but chains of weakly-similar
  answers cannot.
- Canonical `[low,high]` interval normalization before comparison, making range answers
  (`0.5–0.7`, `0.5 - 0.7`, `[0.5, 0.7]`) equivalent.
- Trace-informed, principled weighting: valid citations, verification acceptance /
  rejection counts, executable `calculation_binding`, and a penalty for dropped citations.
- Calibrated non-blank quorum (`--quorum-alpha`, `--singleton-alpha`) so a blank cluster
  cannot win merely by being the plurality.
- Citation reconciliation via majority / intersection / best strategies.

**System 2 — Difficulty-Aware Cascaded Router.** A cheap Tier 1 (optionally with a Tier 1
auxiliary run) gates each question; general automated escalation triggers promote hard
cases to Tier 2:

- Tier 1 abstained,
- citations missing or dropped,
- verification rejections above threshold,
- inter-run disagreement in the consensus.

Tier 2 is either Full-Context with general trace-based citation recovery, or a stronger
campus model (DeepSeek-V4-Pro) re-run as RAG. Every row carries exact per-question token
accounting across both tiers, enabling the Pareto analysis below. No trigger references a
question ID.

**Open-Source Multi-Model Runner.** `CampusModelClient` is an OpenAI-compatible client for
the `hkustgz-aigc` endpoint, supporting `DeepSeek-V4.1-Flash`, `DeepSeek-V4-Pro`,
`Qwen3.8-27B`, `GLM-5.3-Flash` and `Kimi-K3`, with exponential backoff and strict
`validate_rows` compliance.

**AGENTS.md compliance.** Zero hardcoded question IDs; only general automated rules. Every
emitted CSV is validated with `wattbot.pipeline.validate_rows`. All code and tests were
written and edited through
`opencode run -m hkustgz-aigc/DeepSeek-V4.1-Flash --auto`.

## 2. 41-Question Train Accuracy–Token Pareto Frontier

| System | Value Acc | Weighted | Cit. Jaccard | Abstention | Mean Tok/Q | Escalation | Pareto |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Closed-book Direct (Flash) | 36.59% | 0.3354 | 0.0488 | 53.66% | 251 | — | ✓ |
| Pi RAG (train-normalized) | 78.05% | 0.7933 | 0.8171 | 85.37% | 9,241 | — | ✓ |
| Pi RAG (field-semantics) | 75.61% | 0.7628 | 0.7683 | 80.49% | 10,074 | — |  |
| RAG (DeepSeek-V4-Pro) | 75.61% | 0.7713 | 0.7927 | 85.37% | 11,717 | — |  |
| RAG (DeepSeek-V4.1-Flash) | 73.17% | 0.7604 | 0.8252 | 87.80% | 12,227 | — |  |
| RAG (Qwen3.8-27B) | 73.17% | 0.7543 | 0.8008 | 85.37% | 13,486 | — |  |
| Cascade: Flash RAG -> Pro RAG (abstain) | 75.61% | 0.7848 | 0.8496 | 90.24% | 16,609 | 17.07% |  |
| Cascade: Flash+Pi_norm -> Pro RAG (zero-FC) | 82.93% | 0.8463 | 0.8780 | 92.68% | 27,595 | 26.83% | ✓ |
| Cascade: Flash+Qwen -> Pro RAG (zero-FC) | 80.49% | 0.8274 | 0.8740 | 92.68% | 31,909 | 29.27% |  |
| Consensus: 3-Model RAG (Flash+Pro+Qwen) | 78.05% | 0.7927 | 0.8130 | 85.37% | 37,430 | — |  |
| Consensus: 4-Cheap (3-Model RAG+Pi_norm) | 80.49% | 0.8171 | 0.8374 | 87.80% | 46,671 | — |  |
| Cascade: Flash RAG -> Full-Context (abstain) | 78.05% | 0.8134 | 0.8862 | 95.12% | 146,348 | 17.07% |  |
| Cascade: Flash+Pi_norm -> Full-Context | 85.37% | 0.8689 | 0.8902 | 95.12% | 222,654 | 26.83% | ✓ |
| Cascade: Flash+Pi_latest -> Full-Context | 85.37% | 0.8698 | 0.8963 | 95.12% | 223,486 | 26.83% | ✓ |
| Cascade: Flash RAG -> Full-Context (abstain+verif) | 78.05% | 0.8134 | 0.8862 | 95.12% | 230,173 | 29.27% |  |
| Cascade: Pro+Pi_norm -> Full-Context | 85.37% | 0.8698 | 0.8963 | 95.12% | 238,912 | 29.27% |  |
| Full-Context Baseline (Flash, raw) | 80.49% | 0.7988 | 0.6829 | 92.68% | 704,147 | — |  |
| Consensus: 4-Run Flash Baseline | 78.05% | 0.7994 | 0.8415 | 87.80% | 735,689 | — |  |
| Consensus: 4-Path Multi-Model (majority) | 82.93% | 0.8476 | 0.8862 | 92.68% | 737,333 | — |  |
| Consensus: 4-Path Multi-Model (intersection) | 82.93% | 0.8476 | 0.8862 | 92.68% | 737,333 | — |  |
| Consensus: 5-Path Multi-Model | 82.93% | 0.8341 | 0.8293 | 87.80% | 750,818 | — |  |

Key Pareto-optimal operating points:

- **Closed-book Direct (Flash):** 36.59% accuracy, 0.3354 weighted at 251 tok/q — the
  cheapest extreme of the frontier.
- **Pi RAG (train-normalized):** 78.05% (32/41), 0.7933 weighted at 9,241 tok/q — best
  accuracy-per-token among single-path RAG systems.
- **Cascade Flash+Pi_norm -> Pro RAG (zero full-context):** 82.93% (34/41), 0.8463
  weighted, 0.8780 Jaccard, 26.83% escalation at 27,595 tok/q. This is 25.5× cheaper
  (96.1% fewer tokens) than raw Full-Context and still **+0.0475 weighted over it**.
- **Cascade Flash+Pi_latest -> Full-Context:** 85.37% (35/41, 100% of the Union Oracle),
  0.8698 weighted, 0.8963 Jaccard, 95.12% abstention, 26.83% escalation at 223,486 tok/q.
  Saves 68.3% of tokens (3.15× cheaper) versus Full-Context while gaining +0.0710 weighted
  and +0.2134 Jaccard over it, and +0.1094 weighted over baseline RAG (0.7604) / +0.0765 over Pi RAG (0.7933).

Raw **Full-Context (0.7988 @ 704,147 tok/q)** is strictly Pareto-dominated by multiple
cascaded and cross-model systems: both the zero-FC cascade and the Flash+Pi_latest
Full-Context cascade are simultaneously cheaper and more accurate.

## 3. Cross-Model Open-Source Diversity

Single-model RAG on the identical `wattbot-index.json`:

| Model | Correct | Weighted | Mean Tok/Q |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | 30/41 | 0.7604 | 12,227 |
| DeepSeek-V4-Pro | 31/41 | 0.7713 | 11,717 |
| Qwen3.8-27B | 30/41 | 0.7543 | 13,486 |

The 3-model RAG-only **Union Oracle** is 34/41 (82.93%) — no single model exceeds 31/41,
so cross-model diversity is worth ~3 extra correct answers.

Replacing a correlated duplicate Pi run inside a 4-run consensus with DeepSeek-V4-Pro
raised the consensus weighted score from **0.7994 (32/41, Jaccard 0.8415)** to **0.8476
(34/41, Jaccard 0.8862)**, a gain of **+0.0482**. Genuine model/paradigm diversity prevents
shared-retrieval abstentions from outvoting valid minority answers.

A cheap 4-path zero-full-context consensus (Flash + Pro + Qwen + Pi_norm) reaches
**33/41 (80.49%), 0.8171 weighted at 46,671 tok/q** — 15.1× cheaper than Full-Context.

## 4. 282-Question Test Diagnostics & Cascade Generalization

From `test-diagnostics-282q.json`:

- **General trace citation recovery** restored **51/53** dropped-citation Full-Context test
  rows; valid cited answers rose from 206 to **257 of 259 answered (99.2%)**.
- Full-Context answered 259 / abstained 23; RAG answered 236 / abstained 46; Direct answered
  134 / abstained 148.
- **Pairwise RAG vs Full-Context agreement:** 81.56% overall value agreement, 90.13% on the
  233 jointly-answered questions (210/233), 89.72% abstention agreement, and 0.8465 mean
  citation Jaccard.
- Consensus diagnostics over 282 questions: 16 all-blank, 62 unanimous non-blank, 155
  majority cluster, 49 no-majority splits.

From `test-cascade-rag-fc-diag.json` (RAG -> Full-Context on abstain):

- Escalates **46/282 (16.31%)**, closely matching the 17.07% escalation seen on train.
- Cuts total tokens from **194,524,804 to 35,407,173** (125,557 tok/q), an **81.80%**
  reduction.
- Increases schema-valid cited answers from 236 to **262/282**, leaving 20 abstentions.
- Trigger breakdown: `abstain` 46. Resolutions: `tier1_accepted` 236,
  `escalated_tier2` 26, `escalated_abstained` 20.
- **100% `validate_rows` clean.**

## 5. Reproducibility, Ingesting Future Worker CSVs & Verification

Any worker prediction CSV is trivially ingested via `--run name=csv[::trace[::weight]]`:

```bash
# Consensus over arbitrary worker outputs
python3 -m wattbot.ensemble consensus \
  --questions data/raw/train_QA.csv --metadata data/raw/metadata.csv \
  --run flash=outputs/rag-flash.csv::outputs/rag-flash.jsonl::1.0 \
  --run pro=outputs/rag-pro.csv::outputs/rag-pro.jsonl::1.0 \
  --run pi=outputs/pi.csv::outputs/pi.jsonl::1.0 \
  --output outputs/cascade-ensemble/consensus.csv \
  --diagnostics outputs/cascade-ensemble/consensus-diag.json --gold data/raw/train_QA.csv

# Difficulty-aware cascade (Tier 1 -> Tier 2)
python3 -m wattbot.ensemble cascade \
  --questions data/raw/test_Q.csv --metadata data/raw/metadata.csv \
  --tier1 flash=outputs/rag-flash.csv::outputs/rag-flash.jsonl \
  --tier1-aux pi=outputs/pi.csv::outputs/pi.jsonl \
  --tier2 fc=outputs/full-ctx.csv::outputs/full-ctx.jsonl \
  --output outputs/cascade-ensemble/cascade.csv \
  --diagnostics outputs/cascade-ensemble/cascade-diag.json

# Run a new campus model end-to-end (OpenAI-compatible)
python3 -m wattbot.ensemble run-model \
  --model Qwen3.8-27B --questions data/raw/test_Q.csv \
  --metadata data/raw/metadata.csv --index data/processed/wattbot-index.json \
  --output outputs/rag-qwen.csv --trace outputs/rag-qwen.jsonl

# Unlabeled diagnostics and Pareto ranking
python3 -m wattbot.ensemble diagnose \
  --questions data/raw/test_Q.csv --metadata data/raw/metadata.csv \
  --run rag=outputs/rag-flash.csv::outputs/rag-flash.jsonl \
  --run fc=outputs/full-ctx.csv::outputs/full-ctx.jsonl \
  --output outputs/cascade-ensemble/test-diagnostics-282q.json

python3 -m wattbot.ensemble pareto \
  --gold data/raw/train_QA.csv --metadata data/raw/metadata.csv \
  --manifest outputs/cascade-ensemble/pareto-manifest.json \
  --output-md outputs/cascade-ensemble/pareto-train-41q.md
```

Saved, git-ignored artifacts under `outputs/cascade-ensemble/` include the Pareto manifest
and tables, all per-system prediction CSVs and traces, consensus/cascade diagnostics, the
282-question test diagnostics, and the cross-model RAG traces.

**Verification:** 29/29 unit tests pass — 15 new tests in `tests/test_ensemble.py` covering
interval normalization, equivalence clustering, weighting, quorum, reconciliation, cascade
triggers and Pareto ranking, plus the 14 pre-existing tests.
