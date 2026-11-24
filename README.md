# WattBot 2025 — RAG for AI environmental impact

Kaggle [WattBot 2025](https://www.kaggle.com/competitions/WattBot2025): build a
retrieval-augmented generation (RAG) system that answers questions about the
energy, carbon, water and sustainability costs of AI workloads, citing the
provided document corpus.

## Course project requirements (DSA5020)

This project is also the final assignment for DSA5020.

- **Report format**: at most 6 pages, double column, ACM format, including
  references and appendix. See the SIGMOD 2023 research track call for the
  reference format:
  <https://2023.sigmod.org/calls_papers_sigmod_research.shtml>
- **Contributions**: clearly indicate project contributions and the individual
  contributions of each team member in the report.
- **Group**: all members must choose the same group in the system. One member
  submits the report (an optional demo may be included).
- **Grading**: members of the same project may receive different points based on
  presentation performance and individual contributions.
- **Deadline**: 26 December.

## Repository layout

```
wattbot/           Python inference modules, shared retrieval, and scoring
wattbot/baselines/ direct and full-context baselines
pi_rag/            Node.js bridges for the Pi agents (Node.js >= 18)
demo/              Interactive offline Trace Explorer + live Q&A web app
report/            ACM-format LaTeX report sources and figures
docs/              delivery instructions and experiment records
tests/             unit and integration tests (208 tests)
data/raw/          competition CSVs (metadata.csv, train_QA.csv, test_Q.csv)
data/docs/         downloaded source PDFs (one per metadata.csv id)
data/related/      curated retrieval / agentic-RAG reference papers (+ PDFs)
scripts/           data prep, experiment comparison, verification, packaging
outputs/           generated submissions and reports (git-ignored)
LICENSE            MIT license for the source code
```

## Key results summary

All numbers below are Kaggle test-set scores for the single open-weight model
`DeepSeek-V4.1-Flash`. The competition winner (KohakuRAG) scored Public `0.902`
/ Private `0.861`; see the [full 32-submission table and Pareto analysis](docs/kaggle-test-results.md).

| Method | Public | Private | tok/q | Note |
| --- | ---: | ---: | ---: | --- |
| **4-paradigm no-FC consensus (Clean v2)** | **0.904** | 0.864 | 148.0k | Highest Public in the project, above KohakuRAG `0.902` |
| **Pure-RAG cascade `pi_on(v2)+mm → doc`** *(main delivery)* | **0.895** | **0.871** | **77.2k** | Cheapest high-value point; 8.7× cheaper than Full-Context |
| Pure-RAG cascade `pi_on+mm → doc` (v1 index) | 0.895 | 0.872 | 79.7k | Previous main point |
| Pure-RAG 3-way consensus `mhop+mm+pi_off` | 0.888 | **0.865** | **44.5k** | Beats the winner's Private at 44.5k tok/q |
| 5-paradigm `q0.5` → multimodal cascade | 0.896 | **0.891** | 840.8k | Highest Private in the project (Full-Context upper bound) |
| *KohakuRAG (competition winner)* | *0.902* | *0.861* | — | Reference on the leaderboard |

## Method entry points

Run these commands from the repository root with the project Python environment.
Append `--help` to see the required flags. Each method writes to its own output
directory or files.

| Method | CLI entry point | Inputs and experiment notes |
| --- | --- | --- |
| **Main Pure-RAG cascade** | `python -m wattbot pure-rag-cascade` | **Recommended delivery.** Tier-1 `Pi Agent v2 (thinking)` + multimodal, escalating only disagreed/abstained questions to Tier-2 `doc_routing`. **0.895 Public / 0.871 Private** @ 77.2k tok/q on the Clean v2 index (0.895 / 0.872 @ 79.7k with the v1 index). [Cascade design](docs/wattbot-delivery.md) · [All Kaggle test scores](docs/kaggle-test-results.md) |
| Single-stage cascade | `python -m wattbot cascade` | Clean v2 text index → multimodal fallback on abstention (no Tier-2 routing). **0.858 / 0.815**. [Delivery notes](docs/wattbot-delivery.md) |
| Chunk RAG | `python -m wattbot run` | Uses v1 by default. Pass `--index outputs/parser-fix/wattbot-index-v2.json` for clean v2. [Index experiment](docs/exp-parser-fix.md) |
| Standalone multimodal | `python -m wattbot.multimodal run --mode mode2-only` | Uses the v1 index and source PDFs. [Visual ablation](docs/wattbot-delivery.md#多模态消融) |
| Document routing | `python -m wattbot.doc_routing run` | Uses document cards and extracted per-document Markdown. [Document routing experiment](docs/exp-doc-routing.md) |
| Multihop retrieval | `python -m wattbot.multihop_rag run` | Uses its own contextual index. [Multihop experiment](docs/exp-multihop-rag.md) |
| Pi agent | `python -m wattbot.pi_agent` or `python -m wattbot.pi_agent_v2` | Requires Node.js >= 18; install the dependencies with `npm ci --prefix pi_rag`. [Pi v1](docs/wattbot-pi-rag.md) · [Pi v2](docs/exp-pi-v2.md) |
| Ensemble and cascade replay | `python -m wattbot.ensemble` | Combines saved prediction CSVs and traces. [Ensemble experiment](docs/exp-ensemble.md) |
| Direct and full-context baselines | `python -m wattbot.baselines.direct` or `python -m wattbot.baselines.full_context` | [Direct experiment](docs/wattbot-direct-baseline.md) · [Full-context experiment](docs/wattbot-full-context-baseline.md) |
| Interactive demo | `python demo/app.py --host 127.0.0.1 --port 8501` | Offline Trace Explorer + live single-question Q&A. See [demo/README.md](demo/README.md). |
| Submission packaging | `python3 scripts/package_dsa5020_submission.py` | Builds the graded ZIP (report PDF, code, dataset, demo outputs). |

```bash
# Main pipeline: run it end to end
.venv/bin/python -m wattbot index                 # v1 index
.venv/bin/python -m wattbot.clean_index           # clean v2 index
.venv/bin/python -m wattbot pure-rag-cascade \
  --pi-workers 4 --multimodal-workers 2 --tier2-workers 2 \
  --output-dir outputs/pure-rag-cascade
.venv/bin/python -m wattbot verify --output outputs/pure-rag-cascade/predictions.csv
```

## Interactive demo

`demo/app.py` is a self-contained presentation app that needs only the Python
standard library plus `pymupdf` (already in `requirements-rag.txt`). It has two
tabs:

- **Trace Explorer** — fully offline. Reads the saved cascade predictions,
  per-stage JSONL traces and PDF corpus and renders the three-stage pipeline
  (Pi v2 tool calls + Decimal AST sandbox, multimodal VLM with rasterised page
  thumbnails, Tier-2 doc routing), escalation reasons, and a token-cost bar
  against the Full-Context baseline.
- **Live Q&A** — runs the real server-side pipeline for a custom question with a
  selectable solver. Requires API credentials and pre-built indexes.

```bash
.venv/bin/python demo/app.py --host 127.0.0.1 --port 8501
# open http://127.0.0.1:8501
```

See [demo/README.md](demo/README.md) for the endpoints, filter chips and
presentation quick picks. The demo tests run in the normal unit-test suite.

## Report and submission packaging

The 6-page ACM-format report lives in `report/`:

```bash
cd report && tectonic main.tex        # -> report/main.pdf
```

`scripts/package_dsa5020_submission.py` assembles the graded ZIP:

```bash
python3 scripts/package_dsa5020_submission.py
# -> outputs/DSA5020_WattBot2025_Submission.zip
python3 scripts/package_dsa5020_submission.py --no-pdfs
# -> lightweight archive without data/docs/*.pdf, safe for public code sharing
```

The archive contains the compiled report PDF, `report/` LaTeX sources, `code/`
(source, tests, docs, demo), `dataset/` (competition CSVs and optionally the
PDFs), `demo_outputs/` (verified prediction CSVs) and a generated
`README_SUBMISSION.md`. The source code is MIT licensed (`LICENSE`); the
competition data stays subject to its non-commercial, no-redistribution terms,
so use `--no-pdfs` (and drop `dataset/`) before any public release.


## Data

```bash
kaggle competitions download -c WattBot2025 -p data/raw
unzip -o data/raw/WattBot2025.zip -d data/raw
python3 scripts/download_docs.py                 # 32 source documents -> data/docs
python3 scripts/collect_retrieval_papers.py      # optional: related papers
```

Kaggle credentials are read from `~/.kaggle/access_token`.

## Competition rules and constraints

Source: competition Description / Evaluation / Rules pages (WattBot 2025).
Re-check the current rules page before each competition, since WattBot 2026 may
change them.

### Models

- Any model is allowed: open or closed, local or API, any LLM, embedding model
  or reranker.
- Open-source methods are preferred, but hosted APIs (OpenAI, OpenRouter, etc.)
  are permitted.
- The system must be fully automated and produced by code.
- Manual lookup is not allowed as the solution: do not read the papers by hand
  or manually prompt ChatGPT and paste answers into the submission. Interactive
  experimentation during development is fine; the final approach must be
  automated and scale to the full benchmark.

### Submission environment

- File-based CSV competition, not a Code Competition. Run the pipeline anywhere
  (local, server, Colab) and upload one CSV via **Make Submission**.
- No runtime, offline, or dependency restrictions are imposed by the platform;
  only the submitted CSV is evaluated.
- Only three data files are provided (`metadata.csv`, `train_QA.csv`,
  `test_Q.csv`). There is no `sample_submission.csv`; `train_QA.csv` defines the
  schema. No preprocessed text is provided: participants must download, parse,
  chunk, embed and retrieve from the documents themselves.
- Up to 5 submissions per day. At the deadline, the 3 most recent submissions
  are carried forward and the highest-scoring one counts as the final result.
- The public leaderboard uses a subset of the test set; the final ranking uses a
  hidden portion.

### Submission format

One CSV with the same columns as `train_QA.csv`:

```
id, question, answer, answer_value, answer_unit, ref_id, ref_url,
supporting_materials, explanation
```

- `answer`: natural-language answer. If unanswerable, use
  `Unable to answer with confidence based on the provided documents.`
- `answer_value`: normalized numeric/categorical value, e.g. `1438`,
  `Water consumption`, `1`; ranges as `[low,high]`; no `<`, `>`, `~`.
- `answer_unit`: measurement unit, e.g. `lbs`, `kWh`, `percent`.
- `ref_id`: one or more `metadata.csv` document IDs.
- `ref_url`: URL(s) of the cited documents.
- `supporting_materials`: verbatim quote, table or figure reference.
- `explanation`: short reasoning linking evidence to the answer.
- For unanswerable questions, set `answer_value`, `answer_unit`, `ref_id`,
  `ref_url` and `supporting_materials` to `is_blank`.

### Scoring

Weighted accuracy in [0, 1]:

| Component | Weight | Correctness |
|---|---:|---|
| `answer_value` | 0.75 | numeric within ±0.1% relative tolerance, categorical exact after normalization |
| `ref_id` | 0.15 | Jaccard overlap with the ground-truth ID set |
| `is_NA` | 0.10 | unanswerable questions must have every relevant field set to `is_blank` |

Only `answer_value` and `ref_id` are scored; the other fields are expected but
not evaluated.

### Data license

The dataset is for non-commercial use only: competition participation and
academic research. It may not be redistributed or republished.

### Open source

Winning teams must release their full solution under an open-source license that
permits reuse.

## Minimal submission

```bash
python3 scripts/make_submission.py                 # all is_blank fallback
python3 scripts/make_submission.py --mode unit_hint
kaggle competitions submit -c WattBot2025 \
  -f outputs/submission_minimal.csv -m "schema smoke test"
```

`outputs/submission_minimal.csv` is the smallest schema-valid submission. It can
only score on the `is_NA` component and exists to validate the upload path.

## Reference implementation

KohakuRAG (WattBot 2025 winner) is used as a design reference:
<https://github.com/Kohaku-Lab/KohakuRAG>

- Embeddings: Jina v3 (text) or Jina v4 (text + image).
- Answer/planner LLM: hosted via OpenRouter; ensemble across models.
- Retrieval: hierarchical document tree, dense + optional BM25, multi-query
  planning, dedup and frequency/score reranking.
- Flow: fetch -> index -> answer -> validate.

## KohakuRAG-inspired pipeline

The pipeline uses only the PDF sources listed in `metadata.csv`. It indexes
text blocks and detected table rows under their parent pages. At inference
time, DeepSeek plans alternative queries, the retriever combines ranks across
queries, and matching blocks can be expanded with nearby page text. BM25 is the
default. Add `--hybrid` to also rank passages using the local BGE encoder.
BGE encodes text for retrieval. All generative calls use DeepSeek Flash 1017.

The default backend uses `DeepSeek-V4.1-Flash` on HKUST(GZ) AIGC, the campus
alias the team confirmed for Flash 1017. The gateway returns the alias and
server fingerprint. The CLI reads the `hkustgz-aigc` key from the local
OpenCode config or `HKUSTGZ_AIGC_API_KEY` from the environment or repository
`.env`. To use the dated OpenRouter model ID, pass `--backend openrouter` with
`OPENROUTER_API_KEY` configured.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-rag.txt
.venv/bin/python -m wattbot index

.venv/bin/python -m wattbot run --questions data/raw/train_QA.csv --workers 4 \
  --output outputs/rag-train.csv --trace outputs/rag-train.jsonl
.venv/bin/python -m wattbot score --output outputs/rag-train.csv \
  --metrics outputs/rag-train-metrics.json

.venv/bin/python -m wattbot run --workers 4 \
  --output outputs/submission_rag.csv --trace outputs/rag-test.jsonl
.venv/bin/python -m wattbot verify --output outputs/submission_rag.csv
.venv/bin/python -m unittest discover -s tests   # 208 tests
```

The index records source PDF hashes and per-document extraction counts. Each
question's JSONL trace records planned queries, retrieval ranks and passages,
model requests and responses, cited quotes, verification decisions, retries
and the final CSV row. It contains corpus excerpts and stays in the ignored
`outputs/` directory. The key is read only for API authentication.

The pipeline checks each cited quote against the retrieved text and maps cited
IDs to metadata URLs. The same DeepSeek model separately audits whether the
citations support the candidate answer. If support is missing, the pipeline
retries with more context before abstaining. Verify checks the CSV schema,
question coverage and order, citation mappings, and the five required
`is_blank` fields on abstentions. Score is a local diagnostic based on the
published weights.

Use `retrieve` with the same question and index options to inspect retrieval
without calling the LLM. A run with `--limit` is a smoke test and writes only
the requested prefix. After an interrupted run, repeat the original command
with the same `--trace` and add `--resume` to reuse completed rows. Keep the
same input files, model backend, retrieval settings, and pipeline code.

To replay a completed trace, check its input and output hashes, and verify
that each accepted answer has quotes in the source index, run:

```bash
.venv/bin/python -m scripts.verify_rag_trace \
  --trace outputs/rag-test.jsonl --output outputs/submission_rag.csv
```

Earlier traces may contain `True` or `False` as answer values. The current
pipeline writes `1` or `0` for explicit true/false questions, following
`train_QA.csv`. It also converts explicit magnitude prefixes such as
`billion parameters` to the hinted base unit when the question does not
request the scaled unit, and converts a missing numeric answer to the
required abstention. For a completed older trace, pass
`--canonical-output outputs/submission_rag_verified.csv` to the replay
command. This produces a separate CSV and a normalization audit without
re-running inference.

## Main Pure-RAG cascade

`python -m wattbot pure-rag-cascade` (also `python -m wattbot.pure_rag_cascade`)
is the delivered system. It is fully automated and needs no Full-Context pass:

1. **Tier 1A** — `Pi Agent v2` with thinking on the clean v2 index: tool calls
   (`search_corpus`, `read_page`, `eval_math`) and a Decimal AST sandbox.
2. **Tier 1B** — multimodal `mode2-only` VLM reads cited PDF pages when text
   abstains or the question is explicitly visual.
3. **Consensus gate** — when Tier-1A and Tier-1B agree, accept (fast path).
4. **Tier 2** — `doc_routing` runs *only* on disagreed or abstained questions
   (24.1% at test time), routing top-3 documents and re-answering with
   adaptive expansion.

```bash
.venv/bin/python -m wattbot pure-rag-cascade \
  --pi-workers 4 --multimodal-workers 2 --tier2-workers 2 \
  --output-dir outputs/pure-rag-cascade
.venv/bin/python -m wattbot verify --output outputs/pure-rag-cascade/predictions.csv
```

`--thinking` defaults to `true`; pass `--pi-index` / `--visual-index` when the
indexes live elsewhere, `--limit` for a smoke test, and `--resume` to continue an
interrupted run. The output directory contains:

```
predictions.csv          final 282-row submission
diagnostics.json         per-stage tokens, escalation reasons, metrics
tier1_pi/                Pi Agent v2 stage (predictions.csv + trace.jsonl)
tier1_multimodal/        multimodal VLM stage (predictions.csv + trace.jsonl)
tier2_doc_routing/       escalation-only doc-routing stage (trace.jsonl)
```

## Clean-index RAG with multimodal fallback

The [system and experiment record](docs/wattbot-delivery.md) describes the
inference stages, evaluation protocol, and historical alternatives.

> This is the earlier **single-stage** cascade (`0.858 Public / 0.815 Private`).
> The recommended delivery pipeline is the **Pure-RAG cascade** described under
> [Method entry points](#method-entry-points)
> (`python -m wattbot pure-rag-cascade`, `0.895 / 0.871` @ 77.2k tok/q).

`wattbot cascade` runs text RAG on the clean v2 index. If text RAG abstains,
the question goes to the multimodal solver in `mode2-only`, which can inspect
the retrieved PDF pages when its text pass cannot answer. The multimodal solver
uses the original v1 index. The command validates the final CSV against the input
question order, source metadata, and the required abstention fields.

```bash
.venv/bin/python -m pip install -r requirements-rag.txt
.venv/bin/python -m wattbot index
.venv/bin/python -m wattbot.clean_index
.venv/bin/python -m wattbot cascade --questions data/raw/train_QA.csv --workers 2 \
  --output outputs/integration-cascade/train/predictions.csv \
  --trace outputs/integration-cascade/train/trace.jsonl
.venv/bin/python -m wattbot score --output outputs/integration-cascade/train/predictions.csv
```

Omit `--questions` to use the test split. Use `--index` for the clean v2 index
and `--visual-index` for the v1 index when they are stored elsewhere. The
multimodal stage defaults to the campus OpenCode provider and DeepSeek V4.1
Flash. After an interruption, repeat the same command with `--resume` to reuse
completed rows. Inspect `cascade_route` and the `pipeline_stage` field in the
trace to audit which questions reached the multimodal solver.

## Direct LLM baseline

The closed-book baseline lives in `wattbot/baselines/direct.py`. It uses the
same HKUST(GZ) DeepSeek model as RAG and sends the question and the provided
unit hint. Predictions keep the benchmark column names, with citation fields
set to `is_blank`. Use them for local comparison with the RAG run.

```bash
.venv/bin/python -m wattbot.baselines.direct --split train --workers 8
.venv/bin/python -m wattbot.baselines.direct --split test --workers 16
```

Each split saves a config, JSONL trace, predictions, and a summary under
`outputs/baseline-direct/<split>/`. The train split also saves per-question
scores in `metrics.json`. Use `--output-dir` for a new run or repeat the
original command with `--resume` to reuse completed rows. See the
[baseline experiment record](docs/wattbot-direct-baseline.md) for results and
the comparison protocol.

## Full-context LLM baseline

This baseline sends the same 32 documents in fixed document and page order
with each question. It uses all extracted page text and table rows from the
existing index, without ranking, query rewriting, or a second LLM verifier.
It checks the returned quotation against the named page. The text protocol
does not include rendered figure images.

```bash
.venv/bin/python -m wattbot index
.venv/bin/python -m wattbot.baselines.full_context --split train --limit 1
.venv/bin/python -m scripts.verify_direct_baseline outputs/baseline-full-context/train-first-1
.venv/bin/python -m scripts.summarize_token_usage outputs/baseline-full-context/train-first-1/trace.jsonl
```

For a full split run, omit `--limit` and use `--split train` or `--split test`.
Each request includes the complete extracted corpus. The output directory
stores `corpus.txt` once and references its SHA-256 in each request event,
alongside `config.json`, `predictions.csv`, `trace.jsonl`, and `summary.json`.
Training runs also save `metrics.json`. All of these outputs are Git ignored.

## Pi tool-using RAG

The Pi experiment uses the existing competition index and the HKUST(GZ)
DeepSeek Flash 1017 model. Python provides and validates the tools.
`pi_rag/agent.mjs` runs the conversation in Pi Agent Core. The agent starts
with passages retrieved by the original question and can submit a cited
answer for the existing DeepSeek verifier to check.

```bash
npm ci --prefix pi_rag
.venv/bin/python -m wattbot.pi_agent --mode calculator --split train --workers 6
.venv/bin/python -m wattbot.pi_agent --mode pi --split train --workers 6
.venv/bin/python -m scripts.verify_rag_trace --questions data/raw/train_QA.csv \
  --trace outputs/pi-rag/pi/train/trace.jsonl --output outputs/pi-rag/pi/train/predictions.csv
```

Both variants can run a Python `Decimal` calculator with cited operands.
The `pi` variant additionally exposes corpus search and page reading. Source
access is limited to the competition index. Runs keep the settings,
predictions, per-question training scores, model calls, tool arguments and
results under `outputs/pi-rag/`. For a fresh run, choose a new
`--output-dir`. See the [Pi experiment results](docs/wattbot-pi-rag.md).
