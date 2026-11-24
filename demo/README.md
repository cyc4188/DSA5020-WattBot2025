# WattBot 2025 — Interactive Demo

A self-contained demo web app for the DSA5020 presentation. It uses **only the
Python standard library** for the server (`http.server`), plus `pymupdf` — which
is already installed in `.venv` — to rasterise cited PDF pages on demand. No
Streamlit, Gradio, Flask, FastAPI or extra `pip install` is required.

There are two modes:

1. **Trace Explorer (recommended for the live presentation).** Reads the saved
   cascade predictions, per-stage JSONL traces and PDF corpus from `outputs/`
   and `data/`. Completely offline and zero-risk: no API keys are needed.
2. **Live Q&A.** Runs the real server-side `wattbot` pipeline for a custom
   question with a selectable solver. Requires API credentials and pre-built
   indexes.

## Launch

From the project root:

```bash
.venv/bin/python demo/app.py --host 127.0.0.1 --port 8501
```

Open <http://127.0.0.1:8501>. Use `--port 0` to let the OS pick an ephemeral
port (the chosen URL is printed at startup). Stop with `Ctrl+C`.

If the app is started from a fresh checkout without `outputs/`, the Trace
Explorer still starts and shows empty/partial states gracefully. Download the
corpus and generate the artefacts first for the full experience:

```bash
python3 scripts/download_docs.py
# ... run the cascade / stage pipelines documented in the top-level README ...
```

## What the Trace Explorer shows

### KPI banner

The header summarises the submitted cascade's Pareto performance:

- Public / Private weighted score: **0.895 / 0.872**
- Mean cost: **79.7k tokens/question**
- **8.7× cheaper** than the Full-Context baseline (689.8k tokens/question)
- **23.4%** Tier-2 escalation rate
- **+0.011 Private** over the KohakuRAG 1st-place result

### Filters, search and showcase picks

The left rail lists every question discovered from `test_Q.csv`, `train_QA.csv`
and the saved traces. Filter chips narrow the list to:

- `All`
- `Escalated (any)`
- `Disagreement`
- `Abstention trigger`
- `Visual / chart`
- `Decimal AST`
- `Abstained (is_blank)`

The search box matches question IDs and question text. **Quick picks** jump to
the best presentation cases:

| ID | Why it is interesting |
| --- | --- |
| `q015` | Chart/table recovery: the text-only passes abstain, the multimodal VLM reads Table 1 and recovers the answer. |
| `q189` | Figure reading: AlexNet 2012 top-1 accuracy recovered from Figure 4a. |
| `q317` | Chart-only value: execution time read from a stacked bar in Fig. 4. |
| Next AST/disagreement/abstained IDs | Auto-detected from the saved traces so the picks never go stale. |

### Three-stage pipeline inspector

Selecting a question renders:

1. **Tier-1 Path A — Pi Agent v2**
   - Step-by-step `search_corpus`, `read_page`, `eval_math` and
     `submit_answer` tool calls with arguments and result previews.
   - A highlighted **Decimal AST sandbox** card showing the expression, bound
     operands (name, value, source quote) and result whenever `eval_math` was
     used.
   - Reasoning-token and total-token counts, verifier outcome and the candidate
     answer.
2. **Tier-1 Path B — Multimodal VLM**
   - Whether vision was triggered and why (e.g. `text_abstained`,
     `explicit_visual_question`).
   - Inline PNG thumbnails of the actual cited/inspected PDF pages, rendered by
     `/api/page-image` through `pymupdf`. Click a thumbnail to zoom.
   - Text-pass decision, visual candidates, decisions and the recovered answer.
3. **Tier-2 — Doc Routing**
   - Whether the consensus gate accepted Tier-1 (fast path) or escalated.
   - Escalation reasons (`disagreement` / `abstain`) and the resolution
     (`tier1_accepted`, `escalated_consensus`,
     `escalated_fallback_tier1_aux`, ...).
   - Top-3 routed documents, BM25 and LLM document choices, router reasoning,
     adaptive-expansion retries, Stage-2 abstention reasons and the Tier-2
     candidate.

Below the stages, a **token-cost bar** compares this question's Tier-1A, Tier-1B,
Tier-2 and cascade totals against the 689,804-token Full-Context baseline.

## Live Q&A tab

Choose a solver and ask any sustainability question:

| Solver | Backing API |
| --- | --- |
| Clean v2 Text RAG | `wattbot.pipeline.solve` + `SearchIndex` on the clean v2 index |
| Multimodal mode2-only | `wattbot.multimodal.solve_multimodal` |
| Doc Routing | `wattbot.doc_routing.solve_question` |

The form returns the structured answer row, citations, trace steps and token
usage. If credentials or indexes are missing, the UI shows a clear error and a
reminder that the Trace Explorer is fully offline.

Live mode expects the same environment as the CLI:

- an API key in `.env` / the OpenCode configuration (`HKUSTGZ_AIGC_API_KEY`);
- `data/processed/wattbot-index.json` and/or
  `outputs/parser-fix/wattbot-index-v2.json`;
- `outputs/doc-routing/corpus_cache/{markdown_corpus,document_cards}.json` for
  Doc Routing.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Single-page dashboard |
| `GET` | `/api/overview` | KPIs, counts, dataset availability, quick picks |
| `GET` | `/api/questions` | Question list with filter metadata |
| `GET` | `/api/question/<qid>` | Full three-stage breakdown for one question |
| `GET` | `/api/page-image?ref_id=...&page=...` | PNG of a corpus PDF page (strict ref-id validation) |
| `POST` | `/api/ask` | Live single-question Q&A (`question`, `solver`, `unit_hint`) |
| `GET` | `/api/health` | Liveness check |

## Tests

```bash
.venv/bin/python -m unittest tests/test_demo.py
.venv/bin/python -m unittest discover -s tests
```

The demo tests start the server on an ephemeral port and cover the HTML page,
overview/questions APIs, the full three-stage question payload, PNG rendering,
path-traversal/invalid-input rejection and live-request validation. A synthetic
temporary fixture keeps the demo tests green even on a checkout without
`outputs/`; the real-output checks are skipped only when those artefacts are
absent.
