# Experiment: Train Replay of the 5-Paradigm Quorum→Multimodal Cascade, and Pi Agent v2 on the Clean v2 Index

**Track:** `pi-v2-clean-v2`
**Status:** Complete. Offline 41-question train replay (zero API calls) plus
`Pi Agent v2 (thinking=true)` runs on both `train_QA.csv` (41 q) and
`test_Q.csv` (282 q) against `outputs/parser-fix/wattbot-index-v2.json`.
No Kaggle submissions were made and no other agents' files were modified.

All artifacts live under `outputs/pi-v2-clean-v2/`. Machine-readable metrics:
`outputs/pi-v2-clean-v2/summary.json`.

---

## 0. Summary of Results

| Experiment | Result |
|---|---|
| **5-Paradigm Quorum (`q0.5`) → Multimodal Cascade, 41 q train** | **0.9018 weighted**, 36/41 value acc, 0.9553 citation Jaccard, 100% abstention, 850,908 tok/q (fills the `---` in Table 1) |
| Pi Agent v2 (thinking) on Clean v2 index, 41 q train | **0.8726** weighted (was 0.8866 on v1 index), 35/41 value acc, 0.9146 Jaccard, **34,326 tok/q** (was 40,414) |
| Main Pure-RAG Cascade `pi_on_v2+mm → doc`, 41 q train | **0.9012** weighted (was 0.8970 with `pi_on_v1`), 36/41 value acc, 0.9512 Jaccard |
| Pi Agent v2 (thinking) on Clean v2 index, 282 q test | **255 answered / 27 blank**, 24 values changed vs v1, **40,205 tok/q** (was 43,182) |
| Updated test ensembles | 3 candidate CSVs built and `wattbot verify`-clean (paths in §6) |

The Clean v2 index **lowers single-path value accuracy by one train question**
(`q124`, table value 5,439,000 → prose value 5,400,000) but **raises citation
quality** (+0.028 Jaccard) at **15% fewer tokens**. In the cascade the citation
and blank-handling gains slightly outweigh the single-path loss
(**+0.0042 weighted**), and the new test cascade answers one more question
(268 vs 267) while abstaining one fewer (14 vs 15).

---

## 1. Compatibility Fix for the Clean v2 Index

`CleanSearchIndex` rejected any index whose `version` was not `1`; the Clean v2
index is `version: 2`. Minimal fix in `wattbot/pi_agent_v2.py`:

```python
# before: if data["version"] != 1:
if data.get("version") not in (1, 2):
    raise ValueError("Rebuild the index for this version of the pipeline")
# before: extra["kind"] == "table_row"
if extra["page_id"] == page_id and extra["kind"] in ("table_row", "table_block"):
```

The version-2 index has identical `sources/pages/nodes` structure plus 183
`table_block` nodes. This mirrors `wattbot/retrieval.py`, which already accepts
`version in (1, 2)`. `python -m unittest tests.test_pi_agent_v2` passes 27/27.

---

## 2. Offline 41-q Train Replay: 5-Paradigm Quorum (`q0.5`) → Multimodal

The five saved 41-question train runs live in the swarm worktrees (they were
git-ignored in the main checkout):

| Paradigm | CSV / trace |
|---|---|
| `pi_on` | `swarm-pi-v2-sandbox-*/outputs/pi-v2-sandbox/pi-v2-thinking-on/train/{predictions.csv,trace.jsonl}` |
| `doc` | `swarm-doc-routing-*/outputs/doc-routing/train/{predictions.csv,trace.jsonl}` |
| `fc_recov` | `swarm-parser-fix-*/outputs/parser-fix/full-context-recovered/train/predictions.csv` + `outputs/baseline-full-context/train/trace.jsonl` |
| `mhop` | `swarm-multihop-retrieval-*/outputs/multihop-retrieval/train/{predictions.csv,trace.jsonl}` |
| `mm` | `swarm-multimodal-vlm-*/outputs/multimodal/train-mode2/{predictions.csv,trace.jsonl}` |

```bash
OUT=outputs/pi-v2-clean-v2/train-5par-q05-mmcascade
python -m wattbot.ensemble consensus \
  --questions data/raw/train_QA.csv --metadata data/raw/metadata.csv \
  --run pi_on=.../train/predictions.csv::.../train/trace.jsonl \
  --run doc=.../train/predictions.csv::.../train/trace.jsonl \
  --run fc_recov=.../full-context-recovered/train/predictions.csv::outputs/baseline-full-context/train/trace.jsonl \
  --run mhop=.../train/predictions.csv::.../train/trace.jsonl \
  --run mm=.../train-mode2/predictions.csv::.../train-mode2/trace.jsonl \
  --quorum-alpha 0.5 --singleton-alpha 0.5 --index data/processed/wattbot-index.json \
  --output $OUT/consensus-q05.csv --diagnostics $OUT/consensus-q05.diag.json \
  --gold data/raw/train_QA.csv

python -m wattbot.ensemble cascade \
  --questions data/raw/train_QA.csv --metadata data/raw/metadata.csv \
  --tier1 consensus=$OUT/consensus-q05.csv \
  --tier2 mm=.../train-mode2/predictions.csv::.../train-mode2/trace.jsonl \
  --no-escalate-missing-citation --no-escalate-disagreement \
  --output $OUT/predictions.csv --diagnostics $OUT/cascade.diag.json \
  --gold data/raw/train_QA.csv
```

**Results (exact):**

| Stage | Answered / Blank | Value acc | Citation Jaccard | Abstention acc | Weighted |
|---|---:|---:|---:|---:|---:|
| Consensus (`q0.5/s0.5`) | 38 / 3 | 35/41 (0.8537) | 0.9309 | 0.9756 | 0.8774 |
| **→ Multimodal cascade on blanks** | **39 / 2** | **36/41 (0.8780)** | **0.9553** | **1.0000** | **0.9018** |

- Escalation: 3/41 (7.32%), all trigger `abstain`; `q215` is recovered by `mm`,
  while the two gold-blank questions `q062` and `q164` correctly stay blank.
- Tokens: consensus sum-of-components **850,908 tok/q** (34,887,217 total),
  plus 128,389 cascade escalation tokens = **854,039 tok/q** end-to-end. This is
  the same accounting convention as the 840,795 tok/q test row.

Per-run token totals: `fc_recov` 28,870,040 · `doc` 3,483,986 · `pi_on`
1,656,974 · `mm` 496,123 · `mhop` 380,094.

---

## 3. Pi Agent v2 (thinking) on the Clean v2 Index — 41-q Train

Smoke test (2 q) first; dense cache was built once and reused
(`outputs/parser-fix/wattbot-index-v2.bge-small.<sha>.npy`), 2/2 correct.

```bash
python -m wattbot.pi_agent_v2 --mode pi --split train --thinking true \
  --index outputs/parser-fix/wattbot-index-v2.json \
  --workers 4 --output-dir outputs/pi-v2-clean-v2/train
```

| Config | Weighted | Value acc | Citation Jaccard | Abstention | False blank | Total tokens | Tok/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pi v2 thinking, **v1 index** (old) | 0.8866 | 36/41 (0.8780) | 0.8862 | 0.9512 | 2/39 | 1,656,974 | 40,414 |
| **Pi v2 thinking, Clean v2 index** | **0.8726** | **35/41 (0.8537)** | **0.9146** | 0.9512 | 2/39 | 1,407,349 | **34,326** |

- Only 2 answer values changed: `q078` (still wrong, different normalization)
  and **`q124` (5,439,000 → 5,400,000, the only correctness loss)**. `q124`
  answered from a prose summary on the Clean v2 index instead of the exact
  table row. `q263` citation moved `luccioni2024 → chung2025`.
- Tool calls: `search_corpus` 59, `read_page` 8, `eval_math` 10,
  `list_documents` 3; 128 API calls, 24,231 reasoning tokens.

### 3.1 Replayed Main Pure-RAG Cascade (`pi_on_v2 + mm → doc_routing`)

```bash
python -m wattbot.ensemble cascade \
  --questions data/raw/train_QA.csv --metadata data/raw/metadata.csv \
  --tier1 pi_on_v2=outputs/pi-v2-clean-v2/train/predictions.csv::outputs/pi-v2-clean-v2/train/trace.jsonl \
  --tier1-aux mm=.../train-mode2/predictions.csv::.../train-mode2/trace.jsonl \
  --tier2 doc=.../doc-routing/train/predictions.csv::.../doc-routing/train/trace.jsonl \
  --output outputs/pi-v2-clean-v2/train-main-cascade/predictions.csv \
  --diagnostics outputs/pi-v2-clean-v2/train-main-cascade/cascade.diag.json \
  --gold data/raw/train_QA.csv
```

| Tier-1 source | Weighted | Value acc | Citation Jaccard | Abstention | Escalation | Tok/q |
|---|---:|---:|---:|---:|---:|---:|
| `pi_on_v1` (old replay, reproduces report 0.8970) | 0.8970 | 36/41 | 0.9228 | 1.000 | 9/41 (21.95%) | 74,503 |
| **`pi_on_v2` Clean v2** | **0.9012** | **36/41** | **0.9512** | **1.000** | 9/41 (21.95%) | **68,414** |

The cascade recovers `q124` (lost by the single path) via the multimodal/aux
route, so the final value accuracy stays 36/41 while citation Jaccard and tokens
both improve relative to the v1 cascade.

---

## 4. Pi Agent v2 (thinking) on the Clean v2 Index — 282-q Test

```bash
python -m wattbot.pi_agent_v2 --mode pi --split test --thinking true \
  --index outputs/parser-fix/wattbot-index-v2.json \
  --workers 4 --output-dir outputs/pi-v2-clean-v2/test
python -m wattbot verify --output outputs/pi-v2-clean-v2/test/predictions.csv
```

- `wattbot verify`: **282/282 rows valid**, question order and schema intact.
- 255 answered / 27 blank (v1: 253 / 29). 946 API calls, 131,938 reasoning
  tokens, `search_corpus` 416, `read_page` 102, `eval_math` 86.
- **40,205 tok/q** (11,337,823 total) versus **43,182 tok/q** for the v1-index run
  — **6.9% cheaper**.

### 4.1 Diff vs old `pi_on_v1` (282 q)

From `wattbot.ensemble diagnose --run pi_on_v2=... --run pi_on_v1=...`:

| Metric | Value |
|---|---|
| Rows with changed value/unit/ref | 28 / 282 |
| Changed `answer_value` | 24 |
| Changed `ref_id` | 16 |
| Blank-status changes | 10 |
| Newly answered vs v1 | `q049, q052, q147, q154, q176, q293` (6) |
| Newly blank vs v1 | `q104, q180, q313, q315` (4) |
| Overall value agreement | 0.9255 |
| Jointly-answered agreement (249 q) | 0.9558 |
| Abstention agreement | 0.9645 |
| Mean citation Jaccard | 0.9527 |
| Mean tokens/q | 40,205 vs 43,182 |

Diagnostics file: `outputs/pi-v2-clean-v2/diagnose-pion-v2-vs-v1.json`.

---

## 5. Updated 282-q Test Ensembles

All three CSVs were produced with the current `wattbot.ensemble` CLI and are
`wattbot verify`-clean.

| Ensemble | Answered / Blank | Tok/q | Escalation | Artifact |
|---|---:|---:|---:|---|
| `rag-cascade-pion-v2-mm-doc` (`pi_on_v2+mm → doc`) | **268 / 14** | **77,228** | 68/282 (24.11%) | `ensembles/rag-cascade-pion-v2-mm-doc.csv` |
| `ens-4paradigm-v2-nofc-bf` (`pi_on_v2+doc+mhop+mm`, `q0/s0`) | **268 / 14** | **148,014** | — | `ensembles/ens-4paradigm-v2-nofc-bf.csv` |
| `ens-5paradigm-v2-q05-mmcascade` (`q0.5 → mm`) | **266 / 16** | **837,818** (+3,117 cascade) | 20/282 (7.09%) | `ensembles/ens-5paradigm-v2-q05-mmcascade.csv` |

Corresponding old (v1 `pi_on`) diagnostics for comparison:

| Ensemble | Old ans / blank | Old tok/q | Old escalation |
|---|---:|---:|---:|
| cascade `pi_on+mm → doc` | 267 / 15 | 79,711 | 66/282 (23.40%) |
| 4-paradigm BF | 267 / 15 | 150,991 | — |
| 5-paradigm `q0.5 → mm` | 266 / 16 | 840,795 | 21/282 (7.45%) |

Cascade trigger breakdown (new): `disagreement` 32, `abstain` 36; resolutions
`tier1_accepted` 214, `escalated_consensus` 34, `escalated_fallback_tier1_aux`
6, `escalated_tier2` 11, `escalated_abstained` 14, `escalated_fallback_tier1` 3.

`5-paradigm` totals include Full-Context (`fc_recov` 194,524,804 tokens, the
dominant term); `4-paradigm` and the cascade exclude Full-Context entirely.

---

## 6. Artifact Index

```
outputs/pi-v2-clean-v2/
  summary.json                          # all exact metrics above
  smoke/                                # 2-q compatibility smoke test
  train/                                # pi_on_v2 clean, 41 q (verified)
  test/                                 # pi_on_v2 clean, 282 q (verified)
  diagnose-pion-v2-vs-v1.json           # test diff diagnostics
  train-5par-q05-mmcascade/
    consensus-q05.csv / .diag.json
    predictions.csv / cascade.diag.json # 0.9018 weighted (verified)
  train-main-cascade/
    predictions.csv / cascade.diag.json # 0.9012 weighted (verified)
  train-main-cascade-old/               # v1 replay, 0.8970 (verified)
  ensembles/
    rag-cascade-pion-v2-mm-doc.csv      # verified, candidate
    ens-4paradigm-v2-nofc-bf.csv        # verified, candidate
    ens-5paradigm-v2-q05-mmcascade.csv  # verified, candidate
    _5par-q05-consensus.csv             # intermediate
```

**Candidate CSVs ready for coordinator review / Kaggle evaluation** (not
submitted here):

1. `outputs/pi-v2-clean-v2/ensembles/rag-cascade-pion-v2-mm-doc.csv`
   — main Pure-RAG cascade, cheapest high-value candidate (77.2k tok/q).
2. `outputs/pi-v2-clean-v2/ensembles/ens-4paradigm-v2-nofc-bf.csv`
   — 4-paradigm no-FC blank-filter.
3. `outputs/pi-v2-clean-v2/ensembles/ens-5paradigm-v2-q05-mmcascade.csv`
   — full-context-inclusive quorum + multimodal ceiling.

---

## 7. Interpretation & Caveats

1. **Train replay fills the missing table cell.** The `5-Paradigm Quorum
   (q0.5) → Multimodal Cascade` scores **0.9018** on the 41-question train split,
   essentially matching the blank-filter 5-paradigm row (0.9024) that is already
   in the table, and above its own consensus stage (0.8774).
2. **Clean v2 is a citation/token win, not a free value win.** The only train
   value regression is `q124`, where the cleaner index returned a prose target
   (5.4M) instead of the table cell (5.439M). The cascade recovers it.
3. **Test-set numbers are unlabeled local outputs.** Only `wattbot verify`
   (schema/order/abstention integrity) was run; no Kaggle score exists yet for
   the three new CSVs. The generated test diffs are descriptive, not accuracy
   measurements.
4. **Kaggle submission was intentionally not performed**; the coordinator should
   choose which candidate(s) to spend the daily quota on.
