# AGENTS.md

## Project

WattBot 2025 Kaggle RAG competition. See `README.md` for the full data layout,
pipeline commands and scoring; the constraints below are non-negotiable.

## Hard constraints (competition rules)

- Any LLM / embedding / reranker is allowed, local or hosted API. Open-source is
  preferred.
- The solution must be a fully automated code pipeline. Never answer questions
  by manual lookup in the papers or by manually chatting with an LLM and
  pasting results.
- Do not redistribute the competition data; it is non-commercial/academic only.
- Submission is a single uploaded CSV produced by code. 5 submissions/day; at
  the deadline the 3 most recent submissions are carried forward.
- Unanswerable questions require `answer_value`, `answer_unit`, `ref_id`,
  `ref_url`, `supporting_materials` all set to `is_blank`.
- Winners must publish their solution under an open-source license.

## Commands

```bash
python3 scripts/download_docs.py
python3 scripts/collect_retrieval_papers.py
python3 scripts/make_submission.py
kaggle competitions submit -c WattBot2025 -f outputs/submission_minimal.csv -m "msg"
```

## Conventions

- Do not commit `data/docs/`, `data/related/pdfs/`, `.seed_cache.json` or
  `outputs/` (already in `.gitignore`).
- Data CSVs are not all UTF-8: `metadata.csv` needs `latin-1`, question CSVs use
  `utf-8-sig`.
