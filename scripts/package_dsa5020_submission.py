"""Package the DSA5020 final project submission ZIP cleanly.

Includes:
  - Final ACM report PDF (top-level + report/ source)
  - Source code (wattbot/, pi_rag/, demo/, scripts/, tests/, docs/)
  - Datasets used (data/raw/, data/docs/, data/related/*.csv)
  - Key verified test prediction CSVs (optional demo artifacts)
  - Top-level README.md, AGENTS.md, LICENSE and requirements-rag.txt
  - A generated README_SUBMISSION.md explaining the archive layout
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIR_NAMES = {
    ".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", "preview",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".aux", ".bbl", ".blg", ".log", ".out", ".fls", ".fdb_latexmk"}

README_SUBMISSION = """\
# DSA5020 Final Project — WattBot 2025 RAG System

This archive contains the complete, reproducible code pipeline, the ACM-format
report, the competition data used, and a set of verified demo outputs for the
WattBot 2025 RAG competition (DSA5020 final project).

## Directory layout

```
DSA5020_Final_Report_WattBot2025.pdf   Final 6-page ACM-format report
report/                                LaTeX sources, vector figures, references
code/                                  All source code, tests, scripts and docs
  README.md, AGENTS.md, LICENSE        Project instructions and MIT license
  requirements-rag.txt                 Python dependencies
  wattbot/                             Python inference / retrieval / scoring modules
  pi_rag/                              Node.js bridges for the Pi agents
  demo/                                Interactive offline + live web demo
  scripts/                             Data preparation, packaging, verification
  tests/                               Unit and integration tests
  docs/                                Experiment records and Kaggle score tables
dataset/                               Competition data (see license note below)
  raw/                                 metadata.csv, train_QA.csv, test_Q.csv
  docs/                                Downloaded source PDFs (omitted with --no-pdfs)
  related/                             Curated retrieval / agentic-RAG reference metadata
demo_outputs/                          Verified prediction CSVs from the test set
```

## Environment

- Python 3.11+ (developed on 3.13)
- Node.js >= 18 (only required for the `pi_rag/` agent bridges)
- Install Python dependencies:
  `python3 -m pip install -r code/requirements-rag.txt`
- Install Node dependencies: `npm ci --prefix code/pi_rag`

An LLM API key is required for live inference. The default backend is the
HKUST(GZ) AIGC gateway; set `HKUSTGZ_AIGC_API_KEY` in the environment or in a
`.env` file. `code/docs/` records the full experiment protocol.

## Run the unit tests

```bash
cd code
python3 -m unittest discover -s tests    # 208 tests
```

## Launch the interactive demo

The demo uses only the Python standard library plus `pymupdf`:

```bash
cd code
python3 demo/app.py --host 127.0.0.1 --port 8501
```

Then open <http://127.0.0.1:8501>. `code/demo/app.py` resolves the corpus and
artefacts relative to `code/` (i.e. `code/data/` and `code/outputs/`). The
**Trace Explorer** starts and degrades gracefully when `outputs/` is absent;
`demo_outputs/` at the archive root holds the compact verified test prediction
CSVs only, while the full per-stage JSONL traces are excluded from this ZIP for
size. To render PDF page images or run the **Live Q&A** tab from the unpacked
archive, first make the corpus visible to the demo, e.g.:

```bash
cd code
ln -s ../dataset data          # or: cp -R ../dataset data
python3 demo/app.py
```

See `code/demo/README.md` for details.

## Reproduce the main Pure-RAG cascade

```bash
cd code
python3 -m wattbot index                 # build the v1 index
python3 -m wattbot.clean_index           # build the clean v2 index
python3 -m wattbot pure-rag-cascade \
  --pi-workers 4 --multimodal-workers 2 --tier2-workers 2 \
  --output-dir outputs/pure-rag-cascade
```

This runs `Pi Agent v2 (thinking)` + multimodal as Tier-1, and escalates only
disagreed/abstained questions to Tier-2 `doc_routing`. It scores
**Public 0.895 / Private 0.871** at **77.2k tokens/question**. The output
directory holds `predictions.csv`, `diagnostics.json` and per-stage
subdirectories (`tier1_pi/`, `tier1_multimodal/`, `tier2_doc_routing/`, each
with its own `trace.jsonl`). Pass `--limit` for a smoke test and `--resume` to
continue an interrupted run.

The earlier single-stage cascade is still available as
`python3 -m wattbot cascade` (Clean v2 text -> multimodal, `0.858 / 0.815`).

## Data license (important)

The WattBot 2025 competition data in `dataset/` is included **strictly for
DSA5020 academic grading**. It is provided by the competition organizers for
non-commercial participation and academic research only; it **must not be
redistributed or republished** beyond this grading submission. Remove
`dataset/` before any public release of this archive. The source code itself is
released under the MIT license (`code/LICENSE`).
"""



def add_path(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> int:
    count = 0
    if src.is_file():
        zf.write(src, arc_prefix)
        return 1
    if not src.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in sorted(dirnames) if d not in SKIP_DIR_NAMES]
        for fname in sorted(filenames):
            if fname.startswith(".") or Path(fname).suffix in SKIP_SUFFIXES:
                continue
            full = Path(dirpath) / fname
            rel = full.relative_to(src)
            zf.write(full, f"{arc_prefix}/{rel.as_posix()}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DSA5020 submission ZIP archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "DSA5020_WattBot2025_Submission.zip",
        help="Destination ZIP path (default: outputs/DSA5020_WattBot2025_Submission.zip)",
    )
    parser.add_argument(
        "--no-pdfs",
        action="store_true",
        help="Exclude data/docs/*.pdf (keep only data/raw CSVs + download script) for a lightweight code+report archive.",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report_pdf = ROOT / "report" / "main.pdf"
    if not report_pdf.exists():
        raise FileNotFoundError(f"Compiled report PDF not found: {report_pdf}")

    total = 0
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Top-level Final Report PDF
        total += add_path(zf, report_pdf, "DSA5020_Final_Report_WattBot2025.pdf")

        # 2. Report LaTeX sources & vector figures
        total += add_path(zf, ROOT / "report", "report")

        # 3. Top-level instructions & requirements
        for top_file in ("README.md", "AGENTS.md", "LICENSE", "requirements-rag.txt"):
            p = ROOT / top_file
            if p.exists():
                total += add_path(zf, p, f"code/{top_file}")

        # 4. Source code, tests, docs, scripts, demo
        for folder in ("wattbot", "pi_rag", "demo", "scripts", "tests", "docs"):
            total += add_path(zf, ROOT / folder, f"code/{folder}")

        # 5. Datasets
        total += add_path(zf, ROOT / "data" / "raw", "dataset/raw")
        for rel_csv in ("papers.csv", "shortlist.csv"):
            p = ROOT / "data" / "related" / rel_csv
            if p.exists():
                total += add_path(zf, p, f"dataset/related/{rel_csv}")
        if not args.no_pdfs:
            total += add_path(zf, ROOT / "data" / "docs", "dataset/docs")

        # 6. Main verified submission CSVs (compact demo outputs)
        demo_csvs = [
            ("outputs/multi-paradigm-ensemble/rag-cascade-pion-mm-doc.csv", "demo_outputs/main_pure_rag_cascade_submission.csv"),
            ("outputs/pi-v2-clean-v2/ensembles/rag-cascade-pion-v2-mm-doc.csv", "demo_outputs/main_pure_rag_cascade_clean_v2_submission.csv"),
            ("outputs/multi-paradigm-ensemble/rag-consensus-mhop-mm-pioff.csv", "demo_outputs/lightweight_pure_rag_3way_submission.csv"),
            ("outputs/pi-v2-sandbox/pi-v2-thinking-on/test/predictions.csv", "demo_outputs/pi_v2_thinking_on_submission.csv"),
            ("outputs/pi-v2-clean-v2/test/predictions.csv", "demo_outputs/pi_v2_clean_v2_thinking_on_submission.csv"),
            ("outputs/doc-routing/test/predictions.csv", "demo_outputs/doc_routing_submission.csv"),
            ("outputs/multihop-retrieval/test/predictions.csv", "demo_outputs/multihop_rag_submission.csv"),
            ("outputs/multimodal/test-mode2/predictions.csv", "demo_outputs/multimodal_mode2_submission.csv"),
            ("outputs/pi-v2-clean-v2/ensembles/ens-4paradigm-v2-nofc-bf.csv", "demo_outputs/public_record_4paradigm_nofc_submission.csv"),
            ("outputs/multi-paradigm-ensemble/ens-5paradigm-q05-mmcascade.csv", "demo_outputs/upper_bound_5paradigm_submission.csv"),
        ]
        for src_rel, dst_arc in demo_csvs:
            p = ROOT / src_rel
            if p.exists():
                total += add_path(zf, p, dst_arc)

        # 7. Generated archive README
        zf.writestr("README_SUBMISSION.md", README_SUBMISSION)
        total += 1

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Created {args.output} ({total} files, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
