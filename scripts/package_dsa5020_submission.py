"""Package the DSA5020 final project submission ZIP cleanly.

Includes:
  - Final ACM report PDF (top-level + report/ source)
  - Source code (wattbot/, pi_rag/ excluding node_modules, scripts/, tests/, docs/)
  - Datasets used (data/raw/, data/docs/, data/related/*.csv)
  - Key verified test prediction CSVs (optional demo artifacts)
  - Top-level README.md and requirements-rag.txt
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
        for top_file in ("README.md", "AGENTS.md", "requirements-rag.txt"):
            p = ROOT / top_file
            if p.exists():
                total += add_path(zf, p, f"code/{top_file}")

        # 4. Source code, tests, docs, scripts
        for folder in ("wattbot", "pi_rag", "scripts", "tests", "docs"):
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
            ("outputs/multi-paradigm-ensemble/rag-consensus-mhop-mm-pioff.csv", "demo_outputs/lightweight_pure_rag_3way_submission.csv"),
            ("outputs/pi-v2-sandbox/pi-v2-thinking-on/test/predictions.csv", "demo_outputs/pi_v2_thinking_on_submission.csv"),
            ("outputs/doc-routing/test/predictions.csv", "demo_outputs/doc_routing_submission.csv"),
            ("outputs/multihop-retrieval/test/predictions.csv", "demo_outputs/multihop_rag_submission.csv"),
            ("outputs/multimodal/test-mode2/predictions.csv", "demo_outputs/multimodal_mode2_submission.csv"),
            ("outputs/multi-paradigm-ensemble/ens-5paradigm-q05-mmcascade.csv", "demo_outputs/upper_bound_5paradigm_submission.csv"),
        ]
        for src_rel, dst_arc in demo_csvs:
            p = ROOT / src_rel
            if p.exists():
                total += add_path(zf, p, dst_arc)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Created {args.output} ({total} files, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
