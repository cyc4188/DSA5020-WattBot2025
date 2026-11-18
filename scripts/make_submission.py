#!/usr/bin/env python3
"""Generate a schema-valid WattBot 2025 submission CSV.

Two modes:
  blank      every field except id/question is the is_blank fallback
             (absolute minimum viable submission)
  unit_hint  keep the answer_unit hints shipped in test_Q.csv, leave the rest blank

Usage:
    python3 scripts/make_submission.py
    python3 scripts/make_submission.py --mode unit_hint --output outputs/submission_unit_hint.csv
"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_ANSWER = "Unable to answer with confidence based on the provided documents."
BLANK = "is_blank"
COLUMNS = [
    "id",
    "question",
    "answer",
    "answer_value",
    "answer_unit",
    "ref_id",
    "ref_url",
    "supporting_materials",
    "explanation",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "data" / "raw" / "test_Q.csv"))
    ap.add_argument("--output", default=str(ROOT / "outputs" / "submission_minimal.csv"))
    ap.add_argument("--mode", choices=["blank", "unit_hint"], default="blank")
    args = ap.parse_args()

    with Path(args.questions).open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": FALLBACK_ANSWER,
                    "answer_value": BLANK,
                    "answer_unit": (row.get("answer_unit") or "").strip()
                    if args.mode == "unit_hint"
                    else BLANK,
                    "ref_id": BLANK,
                    "ref_url": BLANK,
                    "supporting_materials": BLANK,
                    "explanation": BLANK,
                }
            )

    print(f"Wrote {len(rows)} rows -> {out_path} (mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
