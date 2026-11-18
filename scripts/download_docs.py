#!/usr/bin/env python3
"""Download source documents listed in data/raw/metadata.csv into data/docs."""

import csv
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "data" / "raw" / "metadata.csv"
OUT_DIR = ROOT / "data" / "docs"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WattBot2025-downloader/1.0)"}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA.open(encoding="latin-1", newline="") as f:
        rows = list(csv.DictReader(f))

    failed = []
    for i, row in enumerate(rows, 1):
        doc_id = row["id"].strip()
        url = row["url"].strip()
        dest = OUT_DIR / f"{doc_id}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i:02d}/{len(rows)}] skip {doc_id} (exists)")
            continue
        print(f"[{i:02d}/{len(rows)}] {doc_id} <- {url}")
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
                out.write(resp.read())
            print(f"          {dest.stat().st_size / 1e6:.2f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"          FAILED: {exc}", file=sys.stderr)
            failed.append(doc_id)
            dest.unlink(missing_ok=True)
        time.sleep(1)

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(rows)} documents downloaded to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
