"""Replay a WattBot run trace against its index and final CSV."""

import argparse
import json
from collections import Counter
from pathlib import Path

from wattbot.__main__ import atomic_csv
from wattbot.deepseek import CAMPUS_MODEL, OPENROUTER_MODEL
from wattbot.pipeline import BLANK, load_metadata, read_csv, submission_fields, validate_rows
from wattbot.retrieval import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "hkustgz": {CAMPUS_MODEL, "DeepSeek-V4.1-Flash-1017"},
    "openrouter": {OPENROUTER_MODEL, "deepseek/deepseek-v4.1-flash-20261017"},
}


def check_trace(trace, index, output, questions_path, metadata_path):
    with index.open(encoding="utf-8") as stream:
        corpus = json.load(stream)
    nodes = {node["id"]: node for node in corpus["nodes"]}
    metadata = load_metadata(metadata_path)
    columns, rows = read_csv(output)
    _, questions = read_csv(questions_path)
    validate_rows(columns, rows, questions, metadata)
    by_question = {row["id"]: row for row in rows}
    selected = {}
    accepted = {}
    emitted = {}
    fingerprints = Counter()
    stages = Counter()
    endings = Counter()
    backend = None
    complete = False

    with trace.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            name = event["event"]
            question_id = event["question_id"]
            if name == "run_start":
                if backend or event["mode"] != "run":
                    raise ValueError("Expected one answer-generation manifest")
                backend = event["backend"]
                for key, digest in (
                    ("index_sha256", sha256_file(index)),
                    ("questions_sha256", sha256_file(questions_path)),
                ):
                    if event[key] != digest:
                        raise ValueError(f"Trace and input disagree on {key}")
            elif name == "llm_response":
                if event.get("model") not in MODELS[backend]:
                    raise ValueError(f"Unexpected served model for {question_id}")
                stages[event["stage"]] += 1
                if event.get("system_fingerprint"):
                    fingerprints[event["system_fingerprint"]] += 1
            elif name == "candidate":
                selected[(question_id, event["attempt"])] = event
            elif name == "verification":
                endings[question_id] = event["decision"]
                if event["supported"]:
                    candidate = selected[(question_id, event["attempt"])]
                    for citation in candidate["citations"]:
                        node = nodes[citation["node_id"]]
                        source = corpus["pages"][node["page_id"]]
                        if (citation["ref_id"], citation["page"]) != (node["ref_id"], node["page"]):
                            raise ValueError(f"Incorrect citation metadata for {question_id}")
                        if citation["quote"] not in node["text"] and citation["quote"] not in source:
                            raise ValueError(f"Citation absent from source index for {question_id}")
                    accepted[question_id] = candidate["row"]
            elif name == "row":
                row = event["row"]
                if row != by_question.get(question_id):
                    raise ValueError(f"Trace row differs from CSV for {question_id}")
                if row["answer_value"] != BLANK:
                    if row != accepted.get(question_id):
                        raise ValueError(f"No accepted candidate for {question_id}")
                elif endings[question_id] not in {"abstained", "rejected", "abstained_invalid_output"}:
                    raise ValueError(f"No abstention decision for {question_id}")
                emitted[question_id] = row
            elif name == "run_complete":
                complete = (event["rows"] == len(rows)
                            and event["output_sha256"] == sha256_file(output))

    if not complete or emitted != by_question:
        raise ValueError("Trace does not attest to the complete CSV")
    return {
        "rows": len(rows), "backend": backend, "model_calls": dict(stages),
        "fingerprints": dict(fingerprints),
        "final_decisions": dict(Counter(endings.values())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path,
                        default=ROOT / "data" / "processed" / "wattbot-index.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "raw" / "test_Q.csv")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "raw" / "metadata.csv")
    parser.add_argument("--canonical-output", type=Path,
                        help="Write canonical answer values after verifying the source CSV and trace")
    args = parser.parse_args()
    summary = check_trace(args.trace, args.index, args.output, args.questions, args.metadata)
    if args.canonical_output:
        if args.canonical_output.resolve() in {args.output.resolve(), args.trace.resolve()}:
            raise ValueError("Choose a separate canonical output path")
        _, rows = read_csv(args.output)
        _, questions = read_csv(args.questions)
        canonical_rows = [submission_fields(question, row)
                          for question, row in zip(questions, rows, strict=True)]
        changes = [
            {"id": before["id"],
             "fields": {field: {"before": before[field], "after": after[field]}
                        for field in before if before[field] != after[field]}}
            for before, after in zip(rows, canonical_rows, strict=True) if before != after
        ]
        columns, _ = read_csv(args.output)
        validate_rows(columns, canonical_rows, questions, load_metadata(args.metadata))
        atomic_csv(args.canonical_output, canonical_rows)
        audit = args.canonical_output.with_suffix(".normalization.json")
        write_json(audit, {
            "source_csv_sha256": sha256_file(args.output),
            "source_trace_sha256": sha256_file(args.trace),
            "output_sha256": sha256_file(args.canonical_output),
            "changes": changes,
        })
        summary.update({"normalized_rows": len(changes), "normalization_audit": str(audit)})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
