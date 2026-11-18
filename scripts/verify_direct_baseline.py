"""Check a completed direct baseline trace, predictions and local metrics."""

import argparse
import json
from collections import Counter
from pathlib import Path

from wattbot.baselines.full_context import FullContext
from wattbot.deepseek import CAMPUS_MODEL
from wattbot.evaluation import evaluate
from wattbot.pipeline import BLANK, COLUMNS, EVIDENCE_FIELDS, read_csv
from wattbot.retrieval import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def verify(directory, index=None):
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    questions_file = ROOT / "data" / "raw" / {
        "train": "train_QA.csv", "test": "test_Q.csv",
    }[config["split"]]
    _, source = read_csv(questions_file)
    full = config["method"] == "full_context_llm"
    if full:
        source = source[:config["questions"]]
        context = FullContext(index or ROOT / "data/processed/wattbot-index.json",
                              ROOT / "data/raw/metadata.csv")
        if (config["corpus_sha256"] != context.digest
                or config["index_sha256"] != context.index_digest
                or sha256_file(directory / config["corpus_file"]) != context.digest):
            raise ValueError("Full corpus checksum mismatch")
    stage = "full_context_answer" if full else "direct_answer"
    by_id = {row["id"]: row for row in source}
    predictions_file = directory / "predictions.csv"
    columns, rows = read_csv(predictions_file)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (config["method"] not in {"direct_llm", "full_context_llm"} or config["model"] != CAMPUS_MODEL
            or config["input_sha256"] != sha256_file(questions_file)
            or columns != COLUMNS or len(by_id) != len(source)
            or len(rows) != len(source) or config["questions"] != len(rows)):
        raise ValueError("Configuration, input, or prediction schema mismatch")
    for row, question in zip(rows, source, strict=True):
        if row["id"] != question["id"] or row["question"] != question["question"]:
            raise ValueError("Prediction IDs or order differ from the input")
        if not full and any(row[field] != BLANK for field in ("ref_id", "ref_url", "supporting_materials")):
            raise ValueError(f"Unexpected citation for {row['id']}")
        if row["answer_value"] == BLANK and any(row[field] != BLANK for field in EVIDENCE_FIELDS):
            raise ValueError(f"Incomplete abstention for {row['id']}")

    traced_rows = {}
    calls = Counter()
    completed = None
    with (directory / "trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            name = event["event"]
            question_id = event["question_id"]
            if name in {"run_start", "run_resume"}:
                if event["config_sha256"] != sha256_file(config_path):
                    raise ValueError("Trace configuration checksum mismatch")
            elif name == "llm_request":
                question = by_id[question_id]
                messages = event["messages"]
                expected = {"question": question["question"], "unit_hint": question["answer_unit"]}
                if full:
                    expected["corpus_sha256"] = context.digest
                if (event["stage"] != stage or event["model"] != config["model"]
                        or len(messages) != 2
                        or messages[0] != {"role": "system", "content": config["prompt"]}
                        or messages[1]["role"] != "user"
                        or json.loads(messages[1]["content"]) != expected):
                    raise ValueError(f"Unexpected prompt or model for {question_id}")
                calls[question_id] += 1
            elif name == "llm_response":
                if event["stage"] != stage or event["model"] not in {
                    CAMPUS_MODEL, "DeepSeek-V4.1-Flash-1017",
                }:
                    raise ValueError(f"Unexpected model response for {question_id}")
            elif name == "row":
                traced_rows[question_id] = event["row"]
            elif name == "run_complete":
                completed = {key: event[key] for key in summary}

    if (set(calls) != set(by_id) or traced_rows != {row["id"]: row for row in rows}
            or summary != completed or summary["predictions_sha256"] != sha256_file(predictions_file)):
        raise ValueError("Trace or summary does not match the saved predictions")
    if config["split"] == "train":
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics != evaluate(rows, source):
            raise ValueError("Saved metrics differ from the recomputed scores")
    return {"rows": len(rows), "requests": sum(calls.values()), "split": config["split"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.index), indent=2))
