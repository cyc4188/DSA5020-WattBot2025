"""Verify a WattBot cascade trace against its inputs and final predictions CSV."""

import argparse
import json
from collections import Counter
from pathlib import Path

from wattbot.pipeline import BLANK, load_metadata, read_csv, validate_rows
from wattbot.retrieval import sha256_file

EVIDENCE_FIELDS = ("answer_value", "answer_unit", "ref_id", "ref_url", "supporting_materials")
MANIFEST_EVENTS = {"run_start", "model_config", "run_resume", "run_complete", "run_error"}


def _fail(message):
    raise ValueError(message)


def _check_blank(row):
    if any(not isinstance(value, str) or not value.strip() for value in row.values()):
        _fail(f"Noncanonical empty value in trace row for {row.get('id')}")
    if row.get("answer_value") == BLANK:
        fields = [field for field in EVIDENCE_FIELDS if field != "answer_value"]
        if any(row.get(field) != BLANK for field in fields):
            _fail(f"Noncanonical abstention in trace row for {row.get('id')}")


def check_trace(trace, predictions, questions_path, metadata_path, index, visual_index):
    columns, rows = read_csv(predictions)
    _, questions = read_csv(questions_path)
    metadata = load_metadata(metadata_path)
    validate_rows(columns, rows, questions, metadata)

    by_question = {question["id"]: question for question in questions}
    final_rows = {row["id"]: row for row in rows}
    if set(final_rows) != set(by_question):
        _fail("Predictions IDs differ from questions")

    hashes = {
        "questions_sha256": sha256_file(questions_path),
        "metadata_sha256": sha256_file(metadata_path),
        "index_sha256": sha256_file(index),
        "visual_index_sha256": sha256_file(visual_index),
        "predictions_sha256": sha256_file(predictions),
        "trace_sha256": sha256_file(trace),
    }

    run_start = None
    run_complete = None
    completions = 0
    last_event = None
    row_events = {}
    attempts_by_question = {}
    active = {}
    attempts = 0
    failed_attempts = 0
    resumes = 0
    routes = Counter()
    results = Counter()
    stage_calls = Counter()
    stage_detail = Counter()
    usage = Counter()
    usage_missing = 0

    with trace.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                _fail(f"Blank trace line {line_number}")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid trace JSON on line {line_number}") from error
            name = event.get("event")
            if not isinstance(name, str):
                _fail(f"Missing event name on line {line_number}")

            if line_number == 1:
                if name != "run_start":
                    _fail("First trace event must be run_start")
                run_start = event
                if event.get("mode") != "cascade":
                    _fail("Trace mode is not cascade")
                for key, filename in (
                    ("questions_sha256", questions_path),
                    ("index_sha256", index),
                ):
                    if event.get(key) != hashes[f"{key}"]:
                        _fail(f"Trace and input disagree on {key}")
                if event.get("question_count") != len(questions):
                    _fail("Trace question_count differs from the input")
                visual = event.get("visual_fallback")
                if not isinstance(visual, dict):
                    _fail("Trace lacks visual_fallback manifest")
                if visual.get("index_sha256") != hashes["visual_index_sha256"]:
                    _fail("Trace and input disagree on visual index_sha256")
            elif name == "run_start":
                _fail("Trace contains a second run_start")

            if name in MANIFEST_EVENTS:
                if event.get("question_id") is not None:
                    _fail(f"Unexpected question_id on {name}")
                if name == "run_resume":
                    resumes += 1
                    if event.get("completed_rows") != len(row_events):
                        _fail("run_resume completed_rows differs from observed rows")
                elif name == "run_complete":
                    completions += 1
                    run_complete = event
                last_event = name
                continue

            question_id = event.get("question_id")
            if question_id not in by_question:
                _fail(f"Unexpected question id {question_id!r} for {name}")
            last_event = name

            if name == "question_start":
                if question_id in row_events:
                    _fail(f"Question {question_id} restarted after a completed row")
                previous = active.get(question_id)
                if previous is not None and previous["row"] is None:
                    previous["failed"] = True
                    failed_attempts += 1
                attempts += 1
                attempt = {
                    "number": len(attempts_by_question.get(question_id, [])) + 1,
                    "route": None,
                    "result": None,
                    "row": None,
                    "failed": False,
                    "multimodal_events": 0,
                }
                attempts_by_question.setdefault(question_id, []).append(attempt)
                active[question_id] = attempt
                if event.get("question") != by_question[question_id]["question"]:
                    _fail(f"Question text differs for {question_id}")
                continue

            attempt = active.get(question_id)
            if attempt is None:
                _fail(f"{name} for {question_id} is outside an attempt")

            if name == "question_error":
                attempt["failed"] = True
                failed_attempts += 1
                active.pop(question_id, None)
                continue

            stage = event.get("pipeline_stage")
            if stage == "multimodal":
                if attempt["route"] is None or not attempt["route"]["escalated"]:
                    _fail(f"Multimodal event before an escalated route for {question_id}")
                if attempt["result"] is not None:
                    _fail(f"Multimodal event after cascade_result for {question_id}")
                attempt["multimodal_events"] += 1
            elif stage == "rag" and attempt["route"] is not None:
                _fail(f"RAG event after cascade_route for {question_id}")

            if name == "llm_response":
                stage_calls[stage] += 1
                stage_detail[f"{stage}/{event.get('stage')}"] += 1
                payload = event.get("usage")
                if not isinstance(payload, dict):
                    usage_missing += 1
                else:
                    usage["prompt_tokens"] += payload.get("prompt_tokens") or 0
                    usage["completion_tokens"] += payload.get("completion_tokens") or 0
                    usage["total_tokens"] += payload.get("total_tokens") or 0
                    details = payload.get("completion_tokens_details") or {}
                    usage["reasoning_tokens"] += details.get("reasoning_tokens") or 0
                    prompt_details = payload.get("prompt_tokens_details") or {}
                    usage["cached_tokens"] += prompt_details.get("cached_tokens") or 0
            elif name == "cascade_route":
                if attempt["route"] is not None:
                    _fail(f"Duplicate cascade_route for {question_id}")
                escalated = event.get("escalated")
                if not isinstance(escalated, bool):
                    _fail(f"Non-boolean cascade_route.escalated for {question_id}")
                expected = "rag_abstained" if escalated else "rag_answered"
                if event.get("reason") != expected:
                    _fail(f"cascade_route reason contradicts escalated for {question_id}")
                attempt["route"] = event
                routes[expected] += 1
            elif name == "cascade_result":
                if attempt["route"] is None:
                    _fail(f"cascade_result before cascade_route for {question_id}")
                if not attempt["route"]["escalated"]:
                    _fail(f"cascade_result on non-escalated question {question_id}")
                if attempt["result"] is not None:
                    _fail(f"Duplicate cascade_result for {question_id}")
                selected = event.get("selected")
                if selected not in {"abstention", "multimodal"}:
                    _fail(f"Invalid cascade_result.selected for {question_id}")
                attempt["result"] = event
                results[selected] += 1
            elif name == "row":
                if attempt["failed"]:
                    _fail(f"Failed attempt produced a row for {question_id}")
                if question_id in row_events:
                    _fail(f"Duplicate completed row for {question_id}")
                row = event.get("row")
                if row != final_rows.get(question_id):
                    _fail(f"Trace row differs from CSV for {question_id}")
                _check_blank(row)
                if attempt["route"] is None:
                    _fail(f"Row without cascade_route for {question_id}")
                escalated = attempt["route"]["escalated"]
                if escalated:
                    if attempt["result"] is None:
                        _fail(f"Routed row lacks cascade_result for {question_id}")
                    selected = attempt["result"]["selected"]
                    if (selected == "abstention") != (row["answer_value"] == BLANK):
                        _fail(f"cascade_result contradicts row for {question_id}")
                elif attempt["result"] is not None:
                    _fail(f"cascade_result on non-escalated question {question_id}")
                row_events[question_id] = row
                attempt["row"] = row
                active.pop(question_id, None)

    if last_event != "run_complete":
        _fail("Last trace event must be run_complete")
    if completions != 1 or run_complete is None:
        _fail("Trace must contain exactly one run_complete")
    if run_complete.get("rows") != len(rows):
        _fail("run_complete rows differs from the predictions")
    if run_complete.get("output_sha256") != hashes["predictions_sha256"]:
        _fail("run_complete output_sha256 differs from the predictions")

    for question_id, attempt in active.items():
        if attempt["row"] is None:
            attempt["failed"] = True
            failed_attempts += 1
    if set(row_events) != set(final_rows):
        _fail("Trace does not contain exactly one row for every question")
    for question_id, question_attempts in attempts_by_question.items():
        completed = [attempt for attempt in question_attempts if attempt["row"] is not None]
        if len(completed) != 1 or completed[0] is not question_attempts[-1]:
            _fail(f"Final attempt for {question_id} is not the completed attempt")

    return {
        "hashes": hashes,
        "counts": {
            "questions": len(questions),
            "rows": len(rows),
            "attempts": attempts,
            "failed_attempts": failed_attempts,
            "resumes": resumes,
            "routes": dict(routes),
            "cascade_results": dict(results),
            "stage_calls": dict(stage_calls),
            "stage_detail": dict(stage_detail),
            "usage": dict(usage),
            "usage_missing": usage_missing,
        },
        "limits": {
            "top_k": run_start.get("top_k"),
            "max_attempts_per_question": max(
                (len(value) for value in attempts_by_question.values()), default=0
            ),
            "question_count": run_start.get("question_count"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--visual-index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = check_trace(
        args.trace, args.predictions, args.questions, args.metadata,
        args.index, args.visual_index,
    )
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
