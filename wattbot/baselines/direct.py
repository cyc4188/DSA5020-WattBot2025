"""Closed-book DeepSeek baseline for WattBot."""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import Lock

from wattbot.__main__ import atomic_csv
from wattbot.deepseek import DeepSeek
from wattbot.evaluation import evaluate
from wattbot.pipeline import BLANK, abstention, model_value, read_csv, required_text, submission_fields
from wattbot.retrieval import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
PROMPT = """Answer this question using your existing knowledge and reasoning. You are
not given source documents. Do not invent citations or use external tools.
If you can answer reliably, return a JSON object with answerable=true, answer,
answer_value, answer_unit, and explanation. Use the unit hint if relevant.
For answer_value use a plain number, [low,high], or a short categorical label.
For explicit true/false questions use 1 or 0. Use is_blank for an inapplicable
unit. If you cannot answer reliably, return {"answerable": false, "reason": "..."}."""


def solve(question, client, emit, context=None):
    data = {"question": question["question"], "unit_hint": question.get("answer_unit", BLANK)}
    prompt = PROMPT
    stage = "direct_answer"
    if context is not None:
        prompt = context.prompt
        stage = "full_context_answer"
        data = {"corpus": context.text, **data}

        def record(event, **fields):
            if event == "llm_request":
                fields["messages"] = [
                    fields["messages"][0],
                    {"role": "user", "content": json.dumps(
                        {"corpus_sha256": context.digest, "question": data["question"],
                         "unit_hint": data["unit_hint"]}, ensure_ascii=False)},
                ]
            emit(event, **fields)
    else:
        record = emit
    for attempt in (1, 2):
        try:
            reply = client.complete(stage, prompt, data, record)
            if reply.get("answerable") is False:
                emit("decision", decision="abstained", reason=required_text(reply, "reason"))
                return abstention(question)
            if reply.get("answerable") is not True:
                raise ValueError("Expected a boolean answerable")
            prose_answer = reply.get("answer")
            raw_value = reply.get("answer_value")
            if raw_value is None and isinstance(prose_answer, (bool, int, Decimal)):
                raw_value = prose_answer
                emit("answer_value_recovered", source="scalar answer")
            value = model_value(raw_value)
            if not isinstance(prose_answer, str) or not prose_answer.strip():
                prose_answer = value
                emit("answer_text_normalized", answer=prose_answer)
            row = submission_fields(question, {
                **abstention(question), "answer": prose_answer.strip(),
                "answer_value": value,
                "answer_unit": required_text(reply, "answer_unit"),
                "explanation": required_text(reply, "explanation"),
            })
            if context is not None and row["answer_value"] != BLANK:
                row.update(context.citations(reply, emit))
            emit("decision", decision="answered" if row["answer_value"] != BLANK else "abstained")
            return row
        except ValueError as error:
            emit("invalid_output", attempt=attempt, detail=str(error))
    emit("decision", decision="abstained_invalid_output",
         reason="No structured answer passed validation after two attempts.")
    return abstention(question)


def run(split, output, workers, resume, *, context=None, limit=None):
    questions_path = ROOT / "data" / "raw" / {"train": "train_QA.csv", "test": "test_Q.csv"}[split]
    _, source = read_csv(questions_path)
    source = source[:limit]
    question_fields = ("id", "question", "answer_unit")
    questions = [{key: row[key] for key in question_fields} for row in source]
    client = (DeepSeek("hkustgz") if context is None else
              DeepSeek("hkustgz", timeout=600, transport_attempts=1))
    manifest = {
        "method": "direct_llm", "split": split, "input_file": questions_path.name,
        "input_sha256": sha256_file(questions_path), "questions": len(questions),
        "backend": client.backend, "model": client.model,
        "checkpoint_basis": "HKUSTGZ alias confirmed by the user as Flash 1017",
        "temperature": 0, "max_tokens": 2400, "response_format": "json_object",
        "thinking": False, "prompt": PROMPT, "model_input_fields": ["question", "unit_hint"],
        "workers": workers, "max_format_attempts": 2,
        "max_transport_attempts": client.transport_attempts, "timeout_seconds": client.timeout,
        "code_sha256": {path.name: sha256_file(path) for path in (
            Path(__file__), ROOT / "wattbot" / "deepseek.py",
            ROOT / "wattbot" / "pipeline.py", ROOT / "wattbot" / "evaluation.py",
        )},
    }

    if context is not None:
        manifest.update(context.settings())
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    trace_path = output / "trace.jsonl"
    predictions_path = output / "predictions.csv"
    rows = {}
    if resume:
        if json.loads(config_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cannot resume with different inputs, code or settings")
        with trace_path.open(encoding="utf-8") as previous:
            for line in previous:
                event = json.loads(line)
                if event["event"] == "row":
                    rows[event["question_id"]] = event["row"]
    elif config_path.exists() or trace_path.exists():
        raise FileExistsError("Use --resume or choose another output directory")

    lock = Lock()
    with trace_path.open("a" if resume else "x", encoding="utf-8") as trace:
        if not resume:
            write_json(config_path, manifest)
            if context is not None:
                (output / "corpus.txt").write_bytes(context.text.encode("utf-8"))

        def emit(event, question_id=None, **fields):
            line = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event, "question_id": question_id, **fields,
            }, ensure_ascii=False)
            with lock:
                trace.write(line + "\n")
                trace.flush()

        def process(question):
            question_id = question["id"]

            def record(event, **fields):
                emit(event, question_id, **fields)

            try:
                row = solve(question, client, record, context)
            except (ValueError, RuntimeError, OSError) as error:
                record("question_error", error_type=type(error).__name__, detail=str(error))
                raise
            record("row", row=row)
            return row

        emit("run_resume" if resume else "run_start",
             config_sha256=sha256_file(config_path), completed_rows=len(rows))
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process, question): question["id"]
                for question in questions if question["id"] not in rows
            }
            for future in as_completed(futures):
                question_id = futures[future]
                try:
                    rows[question_id] = future.result()
                except (ValueError, RuntimeError, OSError):
                    failures.append(question_id)
        if failures:
            emit("run_error", failed_ids=failures)
            raise RuntimeError(f"{len(failures)} questions failed. Inspect {trace_path} and use --resume.")

        ordered = [rows[question["id"]] for question in questions]
        atomic_csv(predictions_path, ordered)
        summary = dict(Counter("abstained" if row["answer_value"] == BLANK else "answered"
                               for row in ordered))
        summary.update({"rows": len(ordered), "predictions_sha256": sha256_file(predictions_path)})
        if split == "train":
            metrics = evaluate(ordered, source)
            write_json(output / "metrics.json", metrics)
            summary.update({key: value for key, value in metrics.items() if key != "per_question"})
        write_json(output / "summary.json", summary)
        emit("run_complete", **summary)
        print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output_dir or ROOT / "outputs" / "baseline-direct" / args.split
    run(args.split, output, args.workers, args.resume)


if __name__ == "__main__":
    main()
