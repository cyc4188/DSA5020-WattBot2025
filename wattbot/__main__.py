"""Command-line entry points for the WattBot pipeline."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from .deepseek import DeepSeek
from .evaluation import evaluate
from .pipeline import COLUMNS, load_metadata, read_csv, solve, validate_rows
from .retrieval import SearchIndex, build_index, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_INDEX = DATA / "processed" / "wattbot-index.json"
DEFAULT_OUTPUT = ROOT / "outputs"


def atomic_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def generate(args, metadata, questions):
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.resume and args.command != "run":
        raise ValueError("--resume is available for run only")
    if args.top_k < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("--top-k and --limit must be positive")
    if args.output.resolve() == args.trace.resolve():
        raise ValueError("Choose different paths for output and trace")
    index = SearchIndex(args.index, dense=args.hybrid)
    if set(index.sources) != set(metadata) or any(
        index.sources[ref]["url"] != metadata[ref]["url"] for ref in metadata
    ):
        raise ValueError("Index does not match the current metadata")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    active_questions = questions[:args.limit]
    if not active_questions:
        raise ValueError("No questions to process")
    manifest = {
        "index_sha256": index.digest, "question_file": args.questions.name,
        "questions_sha256": sha256_file(args.questions), "question_count": len(active_questions),
        "top_k": args.top_k, "hybrid": args.hybrid, "backend": args.backend, "mode": args.command,
        "code_sha256": {path.name: sha256_file(path)
                        for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    rows = {}
    if args.resume:
        with args.trace.open(encoding="utf-8") as previous:
            first = json.loads(next(previous))
            if first["event"] != "run_start" or any(
                first.get(key) != value for key, value in manifest.items()
            ):
                raise ValueError("Trace belongs to a different corpus, run configuration, or code")
            for line in previous:
                record = json.loads(line)
                if record["event"] == "row":
                    rows[record["question_id"]] = record["row"]

    with args.trace.open("a" if args.resume else "x", encoding="utf-8") as trace:
        lock = Lock()

        def emit(event, question_id=None, **fields):
            line = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(), "question_id": question_id,
                "event": event, **fields,
            }, ensure_ascii=False)
            with lock:
                trace.write(line + "\n")
                trace.flush()

        if args.resume:
            emit("run_resume", completed_rows=len(rows))
        else:
            emit("run_start", **manifest)
        try:
            client = DeepSeek(args.backend) if args.command == "run" else None
            if client:
                emit("model_config", requested_model=client.model,
                     checkpoint="DeepSeek-V4.1-Flash-1017",
                     checkpoint_basis="campus alias confirmed by user"
                     if client.backend == "hkustgz" else "dated provider model ID")
        except (RuntimeError, ValueError, OSError, KeyError) as error:
            emit("run_error", error_type=type(error).__name__, detail=str(error))
            raise

        def process(question):
            def record(event, **fields):
                emit(event, question_id=question["id"], **fields)

            emit("question_start", question_id=question["id"], question=question["question"])
            if client:
                row = solve(question, index, client, record, args.top_k)
                validate_rows(COLUMNS, [row], [question], metadata)
                record("row", row=row)
                return row
            candidates = index.search([question["question"]], args.top_k)
            record("retrieval", candidates=candidates, expanded=index.expand(candidates, 1700))
            return None

        pending = [question for question in active_questions if question["id"] not in rows]
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, question): question for question in pending}
            for completed in as_completed(futures):
                question_id = futures[completed]["id"]
                try:
                    row = completed.result()
                    if row is not None:
                        rows[question_id] = row
                except (RuntimeError, ValueError, OSError, KeyError) as error:
                    emit("question_error", question_id=question_id,
                         error_type=type(error).__name__, detail=str(error))
                    failed.append(question_id)
                    if isinstance(error, RuntimeError):
                        for future in futures:
                            future.cancel()
                        emit("run_error", failures=len(failed), trace=str(args.trace))
                        raise
        if failed:
            emit("run_error", failures=len(failed), trace=str(args.trace))
            raise ValueError(f"Failed questions: {', '.join(failed)}. See {args.trace}")
        if client:
            ordered = [rows[question["id"]] for question in active_questions]
            validate_rows(COLUMNS, ordered, active_questions, metadata)
            atomic_csv(args.output, ordered)
            emit("run_complete", rows=len(rows), output_sha256=sha256_file(args.output))
            print(f"Validated and wrote {len(rows)} rows to {args.output}")
        else:
            emit("run_complete", questions=len(active_questions))
        print(f"Trace: {args.trace}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DATA / "raw" / "metadata.csv")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("index")
    build.add_argument("--docs", type=Path, default=DATA / "docs")
    build.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    for name in ("run", "retrieve", "verify", "score"):
        command = commands.add_parser(name)
        filename = "train_QA.csv" if name == "score" else "test_Q.csv"
        command.add_argument("--questions", type=Path, default=DATA / "raw" / filename)
        command.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "submission_rag.csv")
        if name in {"run", "retrieve"}:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            command.add_argument("--trace", type=Path,
                                 default=DEFAULT_OUTPUT / f"{name}-{timestamp}.jsonl")
            command.add_argument("--index", type=Path, default=DEFAULT_INDEX)
            command.add_argument("--top-k", type=int, default=10)
            command.add_argument("--limit", type=int)
            command.add_argument("--workers", type=int, default=1)
            command.add_argument("--resume", action="store_true")
            command.add_argument("--hybrid", action="store_true")
            command.add_argument("--backend", choices=("hkustgz", "openrouter"), default="hkustgz")
        elif name == "score":
            command.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    if args.command == "index":
        count = build_index(metadata, args.docs, args.index)
        print(f"Wrote {count} index nodes to {args.index}")
        return
    _, questions = read_csv(args.questions)
    if args.command in {"run", "retrieve"}:
        generate(args, metadata, questions)
        return
    columns, rows = read_csv(args.output)
    validate_rows(columns, rows, questions, metadata)
    if args.command == "verify":
        print(f"Validated {len(rows)} rows in {args.output}")
        return
    metrics = evaluate(rows, questions)
    if args.metrics:
        write_json(args.metrics, metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_question"},
                     indent=2))


if __name__ == "__main__":
    main()
