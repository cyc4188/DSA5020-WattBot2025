#!/usr/bin/env python3
"""Paired text-only versus mode2-only visual ablation for WattBot."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wattbot.evaluation import evaluate  # noqa: E402
from wattbot.multimodal import summarize_trace_usage  # noqa: E402
from wattbot.pipeline import (  # noqa: E402
    BLANK, COLUMNS, FALLBACK_ANSWER, load_metadata, read_csv, validate_rows)
from wattbot.retrieval import sha256_file, write_json  # noqa: E402

MODES = ("text-only", "mode2-only")
QUESTION_COLUMNS = ("id", "question", "answer_unit")
LABEL_COLUMNS = ("id", "answer_value", "ref_id", "ref_url",
                 "supporting_materials", "explanation")
ESCALATION_REASON = "rag_abstained"
DEFAULT_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_PROVIDER = "hkustgz-aigc"
MANIFEST_NAME = "input_manifest.json"


def find_project_root(start):
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "raw" / "train_QA.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project data/raw/train_QA.csv")


def read_jsonl(path):
    records = []
    for number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed JSON in {path}:{number}: {error}") from error
    return records


def read_rows(path):
    _, rows = read_csv(path)
    return rows


def rows_to_csv_text(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns),
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(rows_to_csv_text(columns, rows))
    temporary.replace(path)


def file_entry(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def build_input_manifest(args, subset_sha256):
    return {
        "schema": 1,
        "source_questions": file_entry(args.questions_source),
        "source_trace": file_entry(args.trace),
        "index": file_entry(args.index),
        "metadata": file_entry(args.metadata),
        "catalog": file_entry(args.catalog),
        "subset": {"path": str(Path(args.questions).resolve()),
                   "sha256": subset_sha256},
        "model": {"vlm_model": args.model, "provider": args.provider},
        "settings": {
            "top_k": args.top_k,
            "max_images": args.max_images,
            "dpi": args.dpi,
            "workers": 1,
            "hybrid_retrieval": False,
        },
    }


def check_output_manifest(output_dir):
    output_dir = Path(output_dir)
    if not (output_dir.exists() and any(output_dir.iterdir())):
        return None
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Refusing legacy/non-empty output directory without "
            f"{MANIFEST_NAME}: {output_dir}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed {manifest_path}: {error}") from error


def write_fresh_manifest(output_dir, manifest):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / MANIFEST_NAME, manifest)


def derive_abstention_ids(trace_path):
    ids = []
    seen = set()
    for record in read_jsonl(trace_path):
        if (record.get("event") == "cascade_route"
                and record.get("escalated") is True
                and record.get("reason") == ESCALATION_REASON):
            question_id = record.get("question_id")
            if question_id and question_id not in seen:
                seen.add(question_id)
                ids.append(question_id)
    if not ids:
        raise ValueError("No clean RAG abstention IDs found in trace")
    return ids


def build_cohort(source_rows, cohort_ids):
    by_id = {row["id"]: row for row in source_rows}
    missing = [question_id for question_id in cohort_ids
               if question_id not in by_id]
    if missing:
        raise ValueError(f"Cohort IDs missing from source questions: {missing}")
    questions = [{
        "id": question_id,
        "question": by_id[question_id]["question"],
        "answer_unit": by_id[question_id].get("answer_unit", "is_blank"),
    } for question_id in cohort_ids]
    labels = [{
        "id": question_id,
        **{column: by_id[question_id].get(column, "is_blank")
           for column in LABEL_COLUMNS if column != "id"},
    } for question_id in cohort_ids]
    return questions, labels


def expected_config(args, mode):
    return {
        "mode": mode,
        "vlm_model": args.model,
        "provider": args.provider,
        "top_k": args.top_k,
        "max_images": args.max_images,
        "dpi": args.dpi,
        "workers": 1,
        "hybrid_retrieval": False,
        "index": str(args.index),
        "questions": str(args.questions),
        "figure_notes": str(args.figure_notes),
    }


def completed_output_sha256(trace_path):
    completes = [record for record in read_jsonl(trace_path)
                 if record.get("event") == "run_complete"]
    if len(completes) != 1:
        return None
    value = completes[0].get("output_sha256")
    return value if isinstance(value, str) and value else None


def arm_status(output_dir, expected, manifest, questions, metadata):
    output_dir = Path(output_dir)
    predictions = output_dir / "predictions.csv"
    config_path = output_dir / "config.json"
    trace_path = output_dir / "trace.jsonl"
    manifest_path = output_dir / MANIFEST_NAME
    if not (predictions.is_file() and config_path.is_file()
            and trace_path.is_file() and manifest_path.is_file()):
        return "incomplete", None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "incomplete", None
    if existing_manifest != manifest:
        return "manifest_mismatch", config
    if config != expected:
        return "config_mismatch", config
    columns, rows = read_csv(predictions)
    try:
        validate_rows(columns, rows, questions, metadata)
    except ValueError:
        return "invalid_rows", config
    recorded = completed_output_sha256(trace_path)
    if recorded is None:
        return "incomplete", config
    if recorded != sha256_file(predictions):
        return "output_hash_mismatch", config
    return "complete", config


def prepare_arm_dir(output_dir, expected, manifest, questions, metadata, force):
    output_dir = Path(output_dir)
    status, _ = arm_status(output_dir, expected, manifest, questions, metadata)
    if status == "complete":
        return "reuse"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite non-complete arm at {output_dir} "
                f"(status={status}); rerun with --force")
        backups = output_dir.parent / "superseded"
        backups.mkdir(parents=True, exist_ok=True)
        target = backups / f"{output_dir.name}-{sha256_file_path(output_dir)}"
        output_dir.rename(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / MANIFEST_NAME, manifest)
    return "run"


def sha256_file_path(path):
    digest = hashlib.sha256()
    for item in sorted(Path(path).rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode("utf-8"))
            digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()[:12]


def run_command(command, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT,
            check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; "
            f"see {log_path}")


def ensure_catalog(args, repo_root):
    catalog = Path(args.catalog)
    if catalog.is_file():
        return catalog
    run_command([
        sys.executable, "-m", "wattbot.multimodal", "detect",
        "--metadata", str(args.metadata),
        "--docs", str(args.docs),
        "--output", str(catalog),
    ], cwd=repo_root, log_path=args.output_dir / "logs" / "catalog.log")
    return catalog


def run_arm(args, mode, repo_root, manifest, questions, metadata):
    output_dir = args.output_dir / "arms" / mode
    expected = expected_config(args, mode)
    action = prepare_arm_dir(
        output_dir, expected, manifest, questions, metadata, args.force)
    if action == "reuse":
        return {"mode": mode, "action": "reuse", "output_dir": str(output_dir)}
    command = [
        sys.executable, "-m", "wattbot.multimodal", "run",
        "--metadata", str(args.metadata),
        "--docs", str(args.docs),
        "--index", str(args.index),
        "--questions", str(args.questions),
        "--catalog", str(args.catalog),
        "--figure-notes", str(args.figure_notes),
        "--output-dir", str(output_dir),
        "--mode", mode,
        "--vlm-model", args.model,
        "--provider", args.provider,
        "--top-k", str(args.top_k),
        "--max-images", str(args.max_images),
        "--dpi", str(args.dpi),
        "--workers", "1",
    ]
    run_command(command, cwd=repo_root,
                log_path=args.output_dir / "logs" / f"{mode}.log")
    status, _ = arm_status(
        output_dir, expected, manifest, questions, metadata)
    if status != "complete":
        raise RuntimeError(f"Arm {mode} finished in unexpected state {status}")
    return {"mode": mode, "action": "run", "output_dir": str(output_dir)}


def load_arm_predictions(output_dir, questions, metadata):
    columns, rows = read_csv(Path(output_dir) / "predictions.csv")
    validate_rows(columns, rows, questions, metadata)
    return rows


def score_arm(predictions, labels):
    metrics = evaluate(predictions, labels)
    return metrics


def aggregate_only(metrics):
    return {key: value for key, value in metrics.items()
            if key != "per_question"}


def paired_changes(text_metrics, visual_metrics):
    text = {row["id"]: row for row in text_metrics["per_question"]}
    visual = {row["id"]: row for row in visual_metrics["per_question"]}
    rows = []
    for question_id in text:
        if question_id not in visual:
            continue
        delta = {key: visual[question_id][key] - text[question_id][key]
                 for key in ("value", "citation", "abstention", "weighted")}
        rows.append({"id": question_id, **delta})
    summary = {
        f"delta_{key}": sum(row[key] for row in rows) / len(rows)
        for key in ("value", "citation", "abstention", "weighted")
    }
    deltas = [row["weighted"] for row in rows]
    summary["improved"] = sum(value > 0 for value in deltas)
    summary["worsened"] = sum(value < 0 for value in deltas)
    summary["unchanged"] = sum(value == 0 for value in deltas)
    summary["n"] = len(rows)
    summary["caveat"] = (
        f"Exploratory paired comparison on n={len(rows)}; no significance "
        "testing or generalization claim is warranted.")
    return {"summary": summary, "per_question": rows}


def routing_summary(trace_path, mode):
    counts = {}
    escalations = 0
    decisions = {}
    errors = 0
    for record in read_jsonl(trace_path):
        event = record.get("event")
        counts[event] = counts.get(event, 0) + 1
        if event == "visual_escalation" and record.get("escalate") is True:
            escalations += 1
        elif event == "multimodal_decision":
            decision = str(record.get("decision"))
            decisions[decision] = decisions.get(decision, 0) + 1
        elif event in {"question_error", "run_error"}:
            errors += 1
    return {
        "mode": mode,
        "trace": str(trace_path),
        "event_counts": counts,
        "visual_escalations": escalations,
        "multimodal_decisions": decisions,
        "errors": errors,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Paired text-only versus mode2-only visual ablation")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--questions-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--docs", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--figure-notes", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    return parser


def resolve_args(args):
    project = (Path(args.project).resolve() if args.project
               else find_project_root(Path(__file__).resolve().parents[1]))
    args.project = project
    args.questions_source = (
        args.questions_source or project / "data" / "raw" / "train_QA.csv")
    args.output_dir = (
        args.output_dir or project / "outputs" / "wattbot-final" / "ablation")
    args.index = args.index or project / "data" / "processed" / "wattbot-index.json"
    args.metadata = args.metadata or project / "data" / "raw" / "metadata.csv"
    args.docs = args.docs or project / "data" / "docs"
    args.catalog = args.catalog or args.output_dir / "catalog.json"
    args.figure_notes = (
        args.figure_notes or project / "data" / "processed" / "figure_notes.json")
    for key in ("questions_source", "output_dir", "trace", "index", "metadata",
                "docs", "catalog", "figure_notes"):
        setattr(args, key, Path(getattr(args, key)).resolve())
    args.questions = args.output_dir / "questions.csv"
    args.labels = args.output_dir / "labels.local.csv"
    for path in (args.trace, args.questions_source, args.index, args.metadata,
                 args.docs):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing required input: {path}")
    return args


def main(argv=None):
    args = resolve_args(build_parser().parse_args(argv))
    repo_root = Path(__file__).resolve().parents[1]
    existing_manifest = check_output_manifest(args.output_dir)

    cohort_ids = derive_abstention_ids(args.trace)
    source_rows = read_rows(args.questions_source)
    questions, labels = build_cohort(source_rows, cohort_ids)
    subset_bytes = rows_to_csv_text(QUESTION_COLUMNS, questions).encode("utf-8")
    args.cohort_size = len(questions)

    args.catalog = ensure_catalog(args, repo_root)
    metadata = load_metadata(args.metadata)
    manifest = build_input_manifest(
        args, hashlib.sha256(subset_bytes).hexdigest())
    if existing_manifest is None:
        write_fresh_manifest(args.output_dir, manifest)
    elif existing_manifest != manifest:
        raise ValueError(
            f"Input manifest mismatch for existing output directory: "
            f"{args.output_dir}")
    write_rows(args.questions, QUESTION_COLUMNS, questions)
    write_rows(args.labels, LABEL_COLUMNS, labels)

    arm_runs = []
    for mode in MODES:
        arm_runs.append(run_arm(
            args, mode, repo_root, manifest, questions, metadata))

    scores = {}
    routing = {}
    usages = {}
    per_arm = {}
    for mode in MODES:
        output_dir = args.output_dir / "arms" / mode
        predictions = load_arm_predictions(output_dir, questions, metadata)
        metrics = score_arm(predictions, labels)
        per_arm[mode] = metrics
        trace_path = output_dir / "trace.jsonl"
        usages[mode] = summarize_trace_usage(trace_path)
        routing[mode] = routing_summary(trace_path, mode)
        scores[mode] = {
            "aggregate": aggregate_only(metrics),
            "usage": usages[mode],
            "images_sent": usages[mode].get("images_sent", 0),
        }

    scores["paired"] = paired_changes(
        per_arm["text-only"], per_arm["mode2-only"])
    scores["comparison"] = {
        "cohort_size": args.cohort_size,
        "model": args.model,
        "provider": args.provider,
        "top_k": args.top_k,
        "max_images": args.max_images,
        "dpi": args.dpi,
        "workers": 1,
        "modes": list(MODES),
        "paired_direction": "mode2-only minus text-only",
    }

    repro = {
        "project": str(args.project),
        "input_manifest": {
            "path": str(args.output_dir / MANIFEST_NAME),
            "sha256": sha256_file(args.output_dir / MANIFEST_NAME)},
        "trace": {"path": str(args.trace), "sha256": sha256_file(args.trace)},
        "questions": {"path": str(args.questions),
                      "sha256": sha256_file(args.questions)},
        "labels": {"path": str(args.labels),
                   "sha256": sha256_file(args.labels)},
        "index": {"path": str(args.index), "sha256": sha256_file(args.index)},
        "catalog": {"path": str(args.catalog),
                    "sha256": sha256_file(args.catalog)},
        "model": args.model,
        "provider": args.provider,
        "modes": list(MODES),
        "settings": {
            "top_k": args.top_k,
            "max_images": args.max_images,
            "dpi": args.dpi,
            "workers": 1,
            "hybrid_retrieval": False,
        },
        "cohort_ids": cohort_ids,
        "cohort_size": args.cohort_size,
        "arms": arm_runs,
        "python": sys.executable,
    }

    write_json(args.output_dir / "scores.json", scores)
    write_json(args.output_dir / "routing.json", routing)
    write_json(args.output_dir / "repro.json", repro)

    print(f"cohort_size={args.cohort_size}")
    for mode in MODES:
        aggregate = scores[mode]["aggregate"]
        usage = scores[mode]["usage"]
        print(
            f"{mode}: weighted={aggregate['weighted']:.4f} "
            f"value={aggregate['value']:.4f} "
            f"citation={aggregate['citation']:.4f} "
            f"abstention={aggregate['abstention']:.4f} "
            f"requests={usage['requests']} "
            f"images_sent={usage['images_sent']}")
    paired = scores["paired"]["summary"]
    print(
        f"paired_delta_weighted={paired['delta_weighted']:.4f} "
        f"n={paired['n']} improved={paired['improved']} "
        f"worsened={paired['worsened']} unchanged={paired['unchanged']}")
    print(f"artifacts={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
