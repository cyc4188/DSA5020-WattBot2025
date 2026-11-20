#!/usr/bin/env python3
"""Paired text-only versus mode2-only visual ablation for WattBot."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wattbot.evaluation import evaluate  # noqa: E402
from wattbot.multimodal import summarize_trace_usage  # noqa: E402

MODES = ("text-only", "mode2-only")
QUESTION_COLUMNS = ("id", "question", "answer_unit")
LABEL_COLUMNS = ("id", "answer_value", "ref_id", "ref_url",
                 "supporting_materials", "explanation")
ESCALATION_REASON = "rag_abstained"
DEFAULT_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_PROVIDER = "hkustgz-aigc"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_project_root(start):
    current = Path(start).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "raw" / "train_QA.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project data/raw/train_QA.csv")


def read_jsonl(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


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


def arm_status(output_dir, expected, expected_rows):
    output_dir = Path(output_dir)
    predictions = output_dir / "predictions.csv"
    config_path = output_dir / "config.json"
    trace_path = output_dir / "trace.jsonl"
    if not (predictions.is_file() and config_path.is_file()
            and trace_path.is_file()):
        return "incomplete", None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "incomplete", None
    if config != expected:
        return "config_mismatch", config
    rows = read_rows(predictions)
    if len(rows) != expected_rows:
        return "incomplete", config
    return "complete", config


def prepare_arm_dir(output_dir, expected, expected_rows, force):
    output_dir = Path(output_dir)
    status, _ = arm_status(output_dir, expected, expected_rows)
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


def run_arm(args, mode, repo_root):
    output_dir = args.output_dir / "arms" / mode
    expected = expected_config(args, mode)
    action = prepare_arm_dir(output_dir, expected, args.cohort_size, args.force)
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
    status, _ = arm_status(output_dir, expected, args.cohort_size)
    if status != "complete":
        raise RuntimeError(f"Arm {mode} finished in unexpected state {status}")
    return {"mode": mode, "action": "run", "output_dir": str(output_dir)}


def load_arm_predictions(output_dir, cohort_ids):
    rows = read_rows(Path(output_dir) / "predictions.csv")
    by_id = {row["id"]: row for row in rows}
    missing = [question_id for question_id in cohort_ids
               if question_id not in by_id]
    if missing:
        raise ValueError(f"Predictions missing cohort IDs: {missing}")
    return [by_id[question_id] for question_id in cohort_ids]


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
    here = Path(__file__).resolve().parents[1]
    default_project = find_project_root(here)
    parser = argparse.ArgumentParser(
        description="Paired text-only versus mode2-only visual ablation")
    parser.add_argument("--project", type=Path, default=default_project)
    parser.add_argument("--trace", type=Path)
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
    project = Path(args.project).resolve()
    args.project = project
    args.questions_source = (
        args.questions_source or project / "data" / "raw" / "train_QA.csv")
    args.output_dir = (
        args.output_dir or project / "outputs" / "wattbot-final" / "ablation")
    args.trace = (
        args.trace or project / "outputs" / "integration-cascade" / "train"
        / "trace.jsonl")
    args.index = args.index or project / "data" / "processed" / "wattbot-index.json"
    args.metadata = args.metadata or project / "data" / "raw" / "metadata.csv"
    args.docs = args.docs or project / "data" / "docs"
    args.catalog = args.catalog or args.output_dir / "catalog.json"
    args.figure_notes = (
        args.figure_notes or project / "data" / "processed" / "figure_notes.json")
    args.questions = args.output_dir / "questions.csv"
    args.labels = args.output_dir / "labels.local.csv"
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort_ids = derive_abstention_ids(args.trace)
    source_rows = read_rows(args.questions_source)
    questions, labels = build_cohort(source_rows, cohort_ids)
    write_rows(args.questions, QUESTION_COLUMNS, questions)
    write_rows(args.labels, LABEL_COLUMNS, labels)
    args.cohort_size = len(questions)

    args.catalog = ensure_catalog(args, repo_root)

    arm_runs = []
    for mode in MODES:
        arm_runs.append(run_arm(args, mode, repo_root))

    scores = {}
    routing = {}
    usages = {}
    per_arm = {}
    for mode in MODES:
        output_dir = args.output_dir / "arms" / mode
        predictions = load_arm_predictions(output_dir, cohort_ids)
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
