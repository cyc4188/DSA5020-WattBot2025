"""Compare saved WattBot runs with an offline, standard-library manifest.

The evaluator never calls a model.  It loads saved prediction CSVs, aligns them
to the labelled train questions by exact question ID and question text, scores
them with :func:`wattbot.evaluation.evaluate`, reads provider-reported token
usage from saved traces when per-question totals are available, and optionally
replays abstention-only cascades from saved per-question predictions.

Only aggregate diagnostics are printed/written.  Raw questions, answers,
traces and credentials are not included in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wattbot.ensemble import (
    compute_pareto_frontier,
    format_pareto_markdown,
    has_valid_citations,
    is_blank_row,
    load_trace_metadata,
)
from wattbot.evaluation import evaluate
from wattbot.pipeline import COLUMNS, load_metadata, read_csv, validate_rows

DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "train_QA.csv"
DEFAULT_METADATA = ROOT / "data" / "raw" / "metadata.csv"
SCHEMA_VERSION = 1
PROVENANCE_MODES = {"historical_replay", "live"}
PROVENANCE_STATUSES = {"successful", "partial", "unsuccessful", "non_comparable"}
ROLES = {"candidate", "fallback", "live_cascade", "baseline"}


class ManifestError(ValueError):
    """Raised for malformed or inconsistent manifest inputs."""


def _read_json(path: Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve(path_value, base: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (base / path)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ManifestError(f"Missing {label}: {path}")
    return path


def load_questions(path: Path) -> list[dict]:
    columns, questions = read_csv(path)
    if columns != COLUMNS:
        raise ManifestError(f"Question file has unexpected columns: {columns}")
    seen: set[str] = set()
    for question in questions:
        qid = question.get("id")
        if not qid:
            raise ManifestError("Question file contains an empty question ID")
        if qid in seen:
            raise ManifestError(f"Duplicate question ID in question file: {qid}")
        seen.add(qid)
    if not questions:
        raise ManifestError("Question file is empty")
    return questions


def align_predictions(path: Path, questions: list[dict]) -> list[dict]:
    """Return rows in question order, rejecting extra/duplicate/missing IDs."""
    columns, rows = read_csv(path)
    if columns != COLUMNS:
        raise ManifestError(
            f"Prediction file {path} has columns {columns}, expected {COLUMNS}")
    expected = {question["id"]: question for question in questions}
    aligned: dict[str, dict] = {}
    for row in rows:
        if None in row:
            raise ManifestError(f"Prediction file {path} has excess columns or values")
        qid = row.get("id")
        if qid not in expected:
            raise ManifestError(f"Prediction file {path} has an extra question ID: {qid}")
        if qid in aligned:
            raise ManifestError(f"Prediction file {path} has a duplicate question ID: {qid}")
        if row.get("question") != expected[qid]["question"]:
            raise ManifestError(
                f"Prediction file {path} has a question identity mismatch at {qid}")
        aligned[qid] = row
    missing = [question["id"] for question in questions if question["id"] not in aligned]
    if missing:
        raise ManifestError(
            f"Prediction file {path} is missing question IDs: {missing[:5]}")
    if len(rows) != len(questions):
        raise ManifestError(
            f"Prediction file {path} has {len(rows)} rows, expected {len(questions)}")
    return [aligned[question["id"]] for question in questions]


def _find_usage_record(node):
    if not isinstance(node, dict):
        return None
    if isinstance(node.get("total_tokens"), int) or isinstance(
            node.get("mean_total_tokens_per_question"), (int, float)):
        return node
    for key, value in node.items():
        if key == "by_stage":
            continue
        found = _find_usage_record(value)
        if found is not None:
            return found
    return None


def _reported_usage(path: Path, expected_count: int) -> dict:
    record = _find_usage_record(_read_json(path))
    if record is None:
        return {"available": False, "reason": "no aggregate usage record",
                "path": str(path)}
    questions = record.get("questions")
    if isinstance(questions, int) and questions != expected_count:
        return {"available": False,
                "reason": f"reported usage covers {questions} questions, "
                          f"expected {expected_count}", "path": str(path)}
    total = record.get("total_tokens")
    mean = record.get("mean_total_tokens_per_question")
    if mean is None:
        mean = record.get("mean_tokens_per_question")
    if mean is None and isinstance(total, int):
        mean = total / expected_count
    if total is None and isinstance(mean, (int, float)):
        total = mean * expected_count
    if not isinstance(total, (int, float)) and not isinstance(mean, (int, float)):
        return {"available": False, "reason": "aggregate usage has no token totals",
                "path": str(path)}
    return {
        "available": True,
        "per_question_available": False,
        "total_tokens": round(float(total), 3) if total is not None else None,
        "mean_tokens_per_question":
            round(float(mean), 3) if mean is not None else None,
        "token_source": "reported_aggregate",
        "usage_path": str(path),
    }


def token_report(run: dict, expected_ids: list[str], base: Path) -> dict:
    trace_value = run.get("trace")
    if trace_value:
        trace_path = _require_file(_resolve(trace_value, base), "trace")
        metadata = load_trace_metadata(trace_path)
        per_question = {
            qid: int((metadata.get(qid) or {}).get("total_tokens") or 0)
            for qid in expected_ids
        }
        missing = [qid for qid in expected_ids if qid not in metadata]
        zero = [qid for qid in expected_ids if per_question[qid] == 0]
        if not missing and not zero:
            total = sum(per_question.values())
            return {
                "available": True,
                "per_question_available": True,
                "total_tokens": total,
                "mean_tokens_per_question": round(total / len(expected_ids), 3),
                "per_question_tokens": per_question,
                "token_source": "per_question_trace",
                "trace_path": str(trace_path),
            }
        trace_note = (f"trace incomplete for scoring: {len(missing)} missing, "
                      f"{len(zero)} zero-token questions")
    else:
        trace_note = None

    usage_value = run.get("usage")
    if usage_value:
        usage_path = _require_file(_resolve(usage_value, base), "usage")
        reported = _reported_usage(usage_path, len(expected_ids))
        if reported.get("available"):
            if trace_note:
                reported["trace_note"] = trace_note
            return reported
        if trace_note:
            reported["reason"] = f"{trace_note}; {reported.get('reason')}"
        return reported

    return {
        "available": False,
        "per_question_available": False,
        "total_tokens": None,
        "mean_tokens_per_question": None,
        "token_source": "unavailable",
        "reason": trace_note or "no trace or reported usage supplied",
    }


def _provenance(run: dict) -> dict:
    raw = run.get("provenance") or {}
    if isinstance(raw, str):
        raw = {"mode": raw}
    if not isinstance(raw, dict):
        raise ManifestError(f"Run {run['id']} has invalid provenance")
    mode = raw.get("mode", "historical_replay")
    status = raw.get("status", "successful")
    if mode not in PROVENANCE_MODES:
        raise ManifestError(f"Run {run['id']} has unknown provenance mode: {mode}")
    if status not in PROVENANCE_STATUSES:
        raise ManifestError(f"Run {run['id']} has unknown provenance status: {status}")
    return {
        "mode": mode,
        "status": status,
        "live_routing": mode == "live",
        "notes": raw.get("notes") or "",
    }


def _score_rows(rows: list[dict], questions: list[dict], metadata: dict) -> tuple[dict, list[dict]]:
    metrics = evaluate(rows, questions)
    details = metrics.pop("per_question")
    answered = sum(1 for detail in details if not detail["predicted_blank"])
    valid_cited = sum(
        1 for detail, row in zip(details, rows, strict=True)
        if not detail["predicted_blank"] and has_valid_citations(row, metadata)
    )
    schema_valid = 0
    for question, row in zip(questions, rows, strict=True):
        try:
            validate_rows(COLUMNS, [row], [question], metadata)
        except (KeyError, SyntaxError, TypeError, ValueError):
            continue
        schema_valid += 1
    citation_complete = (
        answered > 0
        and valid_cited == answered
        and schema_valid == len(rows)
    )
    metrics.update({
        "answered": answered,
        "abstained": len(rows) - answered,
        "answered_with_valid_citations": valid_cited,
        "schema_valid_rows": schema_valid,
        "citation_schema_valid_rate": (
            round(valid_cited / answered, 4) if answered else None),
        "citation_complete": citation_complete,
    })
    return metrics, details


def _comparability(run: dict) -> dict:
    reasons: list[str] = []
    if run["provenance"]["status"] != "successful":
        reasons.append(f"provenance status is {run['provenance']['status']}")
    if not run["metrics"]["citation_complete"]:
        reasons.append("incomplete or schema-invalid citations")
    score_comparable = run["provenance"]["status"] == "successful"
    pareto_eligible = (
        score_comparable
        and run["metrics"]["citation_complete"]
        and run["tokens"].get("mean_tokens_per_question") is not None
    )
    if run["tokens"].get("mean_tokens_per_question") is None:
        reasons.append("no per-run token measurement")
    return {
        "score_comparable": score_comparable,
        "pareto_eligible": pareto_eligible,
        "reasons": reasons,
    }


def load_run(run: dict, questions: list[dict], metadata: dict, base: Path) -> dict:
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise ManifestError("Every run needs a non-empty string id")
    role = run.get("role", "candidate")
    if role not in ROLES:
        raise ManifestError(f"Run {run_id} has unknown role: {role}")
    predictions_value = run.get("predictions")
    if not predictions_value:
        raise ManifestError(f"Run {run_id} needs a predictions path")
    predictions_path = _require_file(
        _resolve(predictions_value, base), f"predictions for {run_id}")
    rows = align_predictions(predictions_path, questions)
    metrics, details = _score_rows(rows, questions, metadata)
    expected_ids = [question["id"] for question in questions]
    result = {
        "id": run_id,
        "label": run.get("label") or run_id,
        "role": role,
        "predictions_path": str(predictions_path),
        "gold_questions": len(questions),
        "metrics": metrics,
        "tokens": token_report(run, expected_ids, base),
        "provenance": _provenance(run),
        "_rows": rows,
        "_details": details,
    }
    result["comparability"] = _comparability(result)
    return result


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if not key.startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _normalise_manifest(manifest: dict, path: Path) -> dict:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"Manifest schema_version must be {SCHEMA_VERSION}")
    questions_path = _require_file(
        _resolve(manifest.get("questions") or DEFAULT_QUESTIONS, path.parent),
        "question file")
    metadata_path = _require_file(
        _resolve(manifest.get("metadata") or DEFAULT_METADATA, path.parent),
        "metadata file")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ManifestError("Manifest needs a non-empty runs list")
    identifiers: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ManifestError("Every run must be an object")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ManifestError("Every run needs a non-empty string id")
        if run_id in identifiers:
            raise ManifestError(f"Duplicate run id: {run_id}")
        identifiers.add(run_id)
    pairings = manifest.get("pairings") or []
    if not isinstance(pairings, list):
        raise ManifestError("pairings must be a list")
    for pairing in pairings:
        if not isinstance(pairing, dict) or not pairing.get("name"):
            raise ManifestError("Every pairing needs a name")
        for key in ("tier1", "fallback"):
            if pairing.get(key) not in identifiers:
                raise ManifestError(
                    f"Pairing {pairing['name']} references unknown run {pairing.get(key)}")
    return {
        "manifest": manifest,
        "questions_path": questions_path,
        "metadata_path": metadata_path,
        "pairings": pairings,
    }


def _complementary(runs: list[dict], questions: list[dict]) -> tuple[list[dict], list[dict]]:
    eligible = [
        run for run in runs
        if run["comparability"]["score_comparable"]
        and run["metrics"]["citation_complete"]
        and run["role"] != "baseline"
    ]
    correct = {
        run["id"]: {detail["id"] for detail in run["_details"] if detail["value"] == 1}
        for run in eligible
    }
    all_ids = [question["id"] for question in questions]
    pairwise: list[dict] = []
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            left_correct = correct[left["id"]]
            right_correct = correct[right["id"]]
            both = left_correct & right_correct
            left_only = left_correct - right_correct
            right_only = right_correct - left_correct
            both_wrong = set(all_ids) - (left_correct | right_correct)
            pairwise.append({
                "run_a": left["id"], "run_b": right["id"],
                "a_only_correct": len(left_only),
                "b_only_correct": len(right_only),
                "both_correct": len(both),
                "both_wrong": len(both_wrong),
                "union_correct": len(left_correct | right_correct),
                "a_only_correct_ids": sorted(left_only),
                "b_only_correct_ids": sorted(right_only),
            })
    unique: list[dict] = []
    for run in eligible:
        others = set().union(*(correct[other["id"]] for other in eligible
                               if other["id"] != run["id"])) if len(eligible) > 1 else set()
        only = correct[run["id"]] - others
        unique.append({
            "run": run["id"],
            "unique_correct_count": len(only),
            "unique_correct_ids": sorted(only),
            "correct_count": len(correct[run["id"]]),
        })
    return pairwise, unique


def _pareto(runs: list[dict], cascades: list[dict]) -> list[dict]:
    entries: list[dict] = []
    for run in runs:
        if not run["comparability"]["pareto_eligible"]:
            continue
        metrics = run["metrics"]
        tokens = run["tokens"]
        entries.append({
            "id": run["id"],
            "name": run["label"],
            "system": run["label"],
            "role": run["role"],
            "provenance_mode": run["provenance"]["mode"],
            "value": metrics["value"],
            "citation": metrics["citation"],
            "abstention": metrics["abstention"],
            "weighted": metrics["weighted"],
            "mean_tokens_per_question": tokens["mean_tokens_per_question"],
            "token_source": tokens["token_source"],
            "escalation_rate": run.get("escalation_rate"),
        })
    for cascade in cascades:
        metrics = cascade["metrics"]
        tokens = cascade["tokens"]
        if (cascade["provenance"]["status"] != "successful"
                or not metrics["citation_complete"]
                or tokens.get("mean_tokens_per_question") is None):
            continue
        label = f"Simulated: {cascade['name']}"
        entries.append({
            "id": cascade["name"],
            "name": label,
            "system": label,
            "role": "simulated_cascade",
            "provenance_mode": cascade["provenance"]["mode"],
            "value": metrics["value"],
            "citation": metrics["citation"],
            "abstention": metrics["abstention"],
            "weighted": metrics["weighted"],
            "mean_tokens_per_question": tokens["mean_tokens_per_question"],
            "token_source": tokens["token_source"],
            "escalation_rate": cascade["escalation_rate"],
        })
    return compute_pareto_frontier(entries)


def _predicted_abstention(row: dict) -> bool:
    return is_blank_row(row)


def simulate_pairing(pairing: dict, runs_by_id: dict[str, dict],
                     questions: list[dict], metadata: dict) -> dict:
    tier1 = runs_by_id[pairing["tier1"]]
    fallback = runs_by_id[pairing["fallback"]]
    name = pairing["name"]
    status = "successful"
    if (tier1["provenance"]["status"] != "successful"
            or fallback["provenance"]["status"] != "successful"):
        status = "non_comparable"
    rows: list[dict] = []
    escalated: list[str] = []
    resolved_by_fallback = 0
    fallback_abstained = 0
    for question, tier1_row, fallback_row in zip(
            questions, tier1["_rows"], fallback["_rows"], strict=True):
        if _predicted_abstention(tier1_row):
            escalated.append(question["id"])
            rows.append(fallback_row)
            if _predicted_abstention(fallback_row):
                fallback_abstained += 1
            else:
                resolved_by_fallback += 1
        else:
            rows.append(tier1_row)
    metrics, details = _score_rows(rows, questions, metadata)
    tier1_metrics = tier1["metrics"]
    tokens = {
        "available": False,
        "per_question_available": False,
        "total_tokens": None,
        "mean_tokens_per_question": None,
        "token_source": "non_comparable",
        "measurement": "historical_replay",
    }
    if (tier1["tokens"].get("per_question_available")
            and fallback["tokens"].get("per_question_available")):
        tier1_tokens = tier1["tokens"]["per_question_tokens"]
        fallback_tokens = fallback["tokens"]["per_question_tokens"]
        total = sum(tier1_tokens.values()) + sum(
            fallback_tokens[qid] for qid in escalated)
        tokens.update({
            "available": True,
            "per_question_available": True,
            "total_tokens": total,
            "mean_tokens_per_question": round(total / len(questions), 3),
            "token_source": "per_question_trace_replay",
        })
    provenance_raw = pairing.get("provenance") or {}
    if isinstance(provenance_raw, str):
        provenance_raw = {"mode": provenance_raw}
    mode = provenance_raw.get("mode", "historical_replay")
    if mode not in PROVENANCE_MODES:
        raise ManifestError(f"Pairing {name} has unknown provenance mode: {mode}")
    if provenance_raw.get("status"):
        status = provenance_raw["status"]
    result = {
        "name": name,
        "tier1": tier1["id"],
        "fallback": fallback["id"],
        "measurement": "historical_replay",
        "provenance": {
            "mode": "historical_replay",
            "declared_mode": mode,
            "status": status,
            "live_routing": False,
            "notes": provenance_raw.get("notes") or
                     "Abstention-only substitution from saved per-question predictions.",
        },
        "gold_questions": len(questions),
        "escalated": len(escalated),
        "escalation_rate": round(len(escalated) / len(questions), 4),
        "resolved_by_fallback": resolved_by_fallback,
        "fallback_abstained": fallback_abstained,
        "metrics": metrics,
        "tokens": tokens,
        "delta_vs_tier1": {
            key: round(metrics[key] - tier1_metrics[key], 6)
            for key in ("value", "citation", "abstention", "weighted")
        },
        "tier1_metrics": {key: tier1_metrics[key]
                          for key in ("value", "citation", "abstention", "weighted")},
        "fallback_metrics": {key: fallback["metrics"][key]
                             for key in ("value", "citation", "abstention", "weighted")},
    }
    live_reference = pairing.get("live_reference")
    if live_reference and live_reference in runs_by_id:
        live = runs_by_id[live_reference]
        result["live_reference"] = {
            "run": live["id"],
            "label": live["label"],
            "provenance": live["provenance"],
            "metrics": {key: live["metrics"][key]
                        for key in ("value", "citation", "abstention", "weighted")},
            "tokens": live["tokens"],
            "delta_vs_simulation": {
                key: round(live["metrics"][key] - metrics[key], 6)
                for key in ("value", "citation", "abstention", "weighted")
            },
        }
    return result


def build_report(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest must be a JSON object")
    normalised = _normalise_manifest(manifest, manifest_path)
    questions = load_questions(normalised["questions_path"])
    metadata = load_metadata(normalised["metadata_path"])
    runs = [load_run(run, questions, metadata, manifest_path.parent)
            for run in manifest["runs"]]
    runs_by_id = {run["id"]: run for run in runs}
    pairings = [simulate_pairing(pairing, runs_by_id, questions, metadata)
                for pairing in normalised["pairings"]]
    pairwise, unique = _complementary(runs, questions)
    pareto_entries = _pareto(runs, pairings)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "questions_path": str(normalised["questions_path"]),
        "metadata_path": str(normalised["metadata_path"]),
        "gold_questions": len(questions),
        "notes": [
            "Scores are local train diagnostics, not Kaggle leaderboard scores.",
            "historical_replay means offline scoring/substitution from saved "
            "per-question artifacts; live means a saved end-to-end routing run.",
            "Simulated cascades are historical replays, not live routing.",
            "Token means use provider-reported per-question trace usage when "
            "complete; otherwise reported aggregate usage or non-comparable.",
            "Runs with incomplete citations are shown separately and excluded "
            "from Pareto and complementary comparisons.",
        ],
        "runs": [_public(run) for run in runs],
        "baselines_incomplete_citations": [
            _public(run) for run in runs
            if not run["metrics"]["citation_complete"]
        ],
        "pareto_entries": pareto_entries,
        "complementary_errors": pairwise,
        "unique_correct": unique,
        "simulated_cascades": pairings,
        "exclusions": manifest.get("exclusions") or [],
    }
    return report


def _percent(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _number(value, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _tokens(value) -> str:
    if value is None:
        return "—"
    return f"{float(value):,.0f}"


def _run_table(runs: list[dict]) -> list[str]:
    lines = [
        "| Run | Label | Role | Mode | Status | Value | Citation | Abstention | "
        "Weighted | Valid cit. | Schema rows | Mean tok/q | Token source |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        metrics = run["metrics"]
        tokens = run["tokens"]
        lines.append(
            f"| {run['id']} | {run['label']} | {run['role']} | "
            f"{run['provenance']['mode']} | {run['provenance']['status']} | "
            f"{_percent(metrics['value'])} | {_percent(metrics['citation'])} | "
            f"{_percent(metrics['abstention'])} | {_number(metrics['weighted'])} | "
            f"{metrics['answered_with_valid_citations']}/{metrics['answered']} | "
            f"{metrics['schema_valid_rows']}/{run['gold_questions']} | "
            f"{_tokens(tokens.get('mean_tokens_per_question'))} | "
            f"{tokens.get('token_source', '—')} |"
        )
    return lines


def render_markdown(report: dict) -> str:
    lines = [
        "# WattBot offline run comparison",
        "",
        f"Generated: `{report['generated_at']}`  ",
        f"Manifest: `{report['manifest_path']}`  ",
        f"Gold questions: {report['gold_questions']}",
        "",
        "## Measurement notes",
        "",
    ]
    lines.extend(f"- {note}" for note in report["notes"])
    lines.extend([
        "",
        "## Runs",
        "",
        *_run_table(report["runs"]),
        "",
    ])
    incomplete = report["baselines_incomplete_citations"]
    lines.extend(["## Incomplete-citation baselines (listed separately)", ""])
    if incomplete:
        lines.extend(_run_table(incomplete))
    else:
        lines.append("None.")
    lines.extend([
        "",
        "## Pareto frontier (complete-citation runs and simulated cascades)",
        "",
    ])
    if report["pareto_entries"]:
        lines.append(format_pareto_markdown(report["pareto_entries"]))
    else:
        lines.append("No Pareto-eligible runs.")
    lines.extend([
        "",
        "## Simulated abstention-only cascades (historical replay)",
        "",
        "| Pairing | Tier 1 → fallback | Escalated | Value | Citation | Abstention | "
        "Weighted | Δ weighted vs tier 1 | Mean tok/q | Token source | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for cascade in report["simulated_cascades"]:
        metrics = cascade["metrics"]
        tokens = cascade["tokens"]
        delta = cascade["delta_vs_tier1"]["weighted"]
        lines.append(
            f"| {cascade['name']} | {cascade['tier1']} → {cascade['fallback']} | "
            f"{cascade['escalated']} ({_percent(cascade['escalation_rate'])}) | "
            f"{_percent(metrics['value'])} | {_percent(metrics['citation'])} | "
            f"{_percent(metrics['abstention'])} | {_number(metrics['weighted'])} | "
            f"{_number(delta)} | {_tokens(tokens.get('mean_tokens_per_question'))} | "
            f"{tokens.get('token_source', '—')} | "
            f"{cascade['provenance']['status']} |"
        )
    lines.extend([
        "",
        "## Complementary value errors (complete-citation candidate systems)",
        "",
        "| Run A | Run B | A only correct | B only correct | Both correct | "
        "Both wrong | Union correct |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for pair in report["complementary_errors"]:
        lines.append(
            f"| {pair['run_a']} | {pair['run_b']} | {pair['a_only_correct']} | "
            f"{pair['b_only_correct']} | {pair['both_correct']} | "
            f"{pair['both_wrong']} | {pair['union_correct']} |"
        )
    lines.extend([
        "",
        "## Unique correct answers",
        "",
        "| Run | Correct | Unique correct vs other candidate systems |",
        "| --- | ---: | ---: |",
    ])
    for item in report["unique_correct"]:
        lines.append(
            f"| {item['run']} | {item['correct_count']} | "
            f"{item['unique_correct_count']} |"
        )
    lines.extend(["", "## Excluded / non-comparable measurements", ""])
    if report["exclusions"]:
        for item in report["exclusions"]:
            lines.append(f"- `{item.get('id', 'unnamed')}`: {item.get('reason', '')}")
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def _default_output(manifest_path: Path, suffix: str) -> Path:
    return manifest_path.with_name(manifest_path.stem + suffix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/compare_wattbot_runs.py",
        description="Offline comparison of saved WattBot runs from a JSON manifest.")
    parser.add_argument("manifest", type=Path, help="JSON run manifest")
    parser.add_argument("--report-json", type=Path,
                        help="Output JSON report (default: <manifest>.report.json)")
    parser.add_argument("--report-md", type=Path,
                        help="Output Markdown report (default: <manifest>.report.md)")
    args = parser.parse_args(argv)
    manifest_path = args.manifest
    if not manifest_path.is_file():
        parser.error(f"manifest does not exist: {manifest_path}")
    try:
        report = build_report(manifest_path)
    except (ManifestError, KeyError, TypeError, ValueError, SyntaxError) as error:
        parser.error(str(error))
    report_json = args.report_json or _default_output(manifest_path, ".report.json")
    report_md = args.report_md or _default_output(manifest_path, ".report.md")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_md.write_text(render_markdown(report), encoding="utf-8")
    summary = {
        "manifest": str(manifest_path),
        "report_json": str(report_json),
        "report_md": str(report_md),
        "runs": len(report["runs"]),
        "pareto_runs": sum(1 for entry in report["pareto_entries"]
                           if entry.get("pareto_optimal")),
        "simulated_cascades": len(report["simulated_cascades"]),
        "incomplete_citation_baselines": len(report["baselines_incomplete_citations"]),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
