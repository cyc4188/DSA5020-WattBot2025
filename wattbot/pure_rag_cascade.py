"""Online two-tier Pure-RAG Difficulty-Aware Cascade.

The pipeline runs two independent Tier-1 solvers on every question:

* ``pi_v2``    - :mod:`wattbot.pi_agent_v2` (Pi agentic RAG v2)
* ``multimodal`` - :mod:`wattbot.multimodal` in ``mode2-only``

A consensus gate accepts a question immediately when both solvers return a
validly cited answer that agrees within the competition tolerance.  Only the
disagreeing / abstaining / uncited questions are escalated to Tier-2
:mod:`wattbot.doc_routing`.  Tier-2 therefore never sees the questions that
already have a corroborated Tier-1 answer.

Arbitration across the three candidates reuses the exact clustering and
corroboration rules from :func:`wattbot.ensemble.cascaded_route`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from wattbot.__main__ import (
    DEFAULT_CLEAN_INDEX,
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    DATA,
    atomic_csv,
)
from wattbot.ensemble import (
    _build_output_row,
    cluster_candidates,
    has_valid_citations,
    is_blank_row,
    load_trace_metadata,
    score_candidate,
    values_equivalent,
)
from wattbot.evaluation import evaluate
from wattbot.pipeline import (
    BLANK,
    COLUMNS,
    abstention,
    load_metadata,
    read_csv,
    validate_rows,
)
from wattbot.retrieval import write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = DATA / "docs"
DEFAULT_CACHE_DIR = ROOT / "outputs/doc-routing/corpus_cache"
DEFAULT_VLM_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_PROVIDER = "hkustgz-aigc"
DEFAULT_FIGURE_NOTES = DATA / "processed" / "figure_notes.json"

TierSolver = Callable[[Sequence[dict], Path], "StageResult"]


# ---------------------------------------------------------------------------
# Stage results and trace token accounting
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Per-question rows and exact provider token usage for one stage."""

    rows: dict[str, dict]
    tokens: dict[str, int] = field(default_factory=dict)
    trace_path: Path | None = None
    stage_dir: Path | None = None

    def tokens_for(self, question_id: str) -> int:
        return int(self.tokens.get(str(question_id), 0) or 0)


def trace_tokens_by_question(trace_path: Path | None) -> dict[str, int]:
    """Sum provider-reported ``total_tokens`` per question from a JSONL trace."""
    if trace_path is None or not Path(trace_path).is_file():
        return {}
    totals: dict[str, int] = {}
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") != "llm_response":
            continue
        question_id = event.get("question_id")
        if question_id is None:
            continue
        usage = event.get("usage") or {}
        total = usage.get("total_tokens")
        if not isinstance(total, int):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = (prompt if isinstance(prompt, int) else 0) + (
                completion if isinstance(completion, int) else 0
            )
        totals[str(question_id)] = totals.get(str(question_id), 0) + int(total)
    return totals


# ---------------------------------------------------------------------------
# Consensus gate and three-way arbitration
# ---------------------------------------------------------------------------


def _candidate(run_index: int, run_name: str, row: dict, base_weight: float,
               metadata: dict, trace_meta: dict | None = None) -> dict:
    score, _ = score_candidate(row, trace_meta or {}, base_weight, metadata)
    return {
        "run_index": run_index,
        "run_name": run_name,
        "row": row,
        "answer_value": row.get("answer_value", BLANK),
        "score": score,
        "valid_cited": has_valid_citations(row, metadata),
    }


def consensus_gate(question: dict, row_a: dict, row_b: dict, metadata: dict) -> tuple[bool, list[str]]:
    """Return ``(escalate, reasons)`` for a Tier-1 pair.

    A question is accepted when both paths have valid citations and agree within
    the 0.1% numeric tolerance.  Any abstention, missing citation, or value
    disagreement escalates the question to Tier-2.
    """
    a_blank, b_blank = is_blank_row(row_a), is_blank_row(row_b)
    a_valid = has_valid_citations(row_a, metadata)
    b_valid = has_valid_citations(row_b, metadata)
    reasons: list[str] = []
    if a_blank or b_blank:
        reasons.append("abstain")
    if (not a_blank and not a_valid) or (not b_blank and not b_valid):
        reasons.append("missing_citation")
    if (not a_blank and not b_blank
            and not values_equivalent(row_a.get("answer_value", BLANK),
                                      row_b.get("answer_value", BLANK))):
        reasons.append("disagreement")
    return bool(reasons), reasons


def _accepted_tier1_row(question: dict, row_a: dict, row_b: dict, metadata: dict,
                        citation_strategy: str, trace_a: dict | None = None,
                        trace_b: dict | None = None) -> dict:
    candidates = [
        _candidate(0, "tier1_pi", row_a, 1.0, metadata, trace_a),
        _candidate(1, "tier1_multimodal", row_b, 1.0, metadata, trace_b),
    ]
    cited = [candidate for candidate in candidates if candidate["valid_cited"]]
    return _build_output_row(question, cited or candidates, metadata, citation_strategy)


def arbitrate_escalated(question: dict, row_a: dict, row_b: dict, row_t2: dict | None,
                        metadata: dict, citation_strategy: str = "majority",
                        trace_a: dict | None = None, trace_b: dict | None = None,
                        trace_t2: dict | None = None) -> tuple[dict, str]:
    """Resolve an escalated question across ``(Path A, Path B, Tier-2)``.

    Mirrors :func:`wattbot.ensemble.cascaded_route`: prefer a corroborated
    cluster (at least two cited candidates), then a valid Tier-2 answer, then a
    valid Tier-1 fallback, then abstain.
    """
    candidates = [
        _candidate(0, "tier1_pi", row_a, 1.0, metadata, trace_a),
        _candidate(1, "tier1_multimodal", row_b, 1.0, metadata, trace_b),
    ]
    if row_t2 is not None:
        candidates.append(
            _candidate(2, "tier2_doc_routing", row_t2, 1.0, metadata, trace_t2))

    non_blank = [candidate for candidate in candidates if not is_blank_row(candidate["row"])]
    clusters = cluster_candidates(non_blank)
    corroborated = [
        cluster for cluster in clusters
        if len(cluster["members"]) >= 2 and cluster["cited_count"] >= 1
    ]
    if corroborated:
        cited = [member for member in corroborated[0]["members"] if member["valid_cited"]]
        return _build_output_row(question, cited, metadata, citation_strategy), "escalated_consensus"

    tier2 = candidates[2] if row_t2 is not None else None
    if tier2 is not None and tier2["valid_cited"] and not is_blank_row(tier2["row"]):
        return _build_output_row(question, [tier2], metadata, citation_strategy), "escalated_tier2"
    if candidates[0]["valid_cited"] and not is_blank_row(candidates[0]["row"]):
        return (_build_output_row(question, [candidates[0]], metadata, citation_strategy),
                "escalated_fallback_tier1")
    if candidates[1]["valid_cited"] and not is_blank_row(candidates[1]["row"]):
        return (_build_output_row(question, [candidates[1]], metadata, citation_strategy),
                "escalated_fallback_multimodal")
    return abstention(question), "escalated_abstained"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_stage_artifact(stage_dir: Path, required_ids: Sequence[str],
                         resume: bool) -> StageResult | None:
    if not resume:
        return None
    artifact = stage_dir / "_stage_result.json"
    if not artifact.is_file():
        return None
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except ValueError:
        return None
    rows = payload.get("rows") or {}
    if not all(question_id in rows for question_id in required_ids):
        return None
    tokens = payload.get("tokens") or {}
    trace_path = payload.get("trace_path")
    return StageResult(
        rows={question_id: rows[question_id] for question_id in required_ids},
        tokens={question_id: int(tokens.get(question_id, 0) or 0)
                for question_id in required_ids},
        trace_path=Path(trace_path) if trace_path else None,
        stage_dir=stage_dir,
    )


def _run_stage(name: str, solver: TierSolver, questions: Sequence[dict], stage_dir: Path,
               required_ids: Sequence[str], resume: bool) -> StageResult:
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_stage_artifact(stage_dir, required_ids, resume)
    if cached is not None:
        return cached

    by_id = {str(question["id"]): question for question in questions}
    subset = [by_id[question_id] for question_id in required_ids if question_id in by_id]
    if len(subset) != len(required_ids):
        missing = [question_id for question_id in required_ids if question_id not in by_id]
        raise ValueError(f"Stage {name} is missing questions: {missing[:5]}")

    result = solver(subset, stage_dir)
    if not isinstance(result, StageResult):
        raise TypeError(f"Stage {name} solver must return a StageResult")
    missing = [question_id for question_id in required_ids if question_id not in result.rows]
    if missing:
        raise ValueError(f"Stage {name} did not produce rows for: {missing[:5]}")

    rows = {question_id: result.rows[question_id] for question_id in required_ids}
    tokens = {question_id: result.tokens_for(question_id) for question_id in required_ids}
    write_json(stage_dir / "_stage_result.json", {
        "stage": name,
        "rows": rows,
        "tokens": tokens,
        "trace_path": str(result.trace_path) if result.trace_path else None,
    })
    return StageResult(rows=rows, tokens=tokens, trace_path=result.trace_path,
                       stage_dir=stage_dir)


def run_cascade(
    questions: Sequence[dict],
    metadata: dict,
    *,
    tier1_a_solver: TierSolver,
    tier1_b_solver: TierSolver,
    tier2_solver: TierSolver,
    output_dir: Path,
    citation_strategy: str = "majority",
    resume: bool = False,
    tier2_full_estimate_tokens: float | None = None,
    gold_rows: Sequence[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Run the online two-tier cascade with injectable stage solvers.

    Each solver receives the question subset and a stage directory, and must
    return a :class:`StageResult`.  Tier-2 is invoked exactly once, only for the
    escalated question subset, and never when no question escalates.
    """
    questions = list(questions)
    if not questions:
        raise ValueError("No questions to process")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_ids = [str(question["id"]) for question in questions]
    by_id = {str(question["id"]): question for question in questions}

    stage_a = _run_stage(
        "tier1_pi", tier1_a_solver, questions, output_dir / "tier1_pi", all_ids, resume)
    stage_b = _run_stage(
        "tier1_multimodal", tier1_b_solver, questions, output_dir / "tier1_multimodal",
        all_ids, resume)
    trace_a = load_trace_metadata(stage_a.trace_path)
    trace_b = load_trace_metadata(stage_b.trace_path)

    gate: dict[str, tuple[bool, list[str]]] = {}
    escalated: list[str] = []
    trigger_counts: dict[str, int] = {}
    for question in questions:
        question_id = str(question["id"])
        is_escalated, reasons = consensus_gate(
            question, stage_a.rows[question_id], stage_b.rows[question_id], metadata)
        gate[question_id] = (is_escalated, reasons)
        if is_escalated:
            escalated.append(question_id)
            for reason in reasons:
                trigger_counts[reason] = trigger_counts.get(reason, 0) + 1

    tier2_dir = output_dir / "tier2_doc_routing"
    if escalated:
        tier2_questions = [by_id[question_id] for question_id in escalated]
        stage_t2 = _run_stage(
            "tier2_doc_routing", tier2_solver, tier2_questions, tier2_dir, escalated, resume)
    else:
        stage_t2 = StageResult(rows={}, tokens={}, stage_dir=tier2_dir)
        tier2_dir.mkdir(parents=True, exist_ok=True)
        write_json(tier2_dir / "_stage_result.json", {
            "stage": "tier2_doc_routing",
            "rows": {},
            "tokens": {},
            "trace_path": None,
            "skipped": True,
        })
    trace_t2 = load_trace_metadata(stage_t2.trace_path)

    ordered: list[dict] = []
    per_question: list[dict] = []
    resolution_counts: dict[str, int] = {}
    answered = abstained = 0
    total_tokens = 0
    tier2_actual_tokens = 0

    for question in questions:
        question_id = str(question["id"])
        is_escalated, reasons = gate[question_id]
        row_a = stage_a.rows[question_id]
        row_b = stage_b.rows[question_id]
        row_t2 = stage_t2.rows.get(question_id) if is_escalated else None
        trace_meta_a = trace_a.get(question_id) or {}
        trace_meta_b = trace_b.get(question_id) or {}
        trace_meta_t2 = trace_t2.get(question_id) or {}

        if is_escalated:
            row, resolution = arbitrate_escalated(
                question, row_a, row_b, row_t2, metadata, citation_strategy,
                trace_a=trace_meta_a, trace_b=trace_meta_b, trace_t2=trace_meta_t2)
        else:
            row = _accepted_tier1_row(
                question, row_a, row_b, metadata, citation_strategy,
                trace_a=trace_meta_a, trace_b=trace_meta_b)
            resolution = "tier1_accepted"

        tokens_a = stage_a.tokens_for(question_id)
        tokens_b = stage_b.tokens_for(question_id)
        tokens_t2 = stage_t2.tokens_for(question_id) if is_escalated else 0
        tokens_q = tokens_a + tokens_b + tokens_t2

        total_tokens += tokens_q
        tier2_actual_tokens += tokens_t2
        resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        if is_blank_row(row):
            abstained += 1
        else:
            answered += 1

        ordered.append(row)
        per_question.append({
            "id": question_id,
            "escalated": is_escalated,
            "reasons": reasons,
            "resolution": resolution,
            "tokens": {
                "tier1_pi": tokens_a,
                "tier1_multimodal": tokens_b,
                "tier2_doc_routing": tokens_t2,
                "total": tokens_q,
            },
        })

    validate_rows(COLUMNS, ordered, questions, metadata)
    atomic_csv(output_dir / "predictions.csv", ordered)

    if tier2_full_estimate_tokens is not None:
        tier2_full = float(tier2_full_estimate_tokens)
    elif escalated:
        tier2_full = tier2_actual_tokens / len(escalated) * len(questions)
    else:
        tier2_full = 0.0
    if tier2_full > 0:
        savings = (1.0 - tier2_actual_tokens / tier2_full) * 100.0
    else:
        savings = 100.0 if not escalated else 0.0

    stage_tokens = {
        "tier1_pi": sum(stage_a.tokens_for(question_id) for question_id in all_ids),
        "tier1_multimodal": sum(stage_b.tokens_for(question_id) for question_id in all_ids),
        "tier2_doc_routing": tier2_actual_tokens,
    }
    summary = {
        "questions": len(questions),
        "answered": answered,
        "abstained": abstained,
        "escalation_count": len(escalated),
        "escalation_rate": round(len(escalated) / len(questions), 4),
        "tier1_accepted_count": len(questions) - len(escalated),
        "trigger_counts": trigger_counts,
        "resolution_counts": resolution_counts,
        "stage_tokens": stage_tokens,
        "total_tokens": total_tokens,
        "mean_tokens_per_question": round(total_tokens / len(questions), 2),
        "tier2_actual_tokens": tier2_actual_tokens,
        "tier2_full_estimate_tokens": round(tier2_full, 2),
        "token_savings_vs_full_tier2_pct": round(savings, 2),
        "tier2_invoked": bool(escalated),
        "tier2_question_ids": list(escalated),
    }
    diagnostics = {
        "summary": summary,
        "escalated_question_ids": list(escalated),
        "per_question": per_question,
    }
    if gold_rows is not None:
        metrics = evaluate(ordered, list(gold_rows))
        diagnostics["metrics"] = metrics
        summary["metrics"] = {
            key: metrics[key]
            for key in ("questions", "value", "citation", "abstention", "weighted")
        }

    write_json(output_dir / "diagnostics.json", diagnostics)
    return ordered, diagnostics


# ---------------------------------------------------------------------------
# Real stage solvers (subprocess-backed)
# ---------------------------------------------------------------------------


def _write_questions(path: Path, questions: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "question", "answer_unit"])
        writer.writeheader()
        for question in questions:
            writer.writerow({
                "id": question["id"],
                "question": question["question"],
                "answer_unit": question.get("answer_unit", BLANK),
            })


def _run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(
            f"Stage command failed ({completed.returncode}): {' '.join(command)}\n{detail[-4000:]}")


def _read_stage_predictions(output_dir: Path) -> tuple[list[dict], dict[str, int]]:
    _, rows = read_csv(output_dir / "predictions.csv")
    tokens = trace_tokens_by_question(output_dir / "trace.jsonl")
    return rows, tokens


@dataclass
class PiAgentOptions:
    metadata: Path
    index: Path
    docs_dir: Path
    workers: int = 2
    thinking: bool = True
    top_k: int = 10
    max_calls: int = 8
    dense: bool = True
    parallel_tool_calls: bool = True


def make_pi_agent_solver(options: PiAgentOptions) -> TierSolver:
    def solve(questions: Sequence[dict], stage_dir: Path) -> StageResult:
        stage_dir = Path(stage_dir)
        data_dir = stage_dir / "data"
        source = data_dir / "raw" / "test_Q.csv"
        _write_questions(source, questions)
        shutil.copyfile(options.metadata, data_dir / "raw" / "metadata.csv")
        output_dir = stage_dir / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        command = [
            sys.executable, "-m", "wattbot.pi_agent_v2",
            "--mode", "pi", "--split", "test",
            "--workers", str(options.workers),
            "--thinking", "true" if options.thinking else "false",
            "--top-k", str(options.top_k),
            "--max-calls", str(options.max_calls),
            "--dense", "true" if options.dense else "false",
            "--parallel-tool-calls", "true" if options.parallel_tool_calls else "false",
            "--data-dir", str(data_dir),
            "--index", str(options.index),
            "--docs-dir", str(options.docs_dir),
            "--output-dir", str(output_dir),
        ]
        _run_command(command)
        rows, tokens = _read_stage_predictions(output_dir)
        return StageResult(
            rows={str(row["id"]): row for row in rows},
            tokens=tokens,
            trace_path=output_dir / "trace.jsonl",
            stage_dir=stage_dir,
        )

    return solve


@dataclass
class MultimodalOptions:
    metadata: Path
    docs: Path
    index: Path
    figure_notes: Path = DEFAULT_FIGURE_NOTES
    workers: int = 1
    top_k: int = 10
    max_images: int = 3
    dpi: int = 180
    vlm_model: str = DEFAULT_VLM_MODEL
    provider: str = DEFAULT_PROVIDER
    mode: str = "mode2-only"
    resume: bool = False


def make_multimodal_solver(options: MultimodalOptions) -> TierSolver:
    def solve(questions: Sequence[dict], stage_dir: Path) -> StageResult:
        stage_dir = Path(stage_dir)
        questions_path = stage_dir / "questions.csv"
        _write_questions(questions_path, questions)
        output_dir = stage_dir / "output"
        if output_dir.exists() and not options.resume:
            shutil.rmtree(output_dir)
        command = [
            sys.executable, "-m", "wattbot.multimodal", "run",
            "--metadata", str(options.metadata),
            "--docs", str(options.docs),
            "--index", str(options.index),
            "--questions", str(questions_path),
            "--output-dir", str(output_dir),
            "--mode", options.mode,
            "--vlm-model", options.vlm_model,
            "--provider", options.provider,
            "--top-k", str(options.top_k),
            "--max-images", str(options.max_images),
            "--dpi", str(options.dpi),
            "--workers", str(options.workers),
        ]
        if options.mode not in {"mode2-only", "text-only"}:
            # Figure notes are only loaded outside mode2/text-only modes; avoid
            # pointing the run at a non-existent cache file.
            if Path(options.figure_notes).is_file():
                command.extend(["--figure-notes", str(options.figure_notes)])
        if options.resume:
            command.append("--resume")
        _run_command(command)
        rows, tokens = _read_stage_predictions(output_dir)
        return StageResult(
            rows={str(row["id"]): row for row in rows},
            tokens=tokens,
            trace_path=output_dir / "trace.jsonl",
            stage_dir=stage_dir,
        )

    return solve


@dataclass
class DocRoutingOptions:
    metadata: Path
    docs: Path = DEFAULT_DOCS
    cache_dir: Path = DEFAULT_CACHE_DIR
    top_docs: int = 3
    workers: int = 2
    backend: str = "hkustgz"
    no_llm_cards: bool = False
    force_cards: bool = False
    resume: bool = False


def make_doc_routing_solver(options: DocRoutingOptions) -> TierSolver:
    def solve(questions: Sequence[dict], stage_dir: Path) -> StageResult:
        from wattbot.deepseek import DeepSeek
        from wattbot.doc_routing import (
            ResilientClient,
            build_document_cards,
            build_markdown_corpus,
            run_doc_routing_subset,
        )

        stage_dir = Path(stage_dir)
        metadata = load_metadata(options.metadata)
        corpus = build_markdown_corpus(metadata, options.docs, options.cache_dir)
        client = ResilientClient(DeepSeek(options.backend, timeout=180, transport_attempts=4))
        cards = build_document_cards(
            corpus,
            options.cache_dir,
            client=client,
            use_llm=not options.no_llm_cards,
            workers=options.workers,
            force=options.force_cards,
        )
        output_dir = stage_dir / "output"
        use_resume = (
            options.resume
            and (output_dir / "config.json").is_file()
            and (output_dir / "trace.jsonl").is_file()
        )
        if not use_resume and output_dir.exists():
            shutil.rmtree(output_dir)

        def invoke(resume: bool):
            return run_doc_routing_subset(
                list(questions),
                metadata,
                corpus=corpus,
                cards=cards,
                client=client,
                output_dir=output_dir,
                docs=options.docs,
                cache_dir=options.cache_dir,
                top_docs=options.top_docs,
                backend=options.backend,
                workers=options.workers,
                resume=resume,
                gold_rows=None,
                no_llm_cards=options.no_llm_cards,
            )

        try:
            ordered, _summary = invoke(use_resume)
        except (FileExistsError, FileNotFoundError, ValueError):
            if not use_resume:
                raise
            # A stale config/trace from a different Tier-2 subset: start clean.
            shutil.rmtree(output_dir, ignore_errors=True)
            ordered, _summary = invoke(False)
        tokens = trace_tokens_by_question(output_dir / "trace.jsonl")
        return StageResult(
            rows={str(row["id"]): row for row in ordered},
            tokens=tokens,
            trace_path=output_dir / "trace.jsonl",
            stage_dir=stage_dir,
        )

    return solve


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_bool(value) -> bool:
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def add_arguments(parser: argparse.ArgumentParser, *,
                  default_questions: Path = DATA / "raw" / "test_Q.csv",
                  default_output_dir: Path = DEFAULT_OUTPUT / "pure-rag-cascade",
                  include_metadata: bool = True) -> None:
    if include_metadata:
        parser.add_argument("--metadata", type=Path, default=DATA / "raw" / "metadata.csv")
    parser.add_argument("--questions", type=Path, default=default_questions)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--citation-strategy", choices=["majority", "intersection", "best"],
                        default="majority")

    parser.add_argument("--backend", choices=("hkustgz", "openrouter"), default="hkustgz")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)

    parser.add_argument("--thinking", type=parse_bool, default=True,
                        help="Enable Pi Agent v2 thinking (default: true)")
    parser.add_argument("--pi-index", type=Path, default=None)
    parser.add_argument("--pi-docs-dir", type=Path, default=None)
    parser.add_argument("--pi-workers", type=int, default=2)
    parser.add_argument("--pi-top-k", type=int, default=10)
    parser.add_argument("--pi-max-calls", type=int, default=8)
    parser.add_argument("--pi-dense", type=parse_bool, default=True)
    parser.add_argument("--pi-parallel-tool-calls", type=parse_bool, default=True)

    parser.add_argument("--visual-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--multimodal-workers", type=int, default=1)
    parser.add_argument("--multimodal-top-k", type=int, default=10)
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--figure-notes", type=Path, default=DEFAULT_FIGURE_NOTES)

    parser.add_argument("--top-docs", type=int, default=3)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tier2-workers", type=int, default=2)
    parser.add_argument("--no-llm-cards", action="store_true")
    parser.add_argument("--force-cards", action="store_true")


def _resolve_pi_index(path: Path | None) -> Path:
    if path is not None:
        return path
    return DEFAULT_CLEAN_INDEX if DEFAULT_CLEAN_INDEX.is_file() else DEFAULT_INDEX


def _ordered_gold(gold_path: Path, questions: Sequence[dict]) -> list[dict]:
    _, gold_rows = read_csv(gold_path)
    by_id = {str(row["id"]): row for row in gold_rows}
    missing = [question["id"] for question in questions if str(question["id"]) not in by_id]
    if missing:
        raise ValueError(f"Gold file is missing question IDs: {missing[:5]}")
    return [by_id[str(question["id"])] for question in questions]


def run_cli(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if min(args.pi_workers, args.multimodal_workers, args.tier2_workers) < 1:
        raise ValueError("Stage worker counts must be positive")

    metadata = load_metadata(args.metadata)
    _, source_rows = read_csv(args.questions)
    active_source = list(source_rows)
    if args.limit is not None:
        active_source = active_source[:args.limit]
    if not active_source:
        raise ValueError("No questions to process")
    questions = [{
        "id": row["id"],
        "question": row["question"],
        "answer_unit": row.get("answer_unit", BLANK),
    } for row in active_source]
    gold_rows = _ordered_gold(args.gold, questions) if args.gold else None

    pi_solver = make_pi_agent_solver(PiAgentOptions(
        metadata=args.metadata,
        index=_resolve_pi_index(args.pi_index),
        docs_dir=args.pi_docs_dir or args.docs,
        workers=args.pi_workers,
        thinking=args.thinking,
        top_k=args.pi_top_k,
        max_calls=args.pi_max_calls,
        dense=args.pi_dense,
        parallel_tool_calls=args.pi_parallel_tool_calls,
    ))
    multimodal_solver = make_multimodal_solver(MultimodalOptions(
        metadata=args.metadata,
        docs=args.docs,
        index=args.visual_index,
        figure_notes=args.figure_notes,
        workers=args.multimodal_workers,
        top_k=args.multimodal_top_k,
        max_images=args.max_images,
        dpi=args.dpi,
        vlm_model=args.vlm_model,
        provider=args.provider,
        resume=args.resume,
    ))
    doc_routing_solver = make_doc_routing_solver(DocRoutingOptions(
        metadata=args.metadata,
        docs=args.docs,
        cache_dir=args.cache_dir,
        top_docs=args.top_docs,
        workers=args.tier2_workers,
        backend=args.backend,
        no_llm_cards=args.no_llm_cards,
        force_cards=args.force_cards,
        resume=args.resume,
    ))

    rows, diagnostics = run_cascade(
        questions,
        metadata,
        tier1_a_solver=pi_solver,
        tier1_b_solver=multimodal_solver,
        tier2_solver=doc_routing_solver,
        output_dir=args.output_dir,
        citation_strategy=args.citation_strategy,
        resume=args.resume,
        gold_rows=gold_rows,
    )
    print(json.dumps({
        "command": "pure-rag-cascade",
        "output": str(Path(args.output_dir) / "predictions.csv"),
        "diagnostics": str(Path(args.output_dir) / "diagnostics.json"),
        "summary": diagnostics["summary"],
    }, indent=2))
    return rows, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m wattbot.pure_rag_cascade",
        description=__doc__,
    )
    add_arguments(parser, include_metadata=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
