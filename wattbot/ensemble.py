from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from wattbot.__main__ import atomic_csv
from wattbot.deepseek import CAMPUS_BASE, setting
from wattbot.evaluation import evaluate
from wattbot.pipeline import (
    BLANK,
    COLUMNS,
    EVIDENCE_FIELDS,
    NUMBER,
    RANGE,
    abstention,
    compact,
    load_metadata,
    normalize_value,
    parse_list,
    read_csv,
    solve,
    source_quote,
    submission_fields,
    validate_rows,
)
from wattbot.retrieval import SearchIndex, sha256_file, write_json


@dataclass(frozen=True)
class RunSpec:
    name: str
    csv_path: Path
    trace_path: Path | None = None
    weight: float = 1.0


def parse_run_spec(text: str) -> RunSpec:
    name, rest = text.split("=", 1)
    parts = rest.split("::")
    csv_path = Path(parts[0])
    trace_path = Path(parts[1]) if len(parts) > 1 and parts[1] else None
    weight = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
    return RunSpec(name.strip(), csv_path, trace_path, weight)


def _json_content(event: dict) -> dict:
    content = event.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except (TypeError, ValueError):
            return {}
    return content if isinstance(content, dict) else {}


def load_trace_metadata(trace_path: Path | None) -> dict[str, dict]:
    if trace_path is None:
        return {}
    traces: dict[str, dict] = {}
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        qid = event.get("question_id") or event.get("qid") or event.get("id")
        if not qid:
            continue
        meta = traces.setdefault(qid, {
            "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "llm_calls": 0, "verification_accepted": False,
            "verification_rejected_count": 0, "verification_abstained_count": 0,
            "verification_attempts": 0, "has_calculation_binding": False,
            "quote_recovered_count": 0, "citation_dropped_count": 0,
            "invalid_output_count": 0, "raw_evidence": [],
        })
        etype = str(event.get("event") or event.get("type") or "")
        content = _json_content(event)
        if etype == "llm_response":
            usage = event.get("usage") or content.get("usage") or {}
            meta["llm_calls"] += 1
            meta["total_tokens"] += int(usage.get("total_tokens") or 0)
            meta["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            meta["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            evidence = content.get("evidence")
            if isinstance(evidence, list):
                meta["raw_evidence"] = evidence
        elif etype == "verification":
            decision = event.get("decision")
            if decision is None:
                decision = content.get("decision")
            decision = str(decision or "")
            supported = event.get("supported")
            if supported is None:
                supported = content.get("supported")
            meta["verification_accepted"] = (decision == "accepted" or supported is True)
            if decision == "rejected":
                meta["verification_rejected_count"] += 1
            if decision.startswith("abstained"):
                meta["verification_abstained_count"] += 1
            attempt = event.get("attempt")
            if attempt is None:
                attempt = content.get("attempt")
            try:
                meta["verification_attempts"] = max(meta["verification_attempts"], int(attempt or 0))
            except (TypeError, ValueError):
                pass
        elif etype == "calculation_binding":
            meta["has_calculation_binding"] = True
        elif "quote_recovered" in etype:
            meta["quote_recovered_count"] += int(event.get("count") or 1)
        elif "citation_dropped" in etype:
            meta["citation_dropped_count"] += int(event.get("count") or 1)
        elif "invalid_output" in etype:
            meta["invalid_output_count"] += int(event.get("count") or 1)
    return traces


def trace_token_summary(trace_path: Path | None) -> dict:
    traces = load_trace_metadata(trace_path)
    total = sum(m["total_tokens"] for m in traces.values())
    n = len(traces)
    return {
        "questions": n,
        "total_tokens": total,
        "mean_tokens_per_question": round(total / n, 2) if n else 0.0,
        "per_question_tokens": {qid: m["total_tokens"] for qid, m in traces.items()},
    }


class CorpusPageLookup:
    def __init__(self, index_path: Path) -> None:
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.pages: dict[tuple[str, int], str] = {}
        self.page_counts: dict[str, int] = {}
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        table_rows: dict[tuple[str, int], list[str]] = {}
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict) or node.get("kind") != "table_row":
                continue
            match = re.match(r"(.+):p(\d+)$", str(node.get("page_id") or ""))
            if not match:
                continue
            key = (match[1], int(match[2]))
            text = node.get("text") or node.get("content") or ""
            if text:
                table_rows.setdefault(key, []).append(str(text))
        pages = data.get("pages", {})
        if isinstance(pages, dict):
            for page_id, page_text in pages.items():
                match = re.match(r"(.+):p(\d+)$", str(page_id))
                if not match:
                    continue
                ref_id, number = match[1], int(match[2])
                rows = table_rows.get((ref_id, number), [])
                text = str(page_text or "")
                self.pages[(ref_id, number)] = text + ("\n" + "\n".join(rows) if rows else "")
        elif isinstance(pages, list):
            for page in pages:
                if not isinstance(page, dict):
                    continue
                ref_id = str(page.get("ref_id") or page.get("source") or "")
                try:
                    number = int(page.get("page") or page.get("page_number") or 0)
                except (TypeError, ValueError):
                    continue
                if not ref_id or not number:
                    continue
                text = str(page.get("text") or page.get("content") or "")
                rows = list(table_rows.get((ref_id, number), []))
                for row in page.get("table_rows", []) if isinstance(page.get("table_rows"), list) else []:
                    rows.append(str(row.get("text") if isinstance(row, dict) else row))
                self.pages[(ref_id, number)] = text + ("\n" + "\n".join(rows) if rows else "")
        sources = data.get("sources", {})
        if isinstance(sources, dict):
            for ref_id, info in sources.items():
                stats = info.get("stats") if isinstance(info, dict) else None
                count = stats.get("pages") if isinstance(stats, dict) else None
                try:
                    if count is not None:
                        self.page_counts[str(ref_id)] = int(count)
                except (TypeError, ValueError):
                    pass
        for ref_id, number in self.pages:
            self.page_counts[ref_id] = max(self.page_counts.get(ref_id, 0), number)


def _match_quote_on_page(quote: str, text: str) -> str | None:
    if not quote or not text:
        return None
    if m := source_quote(quote, text):
        return m
    rq = re.sub(r"(\w)-\s+(\w)", r"\1\2", compact(quote))
    rt = re.sub(r"(\w)-\s+(\w)", r"\1\2", compact(text))
    if m := source_quote(rq, rt):
        return m
    q_tokens = re.findall(r"[a-z0-9]+", quote.casefold())
    segments = [s.strip() for s in re.split(r"\.\.\.|…", quote)]
    segments = [s for s in segments if len(re.findall(r"[a-z0-9]+", s.casefold())) >= 2]
    if segments:
        matched: list[str] = []
        for segment in segments:
            seg = source_quote(segment, text)
            if not seg:
                seg = source_quote(re.sub(r"(\w)-\s+(\w)", r"\1\2", compact(segment)), rt)
            if seg:
                matched.append(seg)
        if matched and len(matched) == len(segments):
            return " ... ".join(matched)
        token_count = sum(len(re.findall(r"[a-z0-9]+", m.casefold())) for m in matched)
        if matched and token_count >= max(4, int(0.6 * len(q_tokens))):
            return " ... ".join(matched)
    rq_tokens = re.findall(r"[a-z0-9]+", rq.casefold())
    if len(rq_tokens) < 4:
        return None
    t_tokens = re.findall(r"[a-z0-9]+", rt.casefold())
    present = sum(1 for tok in rq_tokens if tok in set(t_tokens)) / len(rq_tokens)
    if present < 0.80:
        return None
    index = 0
    for tok in t_tokens:
        if index < len(rq_tokens) and tok == rq_tokens[index]:
            index += 1
    if index / len(rq_tokens) >= 0.70:
        return compact(quote)
    return None


def recover_row_citations(row: dict, raw_evidence: list[dict], corpus: CorpusPageLookup,
                          metadata: dict) -> tuple[dict, bool]:
    if row.get("answer_value") == BLANK or row.get("ref_id") != BLANK or not raw_evidence:
        return (dict(row), False)
    refs_map: dict[str, str] = {}
    materials: list[str] = []
    for item in raw_evidence:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("ref_id") or "")
        quote = str(item.get("quote") or "").strip()
        if ref_id not in metadata or not quote:
            continue
        try:
            page = int(item.get("page") or 0)
        except (TypeError, ValueError):
            page = 0
        order = [page, page - 1, page + 1, *range(1, corpus.page_counts.get(ref_id, 0) + 1)]
        for number in dict.fromkeys(p for p in order if p >= 1):
            text = corpus.pages.get((ref_id, number), "")
            if matched_quote := _match_quote_on_page(quote, text):
                info = metadata.get(ref_id) or {}
                refs_map[ref_id] = str(info.get("url") or "") if isinstance(info, dict) else ""
                materials.append(f"[{ref_id}, p. {number}] {matched_quote}")
                break
    updated = dict(row)
    if refs_map:
        updated["ref_id"] = repr(list(refs_map))
        updated["ref_url"] = repr(list(refs_map.values()))
        updated["supporting_materials"] = "\n".join(materials)
    return (updated, bool(refs_map))


def is_blank_row(row: dict) -> bool:
    return row.get("answer_value") == BLANK or all(row.get(f) == BLANK for f in EVIDENCE_FIELDS)


def has_valid_citations(row: dict, metadata: dict) -> bool:
    if row.get("answer_value") == BLANK:
        return False
    try:
        normalize_value(row.get("answer_value", ""))
        refs = parse_list(row.get("ref_id", BLANK))
        urls = parse_list(row.get("ref_url", BLANK))
    except (ValueError, SyntaxError):
        return False
    return bool(
        refs
        and len(refs) == len(set(refs))
        and all(ref in metadata for ref in refs)
        and urls == [metadata[ref]["url"] for ref in refs]
        and row.get("supporting_materials") not in (None, "", BLANK)
        and row.get("explanation") not in (None, "", BLANK)
    )


def canonicalize_interval(value: str) -> str:
    text = compact(str(value or ""))
    if text == BLANK:
        return text
    match = re.fullmatch(rf"({NUMBER})\s*(?:[–—-]|\bto\b)\s*({NUMBER})", text, re.I)
    if match:
        try:
            low, high = Decimal(match[1]), Decimal(match[2])
        except InvalidOperation:
            return text
        if low <= high:
            return f"[{low},{high}]"
    return text


def values_equivalent(val_a: str, val_b: str, rel_tol: Decimal = Decimal("0.001")) -> bool:
    a, b = canonicalize_interval(val_a), canonicalize_interval(val_b)
    if a == BLANK or b == BLANK:
        return a == b

    def _num_eq(x: Decimal, y: Decimal, rel_tol: Decimal) -> bool:
        return True if (x == 0 and y == 0) else abs(x - y) <= rel_tol * max(abs(x), abs(y))

    if re.fullmatch(NUMBER, a) and re.fullmatch(NUMBER, b):
        return _num_eq(Decimal(a), Decimal(b), rel_tol)
    if (ma := RANGE.fullmatch(a)) and (mb := RANGE.fullmatch(b)):
        return (_num_eq(Decimal(ma[1]), Decimal(mb[1]), rel_tol)
                and _num_eq(Decimal(ma[2]), Decimal(mb[2]), rel_tol))
    return a.casefold() == b.casefold()


def _candidate_value(candidate: dict) -> str:
    return str(candidate.get("answer_value") or candidate.get("value") or "")


def cluster_candidates(candidates: list[dict], rel_tol: Decimal = Decimal("0.001")) -> list[dict]:
    ordered = sorted(candidates, key=lambda c: (-float(c.get("score") or 0), int(c.get("run_index") or 0)))
    clusters: list[dict] = []
    for candidate in ordered:
        value = _candidate_value(candidate)
        compatible = [
            cluster for cluster in clusters
            if all(values_equivalent(value, _candidate_value(member), rel_tol) for member in cluster["members"])
        ]
        if not compatible:
            clusters.append({"members": [candidate]})
            continue
        compatible[0]["members"].append(candidate)
    for cluster in clusters:
        members = cluster["members"]
        cluster["total_weight"] = sum(float(m.get("score", 0.0)) for m in members)
        cluster["cited_count"] = sum(1 for m in members if m.get("valid_cited", False))
        cluster["max_score"] = max(float(m.get("score", 0.0)) for m in members)
        pool = [m for m in members if m.get("valid_cited", False)] or members
        best = max(pool, key=lambda m: float(m.get("score", 0.0)))
        cluster["representative_value"] = _candidate_value(best)
    return sorted(clusters, key=lambda c: (-c["total_weight"], -c["cited_count"],
                                           -len(c["members"]), -c["max_score"]))


def _safe_refs(value) -> list[str]:
    try:
        return list(parse_list(value if value not in (None, "") else BLANK))
    except (ValueError, SyntaxError):
        return []


def score_candidate(row: dict, trace_meta: dict, base_weight: float, metadata: dict) -> tuple[float, dict]:
    valid_cited = has_valid_citations(row, metadata)
    if is_blank_row(row):
        penalty = 0.85 if (trace_meta.get("invalid_output_count", 0) > 0
                           and not trace_meta.get("verification_accepted")) else 1.0
        score = max(0.05 * base_weight, base_weight * penalty)
        return score, {"blank": True, "valid_cited": False, "base_weight": base_weight}

    score = base_weight
    verif_acc = bool(trace_meta.get("verification_accepted"))
    verif_rej = int(trace_meta.get("verification_rejected_count") or 0)
    inv_out = int(trace_meta.get("invalid_output_count") or 0)
    calc_bind = bool(trace_meta.get("has_calculation_binding"))
    cit_drop = int(trace_meta.get("citation_dropped_count") or 0)
    refs = _safe_refs(row.get("ref_id", BLANK)) if valid_cited else []
    if verif_acc:
        score += 0.20 * base_weight
        if verif_rej == 0 and inv_out == 0:
            score += 0.05 * base_weight
    if verif_rej > 0:
        score -= 0.10 * base_weight * min(verif_rej, 2)
    if calc_bind:
        score += 0.25 * base_weight
    if cit_drop > 0 and valid_cited:
        score -= 0.15 * base_weight
    if len(refs) >= 3:
        score -= 0.05 * base_weight
    if not valid_cited:
        score *= 0.45
    score = max(0.05 * base_weight, score)
    return score, {"blank": False, "valid_cited": valid_cited, "verif_accepted": verif_acc,
                   "calc_bound": calc_bind, "ref_count": len(refs)}


def reconcile_citations(cited_members: list[dict], metadata: dict,
                        strategy: str = "majority") -> tuple[list[str], list[str], str]:
    ordered = sorted(cited_members, key=lambda m: (-float(m.get("score") or 0.0),
                                                   int(m.get("run_index") or 0)))
    if not ordered:
        return [], [], ""
    best = ordered[0]
    best_refs = _safe_refs(best["row"].get("ref_id", BLANK))

    kept: list[str] | None = None
    if strategy == "best" or len(ordered) == 1:
        kept = list(best_refs)
    elif strategy == "intersection":
        rest = [set(_safe_refs(m["row"].get("ref_id", BLANK))) for m in ordered[1:]]
        common = [r for r in best_refs if all(r in s for s in rest)]
        if common:
            kept = common
    if kept is None:
        total_w = sum(float(m.get("score") or 0.0) for m in ordered) or 1.0
        ref_w: dict[str, float] = {}
        first_pos: dict[str, int] = {}
        for m in ordered:
            for r in _safe_refs(m["row"].get("ref_id", BLANK)):
                ref_w[r] = ref_w.get(r, 0.0) + float(m.get("score") or 0.0)
                first_pos.setdefault(r, len(first_pos))
        ranked = sorted(ref_w.items(), key=lambda kv: (-kv[1], first_pos[kv[0]]))
        kept = [r for r, w in ranked if w / total_w >= 0.5]
        if not kept:
            kept = list(best_refs)

    urls = [str(metadata[r].get("url") or "") for r in kept if r in metadata]
    kept = [r for r in kept if r in metadata]

    material_lines: list[str] = []
    seen: set[str] = set()
    for ref in kept:
        found = False
        for member in ordered:
            raw = str(member["row"].get("supporting_materials") or "")
            lines = [ln.strip() for ln in raw.splitlines()
                     if ln.strip() and (ln.strip().startswith(f"[{ref},") or ref in ln)]
            if lines:
                for ln in lines:
                    if ln not in seen:
                        seen.add(ln)
                        material_lines.append(ln)
                found = True
                break
        if not found:
            continue
    if not material_lines:
        material_lines = [ln.strip() for ln in
                          str(best["row"].get("supporting_materials") or "").splitlines()
                          if ln.strip()]
    return kept, urls, "\n".join(material_lines)


def _prepare_runs(questions, runs: list[RunSpec], metadata: dict,
                  index_path: Path | None) -> tuple[list[dict], int]:
    corpus = CorpusPageLookup(index_path) if index_path is not None else None
    qids = [str(q["id"]) for q in questions]
    prepared: list[dict] = []
    total_recovered = 0
    for spec in runs:
        _, rows = read_csv(spec.csv_path)
        by_qid = {str(row.get("id")): row for row in rows}
        missing = [qid for qid in qids if qid not in by_qid]
        if missing:
            raise ValueError(f"Run {spec.name} missing question IDs: {missing[:5]}")
        trace_meta = load_trace_metadata(spec.trace_path)
        recovered = 0
        if corpus is not None:
            for qid in qids:
                row = by_qid[qid]
                if is_blank_row(row) or has_valid_citations(row, metadata):
                    continue
                raw_evidence = (trace_meta.get(qid) or {}).get("raw_evidence") or []
                updated, ok = recover_row_citations(row, raw_evidence, corpus, metadata)
                if ok:
                    by_qid[qid] = updated
                    recovered += 1
        total_recovered += recovered
        prepared.append({"spec": spec, "rows": by_qid, "trace": trace_meta,
                         "recovered_count": recovered})
    return prepared, total_recovered


def _build_output_row(question: dict, cited_members: list[dict], metadata: dict,
                      citation_strategy: str) -> dict:
    ordered = sorted(cited_members, key=lambda m: (-float(m.get("score") or 0.0),
                                                   int(m.get("run_index") or 0)))
    if not ordered:
        return abstention(question)
    best = ordered[0]
    kept, urls, materials = reconcile_citations(ordered, metadata, citation_strategy)
    row = dict(best["row"])
    row["id"] = question["id"]
    row["question"] = question["question"]
    row["answer"] = best["row"].get("answer", "")
    row["answer_value"] = canonicalize_interval(best["row"].get("answer_value", BLANK))
    row["answer_unit"] = best["row"].get("answer_unit", BLANK)
    row["explanation"] = best["row"].get("explanation", BLANK)
    row["ref_id"] = repr(kept)
    row["ref_url"] = repr(urls)
    row["supporting_materials"] = materials
    result = submission_fields(question, row)
    if is_blank_row(result) or not has_valid_citations(result, metadata):
        return abstention(question)
    return result


def consensus_ensemble(questions, runs: list[RunSpec], metadata: dict, *,
                       index_path: Path | None = None, citation_strategy: str = "majority",
                       quorum_alpha: float = 0.75, singleton_alpha: float = 1.0
                       ) -> tuple[list[dict], dict]:
    prepared, total_recovered = _prepare_runs(questions, runs, metadata, index_path)
    out_rows: list[dict] = []
    per_question: list[dict] = []
    total_tokens = 0
    per_run_tokens: dict[str, int] = {spec.name: 0 for spec in runs}
    answered = abstained = unanimous_count = 0

    for question in questions:
        qid = str(question["id"])
        candidates: list[dict] = []
        for run_index, item in enumerate(prepared):
            spec = item["spec"]
            row = item["rows"][qid]
            trace_meta = item["trace"].get(qid) or {}
            score, diag = score_candidate(row, trace_meta, spec.weight, metadata)
            tokens = int(trace_meta.get("total_tokens") or 0)
            total_tokens += tokens
            per_run_tokens[spec.name] = per_run_tokens.get(spec.name, 0) + tokens
            candidates.append({
                "run_index": run_index, "run_name": spec.name, "row": row,
                "answer_value": row.get("answer_value", BLANK), "score": score,
                "valid_cited": diag["valid_cited"], "trace_meta": trace_meta,
                "recovered": item["recovered_count"], "tokens": tokens,
            })

        blank_weight = sum(c["score"] for c in candidates if is_blank_row(c["row"]))
        non_blank = [c for c in candidates if not is_blank_row(c["row"])]
        clusters = cluster_candidates(non_blank)
        valid_clusters = [c for c in clusters if c["cited_count"] >= 1]
        unanimous = bool(non_blank) and len(non_blank) == len(candidates) and \
            len(clusters) == 1

        if valid_clusters:
            top = valid_clusters[0]
            alpha = quorum_alpha if len(top["members"]) >= 2 else singleton_alpha
            if top["total_weight"] >= alpha * blank_weight:
                cited = [m for m in top["members"] if m["valid_cited"]]
                out_row = _build_output_row(question, cited, metadata, citation_strategy)
                if is_blank_row(out_row):
                    decision = "abstained_no_valid_cluster"
                else:
                    decision = "consensus_answered"
            else:
                out_row = abstention(question)
                decision = "abstained_quorum"
        else:
            out_row = abstention(question)
            decision = "abstained_no_valid_cluster"

        if is_blank_row(out_row):
            abstained += 1
        else:
            answered += 1
            unanimous_count += int(unanimous)
        out_rows.append(out_row)
        per_question.append({
            "id": qid, "decision": decision, "cluster_count": len(clusters),
            "valid_cluster_count": len(valid_clusters),
            "blank_weight": round(blank_weight, 4),
            "top_weight": round(valid_clusters[0]["total_weight"], 4) if valid_clusters else 0.0,
            "unanimous": unanimous, "tokens": sum(c["tokens"] for c in candidates),
        })

    validate_rows(COLUMNS, out_rows, questions, metadata)
    diagnostics = {
        "summary": {
            "questions": len(questions), "answered": answered, "abstained": abstained,
            "recovered_citations": total_recovered, "unanimous_count": unanimous_count,
            "total_tokens": total_tokens,
            "mean_tokens_per_question": round(total_tokens / len(questions), 2) if questions else 0.0,
            "per_run_tokens": per_run_tokens,
        },
        "per_question": per_question,
    }
    return out_rows, diagnostics


def cascaded_route(questions, tier1: RunSpec, tier2: RunSpec, metadata: dict, *,
                   tier1_aux: RunSpec | None = None, index_path: Path | None = None,
                   escalate_on_abstain: bool = True,
                   escalate_on_missing_citation: bool = True,
                   escalate_on_verif_reject: int = 0,
                   escalate_on_disagreement: bool = True,
                   citation_strategy: str = "majority") -> tuple[list[dict], dict]:
    run_specs = [tier1] + ([tier1_aux] if tier1_aux is not None else []) + [tier2]
    prepared, total_recovered = _prepare_runs(questions, run_specs, metadata, index_path)
    t1_item = prepared[0]
    aux_item = prepared[1] if tier1_aux is not None else None
    t2_item = prepared[-1]

    out_rows: list[dict] = []
    per_question: list[dict] = []
    total_tokens = tier2_full_tokens = escalation_count = 0
    trigger_counts: dict[str, int] = {}
    resolution_counts: dict[str, int] = {}
    answered = abstained = 0

    for question in questions:
        qid = str(question["id"])
        t1_row = t1_item["rows"][qid]
        t2_row = t2_item["rows"][qid]
        t1_trace = t1_item["trace"].get(qid) or {}
        t2_trace = t2_item["trace"].get(qid) or {}
        t1_tokens = int(t1_trace.get("total_tokens") or 0)
        t2_tokens = int(t2_trace.get("total_tokens") or 0)
        aux_row = aux_item["rows"][qid] if aux_item is not None else None
        aux_trace = (aux_item["trace"].get(qid) or {}) if aux_item is not None else {}
        aux_tokens = int(aux_trace.get("total_tokens") or 0) if aux_item is not None else 0

        t1_score, _ = score_candidate(t1_row, t1_trace, tier1.weight, metadata)
        c_t1 = {"run_index": 0, "run_name": tier1.name, "row": t1_row,
                "answer_value": t1_row.get("answer_value", BLANK),
                "score": t1_score, "valid_cited": has_valid_citations(t1_row, metadata)}
        c_t1_aux = None
        t2_score, _ = score_candidate(t2_row, t2_trace, tier2.weight, metadata)
        c_t2 = {"run_index": len(run_specs) - 1, "run_name": tier2.name, "row": t2_row,
                "answer_value": t2_row.get("answer_value", BLANK),
                "score": t2_score, "valid_cited": has_valid_citations(t2_row, metadata)}
        if aux_item is not None:
            aux_score, _ = score_candidate(aux_row, aux_trace, tier1_aux.weight, metadata)
            c_t1_aux = {"run_index": 1, "run_name": tier1_aux.name, "row": aux_row,
                        "answer_value": aux_row.get("answer_value", BLANK),
                        "score": aux_score,
                        "valid_cited": has_valid_citations(aux_row, metadata)}

        t1_valid = c_t1["valid_cited"]
        reasons: list[str] = []
        if escalate_on_abstain and (is_blank_row(t1_row) or
                                    (aux_row is not None and is_blank_row(aux_row))):
            reasons.append("abstain")
        if escalate_on_missing_citation and ((not is_blank_row(t1_row) and not t1_valid)
                                             or int(t1_trace.get("citation_dropped_count") or 0) > 0):
            reasons.append("missing_citation")
        if escalate_on_verif_reject > 0 and \
                int(t1_trace.get("verification_rejected_count") or 0) >= escalate_on_verif_reject:
            reasons.append("verif_reject")
        if escalate_on_disagreement and aux_row is not None and \
                (not is_blank_row(t1_row)) and (not is_blank_row(aux_row)) and \
                (not values_equivalent(t1_row.get("answer_value", BLANK),
                                       aux_row.get("answer_value", BLANK))):
            reasons.append("disagreement")
        escalated = bool(reasons)
        for reason in reasons:
            trigger_counts[reason] = trigger_counts.get(reason, 0) + 1

        tokens_q = t1_tokens + aux_tokens + (t2_tokens if escalated else 0)
        total_tokens += tokens_q
        tier2_full_tokens += t2_tokens
        escalation_count += int(escalated)

        if not escalated:
            if (c_t1_aux is not None and c_t1["valid_cited"] and c_t1_aux["valid_cited"]
                    and values_equivalent(t1_row.get("answer_value", BLANK),
                                          aux_row.get("answer_value", BLANK))):
                out_row = _build_output_row(question, [c_t1, c_t1_aux], metadata,
                                            citation_strategy)
            elif t1_valid:
                out_row = _build_output_row(question, [c_t1], metadata, citation_strategy)
            else:
                out_row = abstention(question)
            resolution = "tier1_accepted"
        else:
            pool = [c for c in (c_t1, c_t1_aux, c_t2) if c is not None]
            non_blank = [c for c in pool if not is_blank_row(c["row"])]
            clusters = cluster_candidates(non_blank)
            corroborated = [cl for cl in clusters
                            if len(cl["members"]) >= 2 and cl["cited_count"] >= 1]
            if corroborated:
                out_row = _build_output_row(
                    question, [m for m in corroborated[0]["members"] if m["valid_cited"]],
                    metadata, citation_strategy)
                resolution = "escalated_consensus"
            elif c_t2["valid_cited"] and not is_blank_row(t2_row):
                out_row = _build_output_row(question, [c_t2], metadata, citation_strategy)
                resolution = "escalated_tier2"
            elif c_t1["valid_cited"] and not is_blank_row(t1_row):
                out_row = _build_output_row(question, [c_t1], metadata, citation_strategy)
                resolution = "escalated_fallback_tier1"
            elif (c_t1_aux is not None and c_t1_aux["valid_cited"]
                  and not is_blank_row(aux_row)):
                out_row = _build_output_row(question, [c_t1_aux], metadata,
                                            citation_strategy)
                resolution = "escalated_fallback_tier1_aux"
            else:
                out_row = abstention(question)
                resolution = "escalated_abstained"

        if is_blank_row(out_row):
            abstained += 1
        else:
            answered += 1
        resolution_counts[resolution] = resolution_counts.get(resolution, 0) + 1
        out_rows.append(out_row)
        per_question.append({
            "id": qid, "escalated": escalated, "reasons": reasons,
            "resolution": resolution, "tokens": tokens_q,
        })

    validate_rows(COLUMNS, out_rows, questions, metadata)
    savings = (1.0 - total_tokens / tier2_full_tokens) * 100.0 if tier2_full_tokens else 0.0
    diagnostics = {
        "summary": {
            "questions": len(questions), "answered": answered, "abstained": abstained,
            "recovered_citations": total_recovered, "total_tokens": total_tokens,
            "mean_tokens_per_question": round(total_tokens / len(questions), 2) if questions else 0.0,
            "tier2_full_tokens": tier2_full_tokens,
            "token_savings_vs_tier2_pct": round(savings, 2),
            "escalation_count": escalation_count,
            "escalation_rate": round(escalation_count / len(questions), 4) if questions else 0.0,
            "trigger_counts": trigger_counts, "resolution_counts": resolution_counts,
        },
        "per_question": per_question,
    }
    return out_rows, diagnostics


class CampusModelClient:
    def __init__(self, model: str, *, timeout: int = 120, transport_attempts: int = 6,
                 enable_thinking: bool = False):
        self.model = model
        self.timeout = timeout
        self.transport_attempts = transport_attempts
        self.enable_thinking = enable_thinking
        self.backend = "hkustgz"
        key = setting("HKUSTGZ_AIGC_API_KEY")
        base = CAMPUS_BASE
        if not key:
            config_root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            config = config_root / "opencode" / "opencode.json"
            options = json.loads(config.read_text(encoding="utf-8"))[
                "provider"]["hkustgz-aigc"]["options"]
            key = options.get("apiKey")
            base = options.get("baseURL") or CAMPUS_BASE
        if not key or key.startswith("{"):
            raise ValueError("Configure a usable HKUSTGZ_AIGC_API_KEY")
        self.key = key
        self.url = base.rstrip("/") + "/chat/completions"

    @staticmethod
    def _parse_content(content: str | None) -> dict:
        text = (content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            result = json.loads(text, parse_float=Decimal)
        except ValueError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Expected a JSON object in response")
            result = json.loads(text[start:end + 1], parse_float=Decimal)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object in response")
        return result

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(60.0, 2 ** attempt) + random.uniform(0, 1.5)

    def complete(self, stage, system, data, emit):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
        payload = {
            "model": self.model, "messages": messages, "temperature": 0,
            "max_tokens": 2400, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        request = Request(self.url, data=json.dumps(payload).encode(), method="POST",
                          headers={"Authorization": f"Bearer {self.key}",
                                   "Content-Type": "application/json"})
        body = None
        started = time.monotonic()
        for attempt in range(1, self.transport_attempts + 1):
            emit("llm_request", stage=stage, attempt=attempt, model=self.model,
                 backend=self.backend, messages=messages)
            started = time.monotonic()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                break
            except HTTPError as error:
                status = error.code
                emit("llm_http_error", stage=stage, attempt=attempt, status=status)
                if status not in {408, 429, 500, 502, 503, 504} or \
                        attempt == self.transport_attempts:
                    raise RuntimeError(
                        f"LLM endpoint returned HTTP {status} in {stage}") from error
                delay = self._backoff(attempt)
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if retry_after is not None:
                    try:
                        delay = max(delay, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                time.sleep(delay)
            except (URLError, TimeoutError) as error:
                emit("llm_network_error", stage=stage, attempt=attempt, detail=str(error))
                if attempt == self.transport_attempts:
                    raise RuntimeError(f"LLM network error in {stage}: {error}") from error
                time.sleep(self._backoff(attempt))
        if not body or not body.get("choices"):
            emit("llm_response_error", stage=stage, response_id=(body or {}).get("id"))
            raise RuntimeError(f"LLM endpoint returned no choice in {stage}")
        choice = body["choices"][0]
        content = (choice.get("message") or {}).get("content")
        emit("llm_response", stage=stage, model=body.get("model"),
             response_id=body.get("id"), usage=body.get("usage"), content=content,
             finish_reason=choice.get("finish_reason"),
             elapsed_ms=round((time.monotonic() - started) * 1000))
        if choice.get("finish_reason") not in {None, "stop"}:
            raise ValueError(f"Unfinished completion in {stage}")
        return self._parse_content(content)


def run_campus_model_rag(model: str, questions_path: Path, metadata_path: Path,
                         index_path: Path, output_path: Path, trace_path: Path, *,
                         top_k: int = 10, workers: int = 2, hybrid: bool = False,
                         resume: bool = False, limit: int | None = None,
                         enable_thinking: bool = False) -> list[dict]:
    if workers < 1:
        raise ValueError("--workers must be positive")
    if top_k < 1 or (limit is not None and limit < 1):
        raise ValueError("--top-k and --limit must be positive")
    if output_path.resolve() == trace_path.resolve():
        raise ValueError("Choose different paths for output and trace")
    metadata = load_metadata(metadata_path)
    _, questions = read_csv(questions_path)
    active_questions = questions[:limit] if limit is not None else questions
    if not active_questions:
        raise ValueError("No questions to process")
    index = SearchIndex(index_path, dense=hybrid)
    client = CampusModelClient(model, enable_thinking=enable_thinking)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if resume and trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "row" and record.get("row"):
                rows[record["question_id"]] = record["row"]
    with trace_path.open("a" if resume else "x", encoding="utf-8") as trace:
        lock = threading.Lock()

        def emit(event, question_id=None, **fields):
            line = json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": question_id, "event": event, **fields,
            }, ensure_ascii=False, default=str)
            with lock:
                trace.write(line + "\n")
                trace.flush()

        if resume and rows:
            emit("run_resume", completed_rows=len(rows), model=model)
        else:
            emit("run_start", model=model, backend=client.backend,
                 questions=len(active_questions), top_k=top_k, hybrid=hybrid)

        def process(question):
            def record(event, **fields):
                emit(event, question_id=question["id"], **fields)

            emit("question_start", question_id=question["id"],
                 question=question["question"])
            row = solve(question, index, client, record, top_k)
            validate_rows(COLUMNS, [row], [question], metadata)
            record("row", row=row)
            return row

        pending = [q for q in active_questions if q["id"] not in rows]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, q): q for q in pending}
            for future in as_completed(futures):
                question = futures[future]
                try:
                    rows[question["id"]] = future.result()
                except (RuntimeError, ValueError, OSError, KeyError) as error:
                    emit("question_error", question_id=question["id"],
                         error_type=type(error).__name__, detail=str(error))
                    for other in futures:
                        other.cancel()
                    emit("run_error", failures=1, trace=str(trace_path))
                    raise
        ordered = [rows[q["id"]] for q in active_questions]
        validate_rows(COLUMNS, ordered, active_questions, metadata)
        atomic_csv(output_path, ordered)
        emit("run_complete", rows=len(ordered), output=str(output_path),
             output_sha256=sha256_file(output_path))
    return ordered


def compute_unlabeled_diagnostics(questions, runs: list[RunSpec], metadata: dict, *,
                                  index_path: Path | None = None) -> dict:
    prepared, total_recovered = _prepare_runs(questions, runs, metadata, index_path)
    qids = [str(q["id"]) for q in questions]
    run_stats: dict[str, dict] = {}
    values_by_run: dict[str, dict[str, str]] = {}
    refs_by_run: dict[str, dict[str, set[str]]] = {}
    blanks_by_run: dict[str, dict[str, bool]] = {}

    for item in prepared:
        spec = item["spec"]
        answered = abstained = valid_cited = uncited_answered = 0
        total_tokens = 0
        values: dict[str, str] = {}
        refs: dict[str, set[str]] = {}
        blanks: dict[str, bool] = {}
        for qid in qids:
            row = item["rows"][qid]
            trace_meta = item["trace"].get(qid) or {}
            total_tokens += int(trace_meta.get("total_tokens") or 0)
            blank = is_blank_row(row)
            cited = has_valid_citations(row, metadata)
            blanks[qid] = blank
            values[qid] = str(row.get("answer_value", BLANK))
            refs[qid] = set(_safe_refs(row.get("ref_id", BLANK)))
            if blank:
                abstained += 1
            else:
                answered += 1
                if cited:
                    valid_cited += 1
                else:
                    uncited_answered += 1
        run_stats[spec.name] = {
            "answered": answered, "abstained": abstained,
            "valid_cited": valid_cited, "uncited_answered": uncited_answered,
            "recovered_citations": item["recovered_count"],
            "total_tokens": total_tokens,
            "mean_tokens_per_question": round(total_tokens / len(qids), 2) if qids else 0.0,
            "weight": spec.weight,
        }
        values_by_run[spec.name] = values
        refs_by_run[spec.name] = refs
        blanks_by_run[spec.name] = blanks

    pairwise: list[dict] = []
    names = [spec.name for spec in runs]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left, right = names[i], names[j]
            agree = jointly = joint_agree = abstain_agree = 0
            jaccards: list[float] = []
            for qid in qids:
                lv, rv = values_by_run[left][qid], values_by_run[right][qid]
                lb, rb = blanks_by_run[left][qid], blanks_by_run[right][qid]
                if values_equivalent(lv, rv):
                    agree += 1
                if not lb and not rb:
                    jointly += 1
                    if values_equivalent(lv, rv):
                        joint_agree += 1
                if lb == rb:
                    abstain_agree += 1
                lset, rset = refs_by_run[left][qid], refs_by_run[right][qid]
                union = lset | rset
                jaccards.append(1.0 if not union else len(lset & rset) / len(union))
            n = len(qids)
            pairwise.append({
                "run_a": left, "run_b": right,
                "overall_value_agreement_rate": round(agree / n, 4) if n else 0.0,
                "jointly_answered_count": jointly,
                "jointly_answered_agreement_rate":
                    round(joint_agree / jointly, 4) if jointly else 0.0,
                "abstention_agreement_rate": round(abstain_agree / n, 4) if n else 0.0,
                "mean_citation_jaccard":
                    round(sum(jaccards) / len(jaccards), 4) if jaccards else 0.0,
            })

    all_blank = unanimous_non_blank = majority_cluster = no_majority_split = 0
    for qid in qids:
        candidates: list[dict] = []
        for run_index, name in enumerate(names):
            if blanks_by_run[name][qid]:
                continue
            candidates.append({"run_index": run_index, "run_name": name,
                               "answer_value": values_by_run[name][qid],
                               "score": 1.0, "valid_cited": True})
        if not candidates:
            all_blank += 1
            continue
        clusters = cluster_candidates(candidates)
        largest = max(len(cluster["members"]) for cluster in clusters)
        if len(candidates) == len(names) and len(clusters) == 1:
            unanimous_non_blank += 1
        elif largest > len(names) / 2:
            majority_cluster += 1
        else:
            no_majority_split += 1

    return {
        "questions": len(qids),
        "runs": run_stats,
        "pairwise": pairwise,
        "consensus": {
            "all_blank": all_blank,
            "unanimous_non_blank": unanimous_non_blank,
            "majority_cluster": majority_cluster,
            "no_majority_split": no_majority_split,
        },
        "recovered_citations": total_recovered,
    }


def compute_pareto_frontier(entries: list[dict]) -> list[dict]:
    result = [dict(entry) for entry in entries]
    result.sort(key=lambda e: (float(e.get("mean_tokens_per_question", 0.0) or 0.0),
                               -float(e.get("weighted", 0.0) or 0.0)))
    for entry in result:
        e_tok = float(entry.get("mean_tokens_per_question", 0.0) or 0.0)
        e_w = float(entry.get("weighted", 0.0) or 0.0)
        dominated = False
        for other in result:
            if other is entry:
                continue
            o_tok = float(other.get("mean_tokens_per_question", 0.0) or 0.0)
            o_w = float(other.get("weighted", 0.0) or 0.0)
            if o_tok <= e_tok and o_w >= e_w and (o_tok < e_tok or o_w > e_w):
                dominated = True
                break
        entry["pareto_optimal"] = not dominated
    return result


def _format_rate(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _format_fraction(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def format_pareto_markdown(entries: list[dict]) -> str:
    header = ("| System | Value Acc | Weighted | Cit. Jaccard | Abstention | "
              "Mean Tok/Q | Escalation | Pareto |")
    divider = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, divider]
    for entry in entries:
        system = str(entry.get("system") or entry.get("name") or entry.get("run") or "—")
        value = _format_rate(entry.get("value", entry.get("value_acc")))
        weighted = _format_fraction(entry.get("weighted"))
        citation = _format_fraction(entry.get("citation", entry.get("mean_citation_jaccard",
                                                                   entry.get("cit_jaccard"))))
        abstention = _format_rate(entry.get("abstention"))
        try:
            tokens = f"{float(entry.get('mean_tokens_per_question', 0.0) or 0.0):,.0f}"
        except (TypeError, ValueError):
            tokens = "—"
        escalation = _format_rate(entry.get("escalation", entry.get("escalation_rate")))
        pareto = "✓" if entry.get("pareto_optimal") else ""
        lines.append(f"| {system} | {value} | {weighted} | {citation} | {abstention} | "
                     f"{tokens} | {escalation} | {pareto} |")
    return "\n".join(lines)


def _evaluate_if_gold(rows: list[dict], gold_path: Path | None) -> dict | None:
    if gold_path is None:
        return None
    _, gold = read_csv(gold_path)
    return evaluate(rows, gold)


def _write_diagnostics(path: Path | None, diagnostics: dict) -> None:
    if path is not None:
        write_json(path, diagnostics)


def _cmd_consensus(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    _, questions = read_csv(args.questions)
    runs = [parse_run_spec(spec) for spec in args.run]
    rows, diagnostics = consensus_ensemble(
        questions, runs, metadata, index_path=args.index,
        citation_strategy=args.citation_strategy,
        quorum_alpha=args.quorum_alpha, singleton_alpha=args.singleton_alpha)
    atomic_csv(args.output, rows)
    metrics = _evaluate_if_gold(rows, args.gold)
    summary = dict(diagnostics.get("summary", {}))
    if metrics is not None:
        diagnostics["metrics"] = metrics
        summary["metrics"] = {key: metrics[key]
                              for key in ("questions", "value", "citation",
                                          "abstention", "weighted")}
    _write_diagnostics(args.diagnostics, diagnostics)
    print(json.dumps({"command": "consensus", "output": str(args.output),
                      "summary": summary}, indent=2))
    return 0


def _cmd_cascade(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    _, questions = read_csv(args.questions)
    tier1 = parse_run_spec(args.tier1)
    tier1_aux = parse_run_spec(args.tier1_aux) if args.tier1_aux else None
    tier2 = parse_run_spec(args.tier2)
    rows, diagnostics = cascaded_route(
        questions, tier1, tier2, metadata, tier1_aux=tier1_aux,
        index_path=args.index,
        escalate_on_abstain=not args.no_escalate_abstain,
        escalate_on_missing_citation=not args.no_escalate_missing_citation,
        escalate_on_verif_reject=args.verif_reject_threshold,
        escalate_on_disagreement=not args.no_escalate_disagreement,
        citation_strategy=args.citation_strategy)
    atomic_csv(args.output, rows)
    metrics = _evaluate_if_gold(rows, args.gold)
    summary = dict(diagnostics.get("summary", {}))
    if metrics is not None:
        diagnostics["metrics"] = metrics
        summary["metrics"] = {key: metrics[key]
                              for key in ("questions", "value", "citation",
                                          "abstention", "weighted")}
    _write_diagnostics(args.diagnostics, diagnostics)
    print(json.dumps({"command": "cascade", "output": str(args.output),
                      "summary": summary}, indent=2))
    return 0


def _cmd_run_model(args: argparse.Namespace) -> int:
    rows = run_campus_model_rag(
        args.model, args.questions, args.metadata, args.index, args.output,
        args.trace, top_k=args.top_k, workers=args.workers, hybrid=args.hybrid,
        resume=args.resume, limit=args.limit,
        enable_thinking=args.enable_thinking)
    metrics = _evaluate_if_gold(rows, args.gold)
    summary: dict = {"command": "run-model", "rows": len(rows),
                     "output": str(args.output), "trace": str(args.trace)}
    if metrics is not None:
        summary["metrics"] = {key: metrics[key]
                              for key in ("questions", "value", "citation",
                                          "abstention", "weighted")}
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    _, questions = read_csv(args.questions)
    runs = [parse_run_spec(spec) for spec in args.run]
    diagnostics = compute_unlabeled_diagnostics(
        questions, runs, metadata, index_path=args.index)
    if args.output is not None:
        write_json(args.output, diagnostics)
    print(json.dumps(diagnostics, indent=2))
    return 0


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cmd_pareto(args: argparse.Namespace) -> int:
    metadata = load_metadata(args.metadata)
    _, gold = read_csv(args.gold)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("Manifest must be a JSON list of objects")
    entries: list[dict] = []
    for item in manifest:
        if not isinstance(item, dict) or not item.get("name") or not item.get("csv"):
            raise ValueError("Each manifest item needs 'name' and 'csv'")
        _, rows = read_csv(item["csv"])
        metrics = evaluate(rows, gold)
        entry = {
            "name": item["name"], "system": item["name"],
            "value": metrics["value"], "weighted": metrics["weighted"],
            "citation": metrics["citation"], "abstention": metrics["abstention"],
            "questions": metrics["questions"],
        }
        mean_tokens = item.get("mean_tokens_per_question")
        diagnostics = None
        if item.get("diagnostics"):
            diagnostics = _read_json(item["diagnostics"])
        if mean_tokens is None and diagnostics is not None:
            mean_tokens = (diagnostics.get("summary") or {}).get(
                "mean_tokens_per_question")
        if mean_tokens is None and item.get("trace"):
            mean_tokens = trace_token_summary(Path(item["trace"])).get(
                "mean_tokens_per_question")
        entry["mean_tokens_per_question"] = float(mean_tokens or 0.0)
        escalation = item.get("escalation_rate")
        if escalation is None and diagnostics is not None:
            escalation = (diagnostics.get("summary") or {}).get("escalation_rate")
        entry["escalation_rate"] = None if escalation is None else float(escalation)
        if item.get("notes"):
            entry["notes"] = item["notes"]
        entries.append(entry)

    ranked = compute_pareto_frontier(entries)
    markdown = format_pareto_markdown(ranked)
    if args.output_json is not None:
        write_json(args.output_json, {"entries": ranked})
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wattbot.ensemble",
        description="WattBot consensus/cascade ensemble and Pareto tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consensus = subparsers.add_parser(
        "consensus", help="Run weighted consensus ensemble over runs")
    consensus.add_argument("--questions", type=Path, required=True)
    consensus.add_argument("--metadata", type=Path, required=True)
    consensus.add_argument("--run", action="append", required=True,
                           help="name=csv[::trace[::weight]]")
    consensus.add_argument("--output", type=Path, required=True)
    consensus.add_argument("--diagnostics", type=Path)
    consensus.add_argument("--index", type=Path)
    consensus.add_argument("--gold", type=Path)
    consensus.add_argument("--citation-strategy",
                           choices=["majority", "intersection", "best"],
                           default="majority")
    consensus.add_argument("--quorum-alpha", type=float, default=0.75)
    consensus.add_argument("--singleton-alpha", type=float, default=1.0)
    consensus.set_defaults(func=_cmd_consensus)

    cascade = subparsers.add_parser(
        "cascade", help="Run cascaded tier1/tier2 routing")
    cascade.add_argument("--questions", type=Path, required=True)
    cascade.add_argument("--metadata", type=Path, required=True)
    cascade.add_argument("--tier1", type=str, required=True)
    cascade.add_argument("--tier1-aux", type=str)
    cascade.add_argument("--tier2", type=str, required=True)
    cascade.add_argument("--output", type=Path, required=True)
    cascade.add_argument("--diagnostics", type=Path)
    cascade.add_argument("--index", type=Path)
    cascade.add_argument("--gold", type=Path)
    cascade.add_argument("--verif-reject-threshold", type=int, default=0)
    cascade.add_argument("--no-escalate-abstain", action="store_true")
    cascade.add_argument("--no-escalate-missing-citation", action="store_true")
    cascade.add_argument("--no-escalate-disagreement", action="store_true")
    cascade.add_argument("--citation-strategy",
                         choices=["majority", "intersection", "best"],
                         default="majority")
    cascade.set_defaults(func=_cmd_cascade)

    run_model = subparsers.add_parser(
        "run-model", help="Run a single end-to-end campus RAG model")
    run_model.add_argument("--model", type=str, required=True)
    run_model.add_argument("--questions", type=Path, required=True)
    run_model.add_argument("--metadata", type=Path, required=True)
    run_model.add_argument("--index", type=Path, required=True)
    run_model.add_argument("--output", type=Path, required=True)
    run_model.add_argument("--trace", type=Path, required=True)
    run_model.add_argument("--top-k", type=int, default=10)
    run_model.add_argument("--workers", type=int, default=2)
    run_model.add_argument("--hybrid", action="store_true")
    run_model.add_argument("--resume", action="store_true")
    run_model.add_argument("--limit", type=int)
    run_model.add_argument("--enable-thinking", action="store_true")
    run_model.add_argument("--gold", type=Path)
    run_model.set_defaults(func=_cmd_run_model)

    diagnose = subparsers.add_parser(
        "diagnose", help="Compute unlabeled run diagnostics")
    diagnose.add_argument("--questions", type=Path, required=True)
    diagnose.add_argument("--metadata", type=Path, required=True)
    diagnose.add_argument("--run", action="append", required=True)
    diagnose.add_argument("--index", type=Path)
    diagnose.add_argument("--output", type=Path)
    diagnose.set_defaults(func=_cmd_diagnose)

    pareto = subparsers.add_parser(
        "pareto", help="Evaluate and rank a manifest of submissions")
    pareto.add_argument("--gold", type=Path, required=True)
    pareto.add_argument("--metadata", type=Path, required=True)
    pareto.add_argument("--manifest", type=Path, required=True)
    pareto.add_argument("--output-json", type=Path)
    pareto.add_argument("--output-md", type=Path)
    pareto.set_defaults(func=_cmd_pareto)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
