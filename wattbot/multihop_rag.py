"""Contextual hybrid retrieval for multi-hop questions.

Part 1: indexing, tokenisation, contextual hybrid retrieval and a retrieval
benchmark.  The multi-hop solver and CLI live in a later module.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scripts.make_submission import COLUMNS
from scripts.summarize_token_usage import summarize as summarize_token_usage
from wattbot.__main__ import atomic_csv
from wattbot.deepseek import DeepSeek
from wattbot.evaluation import evaluate
from wattbot.pipeline import (
    BLANK,
    MISSING_VALUES,
    NUMBER,
    RANGE,
    SCALE,
    abstention,
    canonical_answer_value,
    compact,
    load_metadata,
    model_value,
    parse_list,
    read_csv,
    required_text,
    source_quote,
    validate_rows,
)
from wattbot.retrieval import SearchIndex, sha256_file, write_json

try:  # pragma: no cover - optional native dependency
    from py_rust_stemmers import SnowballStemmer

    _STEMMER = SnowballStemmer("english")
except Exception:  # pragma: no cover
    _STEMMER = None

STOPWORDS = frozenset(
    "a about above after again against all am an and any are aren't as at be because been "
    "before being below between both but by can cannot could did do does doing down during "
    "each few for from further had has have having he her here hers herself him himself his "
    "how i if in into is it its itself let's me more most my myself no nor not of off on once "
    "only or other ought our ours ourselves out over own same she should so some such than that "
    "the their theirs them themselves then there these they this those through to too under "
    "until up very was we were what when where which while who whom why will with would you "
    "your yours yourself yourselves".split()
)

SHORT_PREFIXES = frozenset(
    "co pre non re sub multi cross self well high low end on off mid in out open full half real "
    "fine zero one two per up down over under all us".split()
)

TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+){0,3}\.?\s+[A-Z][^|]{2,110}$")
UNIT_HINT = re.compile(r"\(\s*[A-Za-z%$€£][^)]*\)")
TABLE_CAPTION = re.compile(
    r"^\s*(?:\*\*|_|\*)*\s*table\s+[A-Za-z0-9]+(?:\.\d+)*\s*[:.\-\s]", re.I
)


def clean_text(text: str) -> str:
    """Normalise markdown-ish text, repairing line breaks and hyphenation."""
    text = unicodedata.normalize("NFKC", text or "")
    lines = []
    for line in text.splitlines():
        if line.count("|") >= 2:
            line = re.sub(r"<br\s*/?>", " ", line, flags=re.I)
        else:
            line = re.sub(r"<br\s*/?>", "\n", line, flags=re.I)
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"</?(?:sup|sub|u)>", "", text, flags=re.I)
    text = re.sub(r"([A-Za-z]{2,})-\s*\n+\s*([a-z]{2,})", r"\1\2", text)

    def inline(match: re.Match[str]) -> str:
        head, tail = match.group(1), match.group(2)
        if head.casefold() in SHORT_PREFIXES:
            return match.group(0)
        return head + tail

    text = re.sub(r"([A-Za-z]{2,})-\s+([a-z]{2,})", inline, text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stem(word: str) -> str:
    if _STEMMER is None:
        return word
    try:
        return _STEMMER.stem_word(word)
    except Exception:  # pragma: no cover - defensive
        return word


def clean_tokens(text: str, stem: bool = True) -> list[str]:
    """Casefolded, stopword-filtered, optionally stemmed search terms."""
    lowered = clean_text(text).casefold()
    found = re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)+", lowered)
    found.extend(re.findall(r"[a-z0-9]+", lowered))
    result = []
    for token in found:
        if token in STOPWORDS:
            continue
        if len(token) == 1 and token.isalpha():
            continue
        if stem and token.isalpha() and len(token) >= 4:
            token = _stem(token)
        result.append(token)
    return result


def safe_eval_decimal(expr: str) -> Decimal | None:
    """Evaluate a trivial arithmetic expression into an exact Decimal."""
    try:
        tree = ast.parse(str(expr).strip(), mode="eval")
    except (SyntaxError, ValueError, TypeError):
        return None

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("unsupported constant")
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.UAdd):
                return +evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("unsupported binary operator")
        raise ValueError("unsupported AST node")

    try:
        return evaluate(tree)
    except (ValueError, ZeroDivisionError, InvalidOperation, TypeError):
        return None


UNIT_ALIASES = {
    "$": "usd",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "us dollars": "usd",
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "l": "liters",
    "liter": "liters",
    "liters": "liters",
    "litre": "liters",
    "litres": "liters",
    "lb": "lbs",
    "lbs": "lbs",
    "pound": "lbs",
    "pounds": "lbs",
    "tco2e": "tco2e",
    "metric tons co2e": "tco2e",
    "tonnes co2e": "tco2e",
}

SCALE_PREFIX = re.compile(
    r"^([KMBT]|thousands?|millions?|billions?|trillions?)\s+(?:of\s+)?(.+)$", re.I
)
CURRENCY_SCALE = re.compile(
    r"^(\$|usd)\s+(thousands?|millions?|billions?|trillions?)$", re.I
)
SI_PAIRS = {
    ("mwh", "kwh"): Decimal(1000),
    ("gwh", "kwh"): Decimal(1000000),
    ("kwh", "mwh"): Decimal("0.001"),
    ("ml", "liters"): Decimal("0.001"),
    ("kg", "g"): Decimal(1000),
    ("g", "kg"): Decimal("0.001"),
    ("metric tons", "kg"): Decimal(1000),
    ("tonnes", "kg"): Decimal(1000),
}
RANGE_PATTERN = re.compile(
    r"^\[\s*([^,\]]+?)\s*,\s*([^,\]]+?)\s*\]$|^([^\-]+?)\s*-\s*([^\-]+?)$"
)


def _canon_unit(unit: str) -> str:
    key = (unit or "").strip().casefold()
    return UNIT_ALIASES.get(key, key)


def _fmt_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _parse_decimal(text) -> Decimal | None:
    try:
        return Decimal(str(text).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _apply_factor(value, factor: Decimal):
    """Scale a single number or ``[low, high]`` range, preserving shape."""
    text = str(value).strip()
    single = _parse_decimal(text)
    if single is not None:
        return _fmt_decimal(single * factor)
    match = RANGE_PATTERN.match(text)
    if match:
        groups = match.groups()
        low = _parse_decimal(groups[0] if groups[0] is not None else groups[2])
        high = _parse_decimal(groups[1] if groups[1] is not None else groups[3])
        if low is not None and high is not None:
            return f"[{_fmt_decimal(low * factor)},{_fmt_decimal(high * factor)}]"
    return None


def _scale_key(prefix: str) -> str:
    if len(prefix) == 1:
        return prefix.upper()
    return prefix.casefold().rstrip("s")


def normalize_submission_fields(question: dict, row: dict) -> dict:
    """Normalise an answer value/unit against the question's unit hint."""
    value = canonical_answer_value(question["question"], row.get("answer_value", BLANK))
    hint = (question.get("answer_unit") or BLANK).strip()
    unit = (row.get("answer_unit") or BLANK).strip()

    if str(value).casefold() in MISSING_VALUES and hint != BLANK:
        return abstention(question)

    if value != BLANK and hint != BLANK:
        if _canon_unit(unit) == _canon_unit(hint):
            return {**row, "answer_value": value, "answer_unit": hint}

        currency = CURRENCY_SCALE.match(unit)
        if currency:
            base = currency.group(1)
            key = _scale_key(currency.group(2))
        else:
            scaled = SCALE_PREFIX.match(unit)
            if not scaled:
                scaled = None
            if scaled:
                base = scaled.group(2)
                key = _scale_key(scaled.group(1))
            else:
                base = None
                key = None

        if base is not None and _canon_unit(base) == _canon_unit(hint):
            factor = SCALE.get(key)
            if factor is not None:
                factor = Decimal(str(factor))
                scaled_value = _apply_factor(value, factor)
                if scaled_value is not None:
                    value = scaled_value
                return {**row, "answer_value": value, "answer_unit": hint}

        factor = SI_PAIRS.get((_canon_unit(unit), _canon_unit(hint)))
        if factor is not None:
            scaled_value = _apply_factor(value, factor)
            if scaled_value is not None:
                value = scaled_value
            return {**row, "answer_value": value, "answer_unit": hint}

    return {**row, "answer_value": value}


PLAN_MULTIHOP = """You are a retrieval planner for a multi-hop question answering system over
technical and sustainability documents.

Read the question and produce a compact JSON plan. Return ONLY JSON with this schema:
{
  "question_type": "single_fact|multi_hop_calculation|comparison|true_false|range_or_list",
  "needs_decomposition": true,
  "fact_slots": [{"slot": "short slot name", "queries": ["query", "query"]}],
  "global_queries": ["query", "query"]
}

Rules:
- Decompose multi-fact, baseline-conversion, cross-table, or comparison questions into
  2-4 independent atomic fact_slots. Each slot must retrieve exactly one fact.
- NEVER mix entities across slots: one slot per entity/metric/baseline so results do not
  contaminate each other.
- Give each slot 1-3 alternative queries that expand domain synonyms, for example
  household / US home / residential, average American / human life, and
  electricity / energy / kWh / MWh.
- Use global_queries for the overall question framing and any terms shared by all slots.
- For a simple single-fact or true/false question, set "needs_decomposition": false and
  provide one fact_slot (or none) plus useful global_queries. Do not invent extra hops.
"""

ANSWER_MULTIHOP = """You answer a question using ONLY the supplied evidence passages.

Return ONLY JSON, in one of these two shapes.
Answerable:
{
  "answerable": true,
  "answer": "short natural-language answer",
  "answer_value": "numeric or literal value",
  "answer_unit": "unit",
  "calculation": "optional arithmetic expression, e.g. 2.5 * 1000 / 3",
  "explanation": "brief justification",
  "evidence": [{"node_id": "...", "quote": "verbatim supporting text"}]
}
Unanswerable:
{
  "answerable": false,
  "found_facts": "what was found so far",
  "missing_facts": ["atomic search query..."],
  "explore_adjacent_pages": ["node_id"],
  "reason": "why it cannot be answered"
}

Rules:
(a) If any required operand or baseline is absent from the evidence, DO NOT guess: emit
    "answerable": false with concrete atomic "missing_facts" queries.
(b) If unit_hint != is_blank, ALWAYS express numeric values in that unit. Apply
    table-header scale multipliers such as "million $", "million L", "MWh", etc.
(c) Prefer exact table numbers over rounded prose approximations, and put the exact
    unrounded arithmetic in "calculation".
(d) True/False semantics: answer_value "1" means supported, "0" means refuted; use
    "answerable": false only when the evidence does not address the claim.
(e) Cite minimally and sufficiently. Prefer a single document when one document proves
    the answer. When several documents report tied values, pick the single passage whose
    wording and framing most closely match the question. Cite multiple documents only for
    genuine multi-hop or explicitly multi-source questions.
(f) Formatting: answer_value must be a nonempty string — plain number (no commas, %, ~, <, >), [low,high] range, 1/0 for True/False, or concise categorical label; use is_blank for answer_unit when not applicable (including True/False).
"""

VERIFY_MULTIHOP = """You verify a proposed answer against its cited evidence passages.

Return ONLY JSON:
{
  "supported": true,
  "error_type": "none|missing_support|factual_mismatch|math_or_unit_error|over_citation|format_error",
  "missing_facts": ["atomic search query..."],
  "keep_node_ids": ["node_id"],
  "reason": "brief justification"
}

Rules:
- Do NOT reject for harmless synonym, capitalization, hyphenation, or section-heading
  wording differences when the cited passage unambiguously matches the asked entity and
  metric.
- DO reject with "math_or_unit_error" when a scale factor is missed or the arithmetic
  is wrong.
- DO reject with "missing_support" and populate "missing_facts" when any required operand
  or baseline is absent from the evidence.
- Mark "over_citation" with "keep_node_ids" ONLY when multiple documents are cited for a
  single-fact or true/false claim that a strict subset already proves; keep that subset.
- Use "supported": true and "error_type": "none" when the answer is correct and supported.
"""


class ResilientDeepSeek:
    """Thin wrapper around ``DeepSeek`` with transport-level retry/backoff."""

    RETRY_TOKENS = ("429", "500", "502", "503", "504", "network", "timed out", "timeout")

    def __init__(self, backend="hkustgz", *, timeout=120, transport_attempts=6,
                 backoff=(5, 10, 20, 35)):
        self.client = DeepSeek(
            backend, timeout=timeout, transport_attempts=transport_attempts
        )
        self.backend = backend
        self.model = getattr(self.client, "model", None)
        self.backoff = tuple(backoff)

    @classmethod
    def _retryable(cls, error):
        message = str(error).casefold()
        return any(token.casefold() in message for token in cls.RETRY_TOKENS)

    def complete(self, stage, system, data, emit):
        error = None
        try:
            return self.client.complete(stage, system, data, emit)
        except RuntimeError as exc:
            if not self._retryable(exc):
                raise
            error = exc
        for retry, delay in enumerate(self.backoff, 1):
            emit(
                "llm_retry_backoff",
                stage=stage,
                retry=retry,
                delay_s=delay,
                detail=str(error),
            )
            time.sleep(delay)
            try:
                return self.client.complete(stage, system, data, emit)
            except RuntimeError as exc:
                if not self._retryable(exc):
                    raise
                error = exc
        raise error


def _row_from_citations(question, reply, citations):
    ref_ids = []
    ref_urls = []
    materials = []
    for citation in citations:
        ref_id = citation["ref_id"]
        if ref_id not in ref_ids:
            ref_ids.append(ref_id)
            ref_urls.append(citation.get("ref_url", ""))
        materials.append(
            f"[{ref_id} | p.{citation.get('page')}] {citation.get('quote', '')}".strip()
        )
    return {
        "id": question.get("id") or question.get("question_id") or "",
        "question": question.get("question", ""),
        "answer": required_text(reply, "answer"),
        "answer_value": model_value(reply.get("answer_value")),
        "answer_unit": str(reply.get("answer_unit") or BLANK).strip(),
        "ref_id": json.dumps(ref_ids),
        "ref_url": json.dumps(ref_urls),
        "supporting_materials": "\n".join(materials),
        "explanation": required_text(reply, "explanation"),
    }


def _gather_evidence(reply):
    evidence = reply.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("evidence must be a non-empty list")
    items = []
    for entry in evidence:
        if not isinstance(entry, dict):
            raise ValueError("evidence entries must be objects")
        node_id = entry.get("node_id")
        quote = entry.get("quote")
        if not node_id or not isinstance(quote, str) or not quote.strip():
            raise ValueError("evidence requires node_id and quote")
        items.append((str(node_id), quote))
    return items


def resolve_candidate(question, reply, context, index, emit, recover_quotes=False):
    """Ground an LLM answer reply against retrieved nodes and build a row."""
    by_id = {node["id"]: node for node in context}
    citations = []
    seen = set()
    for node_id, quote in _gather_evidence(reply):
        node = by_id.get(node_id)
        if node is None:
            if not recover_quotes:
                raise ValueError(f"unknown node_id {node_id}")
            needle = compact(quote)
            matches = [
                candidate for candidate in context
                if needle and needle in compact(candidate.get("text", ""))
            ]
            if len({candidate["ref_id"] for candidate in matches}) != 1:
                continue
            node = matches[0]
            node_id = node["id"]

        text = node.get("text", "")
        page_text = index.pages.get(node.get("page_id"), "")
        if quote and (quote in text or quote in page_text):
            grounded = quote
        else:
            grounded = source_quote(quote, text) or source_quote(quote, page_text)
            if not grounded and recover_quotes:
                grounded = node.get("matched_text") or text[:400]
            if not grounded:
                raise ValueError(f"quote could not be grounded for {node_id}")

        key = (node_id, grounded)
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "node_id": node_id,
            "ref_id": node.get("ref_id"),
            "ref_url": index.sources.get(node.get("ref_id"), {}).get("url", ""),
            "page": node.get("page"),
            "section": node.get("section"),
            "quote": grounded,
            "passage_excerpt": node.get("text", "")[:1200],
        })

    if not citations:
        raise ValueError("no usable citations could be resolved")

    row = _row_from_citations(question, reply, citations)
    verified = [
        {
            "node_id": citation["node_id"],
            "ref_id": citation["ref_id"],
            "page": citation["page"],
            "section": citation["section"],
            "quote": citation["quote"],
            "passage_excerpt": citation["passage_excerpt"],
        }
        for citation in citations
    ]
    return row, verified


def prune_candidate_citations(row, cited, keep_node_ids, sources):
    keep_set = set(keep_node_ids or [])
    keep = [citation for citation in cited if citation["node_id"] in keep_set]
    if not (0 < len(keep) < len(cited)):
        return row, cited
    if not all(citation["ref_id"] in sources for citation in keep):
        return row, cited

    refs = {}
    materials = []
    for citation in keep:
        ref_id = citation["ref_id"]
        if ref_id not in refs:
            refs[ref_id] = sources.get(ref_id, {}).get("url", "")
        materials.append(
            f"[{ref_id} | p.{citation.get('page')}] {citation.get('quote', '')}".strip()
        )
    pruned = dict(row)
    pruned["ref_id"] = json.dumps(list(refs))
    pruned["ref_url"] = json.dumps(list(refs.values()))
    pruned["supporting_materials"] = "\n".join(materials)
    return pruned, keep


def _string_list(value):
    if isinstance(value, str):
        value = [value]
    return [str(item).strip() for item in (value or []) if str(item).strip()]


def _passage_payload(node):
    return {
        "id": node.get("id"),
        "ref_id": node.get("ref_id"),
        "page": node.get("page"),
        "title": node.get("title"),
        "kind": node.get("kind"),
        "section": node.get("section"),
        "lexical_alignment": node.get("lexical_alignment"),
        "matched_slots": node.get("matched_slots"),
        "text": node.get("text"),
    }


def _citation_brief(cited):
    return [
        {key: citation[key] for key in ("node_id", "ref_id", "page", "quote")}
        for citation in cited
    ]


def _calculation_mismatch(reply, row):
    calculation = reply.get("calculation")
    if not isinstance(calculation, str) or not calculation.strip():
        return None
    calc_val = safe_eval_decimal(calculation)
    if calc_val is None or not re.fullmatch(NUMBER, row.get("answer_value", "")):
        return None
    ans_dec = Decimal(row["answer_value"])
    if calc_val == 0:
        return None
    relative = abs(ans_dec - calc_val) / abs(calc_val)
    if relative <= Decimal("0.005"):
        return None
    if ans_dec != 0:
        ratio = abs(ans_dec / calc_val)
        for factor in (
            Decimal("1e3"), Decimal("1e6"), Decimal("1e9"), Decimal("1e12"),
            Decimal("1e-3"), Decimal("1e-6"),
        ):
            if abs(ratio - factor) / factor <= Decimal("0.005"):
                return None
    return calc_val


def solve_multihop(question, index, client, emit, *, top_k=12,
                   retrieval_mode="hybrid_rerank", max_rounds=3):
    query = question.get("question", "")
    unit_hint = question.get("answer_unit") or BLANK

    fact_slots = []
    global_queries = []
    needs_decomposition = False
    try:
        plan = client.complete(
            "plan", PLAN_MULTIHOP,
            {"question": query, "unit_hint": unit_hint}, emit,
        )
    except ValueError:
        plan = {}
    if isinstance(plan, dict):
        raw_slots = plan.get("fact_slots")
        if isinstance(raw_slots, list):
            for slot in raw_slots[:4]:
                if not isinstance(slot, dict) or not slot.get("slot"):
                    continue
                queries = _string_list(slot.get("queries"))
                if queries:
                    fact_slots.append({"slot": str(slot["slot"]), "queries": queries})
        global_queries = _string_list(plan.get("global_queries"))[:3]
        needs_decomposition = bool(plan.get("needs_decomposition"))
    emit(
        "plan",
        needs_decomposition=needs_decomposition,
        fact_slots=fact_slots,
        global_queries=global_queries,
    )

    context = []
    prior_ids = []
    feedback = ""
    pending_missing_facts = []
    pending_adjacent = []
    keep_prior_ids = []

    for attempt in range(1, max_rounds + 1):
        if attempt == 1:
            hits = index.search_multihop(
                query, global_queries, fact_slots, top_k, mode=retrieval_mode
            )
            context = index.expand(hits, 2000)
        elif pending_missing_facts or pending_adjacent:
            fetched = []
            for missing in pending_missing_facts[:4]:
                fetched.extend(
                    index.search_mode(
                        [missing], top_k=6, mode=retrieval_mode, rerank_query=missing
                    )
                )
            fetched.extend(
                index.adjacent_page_nodes(
                    pending_adjacent, exclude_ids=prior_ids, max_nodes=4
                )
            )
            keep_set = set(keep_prior_ids)
            preserved = [node for node in context if node.get("id") in keep_set]
            preserved.extend(
                node for node in context if node.get("id") not in keep_set
            )
            merged = []
            seen = set()
            for node in preserved + fetched:
                node_id = node.get("id")
                if node_id in seen:
                    continue
                seen.add(node_id)
                merged.append(node)
            context = index.expand(merged[: top_k + 8], 2600)
            pending_missing_facts = []
            pending_adjacent = []
        else:
            hits = index.search_multihop(
                query, global_queries, fact_slots, top_k + 6, mode=retrieval_mode
            )
            context = index.expand(hits, 2600)

        prior_ids = [node["id"] for node in context]
        emit(
            "retrieval",
            attempt=attempt,
            count=len(context),
            refs=sorted({node["ref_id"] for node in context}),
        )
        messages = [_passage_payload(node) for node in context]

        try:
            reply = client.complete(
                "answer", ANSWER_MULTIHOP,
                {
                    "question": query,
                    "unit_hint": unit_hint,
                    "fact_slots": fact_slots,
                    "passages": messages,
                    "previous_feedback": feedback,
                },
                emit,
            )

            if reply.get("answerable") is False:
                pending_missing_facts = _string_list(reply.get("missing_facts"))
                pending_adjacent = _string_list(reply.get("explore_adjacent_pages"))
                keep_prior_ids = [node["id"] for node in context][:top_k]
                found_facts = reply.get("found_facts")
                reason = str(reply.get("reason") or "")
                feedback = reason + (
                    f" Found so far: {found_facts}" if found_facts else ""
                )
                emit(
                    "verification",
                    attempt=attempt,
                    supported=False,
                    error_type="missing_support",
                    reason=feedback,
                    decision="abstained" if attempt == max_rounds else "followup_needed",
                )
                if attempt == max_rounds:
                    return abstention(question)
                continue

            if reply.get("answerable") is not True:
                raise ValueError("Expected boolean answerable")

            row, cited = resolve_candidate(
                question, reply, context, index, emit,
                recover_quotes=attempt >= 2,
            )

            canonical = normalize_submission_fields(question, row)
            if canonical != row:
                emit("candidate_normalized", attempt=attempt, row=canonical)
            row = canonical

            if row["answer_value"] == BLANK:
                if attempt == max_rounds:
                    return abstention(question)
                continue

            calc_val = _calculation_mismatch(reply, row)
            if calc_val is not None and attempt < max_rounds:
                emit(
                    "calculation_mismatch",
                    attempt=attempt,
                    calculation=reply.get("calculation"),
                    value=row["answer_value"],
                    evaluated=_fmt_decimal(calc_val),
                )
                feedback = (
                    f'Arithmetic mismatch: calculation {reply["calculation"]} '
                    f'evaluates to {_fmt_decimal(calc_val)}, not {row["answer_value"]}. '
                    "Recheck arithmetic and unit conversion."
                )
                keep_prior_ids = [citation["node_id"] for citation in cited]
                continue

            emit(
                "candidate",
                attempt=attempt,
                row=row,
                citations=_citation_brief(cited),
            )

            review = client.complete(
                "verify", VERIFY_MULTIHOP,
                {
                    "question": query,
                    "unit_hint": unit_hint,
                    "answer": row["answer"],
                    "answer_value": row["answer_value"],
                    "answer_unit": row["answer_unit"],
                    "calculation": reply.get("calculation", ""),
                    "explanation": row["explanation"],
                    "citations": cited,
                },
                emit,
            )

            supported = review.get("supported")
            error_type = str(
                review.get("error_type") or ("none" if supported else "factual_mismatch")
            )
            reason = required_text(review, "reason")
            keep_ids = review.get("keep_node_ids") or []

            if supported is True:
                if error_type == "over_citation" and isinstance(keep_ids, list):
                    pruned_row, pruned = prune_candidate_citations(
                        row, cited, keep_ids, index.sources
                    )
                    if len(pruned) < len(cited):
                        emit(
                            "citations_pruned",
                            attempt=attempt,
                            citations=_citation_brief(pruned),
                        )
                        row = pruned_row
                emit(
                    "verification",
                    attempt=attempt,
                    supported=True,
                    error_type=error_type,
                    reason=reason,
                    decision="accepted",
                )
                return row

            if supported is False and error_type == "over_citation" and isinstance(
                keep_ids, list
            ):
                pruned_row, pruned = prune_candidate_citations(
                    row, cited, keep_ids, index.sources
                )
                if len(pruned) < len(cited):
                    emit(
                        "citations_pruned",
                        attempt=attempt,
                        citations=_citation_brief(pruned),
                    )
                    emit(
                        "verification",
                        attempt=attempt,
                        supported=True,
                        error_type="over_citation",
                        reason=reason,
                        decision="accepted_pruned",
                    )
                    return pruned_row

            emit(
                "verification",
                attempt=attempt,
                supported=False,
                error_type=error_type,
                reason=reason,
                decision="rejected",
            )
            if error_type == "missing_support":
                pending_missing_facts = _string_list(review.get("missing_facts"))
                pending_adjacent = [citation["node_id"] for citation in cited]
                keep_prior_ids = [citation["node_id"] for citation in cited]
            feedback = f"[{error_type}] {reason}"
            if attempt == max_rounds:
                return abstention(question)
        except ValueError as error:
            emit(
                "invalid_output",
                attempt=attempt,
                detail=str(error),
            )
            if attempt == max_rounds:
                return abstention(question)

    return abstention(question)


def _is_subheader(row: str) -> bool:
    if UNIT_HINT.search(row):
        return True
    cells = row.strip().strip("|").split("|")
    return any(not cell.strip() for cell in cells)


def _heading(line: str) -> str | None:
    match = HEADING.match(line)
    if match:
        return match.group(2).strip()
    if len(line) <= 120 and NUMBERED_HEADING.match(line):
        return line.strip()
    return None


def _table_ranges(lines: list[str]) -> list[tuple[int, int, str]]:
    """Locate contiguous markdown tables with their surrounding caption."""
    ranges = []
    position = 0
    while position < len(lines):
        line = lines[position]
        if TABLE_SEPARATOR.match(line) and position > 0 and "|" in lines[position - 1]:
            start = position - 1
            while start > 0 and "|" in lines[start - 1] and lines[start - 1].strip():
                start -= 1
            end = position + 1
            while end < len(lines) and "|" in lines[end] and lines[end].strip():
                end += 1
            caption = ""
            caption_index = -1
            above = [(i, lines[i]) for i in range(max(0, start - 4), start) if lines[i].strip()][-2:]
            below = [(i, lines[i]) for i in range(end, min(len(lines), end + 4)) if lines[i].strip()][:2]
            for i, text in above + below:
                if "|" not in text and TABLE_CAPTION.search(text):
                    caption = compact(text)
                    caption_index = i
                    break
            ranges.append((start, end, caption, caption_index))
            position = end
        else:
            position += 1
    return ranges


def _make_node(ref_id, title, page_num, page_id, kind, section, text, index):
    prefix = f"[{ref_id} | {title} | p.{page_num} | {section}]"
    return {
        "id": f"{page_id}:n{index}",
        "page_id": page_id,
        "ref_id": ref_id,
        "page": page_num,
        "kind": kind,
        "section": section,
        "text": text,
        "context_prefix": prefix,
        "search_text": f"{prefix}\n{text}",
    }


def _table_nodes(ref_id, title, page_num, page_id, block, section, caption, index):
    rows = [compact(re.sub(r"<br\s*/?>", " ", line, flags=re.I)) for line in block if line.strip()]
    if len(rows) < 2:
        return []
    header = [rows[0]]
    body = rows[2:]
    if body and _is_subheader(body[0]):
        header.append(body[0])
        body = body[1:]
    if not body:
        body = rows[2:] or list(header)
    head = "\n".join(([caption] if caption else []) + header)
    chunks = []
    current = head
    for row in body:
        if current != head and len(current) + len(row) + 1 > 1300:
            chunks.append(current)
            current = head
        current = f"{current}\n{row}"
    chunks.append(current)
    nodes = []
    for offset, chunk in enumerate(chunks):
        nodes.append(
            _make_node(ref_id, title, page_num, page_id, "table", section, chunk, index + offset)
        )
    return nodes


def _split_long(paragraph, limit, overlap):
    """Split a single oversized paragraph on word boundaries."""
    pieces = []
    start = 0
    length = len(paragraph)
    while start < length:
        end = min(start + limit, length)
        if end < length:
            cut = paragraph.rfind(" ", start + max(1, limit // 2), end)
            if cut != -1:
                end = cut
        piece = paragraph[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= length:
            break
        next_start = max(end - overlap, start + 1)
        space = paragraph.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return pieces


def _text_windows(paragraphs, limit=950, overlap=150):
    chunks = []
    current = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for piece in _split_long(paragraph, limit, overlap) if len(paragraph) > limit else [paragraph]:
            if current and len(current) + len(piece) + 1 > limit:
                chunks.append(current)
                tail = current[-overlap:]
                cut = tail.find(" ")
                tail = tail[cut + 1:] if cut != -1 else tail
                current = f"{tail}\n{piece}".strip()
            else:
                current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def parse_page_chunks(ref_id, title, page_num, raw_text, current_section):
    """Split one PDF page into table and section-aware text nodes."""
    lines = clean_text(raw_text).split("\n")
    page_id = f"{ref_id}:p{page_num}"
    ranges = _table_ranges(lines)
    starts = {start: (end, caption) for start, end, caption, _ in ranges}
    captions = {index for _, _, _, index in ranges if index >= 0}
    nodes: list[dict] = []
    buffer: list[str] = []
    section = current_section or "Body"

    def flush():
        nonlocal buffer
        if not buffer:
            return
        for chunk in _text_windows(buffer):
            nodes.append(
                _make_node(ref_id, title, page_num, page_id, "text", section, chunk, len(nodes))
            )
        buffer = []

    position = 0
    while position < len(lines):
        if position in starts:
            flush()
            end, caption = starts[position]
            nodes.extend(
                _table_nodes(ref_id, title, page_num, page_id, lines[position:end],
                             section, caption, len(nodes))
            )
            position = end
            continue
        line = lines[position].strip()
        is_caption = position in captions
        position += 1
        if not line or is_caption:
            continue
        heading = _heading(line)
        if heading:
            flush()
            section = heading
            continue
        buffer.append(line)
    flush()
    return nodes, section


def build_contextual_index(metadata, docs, output):
    """Extract every PDF into a version-2 contextual node index."""
    import pymupdf
    import pymupdf4llm

    sources = {}
    pages = {}
    nodes = []
    for ref_id, entry in metadata.items():
        path = Path(docs) / f"{ref_id}.pdf"
        title = entry.get("title", ref_id)
        raw_chunks = []
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                raw_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        except Exception:
            raw_chunks = []
        section = ""
        with pymupdf.open(path) as document:
            for number, page in enumerate(document):
                raw_text = ""
                if number < len(raw_chunks):
                    raw_text = (raw_chunks[number] or {}).get("text") or ""
                if not raw_text.strip():
                    raw_text = "\n".join(
                        block[4] for block in page.get_text("blocks", sort=True)
                        if block[6] == 0 and block[4].strip()
                    )
                page_id = f"{ref_id}:p{number + 1}"
                pages[page_id] = clean_text(raw_text)
                page_nodes, section = parse_page_chunks(
                    ref_id, title, number + 1, raw_text, section
                )
                nodes.extend(page_nodes)
            total_pages = document.page_count
        sources[ref_id] = {
            "title": title,
            "url": entry.get("url", ""),
            "sha256": sha256_file(path),
            "pages": total_pages,
        }
    write_json(output, {"version": 2, "sources": sources, "pages": pages, "nodes": nodes})
    return {"sources": sources, "pages": pages, "nodes": nodes}


class ContextualSearchIndex:
    """BM25 + dense + cross-encoder ranking over contextual nodes."""

    def __init__(self, path, *, embed_model="BAAI/bge-base-en-v1.5",
                 rerank_model="jinaai/jina-reranker-v1-turbo-en",
                 use_dense=True, use_rerank=True, stem=True):
        self.path = Path(path)
        self.stem = stem
        self.digest = sha256_file(self.path)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.sources = data.get("sources", {})
        self.pages = data.get("pages", {})
        self.nodes = data.get("nodes", [])
        if data.get("version") != 2:
            self._upgrade()
        self.positions = {node["id"]: index for index, node in enumerate(self.nodes)}
        self.doc_tokens = [
            clean_tokens(node.get("search_text") or node["text"], stem) for node in self.nodes
        ]
        self.postings = defaultdict(list)
        lengths = []
        for index, terms in enumerate(self.doc_tokens):
            lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((index, frequency))
        total = len(self.doc_tokens) or 1
        self.average = (sum(lengths) / total) or 1.0
        self.norm = [1.2 * (0.25 + 0.75 * length / self.average) for length in lengths]
        self.idf = {
            term: math.log1p((total - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self._lock = threading.Lock()
        self.embedder = None
        self.reranker = None
        self.vectors = None
        self.vector_norm = None
        if use_dense:
            from fastembed import TextEmbedding

            self.embedder = TextEmbedding(embed_model)
            self._load_vectors()
        if use_rerank:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self.reranker = TextCrossEncoder(rerank_model)

    def _upgrade(self):
        for node in self.nodes:
            ref_id = node.get("ref_id", "")
            page = node.get("page", 0)
            section = node.get("section") or "Body"
            title = self.sources.get(ref_id, {}).get("title", ref_id)
            text = clean_text(node.get("text", ""))
            prefix = f"[{ref_id} | {title} | p.{page} | {section}]"
            node.update({"section": section, "text": text, "context_prefix": prefix,
                         "search_text": f"{prefix}\n{text}"})

    def _load_vectors(self):
        import os
        import tempfile

        import numpy as np

        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", self.embed_model).strip("._") or "model"
        cache = self.path.with_name(
            f"{self.path.stem}.ctx-{self.digest[:12]}.{slug}.npy"
        )
        try:
            if cache.exists():
                vectors = np.load(cache, allow_pickle=False)
                if len(vectors) == len(self.nodes):
                    self.vectors = vectors
                    self.vector_norm = np.linalg.norm(vectors, axis=1)
                    return
            texts = [(node.get("search_text") or node["text"])[:1800] for node in self.nodes]
            vectors = np.array(
                list(self.embedder.passage_embed(texts, batch_size=32)), dtype=np.float32
            )
            handle, temp_name = tempfile.mkstemp(
                dir=str(cache.parent), prefix=f".{cache.name}.", suffix=".npy"
            )
            os.close(handle)
            try:
                np.save(temp_name, vectors)
                os.replace(temp_name, cache)
            finally:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            self.vectors = vectors
            self.vector_norm = np.linalg.norm(vectors, axis=1)
        except Exception:  # pragma: no cover - embedding backend failures
            self.vectors = None

    def _node(self, item):
        if isinstance(item, int):
            return self.nodes[item] if 0 <= item < len(self.nodes) else None
        position = self.positions.get(item)
        return self.nodes[position] if position is not None else None

    def bm25(self, query, limit=10):
        scores = defaultdict(float)
        for term in clean_tokens(query, self.stem):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for index, frequency in postings:
                scores[index] += idf * (frequency * 2.2) / (frequency + self.norm[index])
        return [index for index, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]

    def dense(self, query, limit=10):
        import numpy as np

        if self.embedder is None or self.vectors is None:
            return []
        with self._lock:
            query_vector = np.array(list(self.embedder.query_embed([query]))[0], dtype=np.float32)
        query_vector /= (np.linalg.norm(query_vector) + 1e-9)
        scores = self.vectors @ query_vector / (self.vector_norm + 1e-9)
        return np.argsort(-scores)[:limit].tolist()

    def lexical_alignment(self, query, text):
        terms = list(dict.fromkeys(t for t in clean_tokens(query, self.stem) if len(t) > 1))
        if not terms:
            return 0.0
        document = set(clean_tokens(text, self.stem))
        total = hit = 0.0
        for term in terms:
            weight = self.idf.get(term, 1.0)
            total += weight
            if term in document:
                hit += weight
        for left, right in zip(terms, terms[1:]):
            weight = (self.idf.get(left, 1.0) + self.idf.get(right, 1.0)) / 2
            total += weight
            if left in document and right in document:
                hit += weight
        return round(hit / total, 4) if total else 0.0

    def rerank_indices(self, query, indices):
        if self.reranker is None or not indices:
            return list(indices)
        documents = [self.nodes[index].get("search_text") or self.nodes[index]["text"]
                     for index in indices]
        with self._lock:
            scores = list(self.reranker.rerank(query, documents, batch_size=64))
        order = sorted(range(len(indices)), key=lambda k: float(scores[k]), reverse=True)
        return [indices[k] for k in order]

    def _node_dict(self, node):
        record = dict(node)
        ref_id = node.get("ref_id")
        record["title"] = self.sources.get(ref_id, {}).get("title", ref_id)
        return record

    def _result_dict(self, node, *, score, rerank_score, lexical_alignment, hits):
        record = self._node_dict(node)
        record["score"] = float(score)
        record["rerank_score"] = rerank_score
        record["lexical_alignment"] = float(lexical_alignment)
        record["hits"] = int(hits)
        return record

    def search_mode(self, queries, top_k, *, mode="hybrid_rerank", candidate_pool=36,
                    rerank_query=None):
        queries = [query for query in (queries or []) if query and query.strip()]
        if not queries:
            return []
        limit = max(candidate_pool, top_k)
        fused = defaultdict(float)
        hits = defaultdict(int)
        for query in queries:
            if mode in ("bm25", "hybrid", "hybrid_rerank"):
                for rank, index in enumerate(self.bm25(query, limit)):
                    fused[index] += 1.0 / (60.0 + rank + 1)
                    hits[index] += 1
            if mode in ("dense", "hybrid", "hybrid_rerank"):
                for rank, index in enumerate(self.dense(query, limit)):
                    fused[index] += 1.0 / (60.0 + rank + 1)
                    hits[index] += 1
        seen = set()
        ranked = []
        for index, _ in sorted(fused.items(), key=lambda pair: (-pair[1], pair[0])):
            node = self.nodes[index]
            key = (node.get("page_id"), compact(node.get("text", "")).casefold())
            if key in seen:
                continue
            seen.add(key)
            ranked.append(index)
        max_rrf = max(fused.values()) if fused else 1.0
        ranked = ranked[: max(candidate_pool * 2, top_k)]
        lexical_query = rerank_query or queries[0]
        results = []
        if mode == "hybrid_rerank" and self.reranker is not None:
            pool = ranked[:candidate_pool]
            rerank_scores = {}
            if pool:
                documents = [
                    self.nodes[index].get("search_text") or self.nodes[index]["text"]
                    for index in pool
                ]
                with self._lock:
                    raw_scores = list(
                        self.reranker.rerank(lexical_query, documents, batch_size=64)
                    )
                for index, raw in zip(pool, raw_scores):
                    clipped = max(-20.0, min(20.0, float(raw)))
                    rerank_scores[index] = 1.0 / (1.0 + math.exp(-clipped))
            for index in ranked:
                node = self.nodes[index]
                norm_rrf = fused[index] / max_rrf if max_rrf else 0.0
                lex_align = self.lexical_alignment(
                    lexical_query, node.get("search_text") or node["text"]
                )
                rerank_sig = rerank_scores.get(index, 0.0)
                final = rerank_sig + 0.10 * norm_rrf + 0.08 * lex_align
                results.append(
                    self._result_dict(
                        node, score=final, rerank_score=rerank_sig,
                        lexical_alignment=lex_align, hits=hits[index],
                    )
                )
            results.sort(key=lambda record: (-record["score"], record["id"]))
            return results[:top_k]
        for index in ranked:
            node = self.nodes[index]
            norm_rrf = fused[index] / max_rrf if max_rrf else 0.0
            lex_align = self.lexical_alignment(
                lexical_query, node.get("search_text") or node["text"]
            )
            results.append(
                self._result_dict(
                    node, score=norm_rrf, rerank_score=None,
                    lexical_alignment=lex_align, hits=hits[index],
                )
            )
        return results[:top_k]

    def search_multihop(self, original_query, global_queries, fact_slots, top_k, *,
                        candidate_pool=36, mode="hybrid_rerank"):
        global_hits = self.search_mode(
            [original_query] + list(global_queries or []),
            top_k=top_k,
            mode=mode,
            candidate_pool=candidate_pool,
            rerank_query=original_query,
        )
        slot_results = []
        for slot in fact_slots or []:
            if isinstance(slot, dict):
                name = str(slot.get("slot") or slot.get("name") or "")
                raw_queries = slot.get("queries")
                if raw_queries is None:
                    raw_queries = slot.get("query")
                if isinstance(raw_queries, str):
                    raw_queries = [raw_queries]
                slot_queries = [
                    str(query) for query in (raw_queries or []) if str(query).strip()
                ]
            else:
                name = str(slot)
                slot_queries = [str(slot)] if str(slot).strip() else []
            if not slot_queries:
                continue
            hits = self.search_mode(
                slot_queries,
                top_k=max(6, top_k // 2),
                mode=mode,
                candidate_pool=candidate_pool,
                rerank_query=slot_queries[0],
            )
            slot_results.append((name or f"slot{len(slot_results) + 1}", hits))

        slot_ids = {name: {hit["id"] for hit in hits} for name, hits in slot_results}

        def matched_slots(node_id):
            return [name for name, ids in slot_ids.items() if node_id in ids]

        merged = {}
        ordered = []
        seen = set()

        def add(record):
            node_id = record.get("id")
            if node_id in seen:
                return None
            seen.add(node_id)
            enriched = dict(record)
            enriched["matched_slots"] = matched_slots(node_id)
            merged[node_id] = enriched
            ordered.append(enriched)
            return enriched

        if len(slot_results) >= 2:
            per_slot = max(3, top_k // (len(slot_results) + 1))
            taken = {name: 0 for name, _ in slot_results}
            while len(ordered) < top_k:
                progressed = False
                for name, hits in slot_results:
                    if len(ordered) >= top_k:
                        break
                    if taken[name] >= per_slot:
                        continue
                    while taken[name] < len(hits):
                        candidate = hits[taken[name]]
                        taken[name] += 1
                        if add(candidate) is not None:
                            progressed = True
                            break
                if not progressed:
                    break

        remaining = []
        for name, hits in slot_results:
            for hit in hits:
                if hit.get("id") not in seen:
                    remaining.append(hit)
        for hit in global_hits:
            if len(ordered) >= top_k:
                break
            add(hit)
        position = 0
        while len(ordered) < top_k and position < len(remaining):
            add(remaining[position])
            position += 1
        return ordered[:top_k]

    def expand(self, selected, max_chars=2000):
        expanded = []
        for item in selected or []:
            record = dict(item) if isinstance(item, dict) else {}
            node = self._node(item.get("id") if isinstance(item, dict) else item)
            if node is None:
                continue
            text = node.get("text", "")
            kind = node.get("kind") or record.get("kind")
            if kind == "table":
                pass
            else:
                page = self.pages.get(node.get("page_id"), "")
                index = page.find(text) if text else -1
                if index != -1 and len(text) < max_chars:
                    extra = max_chars - len(text)
                    left = max(0, index - extra // 2)
                    right = min(
                        len(page),
                        index + len(text) + (extra - (index - left)),
                    )
                    text = page[left:right]
                elif index == -1 and len(text) > max_chars:
                    text = text[:max_chars]
            base = {
                key: node.get(key)
                for key in ("id", "page_id", "ref_id", "page", "kind", "section",
                            "context_prefix")
            }
            base["title"] = record.get("title") or self.sources.get(
                node.get("ref_id"), {}
            ).get("title", node.get("ref_id"))
            base["matched_text"] = record.get("matched_text", node.get("text", ""))
            base.update(record)
            base["text"] = text
            expanded.append(base)
        return expanded

    def adjacent_page_nodes(self, node_ids, exclude_ids=None, max_nodes=6):
        excluded = set(exclude_ids or []) | set(node_ids or [])
        result = []
        seen = set()

        def push(node):
            node_id = node.get("id")
            if node_id in excluded or node_id in seen:
                return
            seen.add(node_id)
            result.append(self._node_dict(node))

        for item in node_ids or []:
            node = self._node(item)
            if node is None:
                continue
            ref_id = node.get("ref_id")
            page = node.get("page", 0)
            for neighbour_page in (page - 1, page + 1):
                page_id = f"{ref_id}:p{neighbour_page}"
                candidates = [
                    candidate for candidate in self.nodes
                    if candidate.get("page_id") == page_id
                ]
                candidates.sort(
                    key=lambda candidate: (
                        candidate.get("kind") != "table",
                        self.positions.get(candidate["id"], 0),
                    )
                )
                for candidate in candidates:
                    if len(result) >= max_nodes:
                        return result
                    push(candidate)
            same_page = [
                candidate for candidate in self.nodes
                if candidate.get("page_id") == node.get("page_id")
            ]
            anchor = self.positions.get(node["id"], 0)
            same_page.sort(
                key=lambda candidate: (
                    candidate.get("kind") != "table",
                    abs(self.positions.get(candidate["id"], 0) - anchor),
                )
            )
            for candidate in same_page:
                if len(result) >= max_nodes:
                    return result
                if candidate["id"] == node["id"]:
                    continue
                push(candidate)
        return result[:max_nodes]


def benchmark_retrieval(index, questions, top_k=10, v1_index=None):
    """Report fractional document recall and MRR for each configuration."""
    modes = []
    if v1_index is not None:
        modes.append(("v1_bm25", "v1"))
    modes.append(("ctx_bm25", "bm25"))
    if getattr(index, "embedder", None) is not None and getattr(index, "vectors", None) is not None:
        modes.append(("ctx_dense", "dense"))
        modes.append(("ctx_hybrid", "hybrid"))
    if getattr(index, "reranker", None) is not None:
        modes.append(("ctx_hybrid_rerank", "hybrid_rerank"))
    depth = max(20, top_k)
    metrics = {
        name: {
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "mrr": 0.0,
            "count": 0,
            "missed_at_10": [],
        }
        for name, _ in modes
    }

    def v1_refs(query):
        refs = []
        seen = set()
        for hit in v1_index.search([query], depth):
            if isinstance(hit, dict):
                ref = hit.get("ref_id", "")
                key = hit.get("id") or hit.get("page_id") or (
                    ref, hit.get("page"), hit.get("text")
                )
            elif isinstance(hit, int):
                node = v1_index.nodes[hit] if 0 <= hit < len(v1_index.nodes) else {}
                ref = node.get("ref_id", "")
                key = node.get("id") or hit
            else:
                ref = str(hit)
                key = ref
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
        return refs

    for question in questions:
        targets = {
            ref for ref in parse_list(question.get("ref_id") or BLANK) if ref != BLANK
        }
        if not targets:
            continue
        query = question.get("question", "")
        question_id = question.get("question_id") or question.get("id") or ""
        for name, mode in modes:
            if mode == "v1":
                refs = v1_refs(query)
            else:
                hits = index.search_mode(
                    [query], depth, mode=mode, candidate_pool=max(depth, 36)
                )
                refs = [hit.get("ref_id") for hit in hits]
            row = metrics[name]
            row["count"] += 1
            for cutoff in (5, 10, 20):
                row[f"recall@{cutoff}"] += (
                    len(targets & set(refs[:cutoff])) / len(targets)
                )
            for rank, ref in enumerate(refs, 1):
                if ref in targets:
                    row["mrr"] += 1.0 / rank
                    break
            missed = sorted(targets - set(refs[:10]))
            if missed:
                row["missed_at_10"].append((question_id, missed))
    for row in metrics.values():
        count = row.pop("count")
        for key in ("recall@5", "recall@10", "recall@20", "mrr"):
            row[key] = round(row[key] / count, 4) if count else 0.0
        row["questions"] = count
    return metrics


def _question_key(question):
    return str(question.get("id") or question.get("question_id") or "")


def run_command(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.jsonl"
    predictions_path = output_dir / "predictions.csv"

    metadata = load_metadata(args.metadata)
    _, loaded = read_csv(args.questions)
    questions = list(loaded)
    if args.limit:
        questions = questions[: args.limit]

    index = ContextualSearchIndex(
        args.index,
        embed_model=args.embed_model,
        rerank_model=args.rerank_model,
        use_dense=not args.no_dense,
        use_rerank=not args.no_rerank,
        stem=not args.no_stem,
    )
    client = ResilientDeepSeek(args.backend)

    config = {
        "questions": str(args.questions),
        "metadata": str(args.metadata),
        "index": str(args.index),
        "output_dir": str(output_dir),
        "top_k": args.top_k,
        "workers": args.workers,
        "limit": args.limit,
        "backend": args.backend,
        "model": client.model,
        "embed_model": args.embed_model,
        "rerank_model": args.rerank_model,
        "max_rounds": args.max_rounds,
        "resume": args.resume,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "config.json", config)

    trace_lock = threading.Lock()

    def write_trace(event, **fields):
        payload = {
            "event": event,
            "time": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with trace_lock:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str) + "\n")

    completed = {}
    if args.resume and trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "row" and isinstance(event.get("row"), dict):
                key = str(event.get("question_id") or event["row"].get("id") or "")
                if key:
                    completed[key] = event["row"]
    elif trace_path.exists():
        trace_path.unlink()

    def emit_for(question):
        key = _question_key(question)

        def emit(event, **fields):
            write_trace(event, question_id=key, **fields)

        return emit

    def solve_one(question):
        key = _question_key(question)
        emit = emit_for(question)
        row = solve_multihop(
            question, index, client, emit,
            top_k=args.top_k, max_rounds=args.max_rounds,
        )
        validate_rows(COLUMNS, [row], [question], metadata)
        write_trace("row", question_id=key, row=row)
        return key, row

    pending = [
        question for question in questions if _question_key(question) not in completed
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(solve_one, question): question for question in pending}
        for future in as_completed(futures):
            key, row = future.result()
            completed[key] = row

    ordered = [completed[_question_key(question)] for question in questions]
    validate_rows(COLUMNS, ordered, questions, metadata)
    atomic_csv(predictions_path, ordered)

    write_trace(
        "run_complete",
        questions=len(questions),
        predictions=str(predictions_path),
    )

    usage_path = output_dir / "token-usage.json"
    write_json(usage_path, summarize_token_usage(trace_path))

    metrics = None
    if any(str(question.get("answer_value") or "").strip() for question in questions):
        metrics = evaluate(ordered, questions)
        write_json(output_dir / "metrics.json", metrics)

    summary = {
        "questions": len(questions),
        "predictions": str(predictions_path),
        "trace": str(trace_path),
        "token_usage": str(usage_path),
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Contextual multi-hop retrieval tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build the contextual node index.")
    index_parser.add_argument("--metadata", required=True)
    index_parser.add_argument("--docs", required=True)
    index_parser.add_argument("--output", required=True)

    bench_parser = subparsers.add_parser(
        "benchmark-retrieval", help="Benchmark retrieval configurations."
    )
    bench_parser.add_argument("--questions", required=True)
    bench_parser.add_argument("--metadata", required=True)
    bench_parser.add_argument("--index", required=True)
    bench_parser.add_argument("--v1-index")
    bench_parser.add_argument("--top-k", type=int, default=10)
    bench_parser.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    bench_parser.add_argument("--rerank-model", default="jinaai/jina-reranker-v1-turbo-en")
    bench_parser.add_argument("--no-dense", action="store_true")
    bench_parser.add_argument("--no-rerank", action="store_true")
    bench_parser.add_argument("--no-stem", action="store_true")
    bench_parser.add_argument("--output")

    run_parser = subparsers.add_parser("run", help="Solve multi-hop questions end to end.")
    run_parser.add_argument("--questions", required=True)
    run_parser.add_argument("--metadata", required=True)
    run_parser.add_argument("--index", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--top-k", type=int, default=12)
    run_parser.add_argument("--workers", type=int, default=2)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--backend", choices=("hkustgz", "openrouter"), default="hkustgz")
    run_parser.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    run_parser.add_argument("--rerank-model", default="jinaai/jina-reranker-v1-turbo-en")
    run_parser.add_argument("--no-dense", action="store_true")
    run_parser.add_argument("--no-rerank", action="store_true")
    run_parser.add_argument("--no-stem", action="store_true")
    run_parser.add_argument("--max-rounds", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "index":
        metadata = load_metadata(args.metadata)
        build_contextual_index(metadata, Path(args.docs), Path(args.output))
        return 0
    if args.command == "run":
        return run_command(args)

    _, questions = read_csv(args.questions)
    index = ContextualSearchIndex(
        args.index,
        embed_model=args.embed_model,
        rerank_model=args.rerank_model,
        use_dense=not args.no_dense,
        use_rerank=not args.no_rerank,
        stem=not args.no_stem,
    )
    v1_index = SearchIndex(Path(args.v1_index)) if args.v1_index else None
    metrics = benchmark_retrieval(index, questions, top_k=args.top_k, v1_index=v1_index)
    if args.output:
        write_json(Path(args.output), metrics)
    else:
        print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
