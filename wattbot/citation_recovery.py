"""Recover citations from page text and replay full-context traces offline.

The recovery code is independent of any model call: it only maps a quoted span
back into the supplied corpus pages.  A saved trace can therefore be replayed
without spending tokens or rebuilding the index.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from wattbot.__main__ import atomic_csv
from wattbot.pipeline import BLANK, read_csv
from wattbot.retrieval import write_json

TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+", re.UNICODE)
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)+")
MULTI_DIGIT = re.compile(r"\d{2,}")
ELLIPSIS = re.compile(r"\s*(?:\.{3,}|\u2026)\s*")
INTEGER = re.compile(r"[+-]?\d+")

_CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u2015": "-", "\u2212": "-", "\u00d7": "x",
})


def _normalized_with_map(text):
    """Return NFKC-like text plus a normalized-index -> original-index map."""
    normalized = []
    index_map = []
    for position, char in enumerate(text):
        if char == "\u00ad":
            continue
        char = char.translate(_CHAR_MAP)
        for item in unicodedata.normalize("NFKC", char):
            if item == "\u00ad":
                continue
            normalized.append(item)
            index_map.append(position)
    return "".join(normalized), index_map


def normalize_text(text):
    return _normalized_with_map(text)[0]


def _compact(text):
    return " ".join(normalize_text(text).split())


def tokenize_with_spans(text):
    """Yield (normalized lowercase token, original start, original end)."""
    normalized, index_map = _normalized_with_map(text)
    result = []
    for match in TOKEN.finditer(normalized):
        start, end = match.span()
        result.append((match.group().lower(), index_map[start], index_map[end - 1] + 1))
    return result


def _is_word(token):
    return bool(WORD.fullmatch(token))


def _is_number_token(token):
    return bool(NUMBER_TOKEN.fullmatch(token) or MULTI_DIGIT.fullmatch(token))


def _numbers(tokens):
    return {token for token in tokens if _is_number_token(token)}


def _is_hyphen_gap(raw, start, end):
    gap = raw[start:end]
    normalized = normalize_text(gap)
    if re.fullmatch(r"-\s*", normalized):
        return True
    return bool(gap) and "\u00ad" in gap and all(
        char == "\u00ad" or char.isspace() for char in gap)


def _match_sequence(expected, actual, raw, start):
    """Match all expected tokens from an actual start; return exclusive end."""
    memo = {}

    def visit(i, j):
        key = (i, j)
        if key in memo:
            return memo[key]
        if i == len(expected):
            return j
        if j >= len(actual):
            return None
        result = None
        expected_token = expected[i][0]
        if expected_token == actual[j][0]:
            result = visit(i + 1, j + 1)
        if result is None and j + 1 < len(actual):
            first, second = actual[j], actual[j + 1]
            if (_is_word(expected_token) and _is_word(first[0]) and _is_word(second[0])
                    and expected_token == first[0] + second[0]
                    and _is_hyphen_gap(raw, first[2], second[1])):
                result = visit(i + 1, j + 2)
        if result is None and i + 1 < len(expected):
            first, second = expected[i], expected[i + 1]
            if (_is_word(first[0]) and _is_word(second[0])
                    and first[0] + second[0] == actual[j][0]
                    and _is_hyphen_gap(raw, first[2], second[1])):
                result = visit(i + 2, j + 1)
        memo[key] = result
        return result

    return visit(0, start)


def _find_token_span(quote, page_text):
    expected = tokenize_with_spans(quote)
    actual = tokenize_with_spans(page_text)
    if not expected or not actual:
        return None
    for start in range(len(actual)):
        end = _match_sequence(expected, actual, page_text, start)
        if end is not None and end > start:
            return actual[start][1], actual[end - 1][2]
    return None


def _match_exact_or_span(quote, page_text):
    compact_quote = _compact(quote)
    if not compact_quote or not page_text:
        return None, None
    if compact_quote in _compact(page_text):
        stripped = quote.strip()
        if stripped and stripped in page_text:
            return stripped, "exact"
        span = _find_token_span(quote, page_text)
        if span is not None:
            return page_text[span[0]:span[1]], "exact"
        return stripped, "exact"
    span = _find_token_span(quote, page_text)
    if span is not None:
        return page_text[span[0]:span[1]], "token_span"
    return None, None


def _match_ellipsis(quote, page_text):
    segments = [segment.strip() for segment in ELLIPSIS.split(quote)]
    segments = [segment for segment in segments if segment]
    if len(segments) < 2:
        return None
    quote_tokens = [token for token, _, _ in tokenize_with_spans(quote)]
    page_tokens = {token for token, _, _ in tokenize_with_spans(page_text)}
    if not _numbers(quote_tokens) <= page_tokens:
        return None
    matched = []
    matched_tokens = 0
    for segment in segments:
        text, _ = _match_exact_or_span(segment, page_text)
        if text is not None:
            matched.append(text)
            matched_tokens += len(tokenize_with_spans(segment))
    needed = len(segments) if len(segments) == 2 else 2
    if (len(matched) < needed or len(matched) < 2
            or len(matched) / len(segments) < 0.6
            or matched_tokens < 4):
        return None
    return " ... ".join(matched)


def _lcs_length(left, right):
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right, 1):
            if item == other:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def _token_lcs_match(quote, page_text, threshold):
    quote_items = tokenize_with_spans(quote)
    if len(quote_items) < 6:
        return None
    quote_tokens = [token for token, _, _ in quote_items]
    actual_items = tokenize_with_spans(page_text)
    if not actual_items:
        return None
    actual_tokens = [token for token, _, _ in actual_items]
    counts = Counter(actual_tokens)
    anchors = sorted({
        token for token in quote_tokens
        if token in counts and (len(token) >= 4 or _is_number_token(token))
    }, key=lambda token: (counts[token], -len(token), token))
    if not anchors:
        anchors = sorted(
            {token for token in quote_tokens if token in counts},
            key=lambda token: (counts[token], -len(token), token),
        )
    if not anchors:
        return None
    anchors = anchors[:6]
    maximum = int(len(quote_tokens) * 1.4) + 4
    offsets = (0, maximum // 4, maximum // 2, (3 * maximum) // 4, maximum - 1)
    starts = set()
    for anchor in anchors:
        for position, token in enumerate(actual_tokens):
            if token != anchor:
                continue
            for offset in offsets:
                starts.add(max(0, position - offset))
    required_numbers = _numbers(quote_tokens)
    best = None
    for start in sorted(starts):
        end = min(len(actual_tokens), start + maximum)
        if start >= end:
            continue
        window = actual_tokens[start:end]
        if not required_numbers <= set(window):
            continue
        score = _lcs_length(quote_tokens, window)
        if best is None or score > best[0]:
            best = (score, start, end)
    if best is None:
        return None
    score, start, end = best
    ratio = score / len(quote_tokens)
    if ratio < threshold:
        return None
    if not required_numbers <= set(actual_tokens[start:end]):
        return None
    return page_text[actual_items[start][1]:actual_items[end - 1][2]], ratio


def match_quote_in_text(quote, page_text):
    """Return (verbatim quote, method) or (None, None) using four tiers."""
    if not isinstance(quote, str) or not quote.strip() or not isinstance(page_text, str):
        return None, None
    text, method = _match_exact_or_span(quote, page_text)
    if text is not None:
        return text, method
    text = _match_ellipsis(quote, page_text)
    if text is not None:
        return text, "ellipsis_segments"
    token_count = len(tokenize_with_spans(quote))
    if token_count >= 6:
        threshold = 0.80 if token_count >= 10 else 0.85
        match = _token_lcs_match(quote, page_text, threshold)
        if match is not None:
            return match[0], "token_lcs"
    return None, None


def _scope_match(ref_id, page, quote, pages, scope):
    text = pages.get((ref_id, page))
    if text is None:
        return None
    matched, method = match_quote_in_text(quote, text)
    if matched is None:
        return None
    return {"ref_id": ref_id, "page": page, "quote": matched,
            "method": method, "scope": scope}


def recover_citation(ref_id, page, quote, pages, sources):
    result = _scope_match(ref_id, page, quote, pages, "same_page")
    if result is not None:
        return result
    for adjacent in (page - 1, page + 1):
        result = _scope_match(ref_id, adjacent, quote, pages, "adjacent_page")
        if result is not None:
            return result
    remaining = sorted(
        (candidate for candidate_ref, candidate in pages if candidate_ref == ref_id),
        key=lambda candidate: (abs(candidate - page), candidate),
    )
    for candidate in remaining:
        if candidate in (page - 1, page, page + 1):
            continue
        result = _scope_match(ref_id, candidate, quote, pages, "same_doc")
        if result is not None:
            return result
    if len(tokenize_with_spans(quote)) < 6:
        return None
    candidates = []
    for (candidate_ref, candidate_page), text in pages.items():
        matched, method = _match_exact_or_span(quote, text)
        if matched is None:
            strict = _token_lcs_match(quote, text, 0.88)
            if strict is None:
                continue
            matched, method = strict[0], "token_lcs"
        candidates.append({"ref_id": candidate_ref, "page": candidate_page,
                           "quote": matched, "method": method})
    if not candidates:
        return None
    matched_refs = {candidate["ref_id"] for candidate in candidates}
    if len(matched_refs) != 1:
        return None
    rank = {"exact": 3, "token_span": 2, "token_lcs": 1}
    best = max(candidates, key=lambda candidate: (
        rank.get(candidate["method"], 0), -abs(candidate["page"] - page),
        -candidate["page"]))
    return {**best, "scope": "corpus_unique"}


def _evidence_page(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and INTEGER.fullmatch(value.strip()):
        return int(value.strip())
    return None


def recover_evidence(reply, pages, sources, emit=None):
    refs = {}
    materials = []
    evidence = reply.get("evidence") if isinstance(reply, dict) else None
    for item in evidence if isinstance(evidence, list) else []:
        if not isinstance(item, dict):
            continue
        ref_id = item.get("ref_id")
        quote = item.get("quote")
        page = _evidence_page(item.get("page"))
        if not isinstance(ref_id, str) or not isinstance(quote, str) or not quote.strip():
            continue
        if page is None:
            continue
        recovered = recover_citation(ref_id, page, quote, pages, sources)
        if recovered is not None:
            resolved_ref = recovered["ref_id"]
            resolved_page = recovered["page"]
            matched = recovered["quote"]
            method = recovered["method"]
            scope = recovered["scope"]
            refs[resolved_ref] = sources[resolved_ref]["url"]
            material = f"[{resolved_ref}, p. {resolved_page}] {matched.strip()}"
            if material not in materials:
                materials.append(material)
            if emit is not None and not (scope == "same_page" and method == "exact"):
                emit("citation_recovered", ref_id=ref_id, page=page,
                     resolved_ref_id=resolved_ref, resolved_page=resolved_page,
                     method=method, scope=scope)
        elif emit is not None:
            emit("citation_dropped", ref_id=ref_id, page=page,
                 reason="Quote absent from supplied corpus")
    return {"ref_id": repr(list(refs)) if refs else BLANK,
            "ref_url": repr(list(refs.values())) if refs else BLANK,
            "supporting_materials": "\n".join(materials) if materials else BLANK}


def _load_trace(trace_path):
    rows = {}
    replies = {}
    with Path(trace_path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            question_id = event.get("question_id")
            if not isinstance(question_id, str):
                continue
            if event.get("event") == "row" and isinstance(event.get("row"), dict):
                rows[question_id] = event["row"]
            elif event.get("event") == "llm_response":
                content = event.get("content")
                reply = content
                if isinstance(content, str):
                    try:
                        reply = json.loads(content)
                    except ValueError:
                        reply = None
                if isinstance(reply, dict):
                    replies[question_id] = reply
    return rows, replies


def replay_full_context_trace(trace_path, index_path, metadata_path, questions_path,
                              output_dir):
    from wattbot.baselines.full_context import FullContext

    output_dir = Path(output_dir)
    context = FullContext(index_path, metadata_path, recover_citations=True)
    fieldnames, questions = read_csv(questions_path)
    trace_rows, replies = _load_trace(trace_path)
    active = [question for question in questions if question["id"] in trace_rows]
    by_id = {question["id"]: question for question in questions}
    gold_available = "answer_value" in (fieldnames or [])

    before_rows = []
    after_rows = []
    per_question = []
    recovered_count = 0
    dropped_count = 0
    for question in active:
        question_id = question["id"]
        original = dict(trace_rows[question_id])
        row = dict(original)
        events = []

        def collector(event, **fields):
            events.append({"event": event, **fields})

        reply = replies.get(question_id)
        if row.get("answer_value") != BLANK and reply is not None:
            citation = recover_evidence(reply, context.pages, context.sources,
                                        emit=collector)
            row["ref_id"] = citation["ref_id"]
            row["ref_url"] = citation["ref_url"]
            row["supporting_materials"] = citation["supporting_materials"]
        for event in events:
            if event["event"] == "citation_recovered":
                recovered_count += 1
            elif event["event"] == "citation_dropped":
                dropped_count += 1
        before_rows.append(dict(original))
        after_rows.append(row)
        citation_fields = ("ref_id", "ref_url", "supporting_materials")
        per_question.append({
            "id": question_id,
            "before": {key: original.get(key) for key in citation_fields},
            "after": {key: row.get(key) for key in citation_fields},
            "events": events,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "predictions.csv", after_rows)
    report = {
        "questions": len(active),
        "recovered_count": recovered_count,
        "dropped_count": dropped_count,
        "per_question": per_question,
    }
    summary = {
        "questions": len(active),
        "recovered_count": recovered_count,
        "dropped_count": dropped_count,
        "output_dir": str(output_dir),
    }
    if gold_available:
        from wattbot.evaluation import evaluate

        gold = [by_id[question["id"]] for question in active]
        before_metrics = evaluate(before_rows, gold)
        after_metrics = evaluate(after_rows, gold)
        report["before"] = {key: value for key, value in before_metrics.items()
                            if key != "per_question"}
        report["after"] = {key: value for key, value in after_metrics.items()
                           if key != "per_question"}
        write_json(output_dir / "metrics.json", after_metrics)
        summary["after_weighted"] = after_metrics["weighted"]
    write_json(output_dir / "recovery-report.json", report)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = replay_full_context_trace(
        args.trace, args.index, args.metadata, args.questions, args.output_dir
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

