#!/usr/bin/env python3
"""WattBot 2025 interactive demo.

A zero-extra-dependency web application with two modes:

1. **Offline trace explorer** -- reads the saved cascade diagnostics, per-stage
   JSONL traces and static PDF corpus, and renders a rich three-stage pipeline
   inspector.  This is the zero-risk presentation mode.
2. **Live single-question Q&A** -- calls the existing ``wattbot`` Python APIs on
   the server side with an on-demand solver choice.

Only the Python standard library is required for the server itself.  ``pymupdf``
(already a project dependency) is used lazily to rasterise cited PDF pages.

Usage::

    python demo/app.py --host 127.0.0.1 --port 8501
    python demo/app.py --port 0          # pick an ephemeral port
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]

# Presentation constants from the submitted cascade.
FULL_CONTEXT_TOKENS = 689804
PUBLIC_SCORE = 0.895
PRIVATE_SCORE = 0.872
KOHAKU_PRIVATE = 0.861
MEAN_TOKENS = 79710.62

BLANK = "is_blank"
SAFE_REF_ID = re.compile(r"^[A-Za-z0-9_-]+$")
UNIT_PATTERNS = (
    re.compile(r"\bin\s+([A-Za-z][A-Za-z0-9 /%.\-]{0,28}?)(?:\?|$|,)", re.IGNORECASE),
    re.compile(r"\(in\s+([^)]{1,28})\)", re.IGNORECASE),
    re.compile(r"\bhow many\s+([A-Za-z][A-Za-z0-9 /%.\-]{0,24}?)(?:\?|$|,)", re.IGNORECASE),
)


# ---------------------------------------------------------------------------
# Small, forgiving readers


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_jsonl(path):
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def read_csv(path, encoding="utf-8-sig"):
    try:
        with Path(path).open(encoding=encoding, newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def parse_list(value):
    """Turn ``"['a', 'b']"`` / ``'a'`` / ``['a']`` into a clean string list."""
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if not is_blank(item)]
    if is_blank(value):
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if not is_blank(item)]
    if isinstance(parsed, str):
        return [] if is_blank(parsed) else [parsed.strip()]
    return [text]


def is_blank(value):
    return value is None or str(value).strip().casefold() in {"", "none", BLANK}


def number(value, default=0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_unit(question, explicit=None):
    if not is_blank(explicit):
        return str(explicit).strip()
    text = str(question or "")
    for pattern in UNIT_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate and candidate.casefold() not in {
                "how many", "what", "which", "the", "a", "an",
            }:
                return candidate
    return BLANK


def usage_total(events):
    total = 0
    for event in events:
        if event.get("event") != "llm_response":
            continue
        usage = event.get("usage") or {}
        total += int(number(usage.get("total_tokens"), 0))
    return total


def reasoning_tokens(events):
    total = 0
    for event in events:
        usage = (event.get("usage") or {}) if event.get("event") == "llm_response" else {}
        details = usage.get("completion_tokens_details") or {}
        total += int(number(details.get("reasoning_tokens"), 0))
    return total


def page_mentions(text):
    """Extract ``[ref_id, p. 7]`` style references from a supporting quote."""
    found = []
    for ref_id, page in re.findall(r"\[\s*([A-Za-z0-9_-]+)\s*,\s*p{1,2}\.?\s*(\d+)\s*\]", str(text or "")):
        found.append((ref_id, int(page)))
    return found


def trim(value, limit=420):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


# ---------------------------------------------------------------------------
# Demo data model


class DemoData:
    """Lazy, forgiving view over ``outputs/`` and ``data/``."""

    def __init__(self, root=None):
        self.root = Path(root or ROOT).resolve()
        self._loaded = False
        self._lock = threading.Lock()
        self._live_lock = threading.Lock()
        self._image_cache = {}
        self._clients = {}
        self._indexes = {}

    # -- discovery ------------------------------------------------------
    def _paths(self):
        outputs = self.root / "outputs"
        return {
            "cascade_diag": outputs / "multi-paradigm-ensemble" / "rag-cascade-pion-mm-doc.diag.json",
            "cascade_csv": outputs / "multi-paradigm-ensemble" / "rag-cascade-pion-mm-doc.csv",
            "pi_test": outputs / "pi-v2-sandbox" / "pi-v2-thinking-on" / "test" / "trace.jsonl",
            "pi_train": outputs / "pi-v2-sandbox" / "pi-v2-thinking-on" / "train" / "trace.jsonl",
            "mm_test": outputs / "multimodal" / "test-mode2" / "trace.jsonl",
            "mm_train": outputs / "multimodal" / "train" / "trace.jsonl",
            "doc_test": outputs / "doc-routing" / "test" / "trace.jsonl",
            "doc_train": outputs / "doc-routing" / "train" / "trace.jsonl",
            "test_q": self.root / "data" / "raw" / "test_Q.csv",
            "train_qa": self.root / "data" / "raw" / "train_QA.csv",
            "metadata": self.root / "data" / "raw" / "metadata.csv",
            "docs": self.root / "data" / "docs",
        }

    def _trace_events(self, paths, keys):
        merged = {}
        for key in keys:
            current = {}
            for event in read_jsonl(paths[key]):
                qid = event.get("question_id")
                if qid:
                    current.setdefault(str(qid), []).append(event)
            # Prefer the first split (test) and only fill gaps from later ones.
            for qid, group in current.items():
                merged.setdefault(qid, group)
        return merged

    # -- loading --------------------------------------------------------
    def ensure(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True

    def _load(self):
        paths = self._paths()
        self.paths = paths

        # Questions: test first, then any extra rows from train / predictions.
        self.question_rows = {}
        for path, encoding in ((paths["test_q"], "utf-8-sig"), (paths["train_qa"], "utf-8-sig")):
            for index, row in enumerate(read_csv(path, encoding)):
                qid = str(row.get("id") or "").strip()
                if qid and qid not in self.question_rows:
                    self.question_rows[qid] = row
                    self.question_rows[qid]["_order"] = len(self.question_rows)
        self.metadata = {
            str(row.get("id")): row for row in read_csv(paths["metadata"], "latin-1")
            if row.get("id")
        }

        self.final_rows = {
            str(row.get("id")): row for row in read_csv(paths["cascade_csv"])
            if row.get("id")
        }
        diag = read_json(paths["cascade_diag"]) or {}
        self.cascade_summary = diag.get("summary") or {}
        self.cascade = {
            str(item.get("id")): item for item in diag.get("per_question") or []
            if item.get("id")
        }

        self.pi_events = self._trace_events(paths, ("pi_test", "pi_train"))
        self.mm_events = self._trace_events(paths, ("mm_test", "mm_train"))
        self.doc_events = self._trace_events(paths, ("doc_test", "doc_train"))

        # Pick up questions that only appear in a stage trace.
        for events in (self.pi_events, self.mm_events, self.doc_events):
            for qid, group in events.items():
                if qid in self.question_rows:
                    continue
                text = ""
                for event in group:
                    row = event.get("row") or {}
                    candidate = (event.get("candidate") or {})
                    text = (row.get("question") or candidate.get("question")
                            or event.get("question") or text)
                    if text:
                        break
                self.question_rows[qid] = {
                    "id": qid, "question": text, "answer_unit": BLANK,
                    "_order": len(self.question_rows),
                }

        self._details = {}
        self._summaries = None

    # -- public API -----------------------------------------------------
    def overview(self):
        self.ensure()
        summaries = self.summaries()
        counts = Counter()
        for item in summaries:
            flags = item["flags"]
            if flags["escalated"]:
                counts["escalated"] += 1
            if flags["disagreement"]:
                counts["disagreement"] += 1
            if flags["abstain_trigger"]:
                counts["abstain_trigger"] += 1
            if flags["visual"]:
                counts["visual"] += 1
            if flags["ast"]:
                counts["ast"] += 1
            if flags["abstained"]:
                counts["abstained"] += 1
            if flags["fast_path"]:
                counts["fast_path"] += 1
        summary = dict(self.cascade_summary)
        summary.setdefault("questions", len(summaries))
        summary.setdefault("escalation_rate", counts["escalated"] / max(1, len(summaries)))
        summary.setdefault("mean_tokens_per_question", MEAN_TOKENS)
        return {
            "kpis": {
                "public": PUBLIC_SCORE,
                "private": PRIVATE_SCORE,
                "private_gain": PRIVATE_SCORE - KOHAKU_PRIVATE,
                "mean_tokens": number(summary.get("mean_tokens_per_question"), MEAN_TOKENS),
                "full_context_tokens": FULL_CONTEXT_TOKENS,
                "cheaper_factor": FULL_CONTEXT_TOKENS / max(1.0, number(summary.get("mean_tokens_per_question"), MEAN_TOKENS)),
                "escalation_rate": number(summary.get("escalation_rate"), 0.0),
            },
            "summary": summary,
            "counts": {
                "total": len(summaries),
                "escalated": counts["escalated"],
                "disagreement": counts["disagreement"],
                "abstain_trigger": counts["abstain_trigger"],
                "visual": counts["visual"],
                "ast": counts["ast"],
                "abstained": counts["abstained"],
                "fast_path": counts["fast_path"],
            },
            "picks": self.quick_picks(summaries),
            "datasets": {
                "cascade": bool(self.final_rows),
                "pi": bool(self.pi_events),
                "multimodal": bool(self.mm_events),
                "doc_routing": bool(self.doc_events),
                "pdfs": (self.paths["docs"]).is_dir(),
                "questions": bool(self.question_rows),
            },
        }

    def summaries(self):
        self.ensure()
        if self._summaries is not None:
            return self._summaries
        items = []
        for qid, row in sorted(self.question_rows.items(), key=lambda kv: kv[1].get("_order", 10**9)):
            detail = self._detail(qid)
            items.append({
                "id": qid,
                "question": detail["question"],
                "unit_hint": detail["unit_hint"],
                "escalated": detail["cascade"].get("escalated", False),
                "reasons": detail["cascade"].get("reasons", []),
                "resolution": detail["cascade"].get("resolution", ""),
                "tokens": detail["tokens"]["cascade"],
                "flags": detail["flags"],
            })
        self._summaries = items
        return items

    def questions(self):
        return self.summaries()

    def question(self, qid):
        self.ensure()
        if qid not in self.question_rows:
            return None
        return self._detail(qid)

    def quick_picks(self, summaries=None):
        summaries = summaries if summaries is not None else self.summaries()
        by_id = {item["id"]: item for item in summaries}
        picks = []
        seen = set()

        def add(qid, label):
            if qid in by_id and qid not in seen:
                seen.add(qid)
                picks.append({"id": qid, "label": label})

        for qid, label in (
            ("q015", "Chart recovery"),
            ("q189", "Figure reading"),
            ("q317", "Chart-only"),
        ):
            add(qid, label)
        for item in summaries:
            if item["flags"]["ast"]:
                add(item["id"], "Decimal AST")
                break
        for item in summaries:
            if item["flags"]["disagreement"]:
                add(item["id"], "Disagreement")
                break
        for item in summaries:
            if item["flags"]["escalated"]:
                add(item["id"], "Escalated")
                break
        for item in summaries:
            if item["flags"]["abstained"]:
                add(item["id"], "Abstained")
                break
        return picks

    # -- per-question assembly -----------------------------------------
    def _detail(self, qid):
        if qid in self._details:
            return self._details[qid]
        row = self.question_rows.get(qid) or {}
        final_row = self.final_rows.get(qid) or {}
        cascade = self.cascade.get(qid) or {}
        pi = self._build_pi(qid)
        mm = self._build_mm(qid)
        doc = self._build_doc(qid)
        final = self._final_block(qid, row, final_row)
        flags = {
            "escalated": bool(cascade.get("escalated")),
            "disagreement": "disagreement" in (cascade.get("reasons") or []),
            "abstain_trigger": "abstain" in (cascade.get("reasons") or []),
            "visual": bool(mm.get("vision_triggered") or mm.get("visual_candidates")
                           or mm.get("visual_candidate")),
            "ast": bool(pi.get("calculations")),
            "abstained": bool(final.get("abstained")),
        }
        if not cascade and final.get("available"):
            flags["fast_path"] = not flags["escalated"]
        else:
            flags["fast_path"] = bool(final.get("available")) and not flags["escalated"]
        cascade_tokens = int(number(cascade.get("tokens"), pi["tokens"] + mm["tokens"] + doc["tokens"]))
        detail = {
            "id": qid,
            "question": str(row.get("question") or final_row.get("question") or ""),
            "unit_hint": infer_unit(row.get("question"), row.get("answer_unit")),
            "final": final,
            "cascade": {
                "available": bool(cascade),
                "escalated": bool(cascade.get("escalated")),
                "reasons": list(cascade.get("reasons") or []),
                "resolution": str(cascade.get("resolution") or ""),
                "tokens": cascade_tokens,
            },
            "tier1a": pi,
            "tier1b": mm,
            "tier2": doc,
            "flags": flags,
            "tokens": {
                "tier1a": pi["tokens"],
                "tier1b": mm["tokens"],
                "tier2": doc["tokens"],
                "cascade": cascade_tokens,
                "full_context": FULL_CONTEXT_TOKENS,
                "mean": MEAN_TOKENS,
            },
        }
        self._details[qid] = detail
        return detail

    def _final_block(self, qid, row, final_row):
        source = final_row or {}
        ref_ids = parse_list(source.get("ref_id"))
        ref_urls = parse_list(source.get("ref_url"))
        if not ref_urls:
            ref_urls = [
                str(self.metadata.get(ref_id, {}).get("url") or "") for ref_id in ref_ids
            ]
            ref_urls = [url for url in ref_urls if url]
        supported = str(source.get("supporting_materials") or "")
        pages = page_mentions(supported)
        if not pages:
            for event in self.mm_events.get(qid, []):
                candidate = event.get("visual_candidate") or event.get("candidate") or {}
                for citation in candidate.get("citations") or []:
                    ref_id = citation.get("ref_id")
                    page = citation.get("page")
                    if ref_id and page:
                        pages.append((ref_id, int(page)))
        dedup = []
        for ref_id, page in pages:
            if (ref_id, page) not in dedup:
                dedup.append((ref_id, page))
        return {
            "available": bool(source),
            "answer": str(source.get("answer") or ""),
            "answer_value": str(source.get("answer_value") or ""),
            "answer_unit": str(source.get("answer_unit") or ""),
            "ref_id": ref_ids,
            "ref_url": ref_urls,
            "supporting_materials": supported,
            "explanation": str(source.get("explanation") or ""),
            "abstained": bool(source) and is_blank(source.get("answer_value")),
            "pages": [{"ref_id": ref_id, "page": page} for ref_id, page in dedup[:6]],
        }

    def _build_pi(self, qid):
        events = self.pi_events.get(qid, [])
        results = {event.get("tool_id"): event for event in events
                   if event.get("event") == "tool_result"}
        tools = []
        calculations = []
        for event in events:
            if event.get("event") != "tool_request":
                continue
            tool = str(event.get("tool") or "")
            arguments = event.get("arguments") or {}
            result = results.get(event.get("tool_id")) or {}
            tools.append({
                "tool": tool,
                "arguments": self._shrink(arguments, depth=3),
                "ok": result.get("ok"),
                "result": self._tool_preview(tool, result.get("data")),
            })
            if tool == "eval_math":
                calculations.append({
                    "expression": str(arguments.get("expression") or ""),
                    "unit": str(arguments.get("result_unit") or ""),
                    "operands": self._shrink(arguments.get("named_operands") or [], depth=3),
                    "result": self._tool_preview(tool, result.get("data")),
                })
        candidate = None
        citations = []
        for event in events:
            if event.get("event") == "candidate":
                candidate = event.get("row")
                citations = event.get("citations") or []
        verification = {}
        for event in events:
            if event.get("event") == "verification":
                verification = {
                    "supported": event.get("supported"),
                    "decision": event.get("decision"),
                    "reason": event.get("reason"),
                }
        bindings = [
            {"calculation_id": event.get("calculation_id"), "submitted": event.get("submitted"),
             "resolved": event.get("resolved"), "unit": event.get("unit")}
            for event in events if event.get("event") == "calculation_binding"
        ]
        return {
            "available": bool(events),
            "tools": tools,
            "tool_count": len(tools),
            "reasoning_tokens": reasoning_tokens(events),
            "tokens": usage_total(events),
            "candidate": candidate,
            "citations": citations,
            "verification": verification,
            "calculations": calculations,
            "calculation_bindings": bindings,
            "abstained": not candidate or is_blank((candidate or {}).get("answer_value")),
            "stop_reason": next((event.get("stop_reason") for event in reversed(events)
                                 if event.get("event") == "agent_done"), ""),
        }

    def _build_mm(self, qid):
        events = self.mm_events.get(qid, [])
        decisions = [
            {"decision": event.get("decision"), "reason": event.get("reason"),
             "attempt": event.get("attempt")}
            for event in events if event.get("event") == "multimodal_decision"
        ]
        escalation = next((event for event in reversed(events)
                           if event.get("event") == "visual_escalation"), {})
        visual_candidates = []
        for event in events:
            if event.get("event") == "visual_candidates":
                visual_candidates = event.get("candidates") or []
        visual_candidate = None
        visual_citations = []
        for event in events:
            if event.get("event") == "visual_candidate":
                visual_candidate = event.get("row")
                visual_citations = event.get("citations") or []
        text_candidate = None
        text_citations = []
        for event in events:
            if event.get("event") == "candidate":
                text_candidate = event.get("row")
                text_citations = event.get("citations") or []
        pages = []

        def add_page(ref_id, page, reason):
            if not ref_id or not page:
                return
            key = (str(ref_id), int(page))
            if key not in {(item["ref_id"], item["page"]) for item in pages}:
                pages.append({"ref_id": str(ref_id), "page": int(page), "reason": reason})

        for citation in visual_citations:
            add_page(citation.get("ref_id"), citation.get("page"), "cited by VLM")
        for item in visual_candidates[:4]:
            add_page(item.get("ref_id"), item.get("page"), ", ".join(item.get("reasons") or []))
        return {
            "available": bool(events),
            "text_decision": decisions[-1]["decision"] if decisions else "",
            "decisions": decisions,
            "vision_triggered": bool(escalation.get("escalate")),
            "escalation_reason": str(escalation.get("reason") or ""),
            "text_candidate": text_candidate,
            "visual_candidate": visual_candidate,
            "citations": visual_citations or text_citations,
            "visual_candidates": visual_candidates,
            "pages": pages[:6],
            "tokens": usage_total(events),
            "reasoning_tokens": reasoning_tokens(events),
            "images_sent": sum(int(number(event.get("num_images"), 0))
                               for event in events if event.get("event") == "llm_request"),
        }

    def _build_doc(self, qid):
        events = self.doc_events.get(qid, [])
        router = next((event for event in reversed(events)
                       if event.get("event") == "doc_routing"), {})
        decision = next((event for event in reversed(events)
                         if event.get("event") == "decision"), {})
        abstained = next((event for event in reversed(events)
                          if event.get("event") == "stage2_abstained"), {})
        candidate = None
        for event in events:
            if event.get("event") == "row":
                candidate = event.get("row")
        citations = []
        for event in events:
            if event.get("event") == "citation_verified":
                citations.append({"ref_id": event.get("ref_id"), "page": event.get("page"),
                                  "method": event.get("method")})
            elif event.get("event") == "quote_recovered":
                citations.append({"ref_id": event.get("ref_id"), "page": event.get("page"),
                                  "method": event.get("method")})
        return {
            "available": bool(events),
            "router": {
                "bm25_top_docs": router.get("bm25_top_docs") or [],
                "llm_docs": router.get("llm_docs") or [],
                "final_docs": router.get("final_docs") or [],
                "needs_multiple_docs": router.get("needs_multiple_docs"),
                "search_keywords": router.get("search_keywords") or [],
                "reasoning": str(router.get("reasoning") or ""),
            },
            "decision": {
                "decision": str(decision.get("decision") or ""),
                "reason": str(decision.get("reason") or ""),
                "selected_docs": decision.get("selected_docs") or [],
            },
            "abstained_reason": str(abstained.get("reason") or ""),
            "expanded_docs": next((event.get("expanded_docs") for event in reversed(events)
                                   if event.get("event") == "adaptive_expansion_retry"), []),
            "candidate": candidate,
            "citations": citations,
            "tokens": usage_total(events),
            "reasoning_tokens": reasoning_tokens(events),
        }

    @staticmethod
    def _shrink(value, depth=2):
        if depth <= 0:
            return trim(value, 240)
        if isinstance(value, dict):
            return {str(key): DemoData._shrink(item, depth - 1)
                    for key, item in list(value.items())[:24]}
        if isinstance(value, (list, tuple)):
            return [DemoData._shrink(item, depth - 1) for item in list(value)[:12]]
        if isinstance(value, str):
            return trim(value, 500)
        return value

    @staticmethod
    def _tool_preview(tool, data):
        if isinstance(data, list):
            hits = []
            for item in data[:4]:
                if isinstance(item, dict):
                    hits.append({
                        "id": item.get("id") or item.get("page_id"),
                        "ref_id": item.get("ref_id"),
                        "page": item.get("page"),
                        "text": trim(item.get("text") or item.get("quote") or "", 220),
                    })
                else:
                    hits.append(trim(item, 180))
            return {"count": len(data), "hits": hits}
        if isinstance(data, dict):
            return {key: trim(value, 260) for key, value in list(data.items())[:16]}
        return trim(data, 300)

    # -- PDF page rasterisation ----------------------------------------
    def page_image(self, ref_id, page, dpi=110):
        ref_id = str(ref_id or "").strip()
        if not SAFE_REF_ID.fullmatch(ref_id):
            raise ValueError("Invalid ref_id")
        try:
            page = int(page)
        except (TypeError, ValueError):
            raise ValueError("Invalid page number")
        if page < 1:
            raise ValueError("Invalid page number")
        dpi = max(60, min(300, int(dpi or 110)))
        key = (ref_id, page, dpi)
        if key in self._image_cache:
            return self._image_cache[key]
        docs_dir = (self.root / "data" / "docs").resolve()
        pdf_path = (docs_dir / f"{ref_id}.pdf").resolve()
        try:
            pdf_path.relative_to(docs_dir)
        except ValueError:
            raise ValueError("Invalid ref_id")
        if not pdf_path.is_file():
            raise LookupError("Unknown document")
        try:
            import pymupdf
        except ImportError:  # pragma: no cover - older installs
            import fitz as pymupdf
        document = pymupdf.open(pdf_path)
        try:
            if page > document.page_count:
                raise LookupError("Page out of range")
            pixmap = document[page - 1].get_pixmap(dpi=dpi, alpha=False)
            image = pixmap.tobytes("png")
        finally:
            document.close()
        self._image_cache[key] = image
        return image

    # -- live single-question Q&A --------------------------------------
    def ask(self, question, solver="text", unit_hint=BLANK):
        question = " ".join(str(question or "").split())
        if not question:
            return {"ok": False, "error": "Please enter a question."}, 400
        if len(question) > 1200:
            return {"ok": False, "error": "Question is too long (max 1200 characters)."}, 400
        solver = solver if solver in {"text", "multimodal", "doc_routing"} else "text"
        unit = infer_unit(question, unit_hint)
        record = {"id": "live", "question": question, "answer_unit": unit}
        events = []

        def emit(event, *args, **fields):
            entry = {"event": event}
            if args:
                entry["args"] = list(args)
            entry.update(fields)
            events.append(entry)
            return None

        try:
            row = self._run_live(solver, record, emit)
        except Exception as error:  # noqa: BLE001 - surfaced to the UI
            return {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "hint": ("Live mode needs API credentials and pre-built indexes. "
                         "Use the offline Trace Explorer during the presentation, "
                         "or check .env / outputs before retrying."),
                "solver": solver,
            }, 503
        citations = self._live_citations(events)
        steps = []
        for event in events:
            if event.get("event") == "tool_request":
                steps.append({"kind": "tool", "tool": event.get("tool"),
                              "arguments": self._shrink(event.get("arguments") or {}),
                              "result": self._tool_preview(
                                  event.get("tool"), next(
                                      (item.get("data") for item in events
                                       if item.get("event") == "tool_result"
                                       and item.get("tool_id") == event.get("tool_id")), None))})
            elif event.get("event") == "llm_response" and event.get("stage"):
                steps.append({"kind": "llm", "stage": event.get("stage"),
                              "tokens": (event.get("usage") or {}).get("total_tokens")})
            elif event.get("event") in {"decision", "multimodal_decision", "visual_escalation"}:
                steps.append({"kind": event.get("event"),
                              "detail": event.get("decision") or event.get("reason") or ""})
        row = row or {}
        return {
            "ok": True,
            "solver": solver,
            "question": question,
            "unit_hint": unit,
            "row": row,
            "citations": citations,
            "trace": {
                "tokens": usage_total(events),
                "reasoning_tokens": reasoning_tokens(events),
                "events": len(events),
                "steps": steps[-40:],
            },
        }, 200

    def _run_live(self, solver, record, emit):
        if solver == "text":
            from wattbot.deepseek import DeepSeek
            from wattbot.pipeline import solve
            from wattbot.retrieval import SearchIndex

            index_path = self._first_existing(
                self.root / "outputs" / "parser-fix" / "wattbot-index-v2.json",
                self.root / "data" / "processed" / "wattbot-index.json",
            )
            if index_path is None:
                raise RuntimeError("No retrieval index found under data/processed or outputs/parser-fix")
            index = self._cached("text-index", lambda: SearchIndex(index_path))
            client = self._client("text", DeepSeek, "hkustgz")
            return solve(record, index, client, emit, 10)

        if solver == "multimodal":
            from wattbot.multimodal import (
                FigureCatalog, MultimodalSearchIndex, VisionClient,
                detect_figure_pages, solve_multimodal,
            )

            index_path = self._first_existing(
                self.root / "data" / "processed" / "wattbot-index.json",
                self.root / "outputs" / "parser-fix" / "wattbot-index-v2.json",
            )
            if index_path is None:
                raise RuntimeError("No multimodal retrieval index found")
            index = self._cached("mm-index", lambda: MultimodalSearchIndex(index_path, None))
            catalog_path = self.root / "outputs" / "multimodal" / "catalog.json"
            catalog_data = read_json(catalog_path)
            if not catalog_data:
                catalog_data = detect_figure_pages(
                    self.metadata, self.root / "data" / "docs")
            catalog = self._cached("mm-catalog", lambda: FigureCatalog(catalog_data))
            client = self._cached("vision-client", VisionClient)
            return solve_multimodal(
                record, index, catalog, self.root / "data" / "docs", client, emit,
                top_k=10, mode="mode2-only", max_images=3, dpi=180, image_cache={})

        from wattbot.doc_routing import HybridDocumentRouter, ResilientClient, solve_question
        from wattbot.deepseek import DeepSeek

        cache = self.root / "outputs" / "doc-routing" / "corpus_cache"
        corpus = read_json(cache / "markdown_corpus.json")
        cards = read_json(cache / "document_cards.json")
        if not corpus or not cards:
            raise RuntimeError(
                "Doc-routing cache missing (outputs/doc-routing/corpus_cache). "
                "Run `python -m wattbot.doc_routing build` first.")
        router = self._cached("dr-router", lambda: HybridDocumentRouter(corpus, cards))
        client = self._cached(
            "dr-client",
            lambda: ResilientClient(DeepSeek("hkustgz", timeout=180, transport_attempts=4)))
        return solve_question(record, router, client, emit, top_docs=3)

    @staticmethod
    def _first_existing(*paths):
        for path in paths:
            if Path(path).is_file():
                return Path(path)
        return None

    def _cached(self, key, factory):
        with self._live_lock:
            if key not in self._indexes:
                self._indexes[key] = factory()
            return self._indexes[key]

    def _client(self, key, factory, *args):
        with self._live_lock:
            if key not in self._clients:
                self._clients[key] = factory(*args)
            return self._clients[key]

    @staticmethod
    def _live_citations(events):
        found = []

        def add(citation):
            ref_id = citation.get("ref_id")
            page = citation.get("page")
            if not ref_id:
                return
            entry = {"ref_id": str(ref_id), "page": page,
                     "quote": trim(citation.get("quote") or "", 500),
                     "node_id": citation.get("node_id"), "method": citation.get("method")}
            if entry not in found:
                found.append(entry)

        for event in events:
            for key in ("candidate", "visual_candidate"):
                block = event.get(key)
                if isinstance(block, dict):
                    for citation in block.get("citations") or []:
                        add(citation)
            if event.get("event") in {"citation_verified", "quote_recovered"}:
                add(event)
        return found[:12]

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WattBot 2025 · Interactive Demo</title>
<style>
:root{
  --navy:#1B365D; --blue:#0072B2; --teal:#009E73; --orange:#D55E00;
  --bg:#F4F7FB; --card:#FFFFFF; --ink:#172033; --muted:#64748B;
  --line:#DDE5EF; --soft:#EEF3F9; --shadow:0 1px 2px rgba(23,32,51,.05),0 8px 24px rgba(23,32,51,.06);
  --radius:14px;
}
*{box-sizing:border-box}
[hidden]{display:none !important}
html,body{margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
  background:var(--bg);color:var(--ink);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
button,input,select,textarea{font:inherit;color:inherit}
button{cursor:pointer}
a{color:var(--blue);text-decoration:none}
a:hover{text-decoration:underline}
code,pre{font-family:"SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace}
.app{min-height:100vh}
.topbar{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:24px;flex-wrap:wrap;
  padding:14px 26px;background:rgba(27,54,93,.98);color:#fff;box-shadow:0 2px 14px rgba(27,54,93,.22)}
.brand{display:flex;align-items:center;gap:12px;min-width:250px}
.mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--teal),var(--blue));
  display:grid;place-items:center;font-weight:800;font-size:19px;letter-spacing:-.5px}
.brand h1{margin:0;font-size:17px;letter-spacing:.2px}
.brand h1 span{color:#8FD3FF;font-weight:500}
.brand p{margin:1px 0 0;font-size:11.5px;color:#B8CAE2}
.tabs{display:flex;gap:6px;background:rgba(255,255,255,.09);padding:4px;border-radius:10px}
.tabs button{border:0;background:transparent;color:#C9D8EA;padding:7px 15px;border-radius:7px;font-weight:600;font-size:13px}
.tabs button.active{background:#fff;color:var(--navy);box-shadow:0 1px 4px rgba(0,0,0,.15)}
.status{margin-left:auto;display:flex;align-items:center;gap:7px;font-size:12px;color:#C9D8EA}
.dot{width:8px;height:8px;border-radius:50%;background:#38D39F;box-shadow:0 0 0 3px rgba(56,211,159,.18)}
.dot.warn{background:#F5B942;box-shadow:0 0 0 3px rgba(245,185,66,.18)}
main{max-width:1560px;margin:0 auto;padding:22px 26px 60px}
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin-bottom:18px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:14px 16px;
  box-shadow:var(--shadow);position:relative;overflow:hidden}
.kpi:before{content:"";position:absolute;inset:0 auto 0 0;width:4px;background:var(--blue)}
.kpi.teal:before{background:var(--teal)} .kpi.orange:before{background:var(--orange)}
.kpi.navy:before{background:var(--navy)} .kpi.mint:before{background:#27AE60}
.kpi .label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}
.kpi .value{font-size:25px;font-weight:750;letter-spacing:-.6px;margin-top:3px}
.kpi .sub{font-size:11.5px;color:var(--muted);margin-top:2px}
.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.search{position:relative;flex:1 1 320px;max-width:520px}
.search input{width:100%;padding:10px 13px 10px 36px;border:1px solid var(--line);border-radius:10px;background:#fff;outline:none}
.search input:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(0,114,178,.13)}
.search svg{position:absolute;left:11px;top:10px;width:16px;height:16px;color:var(--muted)}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:7px 12px;font-size:12px;
  font-weight:600;color:#475569;display:inline-flex;gap:7px;align-items:center;transition:.15s}
.chip:hover{border-color:#B7C9DE}
.chip.active{background:var(--navy);border-color:var(--navy);color:#fff}
.chip .n{background:var(--soft);color:var(--muted);border-radius:999px;padding:1px 6px;font-size:10.5px}
.chip.active .n{background:rgba(255,255,255,.18);color:#fff}
.quickpicks{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 16px}
.quickpicks .qlabel{font-size:11px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;color:var(--muted)}
.pick{border:1px dashed #BFD2E6;background:#F8FBFF;border-radius:9px;padding:5px 10px;font-size:12px;color:var(--navy);font-weight:600}
.pick:hover{background:#EDF6FF;border-color:var(--blue)}
.pick b{color:var(--blue)}
.workspace{display:grid;grid-template-columns:330px minmax(0,1fr);gap:16px;align-items:start}
.qlist-panel,.inspector{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.qlist-panel{position:sticky;top:86px;max-height:calc(100vh - 108px);display:flex;flex-direction:column;overflow:hidden}
.panel-head{display:flex;align-items:center;justify-content:space-between;padding:13px 15px;border-bottom:1px solid var(--line)}
.panel-head h2{margin:0;font-size:13px;text-transform:uppercase;letter-spacing:.7px;color:var(--navy)}
.panel-head span{font-size:11.5px;color:var(--muted)}
.qlist{overflow:auto;padding:7px}
.qitem{border:1px solid transparent;border-radius:10px;padding:9px 10px;margin-bottom:3px;cursor:pointer}
.qitem:hover{background:var(--soft)}
.qitem.active{background:#EDF6FF;border-color:#B7D8F2}
.qitem .qtop{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.qitem .qid{font-weight:750;color:var(--navy);font-size:12.5px}
.qitem .qtok{margin-left:auto;font-size:10.5px;color:var(--muted)}
.qitem .qtext{font-size:12px;color:#334155;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.qitem .qbadges{display:flex;gap:4px;margin-top:5px;flex-wrap:wrap}
.mini{font-size:9.5px;font-weight:700;padding:1px 5px;border-radius:5px;letter-spacing:.2px;text-transform:uppercase}
.mini.esc{background:#FDEBE2;color:#B0440B}.mini.ast{background:#E6F7F0;color:#007A59}
.mini.vlm{background:#E6F0FA;color:#005D93}.mini.blank{background:#F1F3F6;color:#5B6675}
.mini.fast{background:#EFF8EC;color:#3C7A2B}
.inspector{min-height:420px;padding:0;overflow:hidden}
.empty{padding:80px 30px;text-align:center;color:var(--muted)}
.empty b{display:block;font-size:16px;color:var(--navy);margin-bottom:6px}
.q-head{padding:19px 22px 16px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,#FBFDFF,#fff)}
.q-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:7px}
.qidbig{font-size:13px;font-weight:800;letter-spacing:.4px;color:var(--blue)}
.badge{font-size:10.5px;font-weight:700;padding:2px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.4px}
.badge.esc{background:#FDEBE2;color:#B0440B}.badge.fast{background:#E8F6E3;color:#3C7A2B}
.badge.vlm{background:#E6F0FA;color:#005D93}.badge.ast{background:#E6F7F0;color:#007A59}
.badge.blank{background:#F1F3F6;color:#5B6675}
.q-head h2{margin:0 0 8px;font-size:20px;line-height:1.35;letter-spacing:-.25px;max-width:1100px}
.unitline{font-size:12px;color:var(--muted)}
.unitline code{background:var(--soft);border:1px solid var(--line);border-radius:5px;padding:1px 6px;color:var(--navy)}
.answer-card{margin:16px 22px;border:1px solid #BEDFD2;border-radius:12px;background:linear-gradient(135deg,#F3FBF7,#fff);padding:15px 17px;position:relative}
.answer-card.blank{border-color:#D9E1EA;background:linear-gradient(135deg,#F7F9FB,#fff)}
.answer-top{display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap}
.answer-value{font-size:32px;font-weight:800;letter-spacing:-1px;color:var(--teal);line-height:1.1}
.answer-value small{font-size:15px;font-weight:600;color:#3D5A50;margin-left:5px;letter-spacing:0}
.answer-card.blank .answer-value{font-size:19px;color:#5B6675;letter-spacing:0}
.answer-refs{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto;align-items:center}
.refpill{background:#fff;border:1px solid var(--line);border-radius:7px;padding:3px 8px;font-size:11.5px;font-weight:600}
.refpill a{font-weight:500}
.support{margin-top:11px;border-left:3px solid var(--teal);background:#fff;padding:8px 11px;border-radius:0 7px 7px 0;
  font-size:12.5px;color:#334155;white-space:pre-wrap;max-height:150px;overflow:auto}
.explain{margin-top:9px;font-size:12.5px;color:#475569}
.explain b{color:var(--navy);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.section-label{display:flex;align-items:center;gap:8px;margin:22px 22px 10px;font-size:12px;font-weight:800;
  text-transform:uppercase;letter-spacing:.8px;color:var(--navy)}
.section-label .rule{flex:1;height:1px;background:var(--line)}
.pipeline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:0 22px;align-items:start}
.stage{border:1px solid var(--line);border-radius:12px;background:#fff;overflow:hidden;display:flex;flex-direction:column}
.stage-head{padding:11px 13px;border-bottom:1px solid var(--line);display:flex;gap:9px;align-items:center}
.stage-icon{width:27px;height:27px;border-radius:8px;display:grid;place-items:center;font-weight:800;font-size:12px;color:#fff;flex:none}
.stage-a .stage-icon{background:var(--blue)} .stage-b .stage-icon{background:var(--orange)} .stage-c .stage-icon{background:var(--teal)}
.stage-head h3{margin:0;font-size:12.5px;color:var(--navy)} .stage-head p{margin:0;font-size:10.5px;color:var(--muted)}
.stage-body{padding:11px 13px;font-size:12px;flex:1}
.statrow{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}
.stat{background:var(--soft);border-radius:7px;padding:4px 7px;font-size:10.5px;color:#475569}
.stat b{color:var(--navy)}
.statusline{display:inline-flex;align-items:center;gap:5px;font-weight:700;font-size:11px;padding:2px 8px;border-radius:999px;margin-bottom:8px}
.statusline.ok{background:#E8F6E3;color:#3C7A2B}.statusline.no{background:#F1F3F6;color:#5B6675}
.statusline.warn{background:#FDEBE2;color:#B0440B}
.astbox{border:1px solid #B7E4D4;background:#F2FBF7;border-radius:8px;padding:8px 9px;margin:8px 0}
.astbox .t{font-size:10px;font-weight:800;letter-spacing:.6px;color:#007A59;text-transform:uppercase;margin-bottom:4px}
.ast-expr{font-family:Menlo,Consolas,monospace;font-size:11.5px;background:#fff;border:1px solid #CDE9DE;border-radius:6px;
  padding:6px 7px;overflow:auto;white-space:pre-wrap;color:#0B4F3A}
.operands{margin-top:5px;display:grid;gap:3px}
.op{font-size:10.5px;color:#3D5A50;display:flex;gap:5px;align-items:baseline}
.op code{background:#fff;border:1px solid #D8EDE5;border-radius:4px;padding:0 4px;color:#0B4F3A}
details.tool{border:1px solid var(--line);border-radius:8px;margin:5px 0;overflow:hidden;background:#FCFDFF}
details.tool summary{cursor:pointer;padding:6px 8px;font-size:11.5px;font-weight:600;color:#334155;display:flex;gap:6px;align-items:center}
details.tool summary .tok{font-family:Menlo,monospace;color:var(--blue);font-size:11px}
details.tool summary .ok{margin-left:auto;font-size:9.5px;font-weight:800;border-radius:4px;padding:1px 5px}
details.tool summary .ok.y{background:#E8F6E3;color:#3C7A2B}details.tool summary .ok.n{background:#FDEBE2;color:#B0440B}
details.tool .body{padding:0 8px 8px;border-top:1px solid var(--line)}
details.tool pre{margin:6px 0 0;font-size:10.5px;background:#F6F8FB;border:1px solid var(--line);border-radius:6px;
  padding:6px;max-height:180px;overflow:auto;white-space:pre-wrap;color:#334155}
.docchips{display:flex;gap:4px;flex-wrap:wrap;margin:3px 0 7px}
.doc{font-size:10.5px;border-radius:5px;padding:2px 6px;background:var(--soft);color:#475569;font-weight:600}
.doc.final{background:#E6F7F0;color:#007A59}.doc.llm{background:#E6F0FA;color:#005D93}
.docrow{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);font-weight:700;margin-top:6px}
.note{background:#F8FAFC;border:1px dashed var(--line);border-radius:8px;padding:8px 9px;color:#526071;font-size:11.5px;margin:7px 0}
.note.warn{background:#FFF8EC;border-color:#F0D8A8;color:#8A5A00}
.note.ok{background:#F2FBF7;border-color:#BFE5D6;color:#0B5E45}
.figgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:8px 0}
.page-thumb{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fff;cursor:zoom-in;position:relative}
.page-thumb img{width:100%;height:118px;object-fit:cover;object-position:top;display:block;background:#F1F3F6}
.page-thumb span{position:absolute;left:5px;bottom:5px;background:rgba(23,32,51,.82);color:#fff;font-size:9.5px;
  padding:2px 5px;border-radius:4px;font-weight:600}
.page-thumb .why{position:absolute;right:5px;top:5px;background:rgba(0,114,178,.9);color:#fff;font-size:8.5px;
  padding:1px 4px;border-radius:4px;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.rowcard{border:1px solid var(--line);border-radius:8px;padding:8px 9px;background:#FCFDFF;margin-top:7px}
.rowcard .val{font-weight:800;color:var(--navy);font-size:16px}
.rowcard .val small{font-size:11px;color:var(--muted);font-weight:600;margin-left:3px}
.rowcard .ans{font-size:11.5px;color:#475569;margin-top:3px}
.rowcard .refs{margin-top:4px;display:flex;gap:4px;flex-wrap:wrap}
.consensus{margin:16px 22px 0;border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:#fff}
.consensus-head{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin-bottom:8px}
.consensus-head h3{margin:0;font-size:13px;color:var(--navy)}
.res-pill{font-size:11px;font-weight:800;border-radius:999px;padding:3px 9px;letter-spacing:.2px}
.res-fast{background:#E8F6E3;color:#3C7A2B}.res-esc{background:#FDEBE2;color:#B0440B}
.flow{display:flex;align-items:center;gap:7px;flex-wrap:wrap;font-size:11.5px;color:#475569;margin-top:7px}
.flow .node{border:1px solid var(--line);border-radius:7px;padding:4px 8px;background:var(--soft);font-weight:600}
.flow .node.hot{background:#E6F0FA;border-color:#B7D8F2;color:#005D93}
.flow .arrow{color:#94A3B8}
.tokgrid{margin:16px 22px 22px;border:1px solid var(--line);border-radius:12px;padding:14px 16px;background:#fff}
.tokrow{display:grid;grid-template-columns:110px minmax(0,1fr) 92px;gap:10px;align-items:center;margin:7px 0;font-size:11.5px}
.tokrow .tl{color:#475569;font-weight:600;text-align:right}
.tokbar{height:17px;background:#F1F5F9;border-radius:5px;overflow:hidden;position:relative}
.tokbar i{display:block;height:100%;border-radius:5px;background:var(--blue);min-width:3px}
.tokbar i.a{background:var(--blue)}.tokbar i.b{background:var(--orange)}.tokbar i.c{background:var(--teal)}
.tokbar i.t{background:var(--navy)}.tokbar i.f{background:#94A3B8}
.tokrow .tv{font-family:Menlo,monospace;font-size:10.5px;color:var(--muted);text-align:right}
.tok-foot{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:9px;margin-top:8px;font-size:11.5px;color:var(--muted)}
.live-grid{display:grid;grid-template-columns:minmax(0,460px) minmax(0,1fr);gap:16px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px}
.card h2{margin:0 0 4px;font-size:17px;color:var(--navy)}
.card .sub{margin:0 0 14px;font-size:12px;color:var(--muted)}
.live-form textarea{width:100%;min-height:104px;resize:vertical;border:1px solid var(--line);border-radius:10px;padding:10px 12px;outline:none}
.live-form textarea:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(0,114,178,.13)}
.live-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:11px 0}
.live-row label{display:block;font-size:11px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.live-row select,.live-row input{width:100%;border:1px solid var(--line);border-radius:8px;padding:8px 9px;background:#fff}
.btn{border:0;border-radius:9px;padding:10px 18px;font-weight:700;background:var(--blue);color:#fff;box-shadow:0 2px 8px rgba(0,114,178,.25)}
.btn:hover{background:#005D93}.btn:disabled{opacity:.55;cursor:progress}
.hint{font-size:11.5px;color:var(--muted);margin:10px 0 0}
.examples{display:flex;gap:6px;flex-wrap:wrap;margin-top:12px}
.example{border:1px solid var(--line);background:#F8FAFC;border-radius:8px;padding:6px 9px;font-size:11px;color:#475569;text-align:left}
.example:hover{border-color:var(--blue);color:var(--blue)}
.live-result{min-height:300px}
.live-status{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--muted)}
.spinner{width:15px;height:15px;border:2px solid #D8E2EE;border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.errbox{border:1px solid #F1C9B5;background:#FFF7F2;border-radius:10px;padding:13px 15px;color:#8A3B12}
.errbox b{display:block;margin-bottom:4px}
.tracesteps{display:grid;gap:5px;margin-top:8px}
.tstep{border:1px solid var(--line);border-radius:7px;padding:6px 8px;font-size:11px;background:#FBFDFF;display:flex;gap:7px;align-items:baseline}
.tstep .k{font-size:9px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;border-radius:4px;padding:1px 4px;background:var(--soft);color:var(--muted);flex:none}
.tstep.tool .k{background:#E6F0FA;color:#005D93}.tstep.llm .k{background:#E6F7F0;color:#007A59}
.tstep code{font-size:10.5px;color:#334155;overflow-wrap:anywhere}
.lightbox{position:fixed;inset:0;z-index:100;background:rgba(15,23,42,.78);display:grid;place-items:center;padding:30px;cursor:zoom-out}
.lightbox img{max-width:94vw;max-height:92vh;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,.5);background:#fff}
.lightbox button{position:absolute;top:16px;right:18px;border:0;background:#fff;border-radius:8px;padding:6px 11px;font-weight:700}
@media (max-width:1180px){
  .kpis{grid-template-columns:repeat(3,minmax(0,1fr))}
  .pipeline{grid-template-columns:1fr}
  .live-grid{grid-template-columns:1fr}
}
@media (max-width:820px){
  .topbar{padding:12px 16px;gap:12px}
  main{padding:16px}
  .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
  .workspace{grid-template-columns:1fr}
  .qlist-panel{position:static;max-height:340px}
  .answer-card,.pipeline,.consensus,.tokgrid,.section-label{margin-left:14px;margin-right:14px}
  .pipeline{padding:0 14px;gap:10px}
  .tokrow{grid-template-columns:82px minmax(0,1fr) 72px}
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <div class="mark">W</div>
      <div>
        <h1>WattBot <span>2025</span></h1>
        <p>Pareto-efficient RAG for AI sustainability</p>
      </div>
    </div>
    <nav class="tabs" role="tablist">
      <button id="tab-explorer" class="active" data-tab="explorer">Trace Explorer</button>
      <button id="tab-live" data-tab="live">Live Q&amp;A</button>
    </nav>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Loading data…</span></div>
  </header>
  <main>
    <section id="view-explorer">
      <div class="kpis" id="kpis"></div>
      <div class="toolbar">
        <div class="search">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
          <input id="search" type="search" placeholder="Search by question ID or text…" autocomplete="off">
        </div>
        <div class="chips" id="chips"></div>
      </div>
      <div class="quickpicks" id="picks"></div>
      <div class="workspace">
        <aside class="qlist-panel">
          <div class="panel-head"><h2>Questions</h2><span id="qcount">—</span></div>
          <div class="qlist" id="qlist"></div>
        </aside>
        <section class="inspector" id="inspector">
          <div class="empty"><b>Loading the trace explorer…</b>Reading saved cascade predictions and stage traces.</div>
        </section>
      </div>
    </section>
    <section id="view-live" hidden>
      <div class="live-grid">
        <div class="card live-form">
          <h2>Ask a sustainability question</h2>
          <p class="sub">Runs the real server-side <code>wattbot</code> pipeline and streams the structured answer, citations and trace steps back.</p>
          <textarea id="liveQuestion" placeholder="e.g. How much energy did GPT-3 training consume, in MWh?"></textarea>
          <div class="live-row">
            <label>Solver
              <select id="liveSolver">
                <option value="text">Clean v2 Text RAG (fast)</option>
                <option value="multimodal">Multimodal mode2-only</option>
                <option value="doc_routing">Doc Routing</option>
              </select>
            </label>
            <label>Unit hint
              <input id="liveUnit" placeholder="auto-detect">
            </label>
          </div>
          <button id="liveAsk" class="btn">Run pipeline</button>
          <p class="hint">Live mode needs API credentials and pre-built indexes. If they are missing, the UI explains what to configure; the Trace Explorer always works offline.</p>
          <div class="examples" id="liveExamples"></div>
        </div>
        <div class="card live-result" id="liveResult">
          <div class="empty"><b>No live query yet</b>Pick a solver, ask a question, and the verified answer will appear here.</div>
        </div>
      </div>
    </section>
  </main>
</div>
<div id="lightbox" class="lightbox" hidden><button id="lightboxClose">Close ×</button><img id="lightboxImg" alt="PDF page"></div>
<script>
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (n) => { n = Number(n || 0); if (n >= 1e6) return (n/1e6).toFixed(n>=1e7?1:2)+"M"; if (n >= 1e3) return Math.round(n).toLocaleString("en-US"); return String(Math.round(n)); };
const fmtFull = (n) => Number(n || 0).toLocaleString("en-US");
function asList(v) {
  if (Array.isArray(v)) return v.filter((x) => x && String(x) !== "is_blank");
  if (!v || String(v).trim() === "" || String(v) === "is_blank") return [];
  try { const parsed = JSON.parse(String(v).replace(/'/g, '"')); return Array.isArray(parsed) ? parsed : [String(parsed)]; }
  catch (e) { return [String(v)]; }
}
function pageUrl(refId, page, dpi) {
  return "/api/page-image?ref_id=" + encodeURIComponent(refId) + "&page=" + encodeURIComponent(page) + (dpi ? "&dpi=" + dpi : "");
}
const state = { questions: [], overview: null, filter: "all", search: "", selected: null, cache: new Map() };
const FILTERS = [
  { key: "all", label: "All" },
  { key: "escalated", label: "Escalated (any)" },
  { key: "disagreement", label: "Escalated (Disagreement)" },
  { key: "abstain_trigger", label: "Escalated (Abstention)" },
  { key: "visual", label: "Visual / Chart Triggered" },
  { key: "ast", label: "Decimal AST Sandbox Used" },
  { key: "abstained", label: "Abstained (is_blank)" }
];

async function api(path, options) {
  const response = await fetch(path, options);
  let data = {};
  try { data = await response.json(); } catch (e) { data = { error: "Invalid JSON response" }; }
  if (!response.ok) { const error = new Error(data.error || ("HTTP " + response.status)); error.payload = data; throw error; }
  return data;
}

function setStatus(text, warn) {
  $("#statusText").textContent = text;
  $("#statusDot").classList.toggle("warn", !!warn);
}

function renderKpis() {
  const k = state.overview.kpis;
  const counts = state.overview.counts;
  const denom = (state.overview.summary && state.overview.summary.questions) || counts.total;
  const cards = [
    { cls: "navy", label: "Leaderboard", value: k.public.toFixed(3) + " / " + k.private.toFixed(3), sub: "Public / Private weighted score" },
    { cls: "", label: "Mean tokens", value: (k.mean_tokens / 1000).toFixed(1) + "k", sub: "tokens per question (cascade)" },
    { cls: "teal", label: "Cost efficiency", value: k.cheaper_factor.toFixed(1) + "×", sub: "cheaper than full-context (" + fmt(k.full_context_tokens) + " tok/q)" },
    { cls: "orange", label: "Tier-2 escalation", value: (k.escalation_rate * 100).toFixed(1) + "%", sub: (counts.escalated || 0) + " of " + denom + " test questions invoked Tier-2" },
    { cls: "mint", label: "Private lead", value: "+" + k.private_gain.toFixed(3), sub: "over KohakuRAG 1st place (" + (k.private - k.private_gain).toFixed(3) + ")" }
  ];
  $("#kpis").innerHTML = cards.map((c) => `
    <div class="kpi ${c.cls}">
      <div class="label">${esc(c.label)}</div>
      <div class="value">${esc(c.value)}</div>
      <div class="sub">${esc(c.sub)}</div>
    </div>`).join("");
}

function renderChips() {
  const counts = state.overview.counts;
  $("#chips").innerHTML = FILTERS.map((f) => {
    const n = f.key === "all" ? counts.total : (counts[f.key] || 0);
    return `<button class="chip ${state.filter === f.key ? "active" : ""}" data-filter="${f.key}">${esc(f.label)}<span class="n">${n}</span></button>`;
  }).join("");
}

function renderPicks() {
  const picks = state.overview.picks || [];
  $("#picks").innerHTML = `<span class="qlabel">Quick picks</span>` + picks.map((p) =>
    `<button class="pick" data-qid="${esc(p.id)}"><b>${esc(p.id)}</b> · ${esc(p.label)}</button>`).join("");
}

function matchesFilter(item) {
  const f = item.flags || {};
  switch (state.filter) {
    case "escalated": return !!f.escalated;
    case "disagreement": return !!f.disagreement;
    case "abstain_trigger": return !!f.abstain_trigger;
    case "visual": return !!f.visual;
    case "ast": return !!f.ast;
    case "abstained": return !!f.abstained;
    default: return true;
  }
}

function badgesFor(item) {
  const f = item.flags || {};
  const out = [];
  if (f.escalated) out.push('<span class="mini esc">esc</span>');
  if (f.disagreement) out.push('<span class="mini esc">disagree</span>');
  if (f.ast) out.push('<span class="mini ast">ast</span>');
  if (f.visual) out.push('<span class="mini vlm">vlm</span>');
  if (f.abstained) out.push('<span class="mini blank">blank</span>');
  if (f.fast_path) out.push('<span class="mini fast">fast</span>');
  return out.join("");
}

function renderList() {
  const needle = state.search.trim().toLowerCase();
  const items = state.questions.filter((item) => matchesFilter(item) &&
    (!needle || item.id.toLowerCase().includes(needle) || (item.question || "").toLowerCase().includes(needle)));
  $("#qcount").textContent = items.length + " of " + state.questions.length;
  if (!items.length) {
    $("#qlist").innerHTML = '<div class="empty" style="padding:30px 10px">No question matches this filter.</div>';
    return;
  }
  $("#qlist").innerHTML = items.map((item) => `
    <div class="qitem ${state.selected === item.id ? "active" : ""}" data-qid="${esc(item.id)}">
      <div class="qtop"><span class="qid">${esc(item.id)}</span>${badgesFor(item)}<span class="qtok">${fmt(item.tokens)} tok</span></div>
      <div class="qtext">${esc(item.question)}</div>
      <div class="qbadges">${item.reasons && item.reasons.length ? item.reasons.map((r) => '<span class="mini esc">' + esc(r) + "</span>").join("") : ""}</div>
    </div>`).join("");
}

function selectQuestion(qid) {
  if (!qid) return;
  state.selected = qid;
  renderList();
  loadQuestion(qid);
}

async function loadQuestion(qid) {
  const inspector = $("#inspector");
  const cached = state.cache.get(qid);
  if (cached) { renderInspector(cached); return; }
  inspector.innerHTML = '<div class="empty"><b>' + esc(qid) + '</b>Loading stage traces…</div>';
  try {
    const detail = await api("/api/question/" + encodeURIComponent(qid));
    state.cache.set(qid, detail);
    if (state.selected === qid) renderInspector(detail);
  } catch (e) {
    inspector.innerHTML = '<div class="empty"><b>Could not load ' + esc(qid) + '</b>' + esc(e.message) + "</div>";
  }
}

function refPills(row) {
  const refs = asList(row && row.ref_id);
  const urls = asList(row && row.ref_url);
  if (!refs.length) return "";
  return refs.map((ref, i) => `<span class="refpill">${esc(ref)}${urls[i] ? ' · <a href="' + esc(urls[i]) + '" target="_blank" rel="noopener">source ↗</a>' : ""}</span>`).join("");
}

function rowSummary(row, label) {
  if (!row) return '<div class="note">No candidate supplied by this stage.</div>';
  const blank = !row.answer_value || String(row.answer_value) === "is_blank";
  return `<div class="rowcard">
    <div class="val">${blank ? "is_blank" : esc(row.answer_value)}${blank ? "" : "<small>" + esc(row.answer_unit || "") + "</small>"}</div>
    <div class="ans">${esc(row.answer || row.explanation || "")}</div>
    <div class="refs">${refPills(row)}</div>
  </div>`;
}

function operandList(operands) {
  if (!operands) return [];
  if (Array.isArray(operands)) return operands;
  return Object.entries(operands).map(([name, value]) =>
    (value && typeof value === "object") ? { name, ...value } : { name, value });
}

function operandHtml(op) {
  if (op && typeof op === "object") {
    const name = op.name || op.id || op.node_id || op.operand || "value";
    const val = op.value ?? op.quantity ?? op.amount ?? "";
    const quote = op.quote || op.text || "";
    return `<div class="op"><code>${esc(name)}</code>${val !== "" ? "<b>= " + esc(val) + "</b>" : ""}${quote ? '<span>“' + esc(quote) + '”</span>' : ""}</div>`;
  }
  return `<div class="op"><code>${esc(op)}</code></div>`;
}

function pageGrid(pages) {
  if (!pages || !pages.length) return "";
  return '<div class="figgrid">' + pages.map((p) => `
    <div class="page-thumb" data-src="${esc(pageUrl(p.ref_id, p.page, 220))}">
      <img loading="lazy" src="${esc(pageUrl(p.ref_id, p.page, 110))}" alt="${esc(p.ref_id)} page ${esc(p.page)}"
           onerror="this.closest('.page-thumb').style.display='none'">
      <span>${esc(p.ref_id)} · p.${esc(p.page)}</span>
      ${p.reason ? '<span class="why">' + esc(p.reason) + "</span>" : ""}
    </div>`).join("") + "</div>";
}

function stageShell(cls, icon, title, subtitle, body) {
  return `<article class="stage ${cls}">
    <div class="stage-head"><div class="stage-icon">${icon}</div><div><h3>${esc(title)}</h3><p>${esc(subtitle)}</p></div></div>
    <div class="stage-body">${body}</div>
  </article>`;
}

function renderStageA(d) {
  const a = d.tier1a;
  if (!a.available) return stageShell("stage-a", "A", "Tier-1 · Path A", "Pi Agent v2 · tool use", '<div class="note">No Pi-v2 trace for this question.</div>');
  const blank = !a.candidate || !a.candidate.answer_value || String(a.candidate.answer_value) === "is_blank";
  const calcs = (a.calculations || []).map((c) => `
    <div class="astbox">
      <div class="t">Decimal AST sandbox</div>
      <div class="ast-expr">${esc(c.expression || "(no expression)")}${c.unit ? "   →  " + esc(c.unit) : ""}</div>
      <div class="operands">${operandList(c.operands).map(operandHtml).join("")}</div>
      <div class="op" style="margin-top:4px"><code>result</code><b>= ${esc(typeof c.result === "object" ? JSON.stringify(c.result) : c.result)}</b></div>
    </div>`).join("");
  const bindings = (a.calculation_bindings || []).length
    ? `<div class="note">Bound calculation: <code>${esc(a.calculation_bindings.map((b) => (b.calculation_id || "?") + " → " + (b.resolved || b.submitted || "") + " " + (b.unit || "")).join(" · "))}</code></div>` : "";
  const tools = (a.tools || []).map((tool, i) => {
    const args = typeof tool.arguments === "object" ? JSON.stringify(tool.arguments, null, 2) : String(tool.arguments || "");
    const result = typeof tool.result === "object" ? JSON.stringify(tool.result, null, 2) : String(tool.result || "");
    return `<details class="tool"><summary><span class="tok">${i + 1}</span> ${esc(tool.tool)}
      <span class="ok ${tool.ok === false ? "n" : "y"}">${tool.ok === false ? "fail" : "ok"}</span></summary>
      <div class="body"><pre>${esc(args)}</pre>${result ? "<pre>" + esc(result) + "</pre>" : ""}</div></details>`;
  }).join("");
  const verify = a.verification && a.verification.reason
    ? `<div class="note ${a.verification.supported ? "ok" : "warn"}"><b>Verifier:</b> ${esc(a.verification.reason)}</div>` : "";
  return stageShell("stage-a", "A", "Tier-1 · Path A", "Pi Agent v2 · tool use", `
    <div class="statusline ${blank ? "no" : "ok"}">${blank ? "Abstained / no candidate" : "Candidate answer"}</div>
    <div class="statrow">
      <span class="stat"><b>${a.tool_count}</b> tool calls</span>
      <span class="stat"><b>${fmt(a.tokens)}</b> tok</span>
      <span class="stat"><b>${fmt(a.reasoning_tokens)}</b> reasoning</span>
    </div>
    ${calcs}${bindings}
    ${tools || '<div class="note">No tool calls recorded.</div>'}
    ${verify}
    ${rowSummary(a.candidate)}
  `);
}

function renderStageB(d) {
  const b = d.tier1b;
  if (!b.available) return stageShell("stage-b", "B", "Tier-1 · Path B", "Multimodal VLM · visual fallback", '<div class="note">No multimodal trace for this question.</div>');
  const decisions = (b.decisions || []).map((x) =>
    `<div class="note">${esc(x.decision || "")}${x.reason ? " · " + esc(x.reason) : ""}</div>`).join("");
  const vision = b.vision_triggered
    ? `<div class="statusline warn">Vision triggered · ${esc(b.escalation_reason || "visual")}</div>`
    : '<div class="statusline ok">Text pass accepted · vision not needed</div>';
  const candidate = b.visual_candidate || b.text_candidate;
  return stageShell("stage-b", "B", "Tier-1 · Path B", "Multimodal VLM · visual fallback", `
    ${vision}
    <div class="statrow">
      <span class="stat"><b>${fmt(b.tokens)}</b> tok</span>
      <span class="stat"><b>${b.images_sent || 0}</b> images</span>
      <span class="stat">text: <b>${esc(b.text_decision || "—")}</b></span>
    </div>
    ${pageGrid(b.pages)}
    ${decisions}
    ${rowSummary(candidate)}
  `);
}

function docRow(label, docs, cls) {
  if (!docs || !docs.length) return "";
  return `<div class="docrow">${esc(label)}</div><div class="docchips">${docs.map((x) => `<span class="doc ${cls || ""}">${esc(x)}</span>`).join("")}</div>`;
}

function renderStageC(d) {
  const c = d.tier2;
  const fast = !d.cascade.escalated;
  if (!c.available) {
    return stageShell("stage-c", "C", "Tier-2 · Doc Routing", "Escalation-only arbiter", fast
      ? '<div class="note ok">Tier-2 bypassed — consensus gate accepted Tier-1 and saved tokens.</div>'
      : '<div class="note">No Tier-2 trace for this question.</div>');
  }
  const r = c.router || {};
  const decision = c.decision || {};
  const body = `
    ${fast ? '<div class="note ok">Fast path: Tier-2 was optional and not needed.</div>' : ""}
    ${docRow("Top-3 documents selected", r.final_docs, "final")}
    ${docRow("Hybrid BM25 candidates", (r.bm25_top_docs || []).slice(0, 8), "")}
    ${docRow("LLM routing decision", r.llm_docs, "llm")}
    ${r.reasoning ? '<div class="note"><b>Router reasoning:</b> ' + esc(r.reasoning) + "</div>" : ""}
    ${r.needs_multiple_docs ? '<div class="note warn">Multi-document question flagged.</div>' : ""}
    ${decision.decision ? '<div class="note ' + (decision.decision.includes("abstain") ? "warn" : "ok") + '"><b>Decision:</b> ' + esc(decision.decision) + (decision.reason ? " · " + esc(decision.reason) : "") + "</div>" : ""}
    ${c.abstained_reason ? '<div class="note warn"><b>Stage-2 abstention:</b> ' + esc(c.abstained_reason) + "</div>" : ""}
    ${(c.expanded_docs || []).length ? docRow("Adaptive expansion retry", c.expanded_docs, "") : ""}
    <div class="statrow"><span class="stat"><b>${fmt(c.tokens)}</b> tok</span><span class="stat"><b>${fmt(c.reasoning_tokens)}</b> reasoning</span></div>
    ${rowSummary(c.candidate)}
  `;
  return stageShell("stage-c", "C", "Tier-2 · Doc Routing", "Escalation-only arbiter", body);
}

function renderConsensus(d) {
  const c = d.cascade;
  const fast = !c.escalated;
  const resolution = c.resolution || (fast ? "tier1_accepted" : "escalated");
  const reasons = (c.reasons || []).length
    ? '<span class="badge esc">' + c.reasons.map(esc).join(" + ") + "</span>" : '<span class="badge fast">no trigger</span>';
  const flow = `<div class="flow">
    <span class="node">Tier-1 A · Pi</span><span class="arrow">→</span>
    <span class="node">Tier-1 B · VLM</span><span class="arrow">→</span>
    <span class="node ${fast ? "" : "hot"}">Consensus gate</span><span class="arrow">→</span>
    <span class="node ${fast ? "" : "hot"}">${fast ? "Tier-2 bypassed" : "Tier-2 invoked"}</span><span class="arrow">→</span>
    <span class="node">Final CSV</span>
  </div>`;
  const saving = fast
    ? '<div class="note ok">Fast path accepted Tier-1, avoiding the Tier-2 call and its ~24.3k token average.</div>'
    : '<div class="note warn">Tier-1 did not satisfy the gate, so the saved Tier-2 trace is shown for arbitration.</div>';
  return `<div class="consensus">
    <div class="consensus-head"><h3>Consensus gate &amp; Tier-2 arbitration</h3>
      <span class="res-pill ${fast ? "res-fast" : "res-esc"}">${esc(resolution)}</span>${reasons}</div>
    ${flow}${saving}
  </div>`;
}

function renderTokens(d) {
  const t = d.tokens;
  const rows = [
    { label: "Tier-1A Pi", value: t.tier1a, cls: "a" },
    { label: "Tier-1B VLM", value: t.tier1b, cls: "b" },
    { label: "Tier-2 Doc", value: t.tier2, cls: "c" },
    { label: "Cascade total", value: t.cascade, cls: "t" },
    { label: "Full-context", value: t.full_context, cls: "f" }
  ];
  const max = Math.max(1, t.full_context || 1);
  const body = rows.map((r) => {
    const width = r.value > 0 ? Math.max(0.8, Math.min(100, (r.value / max) * 100)) : 0;
    return `<div class="tokrow"><div class="tl">${esc(r.label)}</div>
      <div class="tokbar"><i class="${r.cls}" style="width:${width.toFixed(2)}%"></i></div>
      <div class="tv">${fmtFull(r.value)}</div></div>`;
  }).join("");
  const ratio = t.cascade > 0 ? (t.full_context / t.cascade) : 0;
  return `<div class="tokgrid">${body}
    <div class="tok-foot"><span>This question: <b>${fmtFull(t.cascade)}</b> cascade tokens.</span>
    <span>${ratio ? ratio.toFixed(1) + "× cheaper than full-context" : "No cascade token record"} · full-context baseline ${fmtFull(t.full_context)} tok/q</span></div>
  </div>`;
}

function renderFinal(final) {
  if (!final.available) {
    return '<div class="answer-card blank"><div class="answer-top"><div class="answer-value">No verified CSV row</div></div><div class="explain">The final cascade prediction CSV is not available for this question.</div></div>';
  }
  if (final.abstained) {
    return `<div class="answer-card blank">
      <div class="answer-top"><div class="answer-value">Not answerable · is_blank</div>
      <div class="answer-refs">${refPills(final)}</div></div>
      <div class="explain"><b>Explanation</b><br>${esc(final.explanation || "The system abstained because the corpus did not support an answer.")}</div>
    </div>`;
  }
  return `<div class="answer-card">
    <div class="answer-top">
      <div class="answer-value">${esc(final.answer_value)}<small>${esc(final.answer_unit)}</small></div>
      <div class="answer-refs">${refPills(final)}</div>
    </div>
    ${final.answer ? '<div class="support">' + esc(final.answer) + "</div>" : ""}
    ${final.supporting_materials ? '<div class="support">' + esc(final.supporting_materials) + "</div>" : ""}
    ${final.explanation ? '<div class="explain"><b>Explanation</b><br>' + esc(final.explanation) + "</div>" : ""}
  </div>`;
}

function renderInspector(d) {
  const f = d.flags || {};
  const badges = [];
  badges.push(f.escalated ? '<span class="badge esc">escalated</span>' : '<span class="badge fast">fast path</span>');
  if (f.visual) badges.push('<span class="badge vlm">visual triggered</span>');
  if (f.ast) badges.push('<span class="badge ast">decimal AST</span>');
  if (f.abstained) badges.push('<span class="badge blank">is_blank</span>');
  $("#inspector").innerHTML = `
    <div class="q-head">
      <div class="q-meta"><span class="qidbig">${esc(d.id)}</span>${badges.join("")}</div>
      <h2>${esc(d.question)}</h2>
      <div class="unitline">Unit hint: <code>${esc(d.unit_hint || "is_blank")}</code> · final answer from the submitted cascade CSV</div>
    </div>
    ${renderFinal(d.final)}
    <div class="section-label">Three-stage pipeline inspector <span class="rule"></span></div>
    <div class="pipeline">${renderStageA(d)}${renderStageB(d)}${renderStageC(d)}</div>
    ${renderConsensus(d)}
    <div class="section-label">Token cost of this question <span class="rule"></span></div>
    ${renderTokens(d)}
  `;
}

const EXAMPLES = [
  "How much electricity (in MWh) did GPT-3 training consume?",
  "What was the average global data center PUE in 2023?",
  "Approximately how many premature deaths in 2030 could be caused by scope-2 pollutants from U.S. data centers alone?",
  "What percentage of e-waste has been formally collected and recycled?"
];

function renderLiveError(payload, fallback) {
  const error = (payload && payload.error) || fallback || "Live pipeline failed.";
  const hint = (payload && payload.hint) || "The offline Trace Explorer remains fully available.";
  $("#liveResult").innerHTML = `<div class="errbox"><b>Live query could not complete</b>${esc(error)}<div class="explain" style="margin-top:7px">${esc(hint)}</div></div>`;
}

function renderLiveTrace(trace) {
  if (!trace) return "";
  const steps = (trace.steps || []).map((s) => {
    if (s.kind === "tool") {
      const args = s.arguments ? JSON.stringify(s.arguments) : "";
      return `<div class="tstep tool"><span class="k">tool</span><code>${esc(s.tool)} ${esc(args.slice(0, 260))}</code></div>`;
    }
    if (s.kind === "llm") {
      return `<div class="tstep llm"><span class="k">llm</span><code>${esc(s.stage)} · ${fmt(s.tokens)} tok</code></div>`;
    }
    return `<div class="tstep"><span class="k">${esc(s.kind)}</span><code>${esc(s.detail || "")}</code></div>`;
  }).join("");
  return `<div class="statrow" style="margin-top:12px">
      <span class="stat"><b>${fmt(trace.tokens)}</b> total tokens</span>
      <span class="stat"><b>${fmt(trace.reasoning_tokens)}</b> reasoning</span>
      <span class="stat"><b>${trace.events}</b> events</span>
    </div><div class="tracesteps">${steps || '<div class="note">No trace steps recorded.</div>'}</div>`;
}

function renderLive(payload) {
  if (!payload || payload.ok === false) { renderLiveError(payload); return; }
  const row = payload.row || {};
  const blank = !row.answer_value || String(row.answer_value) === "is_blank";
  const citations = payload.citations || [];
  const pages = citations.filter((c) => c.ref_id && c.page)
    .map((c) => ({ ref_id: c.ref_id, page: c.page, reason: "citation" }));
  $("#liveResult").innerHTML = `
    <div class="statusline ok">${esc(payload.solver)} · verified answer</div>
    <div class="answer-card ${blank ? "blank" : ""}" style="margin:12px 0 0">
      <div class="answer-top">
        <div class="answer-value">${blank ? "Not answerable · is_blank" : esc(row.answer_value) + "<small>" + esc(row.answer_unit || "") + "</small>"}</div>
        <div class="answer-refs">${refPills(row)}</div>
      </div>
      ${row.answer ? '<div class="support">' + esc(row.answer) + "</div>" : ""}
      ${row.supporting_materials ? '<div class="support">' + esc(row.supporting_materials) + "</div>" : ""}
      ${row.explanation ? '<div class="explain"><b>Explanation</b><br>' + esc(row.explanation) + "</div>" : ""}
    </div>
    ${pageGrid(pages.slice(0, 4))}
    ${citations.length ? '<div class="docrow">Verified citations</div>' + citations.map((c) =>
      '<div class="note"><b>' + esc(c.ref_id) + (c.page ? " · p." + esc(c.page) : "") + "</b> " +
      (c.method ? "(" + esc(c.method) + ") " : "") + esc(c.quote || "") + "</div>").join("") : ""}
    ${renderLiveTrace(payload.trace)}
  `;
}

async function askLive() {
  const button = $("#liveAsk");
  const question = $("#liveQuestion").value.trim();
  if (!question) {
    renderLiveError({ error: "Please enter a question first.", hint: "Try one of the example chips below the form." });
    return;
  }
  button.disabled = true;
  $("#liveResult").innerHTML = '<div class="live-status"><span class="spinner"></span>Running the selected pipeline on the server…</div>';
  try {
    const payload = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        solver: $("#liveSolver").value,
        unit_hint: $("#liveUnit").value.trim() || "is_blank"
      })
    });
    renderLive(payload);
  } catch (error) {
    renderLiveError(error.payload, error.message);
  } finally {
    button.disabled = false;
  }
}

function openLightbox(src) {
  if (!src) return;
  $("#lightboxImg").src = src;
  $("#lightbox").hidden = false;
}

function bindEvents() {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b === button));
      $("#view-explorer").hidden = tab !== "explorer";
      $("#view-live").hidden = tab !== "live";
    });
  });
  $("#search").addEventListener("input", (event) => { state.search = event.target.value; renderList(); });
  $("#chips").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-filter]");
    if (!chip) return;
    state.filter = chip.dataset.filter;
    renderChips();
    renderList();
  });
  $("#picks").addEventListener("click", (event) => {
    const pick = event.target.closest("[data-qid]");
    if (pick) selectQuestion(pick.dataset.qid);
  });
  $("#qlist").addEventListener("click", (event) => {
    const item = event.target.closest("[data-qid]");
    if (item) selectQuestion(item.dataset.qid);
  });
  $("#liveAsk").addEventListener("click", askLive);
  $("#liveQuestion").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) askLive();
  });
  $("#liveExamples").innerHTML = EXAMPLES.map((q) => `<button class="example">${esc(q)}</button>`).join("");
  $("#liveExamples").addEventListener("click", (event) => {
    const example = event.target.closest(".example");
    if (example) { $("#liveQuestion").value = example.textContent; $("#liveQuestion").focus(); }
  });
  document.addEventListener("click", (event) => {
    const thumb = event.target.closest(".page-thumb");
    if (thumb) openLightbox(thumb.dataset.src);
  });
  $("#lightbox").addEventListener("click", () => { $("#lightbox").hidden = true; });
  $("#lightboxClose").addEventListener("click", () => { $("#lightbox").hidden = true; });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") $("#lightbox").hidden = true;
  });
}

async function init() {
  bindEvents();
  try {
    const [overview, questions] = await Promise.all([api("/api/overview"), api("/api/questions")]);
    state.overview = overview;
    state.questions = questions.questions || [];
    renderKpis();
    renderChips();
    renderPicks();
    renderList();
    const counts = overview.counts || {};
    setStatus((counts.total || state.questions.length) + " questions · " + (counts.escalated || 0) + " escalated · saved traces loaded", false);
    const first = (overview.picks && overview.picks[0] && overview.picks[0].id) ||
      (state.questions[0] && state.questions[0].id);
    if (first) {
      selectQuestion(first);
    } else {
      $("#inspector").innerHTML = '<div class="empty"><b>No saved traces found</b>Run the cascade pipelines (or download <code>outputs/</code>) to populate the Trace Explorer. The app is otherwise ready.</div>';
      setStatus("Server running · no trace data in outputs/", true);
    }
  } catch (error) {
    setStatus("Could not load demo data", true);
    $("#inspector").innerHTML = '<div class="empty"><b>Demo data could not be loaded</b>' + esc(error.message) + "</div>";
  }
}

init();



</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server


def create_server(root=None, host="127.0.0.1", port=8501):
    """Create (but do not start) the demo HTTP server."""
    data = DemoData(root)

    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "WattBotDemo/1.0"

        def log_message(self, *args):  # keep the presentation terminal clean
            if os.environ.get("WATBOT_DEMO_DEBUG"):
                super().log_message(*args)

        def _send(self, status, body, content_type, extra=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, payload, status=200):
            self._send(status, json.dumps(payload, ensure_ascii=False, default=str),
                       "application/json; charset=utf-8")

        def _not_found(self):
            self._json({"error": "Not found"}, 404)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                if path in {"/", "/index.html"}:
                    self._send(200, INDEX_HTML, "text/html; charset=utf-8")
                elif path == "/api/overview":
                    self._json(data.overview())
                elif path == "/api/questions":
                    self._json({"questions": data.questions()})
                elif path.startswith("/api/question/"):
                    qid = path[len("/api/question/"):]
                    detail = data.question(qid)
                    if detail is None:
                        self._json({"error": f"Unknown question {qid}"}, 404)
                    else:
                        self._json(detail)
                elif path == "/api/page-image":
                    query = parse_qs(parsed.query)
                    ref_id = (query.get("ref_id") or [""])[0]
                    page = (query.get("page") or [""])[0]
                    dpi = (query.get("dpi") or ["110"])[0]
                    try:
                        image = data.page_image(ref_id, page, dpi)
                    except ValueError as error:
                        self._json({"error": str(error)}, 400)
                    except LookupError as error:
                        self._json({"error": str(error)}, 404)
                    else:
                        self._send(200, image, "image/png")
                elif path == "/api/health":
                    self._json({"ok": True})
                else:
                    self._not_found()
            except Exception as error:  # noqa: BLE001 - never kill the server
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/ask":
                self._not_found()
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > 1_000_000:
                self._json({"ok": False, "error": "Invalid request body"}, 400)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._json({"ok": False, "error": "Expected a JSON body"}, 400)
                return
            if not isinstance(payload, dict):
                self._json({"ok": False, "error": "Expected a JSON object"}, 400)
                return
            result, status = data.ask(
                payload.get("question", ""),
                payload.get("solver") or "text",
                payload.get("unit_hint") or BLANK,
            )
            self._json(result, status)

    server = ThreadingHTTPServer((host, port), DemoHandler)
    server.daemon_threads = True
    server.demo_data = data
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="WattBot 2025 interactive demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501,
                        help="TCP port (0 picks an ephemeral port)")
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="project root containing data/ and outputs/")
    args = parser.parse_args(argv)
    server = create_server(args.root, args.host, args.port)
    host, port = server.server_address[:2]
    print(f"WattBot demo running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
