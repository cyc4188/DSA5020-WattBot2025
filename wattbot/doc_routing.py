"""Two-Stage Document-Level Routing and Full-Document Reading for WattBot 2025."""

import argparse
import csv
import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Lock

from scripts.summarize_token_usage import summarize as summarize_token_usage
from wattbot.__main__ import atomic_csv
from wattbot.deepseek import DeepSeek
from wattbot.evaluation import evaluate
from wattbot.pipeline import (
    BLANK, COLUMNS, EVIDENCE_FIELDS, MISSING_VALUES, NUMBER, RANGE, SCALE,
    abstention, canonical_answer_value, compact, load_metadata, model_value,
    normalize_value, parse_list, read_csv, required_text, source_quote, validate_rows,
)
from wattbot.retrieval import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]

RETRY_MARKERS = ("429", "408", "500", "502", "503", "504", "network error", "timed out")
CARD_KEYS = (
    "models", "hardware_and_infra", "datasets_and_benchmarks",
    "metrics_and_topics", "reference_baselines",
)

CARD_PROMPT = """You summarize one research paper for a retrieval router.
Return ONLY a JSON object with keys: summary (string), models, hardware_and_infra,
datasets_and_benchmarks, metrics_and_topics, reference_baselines (arrays of short strings).
Capture every concrete AI model, hardware/cloud infrastructure item, dataset, benchmark,
law, policy, metric, and reference conversion constant or baseline mentioned in the paper
(for example household electricity, annual per-capita CO2e, flights, PUE/WUE).
Keep names verbatim; do not invent facts; use [] when a category is absent."""


def _emit(emit, event, **fields):
    if emit is None:
        return
    for call in (lambda: emit(event, **fields), lambda: emit({"event": event, **fields}), lambda: emit(event)):
        try:
            call()
            return
        except TypeError:
            continue
        except Exception:
            return


def clean_markdown_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"</?(?:u|sup|sub)>", "", text, flags=re.I)
    lines = text.split("\n")
    joined, i = [], 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            if (not cur.startswith("|") and re.search(r"[A-Za-z]{2,}-$", cur)
                    and not nxt.startswith("|") and re.match(r"^[a-z]{2,}", nxt)):
                joined.append(cur[:-1] + nxt)
                i += 2
                continue
        joined.append(cur)
        i += 1
    lines = [re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in joined]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def routing_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    tokens = re.findall(r"[^\W_]+", text)
    for dec in re.findall(r"\d+\.\d+[a-z0-9]*", text):
        tokens.append(dec)
        tokens.append(dec.replace(".", ""))
    for token in re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)+", text):
        joined = re.sub(r"[-/]", "", token)
        if joined != token:
            tokens.append(joined)
    return tokens


def extract_authors_summary(citation: str) -> str:
    raw = re.sub(r"\s+", " ", citation or "").strip()
    if not raw:
        return ""
    cut = re.search(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)", raw)
    raw = (raw[:cut.start()] if cut else raw[:120]).strip().rstrip(".,;:")
    authors = [part.strip() for part in re.split(r",|&|\band\b", raw) if part.strip()]
    if len(authors) >= 2:
        return f"{authors[0]} et al. ({raw})"[:160]
    return raw[:160]


def extract_year_aliases(ref_id: str, meta_row: dict, page1_text: str, pdf_metadata: dict | None = None) -> list[str]:
    base_years = set()
    for value in (meta_row.get("year"), ref_id, meta_row.get("citation")):
        base_years.update(re.findall(r"(?:19|20)\d{2}", str(value or "")))
    aliases = set(base_years)
    if not base_years:
        return sorted(aliases)
    min_y = min(int(year) for year in base_years)
    max_y = max(int(year) for year in base_years)
    for key in ("creationDate", "modDate"):
        for year in re.findall(r"(?:19|20)\d{2}", str((pdf_metadata or {}).get(key) or "")):
            if min_y - 2 <= int(year) <= max_y + 2:
                aliases.add(year)
    marker = re.compile(
        r"arxiv|copyright|©|proceedings|published|accepted|conference|asplos|acl|icml|neurips|ieee|acm|preprint|report|"
        r"january|february|march|april|may|june|july|august|september|october|november|december", re.I)
    lines = (page1_text or "").splitlines()
    total = len(lines)
    for index, line in enumerate(lines):
        if not (index < 15 or index >= total - 10 or marker.search(line)):
            continue
        for year in re.findall(r"(?:19|20)\d{2}", line):
            if min_y - 6 <= int(year) <= max_y + 2:
                aliases.add(year)
    return sorted(aliases)


def _unique(items: list[str], limit: int) -> list[str]:
    seen, out = set(), []
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def extract_headings_and_captions(pages: list[dict]) -> tuple[list[str], list[str]]:
    headings, captions = [], []
    heading_re = re.compile(r"^#{1,4}\s+(.+)")
    caption_re = re.compile(r"^(?:\*\*|__)?(?:Table|Tab\.|Figure|Fig\.)\s+[A-Za-z0-9]+(?:\.\d+)?(?:[:.\s].*)")
    for page in pages:
        for line in (page.get("text") or "").splitlines():
            line = line.strip()
            match = heading_re.match(line)
            if match:
                value = re.sub(r"[*_#]+", "", match.group(1)).strip()
                if value and len(value) <= 100:
                    headings.append(value)
            match = caption_re.match(line)
            if match:
                value = match.group(0).replace("*", "").replace("_", "").strip()
                if value and len(value) <= 160:
                    captions.append(value)
    return _unique(headings, 20), _unique(captions, 20)


class ResilientClient:
    def __init__(self, client, *, max_retries: int = 5, backoff_base: float = 5.0):
        self.client = client
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backend = getattr(client, "backend", "custom")
        self.model = getattr(client, "model", "custom")

    def complete(self, stage, system, data, emit):
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.complete(stage, system, data, emit)
            except RuntimeError as exc:
                message = str(exc).lower()
                if attempt >= self.max_retries or not any(marker in message for marker in RETRY_MARKERS):
                    raise
                delay = self.backoff_base * (2 ** (attempt - 1))
                _emit(emit, "llm_retry_backoff", attempt=attempt, delay=delay, error=str(exc)[:300])
                time.sleep(delay)


def extract_document_markdown(pdf_path: Path) -> tuple[list[dict], dict]:
    import pymupdf
    import pymupdf4llm

    raw_blocks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, use_ocr=False)
    pages = []
    doc = pymupdf.open(pdf_path)
    try:
        pdf_meta = dict(doc.metadata or {})
        for index, block in enumerate(raw_blocks or []):
            raw = block.get("text") or ""
            cleaned = clean_markdown_page(raw)
            try:
                raw_blocks_text = "\n".join(
                    " ".join(b[4].split())
                    for b in doc[index].get_text("blocks", sort=True)
                    if b[6] == 0 and b[4].strip()
                )
            except (IndexError, TypeError, ValueError):
                raw_blocks_text = ""
            if len(raw_blocks_text) > 200 and len(cleaned) < 0.35 * len(raw_blocks_text):
                cleaned = clean_markdown_page(raw_blocks_text)
            pages.append({"page": index + 1, "text": cleaned})
    finally:
        doc.close()
    return pages, pdf_meta


def _metadata_rows(metadata: dict) -> dict:
    items = metadata.items() if isinstance(metadata, dict) else (
        (row.get("ref_id") or row.get("id"), row) for row in metadata or [])
    rows = {}
    for key, row in items:
        if not isinstance(row, dict):
            continue
        ref_id = str(row.get("ref_id") or row.get("id") or key or "").strip()
        if ref_id:
            rows[ref_id] = row
    return rows


def _pdf_path(ref_id: str, row: dict, docs_dir: Path) -> Path | None:
    base = Path(docs_dir)
    for key in ("file", "filename", "file_name", "pdf", "pdf_path", "path", "local_path"):
        value = row.get(key)
        if value:
            candidate = Path(str(value))
            for path in (candidate, base / candidate.name):
                if path.exists():
                    return path
    for pattern in (f"{ref_id}.pdf", f"*{ref_id}*.pdf"):
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def _doc_pages(value) -> list[dict]:
    if isinstance(value, dict):
        value = value.get("pages")
    return [page for page in (value or []) if isinstance(page, dict)]


def build_markdown_corpus(metadata: dict, docs_dir: Path, cache_dir: Path, force: bool = False) -> dict:
    docs_dir, cache_dir = Path(docs_dir), Path(cache_dir)
    cache_path = cache_dir / "markdown_corpus.json"
    rows, pdfs, sources = _metadata_rows(metadata), {}, {}
    for ref_id, row in rows.items():
        path = _pdf_path(ref_id, row, docs_dir)
        if path is None:
            continue
        pdfs[ref_id] = path
        sources[ref_id] = {
            "title": str(row.get("title") or ref_id),
            "url": str(row.get("url") or ""),
            "year": row.get("year"),
            "citation": str(row.get("citation") or ""),
            "sha256": sha256_file(path),
        }
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_sources = cached.get("sources") or {}
            if (cached.get("version") == 1 and set(cached_sources) == set(sources)
                    and all(cached_sources[ref_id].get("url") == sources[ref_id]["url"]
                            and cached_sources[ref_id].get("sha256") == sources[ref_id]["sha256"]
                            for ref_id in sources)):
                return cached
        except (OSError, ValueError, TypeError):
            pass
    docs = {}
    for ref_id, path in pdfs.items():
        pages, pdf_meta = extract_document_markdown(path)
        headings, captions = extract_headings_and_captions(pages)
        page1_text = pages[0]["text"] if pages else ""
        source = sources[ref_id]
        source.update({
            "pages": len(pages),
            "total_chars": sum(len(page["text"]) for page in pages),
            "authors": extract_authors_summary(source["citation"]),
            "year_aliases": extract_year_aliases(ref_id, rows[ref_id], page1_text, pdf_meta),
            "headings": headings,
            "table_figure_titles": captions,
        })
        docs[ref_id] = pages
    corpus = {"version": 1, "sources": sources, "docs": docs}
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, corpus)
    return corpus


def sample_document_for_card(pages: list[dict], max_chars: int = 28000) -> str:
    by_page = {int(page.get("page") or index + 1): page.get("text") or ""
               for index, page in enumerate(pages)}
    if not by_page:
        return ""
    selected = set(sorted(by_page)[:3])
    selected.update(page_no for page_no, text in by_page.items() if "|---|" in text)
    selected.update(sorted(by_page)[-2:])
    chunks, used = [], 0
    for page_no in sorted(selected):
        block = f"[p. {page_no}]\n{by_page[page_no]}"
        if used + len(block) + 2 > max_chars:
            remaining = max_chars - used - 2
            if remaining > 100:
                chunks.append(block[:remaining].rsplit("\n", 1)[0].rstrip())
            break
        chunks.append(block)
        used += len(block) + 2
    return "\n\n".join(chunks).strip()


def corpus_fingerprint(corpus: dict) -> str:
    sources, docs = corpus.get("sources") or {}, corpus.get("docs") or {}
    payload = []
    for ref_id in sorted(sources):
        pages = docs.get(ref_id) or []
        if isinstance(pages, dict):
            pages = pages.get("pages") or []
        payload.append([ref_id, sources[ref_id].get("sha256", ""), len(pages)])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _as_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value or [] if str(item).strip()] if isinstance(value, list) else []


def _clean_card_items(value, limit: int = 40) -> list[str]:
    items, seen = [], set()
    for item in _as_list(value):
        item = re.sub(r"\s+", " ", item).strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            items.append(item[:120])
        if len(items) >= limit:
            break
    return items


def _base_card(ref_id: str, source: dict, pages: list[dict]) -> dict:
    headings = _as_list(source.get("headings"))[:20]
    captions = _as_list(source.get("table_figure_titles"))[:20]
    title = str(source.get("title") or ref_id)
    authors = str(source.get("authors") or "")
    summary = ". ".join(part for part in (title, authors, str(source.get("year") or "")) if part)
    if headings:
        summary += ". Sections: " + "; ".join(headings[:8])
    page1_text = str(pages[0].get("text") or "") if pages else ""
    if page1_text:
        excerpt = re.sub(r"\s+", " ", page1_text).strip()[:280]
        if excerpt:
            summary += ". Excerpt: " + excerpt
    return {
        "ref_id": ref_id, "title": title, "url": source.get("url", ""),
        "year": source.get("year"), "authors": authors,
        "year_aliases": _as_list(source.get("year_aliases")),
        "headings": headings, "table_figure_titles": captions,
        "page_count": len(pages), "pages": len(pages),
        "total_chars": source.get("total_chars", sum(len(p.get("text", "")) for p in pages)),
        "summary": summary[:1200],
        "models": [], "hardware_and_infra": [], "datasets_and_benchmarks": [],
        "metrics_and_topics": [], "reference_baselines": [], "llm_enriched": False,
    }


def _parse_card_response(response) -> dict:
    if isinstance(response, dict):
        data = response
    else:
        text = re.sub(r"^```(?:json)?|```$", "", str(response or "").strip(), flags=re.I | re.M).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in card response")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("card response is not an object")
    parsed = {key: _clean_card_items(data.get(key)) for key in CARD_KEYS}
    summary = str(data.get("summary") or "").strip()
    if summary:
        parsed["summary"] = summary[:1500]
    return parsed


def _append_trace(trace_path: Path, payload: dict) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _card_emitter(lock: Lock, trace_path: Path, ref_id: str):
    def emit(event, **fields):
        with lock:
            _append_trace(trace_path, {"ref_id": ref_id, "event": event, **fields})
    return emit


def _write_token_summary(cache_dir: Path, trace_path: Path) -> None:
    try:
        summary = summarize_token_usage(Path(trace_path))
    except Exception:
        summary = {}
    try:
        write_json(Path(cache_dir) / "card-build-token-usage.json", summary)
    except Exception:
        pass


def build_document_cards(corpus: dict, cache_dir: Path, client=None, use_llm: bool = True,
                         workers: int = 2, force: bool = False) -> dict:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "document_cards.json"
    fingerprint = corpus_fingerprint(corpus)
    want_llm = bool(use_llm and client is not None)
    sources, docs = corpus.get("sources") or {}, corpus.get("docs") or {}
    if cache_path.exists() and not force:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (cached.get("version") == 1 and cached.get("corpus_sha256") == fingerprint
                    and bool(cached.get("llm_enriched")) == want_llm
                    and set(sources) <= set(cached.get("cards") or {})):
                return cached
        except (OSError, ValueError, TypeError):
            pass
    cards = {ref_id: _base_card(ref_id, source, _doc_pages(docs.get(ref_id)))
             for ref_id, source in sources.items()}
    trace_path = cache_dir / "card-build-trace.jsonl"
    if want_llm:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if trace_path.exists():
            trace_path.unlink()
        lock = Lock()

        def work(ref_id, source, pages):
            started = time.time()
            try:
                data = {"ref_id": ref_id, "title": source.get("title"),
                        "year": source.get("year"),
                        "authors": source.get("authors"),
                        "year_aliases": _as_list(source.get("year_aliases")),
                        "headings": _as_list(source.get("headings")),
                        "table_figure_titles": _as_list(source.get("table_figure_titles")),
                        "sample": sample_document_for_card(pages)}
                response = client.complete("document_card", CARD_PROMPT, data,
                                           _card_emitter(lock, trace_path, ref_id))
                return ref_id, _parse_card_response(response), None, time.time() - started
            except Exception as exc:
                return ref_id, None, f"{type(exc).__name__}: {exc}"[:300], time.time() - started

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(work, ref_id, source, _doc_pages(docs.get(ref_id)))
                       for ref_id, source in sources.items()]
            for future in as_completed(futures):
                ref_id, parsed, error, elapsed = future.result()
                with lock:
                    _append_trace(trace_path, {"ref_id": ref_id, "event": "card_done",
                                               "status": "error" if error else "ok",
                                               "error": error, "seconds": round(elapsed, 3)})
                if parsed:
                    cards[ref_id].update(parsed)
                    cards[ref_id]["llm_enriched"] = True
        _write_token_summary(cache_dir, trace_path)
    result = {"version": 1, "corpus_sha256": fingerprint, "llm_enriched": want_llm, "cards": cards}
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, result)
    return result


ROUTE_PROMPT = """You route a question to the most relevant documents in a research corpus.
You receive a JSON payload with:
- question: the user's question.
- unit_hint: the expected answer unit (or "is_blank" when unspecified).
- bm25_top_candidates: full-text search hits, each with a ref_id, score, top page
  numbers, and a short snippet. These are lexical hints, not the final answer.
- document_cards: one compact card per corpus document, with fields ref_id, title,
  authors, year_aliases, summary, models, hardware_and_infra, datasets_and_benchmarks,
  metrics_and_topics, reference_baselines, and table_figure_titles.

Match documents carefully:
- Match authors and year_aliases, not just titles. A paper's ref_id year can differ
  from its arXiv/preprint or conference publication year listed in year_aliases.
- Use semantic understanding of all document_cards AND the bm25_top_candidates snippets.
- Detect multi-document questions and set needs_multiple_docs to true. Examples:
  converting a model's energy or emissions from Paper A into a reference baseline unit
  defined in Paper B (such as U.S. household-years, average American life days, or
  flights), or comparing and synthesizing claims across two papers. In such cases
  select ALL required source documents.
- Prefer the smallest set of documents that fully answers the question. If multiple
  documents are genuinely required, include each with an appropriate role.

Return ONLY a JSON object, with no prose or code fences, in exactly this shape:
{"needs_multiple_docs": true/false, "reasoning": "brief explanation", "selected_docs": [{"ref_id": "...", "role": "primary|secondary|conversion_baseline|comparison", "reason": "..."}], "search_keywords": ["short keyword or phrase", ...]}"""


def _build_bm25_index(texts: list[str]) -> tuple[dict, int, list[float]]:
    postings = defaultdict(dict)
    lengths = []
    for index, text in enumerate(texts):
        tokens = routing_tokens(text)
        lengths.append(len(tokens))
        for term, count in Counter(tokens).items():
            postings[term][index] = count
    n_docs = len(texts)
    avgdl = (sum(lengths) / n_docs) if n_docs else 0.0
    norms = [(length / avgdl if avgdl else 0.0) for length in lengths]
    return dict(postings), n_docs, norms


class HybridDocumentRouter:
    def __init__(self, corpus: dict, cards: dict):
        self.sources = corpus.get("sources") or {}
        self.docs = corpus.get("docs") or {}
        self.cards = cards.get("cards") or {}
        self.k1 = 1.2
        self.b = 0.75

        self.page_entries: list[dict] = []
        self.doc_page_indices: dict[str, list[int]] = defaultdict(list)
        self.page_text_by_key: dict[tuple[str, int], str] = {}
        page_texts: list[str] = []
        for ref_id in sorted(self.sources):
            source = self.sources.get(ref_id) or {}
            title = str(source.get("title") or ref_id)
            authors = str(source.get("authors") or "")
            years = " ".join(_as_list(source.get("year_aliases")))
            for index, page in enumerate(_doc_pages(self.docs.get(ref_id))):
                page_no = int(page.get("page") or index + 1)
                text = page.get("text") or ""
                entry_index = len(self.page_entries)
                self.page_entries.append({"ref_id": ref_id, "page": page_no, "text": text})
                self.doc_page_indices[ref_id].append(entry_index)
                self.page_text_by_key[(ref_id, page_no)] = text
                page_texts.append(f"{ref_id} {title} {authors} {years}\n{text}")
        self.page_postings, self.page_n_docs, self.page_norms = _build_bm25_index(page_texts)

        self.card_ref_ids = sorted(self.cards)
        card_texts: list[str] = []
        for ref_id in self.card_ref_ids:
            card = self.cards.get(ref_id) or {}
            fields = [ref_id, str(card.get("title") or ""), str(card.get("authors") or "")]
            fields.extend(_as_list(card.get("year_aliases")))
            fields.extend(_as_list(card.get("headings")))
            fields.append(str(card.get("summary") or ""))
            for key in ("models", "hardware_and_infra", "datasets_and_benchmarks",
                        "metrics_and_topics", "reference_baselines", "table_figure_titles"):
                fields.extend(_as_list(card.get(key)))
            card_texts.append(" ".join(fields))
        self.card_postings, self.card_n_docs, self.card_norms = _build_bm25_index(card_texts)

        self.compact_cards: list[dict] = []
        for ref_id in self.card_ref_ids:
            card = self.cards.get(ref_id) or {}
            self.compact_cards.append({
                "ref_id": ref_id,
                "title": str(card.get("title") or ref_id),
                "authors": str(card.get("authors") or ""),
                "year_aliases": _as_list(card.get("year_aliases"))[:8],
                "pages": card.get("pages") or card.get("page_count") or 0,
                "summary": str(card.get("summary") or "")[:450],
                "models": _clean_card_items(card.get("models"), 20),
                "hardware_and_infra": _clean_card_items(card.get("hardware_and_infra"), 15),
                "datasets_and_benchmarks": _clean_card_items(card.get("datasets_and_benchmarks"), 15),
                "metrics_and_topics": _clean_card_items(card.get("metrics_and_topics"), 20),
                "reference_baselines": _clean_card_items(card.get("reference_baselines"), 15),
                "table_figure_titles": _clean_card_items(card.get("table_figure_titles"), 8),
            })
        self.cards_sha256 = hashlib.sha256(
            json.dumps(self.compact_cards, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _term_idf(self, term, postings, n_docs) -> float:
        df = len(postings.get(term) or {})
        if df == 0 or n_docs <= 0:
            return 0.0
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def _bm25_scores(self, query: str, postings, n_docs: int, norms: list[float],
                     allowed: set[int] | None = None) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(routing_tokens(query)):
            term_postings = postings.get(term)
            if not term_postings:
                continue
            idf = self._term_idf(term, postings, n_docs)
            for doc_index, tf in term_postings.items():
                if allowed is not None and doc_index not in allowed:
                    continue
                norm = norms[doc_index] if doc_index < len(norms) else 0.0
                denominator = tf + self.k1 * (1 - self.b + self.b * norm)
                if denominator <= 0:
                    continue
                scores[doc_index] += idf * tf * (self.k1 + 1) / denominator
        return dict(scores)

    def _best_snippet(self, page_text: str, query_terms: list[str], max_chars: int = 240) -> str:
        text = re.sub(r"\s+", " ", page_text or "").strip()
        if not text:
            return ""
        lowered = text.casefold()
        best_term, best_pos, best_idf = None, -1, -1.0
        for term in dict.fromkeys(term for term in query_terms if term):
            pos = lowered.find(term.casefold())
            if pos < 0:
                continue
            idf = self._term_idf(term, self.page_postings, self.page_n_docs)
            if idf > best_idf:
                best_idf, best_term, best_pos = idf, term, pos
        if best_pos < 0:
            return text[:max_chars].strip()
        start = max(0, best_pos - max_chars // 2)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        return text[start:end].strip()

    def bm25_rank_documents(self, queries: list[str], top_n: int = 32) -> list[dict]:
        page_scores: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        card_scores: dict[str, float] = defaultdict(float)
        query_terms: list[str] = []
        seen_terms: set[str] = set()
        for query in queries or []:
            if not query:
                continue
            for term in routing_tokens(query):
                if term not in seen_terms:
                    seen_terms.add(term)
                    query_terms.append(term)
            for entry_index, score in self._bm25_scores(
                    query, self.page_postings, self.page_n_docs, self.page_norms).items():
                entry = self.page_entries[entry_index]
                page_scores[entry["ref_id"]][entry["page"]] += score
            if self.card_ref_ids:
                for entry_index, score in self._bm25_scores(
                        query, self.card_postings, self.card_n_docs, self.card_norms).items():
                    card_scores[self.card_ref_ids[entry_index]] += score
        results = []
        for ref_id in set(self.sources) | set(card_scores):
            ranked_pages = sorted(page_scores.get(ref_id, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            top_pages = [page for page, _ in ranked_pages[:3]]
            weighted = sum(weight * score for weight, (_, score) in zip((1.0, 0.5, 0.25), ranked_pages[:3]))
            score = weighted + 0.35 * card_scores.get(ref_id, 0.0)
            snippet = ""
            if top_pages:
                snippet = self._best_snippet(
                    self.page_text_by_key.get((ref_id, top_pages[0]), ""), query_terms)
            results.append({"ref_id": ref_id, "score": round(score, 4),
                            "top_pages": top_pages, "snippet": snippet})
        results.sort(key=lambda row: (-row["score"], row["ref_id"]))
        return results[:top_n]

    def _mask_document_cards_value(self, value):
        if isinstance(value, str):
            if '"document_cards"' not in value:
                return value
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                return value
            return json.dumps(self._mask_document_cards_value(parsed), ensure_ascii=False)
        if isinstance(value, dict):
            masked = {}
            for key, item in value.items():
                if key == "document_cards":
                    masked[key] = {
                        "document_cards_sha256": self.cards_sha256,
                        "document_cards_count": len(self.compact_cards),
                    }
                else:
                    masked[key] = self._mask_document_cards_value(item)
            return masked
        if isinstance(value, list):
            return [self._mask_document_cards_value(item) for item in value]
        return value

    def route(self, question: dict, client, emit, top_docs: int = 3) -> dict:
        bm25_ranked = self.bm25_rank_documents([question["question"]], top_n=len(self.sources))
        payload = {
            "question": question["question"],
            "unit_hint": question.get("answer_unit") or BLANK,
            "top_docs": top_docs,
            "bm25_top_candidates": bm25_ranked[:8],
            "document_cards": self.compact_cards,
        }

        def masked_emit(event, **fields):
            if event == "llm_request":
                fields = self._mask_document_cards_value(fields)
            _emit(emit, event, **fields)

        try:
            reply = client.complete("route", ROUTE_PROMPT, payload, masked_emit)
        except ValueError as exc:
            _emit(emit, "invalid_route", error=str(exc)[:300])
            reply = {}
        if isinstance(reply, dict):
            data = reply
        else:
            text = re.sub(r"^```(?:json)?|```$", "", str(reply or "").strip(), flags=re.I | re.M).strip()
            start, end = text.find("{"), text.rfind("}")
            try:
                data = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
            except (ValueError, TypeError):
                data = {}
            if not isinstance(data, dict):
                data = {}
            if not data:
                _emit(emit, "invalid_route", error="no JSON object in route response")

        llm_docs: list[str] = []
        seen_docs: set[str] = set()
        for item in data.get("selected_docs") or []:
            ref_id = str(item.get("ref_id") if isinstance(item, dict) else item or "").strip()
            if ref_id in self.sources and ref_id not in seen_docs:
                seen_docs.add(ref_id)
                llm_docs.append(ref_id)
        needs_multiple_docs = bool(data.get("needs_multiple_docs"))
        raw_keywords = data.get("search_keywords")
        if isinstance(raw_keywords, str):
            raw_keywords = [raw_keywords]
        search_keywords = [str(keyword).strip() for keyword in (raw_keywords or [])
                           if str(keyword).strip()][:8]
        reasoning = str(data.get("reasoning") or "")[:1000]

        final_docs = list(llm_docs[:top_docs])
        bm25_top1 = bm25_ranked[0]["ref_id"] if bm25_ranked and bm25_ranked[0]["score"] > 0 else None
        if bm25_top1 and bm25_top1 not in final_docs:
            if len(final_docs) < top_docs:
                final_docs.append(bm25_top1)
            elif not needs_multiple_docs and top_docs >= 3:
                final_docs = [final_docs[0], final_docs[1], bm25_top1]
        target = min(top_docs, len(self.sources))
        for row in bm25_ranked:
            if len(final_docs) >= target:
                break
            if row["ref_id"] not in final_docs:
                final_docs.append(row["ref_id"])

        bm25_top_docs = [row["ref_id"] for row in bm25_ranked[:8]]
        _emit(emit, "doc_routing", bm25_top_docs=bm25_top_docs, llm_docs=llm_docs,
              final_docs=final_docs, needs_multiple_docs=needs_multiple_docs,
              search_keywords=search_keywords, reasoning=reasoning)
        return {
            "bm25_top_docs": bm25_top_docs,
            "llm_docs": llm_docs,
            "final_docs": final_docs,
            "needs_multiple_docs": needs_multiple_docs,
            "search_keywords": search_keywords,
            "reasoning": reasoning,
            "bm25_ranked": bm25_ranked,
        }

    def select_pages_for_doc(self, ref_id: str, queries: list[str], max_full_pages: int = 35,
                             max_full_chars: int = 130_000, page_budget: int = 22) -> list[int]:
        pages = _doc_pages(self.docs.get(ref_id))
        page_map = {}
        for index, page in enumerate(pages):
            page_map[int(page.get("page") or index + 1)] = page.get("text") or ""
        if not page_map:
            return []
        total_chars = sum(len(text) for text in page_map.values())
        if len(page_map) <= max_full_pages and total_chars <= max_full_chars:
            return sorted(page_map)
        allowed = set(self.doc_page_indices.get(ref_id) or [])
        scores: dict[int, float] = defaultdict(float)
        for query in queries or []:
            for entry_index, score in self._bm25_scores(
                    query, self.page_postings, self.page_n_docs, self.page_norms,
                    allowed=allowed).items():
                scores[entry_index] += score
        valid_pages = set(page_map)
        chosen = {page for page in (1, 2) if page in valid_pages}
        for entry_index, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(chosen) >= page_budget:
                break
            entry = self.page_entries[entry_index]
            for candidate in (entry["page"], entry["page"] - 1, entry["page"] + 1):
                if candidate in valid_pages and candidate not in chosen:
                    chosen.add(candidate)
                    if len(chosen) >= page_budget:
                        break
        if not chosen:
            chosen.update(sorted(valid_pages)[:page_budget])
        return sorted(chosen)

    def build_context(self, selected_ref_ids: list[str], queries: list[str],
                      total_char_budget: int = 360_000, max_full_pages: int = 35,
                      max_full_chars: int = 130_000,
                      page_budget: int = 22) -> tuple[str, dict[str, dict[int, str]], dict]:
        selected: list[str] = []
        for ref_id in selected_ref_ids:
            if ref_id in self.sources and ref_id not in selected:
                selected.append(ref_id)
        full_maps: dict[str, dict[int, str]] = {}
        for ref_id in selected:
            full_maps[ref_id] = {
                int(page.get("page") or index + 1): page.get("text") or ""
                for index, page in enumerate(_doc_pages(self.docs.get(ref_id)))
            }

        def render(pages_by_ref: dict[str, list[int]]):
            supplied_docs: dict[str, dict[int, str]] = {}
            sections: list[str] = []
            for ref_id in selected:
                source = self.sources.get(ref_id) or {}
                page_map = full_maps[ref_id]
                all_pages = sorted(page_map)
                if not all_pages:
                    supplied_docs[ref_id] = {}
                    continue
                page_numbers = sorted(set(pages_by_ref.get(ref_id) or []) & set(page_map))
                supplied_docs[ref_id] = {number: page_map[number] for number in page_numbers}
                windowed = not (len(all_pages) <= max_full_pages
                                and sum(len(text) for text in page_map.values()) <= max_full_chars)
                title = str(source.get("title") or ref_id)
                authors = str(source.get("authors") or "")
                years = ", ".join(_as_list(source.get("year_aliases"))) or str(source.get("year") or "")
                parts = [f"=== DOCUMENT {ref_id} | Title: {title} | Authors: {authors} | Years: {years} ==="]
                previous = None
                for number in page_numbers:
                    if windowed:
                        if previous is None and number > all_pages[0]:
                            parts.append(f"[... pages {all_pages[0]}-{number - 1} omitted ...]")
                        elif previous is not None and number - previous > 1:
                            parts.append(f"[... pages {previous + 1}-{number - 1} omitted ...]")
                    parts.append(f"[{ref_id}, p. {number}]\n{page_map[number]}")
                    previous = number
                if windowed and previous is not None and previous < all_pages[-1]:
                    parts.append(f"[... pages {previous + 1}-{all_pages[-1]} omitted ...]")
                sections.append("\n\n".join(parts))
            return "\n\n".join(sections), supplied_docs

        plan = {
            ref_id: self.select_pages_for_doc(
                ref_id, queries, max_full_pages, max_full_chars, page_budget)
            for ref_id in selected
        }
        context_text, supplied_docs = render(plan)
        truncated = False
        if len(context_text) > total_char_budget:
            for smaller_budget in (14, 10):
                if len(context_text) <= total_char_budget:
                    break
                plan = {
                    ref_id: self.select_pages_for_doc(
                        ref_id, queries, max_full_pages, max_full_chars,
                        min(page_budget, smaller_budget))
                    for ref_id in selected
                }
                context_text, supplied_docs = render(plan)
                truncated = True
        while len(context_text) > total_char_budget:
            removed = False
            for ref_id in reversed(selected):
                numbers = sorted(plan.get(ref_id) or [])
                if not numbers:
                    continue
                plan[ref_id] = numbers[:-1]
                removed = True
                break
            if not removed:
                break
            context_text, supplied_docs = render(plan)
            truncated = True
        if len(context_text) > total_char_budget:
            context_text = context_text[:total_char_budget]
            truncated = True
        stats = {
            "documents": len(supplied_docs),
            "pages": sum(len(pages) for pages in supplied_docs.values()),
            "chars": len(context_text),
            "char_budget": total_char_budget,
            "truncated": truncated,
        }
        return context_text, supplied_docs, stats


READ_PROMPT = """You are a strict, general-purpose research reader. Answer the user's question using ONLY the supplied documents. Do not use outside knowledge or guess.

Rules:
- Prefer exact primary/appendix TABLE figures and units over rounded prose summaries.
- Keep full arithmetic precision: report at least 6 significant digits and do not round prematurely, unless the source itself states a rounded value.
- If a unit_hint is provided (not is_blank), convert the answer to match it and set answer_unit to that unit_hint; otherwise set answer_unit to is_blank when no unit applies.
- True/False answers: set answer_value to "1" for True and "0" for False, with answer_unit set to is_blank.
- Categorical or named-entity answers: give the concise canonical name, omitting redundant trailing category nouns already present in the question.
- Ranges: format as [low,high] with no spaces; do not include commas, currency, percent, or approximation symbols in the numeric answer_value.
- Cite the minimal sufficient ref_id set: usually 1 source, or 2+ only when combining distinct facts or constants across papers.

Return ONLY JSON in one of these forms:
{"answerable": true, "answer": "...", "answer_value": "...", "answer_unit": "...", "explanation": "...", "evidence": [{"ref_id": "...", "page": 1, "quote": "..."}]}
{"answerable": false, "reason": "..."}"""

_MD_TABLE = str.maketrans({c: " " for c in "|*_#`"})

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "which", "with", "we", "our", "they", "these",
    "those", "can", "may", "but", "not", "also", "than", "then", "into", "such",
}


def _strip_md(text):
    if not text:
        return ""
    return " ".join(text.translate(_MD_TABLE).split())


def _compact(text):
    return "".join((text or "").lower().split())


def _quote_match_in_page(quote, page_text):
    if not quote or not page_text:
        return None
    if quote in page_text:
        return quote.strip()
    if compact(quote) and compact(quote) in compact(page_text):
        return source_quote(quote, page_text) or quote.strip()
    try:
        return source_quote(quote, page_text) or source_quote(_strip_md(quote), page_text)
    except Exception:
        return None


def _content_tokens(text):
    out = []
    for tok in re.findall(r"[a-z0-9][a-z0-9.,%/+-]*", (text or "").lower()):
        tok = tok.strip(".,%/+/-")
        if len(tok) >= 2 and tok not in _STOPWORDS:
            out.append(tok)
    return out


def _best_overlap_snippet(quote, page_text):
    qset = {t for t in routing_tokens(quote) if len(t) >= 2 and t not in _STOPWORDS}
    if not qset or not page_text:
        return 0.0, 0, False, (page_text or "")[:300]
    page_tokens = set(routing_tokens(page_text))
    matched = qset & page_tokens
    ratio = len(matched) / float(len(qset))
    has_num = any(any(ch.isdigit() for ch in t) and len(t) >= 2 for t in matched)
    lines = [ln for ln in page_text.splitlines() if ln.strip()]
    best_count, best_snippet = 0, ""
    for i in range(len(lines)):
        for span in (1, 2, 3):
            window = lines[i:i + span]
            if not window:
                continue
            snippet = " ".join(window).strip()
            if len(snippet) > 300:
                snippet = snippet[:300]
            count = len(qset & set(routing_tokens(snippet)))
            if count > best_count:
                best_count, best_snippet = count, snippet
    if not best_snippet:
        matching = [ln.strip() for ln in lines if qset & set(routing_tokens(ln))]
        best_snippet = " ... ".join(matching[:2])
    if not best_snippet:
        best_snippet = (page_text or "")[:300]
    return ratio, len(matched), has_num, best_snippet


def verify_and_recover_citations(reply, supplied_docs, sources, emit):
    blank = {"ref_id": "is_blank", "ref_url": "is_blank", "supporting_materials": "is_blank"}
    if not isinstance(reply, dict):
        return blank, []
    verified = []
    seen = set()

    def _record(ref_id, page, quote, method, event):
        key = (ref_id, int(page) if isinstance(page, int) else page, compact(quote))
        if key in seen:
            return
        seen.add(key)
        verified.append({"ref_id": ref_id, "page": page, "quote": quote, "method": method})
        _emit(emit, event, ref_id=ref_id, page=page, method=method)

    for ev in (reply.get("evidence") or []):
        if not isinstance(ev, dict):
            continue
        ref_id = ev.get("ref_id")
        quote = ev.get("quote") or ""
        if not quote.strip():
            continue
        try:
            page = int(ev.get("page"))
        except (TypeError, ValueError):
            page = None

        if ref_id not in supplied_docs or ref_id not in sources:
            hits = []
            for rid, pages in supplied_docs.items():
                for pno in sorted(pages):
                    matched_quote = _quote_match_in_page(quote, pages[pno])
                    if matched_quote:
                        hits.append((rid, pno, matched_quote))
            unique_refs = {rid for rid, _pno, _q in hits if rid in sources}
            if len(unique_refs) == 1:
                rid = next(iter(unique_refs))
                _rid, pno, matched_quote = next(h for h in hits if h[0] == rid)
                _record(rid, pno, matched_quote, "unique_quote_match", "citation_recovered")
            else:
                _emit(emit, "citation_dropped", ref_id=ref_id, page=page, reason="unknown_ref")
            continue

        pages = supplied_docs.get(ref_id) or {}
        if page is not None and page in pages:
            matched_quote = _quote_match_in_page(quote, pages[page])
            if matched_quote:
                _record(ref_id, page, matched_quote, "exact", "citation_verified")
                continue

        if page is not None:
            for adj in (page - 1, page + 1):
                if adj in pages:
                    matched_quote = _quote_match_in_page(quote, pages[adj])
                    if matched_quote:
                        _record(ref_id, adj, matched_quote, "adjacent_page", "quote_recovered")
                        break
            else:
                adj = None
            if adj is not None and any(v["ref_id"] == ref_id and v["page"] == adj for v in verified):
                continue

        found = False
        for pno in sorted(pages):
            matched_quote = _quote_match_in_page(quote, pages[pno])
            if matched_quote:
                _record(ref_id, pno, matched_quote, "doc_wide_page", "quote_recovered")
                found = True
                break
        if found:
            continue

        if page is not None and page in pages:
            ratio, matched, has_num, snippet = _best_overlap_snippet(quote, pages[page])
            if ratio >= 0.50 and (matched >= 2 or has_num):
                _record(ref_id, page, snippet, "page_token_overlap", "quote_recovered")
                continue

        best_page, best_ratio, best_matched, best_snip = None, 0.0, 0, ""
        for pno in sorted(pages):
            ratio, matched, _hn, snippet = _best_overlap_snippet(quote, pages[pno])
            if ratio > best_ratio or (ratio == best_ratio and matched > best_matched):
                best_page, best_ratio, best_matched, best_snip = pno, ratio, matched, snippet
        if best_page is not None and best_ratio >= 0.60 and best_matched >= 2:
            _record(ref_id, best_page, best_snip, "page_token_overlap", "quote_recovered")
            continue

        _emit(emit, "citation_dropped", ref_id=ref_id, page=page, reason="no_support")

    if not verified:
        return blank, []
    refs = {}
    for item in verified:
        refs[item["ref_id"]] = (sources.get(item["ref_id"]) or {}).get("url", "is_blank")
    return {
        "ref_id": repr(list(refs)),
        "ref_url": repr(list(refs.values())),
        "supporting_materials": "\n".join(
            f"[{item['ref_id']}, p. {item['page']}] {item['quote']}" for item in verified
        ),
    }, verified


def _format_decimal(value: Decimal) -> str:
    s = format(value, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in {"", "-0", "+0"} else s


def _unit_canonical(unit: str) -> str:
    key = compact(unit or "").casefold()
    mapping = {
        "%": "percent", "pct": "percent", "percentage": "percent",
        "$": "usd", "usd": "usd", "dollar": "usd", "dollars": "usd", "us dollars": "usd",
        "l": "liters", "liter": "liters", "liters": "liters", "litre": "liters", "litres": "liters",
        "tco2e": "tco2e", "metric tons co2e": "tco2e", "tonnes co2e": "tco2e", "tons co2e": "tco2e",
        "x": "multiplier", "times": "multiplier", "fold": "multiplier", "multiplier": "multiplier",
        "lb": "lbs", "lbs": "lbs", "pound": "lbs", "pounds": "lbs",
        "w": "w", "watt": "w", "watts": "w",
    }
    return mapping.get(key, key)


def _clean_raw_answer_value(raw_value, unit_hint: str):
    if not isinstance(raw_value, str):
        return raw_value
    s = compact(raw_value)
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    hint = _unit_canonical(unit_hint)
    if hint == "percent" and s.endswith(("%", "％")) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?[%％]", s):
        s = s[:-1]
    if hint == "usd" and s.startswith("$") and re.fullmatch(r"\$[+-]?\d+(?:\.\d+)?", s):
        s = s[1:]
    return s


def normalize_doc_routing_fields(question: dict, row: dict) -> dict:
    value = _clean_raw_answer_value(row.get("answer_value"), question.get("answer_unit") or BLANK)
    value = canonical_answer_value(question["question"], value)
    hint = (question.get("answer_unit") or BLANK).strip()
    unit = compact(str(row.get("answer_unit") or BLANK))

    if str(value).casefold() in MISSING_VALUES and hint != BLANK:
        return abstention(question)
    if value == BLANK:
        return {**row, "answer_value": BLANK}
    if hint == BLANK:
        return {
            **row,
            "answer_value": normalize_value(value),
            "answer_unit": BLANK if unit.casefold() in {"", "none", "n/a", "is_blank"} else unit,
        }
    if _unit_canonical(unit) == _unit_canonical(hint):
        return {**row, "answer_value": normalize_value(value), "answer_unit": hint}

    m1 = re.fullmatch(
        r"(K|M|B|T|thousands?|millions?|billions?|trillions?)\s+(?:of\s+)?(.+)", unit, re.I
    )
    m2 = re.fullmatch(r"\$\s*(K|M|B|T|thousands?|millions?|billions?|trillions?)", unit, re.I)
    key = None
    if m1 and _unit_canonical(m1.group(2)) == _unit_canonical(hint):
        key = m1.group(1)
    elif m2 and _unit_canonical(hint) == "usd":
        key = m2.group(1)
    factor = None
    if key:
        lookup = key.upper() if len(key) == 1 else key.lower().rstrip("s")
        factor = SCALE.get(lookup)
    if factor is not None:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                low = _format_decimal(Decimal(str(value[0])) * factor)
                high = _format_decimal(Decimal(str(value[1])) * factor)
                return {**row, "answer_value": f"[{low},{high}]", "answer_unit": hint}
            except Exception:
                pass
        range_match = re.fullmatch(
            r"\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]", str(value).strip()
        )
        if range_match:
            try:
                low = _format_decimal(Decimal(range_match.group(1)) * factor)
                high = _format_decimal(Decimal(range_match.group(2)) * factor)
                return {**row, "answer_value": f"[{low},{high}]", "answer_unit": hint}
            except Exception:
                pass
        try:
            scaled = _format_decimal(Decimal(str(value)) * factor)
            return {**row, "answer_value": normalize_value(scaled), "answer_unit": hint}
        except Exception:
            pass
    return {**row, "answer_value": normalize_value(value), "answer_unit": unit}


def solve_question(question: dict, router, client, emit, top_docs: int = 3) -> dict:
    route_info = router.route(question, client, emit, top_docs=top_docs)
    base_queries = [question["question"]] + route_info.get("search_keywords", [])

    def _read_docs(selected_docs, queries, feedback=None):
        context_text, supplied_docs, ctx_stats = router.build_context(selected_docs, queries)
        ctx_sha = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
        mask = {
            "documents_sha256": ctx_sha,
            "selected_docs": list(selected_docs),
            "context_stats": ctx_stats,
        }

        def _mask_value(obj):
            if isinstance(obj, str):
                if '"documents"' in obj:
                    try:
                        parsed = json.loads(obj)
                    except (ValueError, TypeError):
                        return obj
                    return json.dumps(_mask_value(parsed), ensure_ascii=False)
                return obj
            if isinstance(obj, dict):
                return {
                    k: (dict(mask) if k == "documents" else _mask_value(v))
                    for k, v in obj.items()
                }
            if isinstance(obj, list):
                return [_mask_value(v) for v in obj]
            return obj

        def masked_emit(event, *args, **kwargs):
            if event == "llm_request":
                masked_args = [_mask_value(a) for a in args]
                masked_kwargs = {k: _mask_value(v) for k, v in kwargs.items()}
                return _emit(emit, event, *masked_args, **masked_kwargs)
            return emit(event, *args, **kwargs)

        reply = client.complete(
            "read_answer",
            READ_PROMPT,
            {
                "documents": context_text,
                "question": question["question"],
                "unit_hint": question.get("answer_unit") or BLANK,
                "previous_feedback": feedback,
            },
            masked_emit,
        )
        if not isinstance(reply, dict):
            raise ValueError("Expected JSON object reply")
        if reply.get("answerable") is False:
            return "abstained", str(reply.get("reason") or "Insufficient evidence."), None
        if reply.get("answerable") is not True:
            raise ValueError("Expected boolean answerable")
        raw_val = _clean_raw_answer_value(reply.get("answer_value"), question.get("answer_unit") or BLANK)
        val = model_value(raw_val)
        answer_text = str(reply.get("answer") or "").strip() or val
        answer_unit = str(reply.get("answer_unit") or "").strip() or (question.get("answer_unit") or BLANK)
        explanation = required_text(reply, "explanation")
        cit_fields, verified = verify_and_recover_citations(reply, supplied_docs, router.sources, emit)
        if not verified:
            raise ValueError("No cited evidence could be verified against supplied documents")
        row = {
            **abstention(question),
            "answer": answer_text,
            "answer_value": val,
            "answer_unit": answer_unit,
            "explanation": explanation,
            **cit_fields,
        }
        canonical = normalize_doc_routing_fields(question, row)
        if canonical != row:
            _emit(emit, "candidate_normalized", before=row, after=canonical)
        row = canonical
        if row["answer_value"] == BLANK:
            return "abstained", "Missing requested value", None
        return "answered", None, row

    final_docs = route_info["final_docs"]
    try:
        try:
            status, reason, row = _read_docs(final_docs, base_queries)
        except ValueError as err:
            _emit(emit, "invalid_output", attempt=1, detail=str(err))
            status, reason, row = _read_docs(final_docs, base_queries, feedback=str(err))
    except ValueError as err:
        _emit(emit, "decision", decision="abstained_invalid_output", detail=str(err))
        return abstention(question)

    if status == "answered":
        _emit(emit, "decision", decision="answered", attempt=1, selected_docs=final_docs)
        return row

    _emit(emit, "stage2_abstained", attempt=1, reason=reason, selected_docs=final_docs)
    unexamined = [
        r["ref_id"]
        for r in route_info.get("bm25_ranked", [])
        if r["score"] > 0 and r["ref_id"] not in final_docs
    ]
    oversized = any(len(_doc_pages(router.docs.get(rid))) > 35 for rid in final_docs)
    if unexamined or oversized:
        if unexamined:
            primary = [final_docs[0]] if final_docs else []
            expanded_docs = primary + [
                rid for rid in unexamined[:max(1, top_docs - 1)] if rid not in primary
            ]
        else:
            expanded_docs = list(final_docs)
        deduped = []
        seen_docs = set()
        for rid in expanded_docs:
            if rid not in seen_docs:
                seen_docs.add(rid)
                deduped.append(rid)
        expanded_docs = deduped[:top_docs] or list(final_docs)
        expanded_queries = base_queries + ([reason] if reason else [])
        _emit(emit, "adaptive_expansion_retry", expanded_docs=expanded_docs, reason=reason)
        feedback = (
            f"Previous attempt abstained: {reason}. "
            "Check newly supplied documents/pages carefully."
        )
        try:
            status2, reason2, row2 = _read_docs(expanded_docs, expanded_queries, feedback=feedback)
        except ValueError as err:
            _emit(emit, "invalid_output", attempt=2, detail=str(err))
            status2, reason2, row2 = "abstained", str(err), None
        if status2 == "answered":
            _emit(
                emit,
                "decision",
                decision="answered_after_expansion",
                attempt=2,
                selected_docs=expanded_docs,
            )
            return row2
        reason = reason2 or reason
    _emit(emit, "decision", decision="abstained", reason=reason)
    return abstention(question)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_router_recall(routing_records: list[dict], gold_rows: list[dict]) -> dict:
    gold_by_id = {row["id"]: row for row in gold_rows}
    usable: list[tuple[dict, dict]] = []
    for record in routing_records:
        gold = record.get("gold") or gold_by_id.get(record.get("question_id") or record.get("id"))
        if not gold or not parse_list(gold.get("ref_id")):
            continue
        usable.append((record, gold))

    strategies: dict = {}
    for name, field in (
        ("bm25_only", "bm25_top_docs"),
        ("llm_only", "llm_docs"),
        ("hybrid", "final_docs"),
    ):
        per_k: dict = {}
        for k in (1, 2, 3):
            recalls: list[float] = []
            coverage: list[float] = []
            exact_hits = 0
            missed: list[dict] = []
            for record, gold in usable:
                gold_set = set(parse_list(gold["ref_id"]))
                preds = list(record.get(field) or [])
                top = preds[:k]
                recalls.append(len(gold_set & set(top)) / len(gold_set))
                hit = 1.0 if gold_set <= set(top) else 0.0
                coverage.append(hit)
                exact_hits += int(hit)
                if not (gold_set <= set(preds[:3])):
                    missed.append(
                        {
                            "id": gold.get("id"),
                            "gold_refs": sorted(gold_set),
                            "pred_top3": preds[:3],
                        }
                    )
            per_k[f"top_{k}"] = {
                "mean_recall": sum(recalls) / len(recalls) if recalls else 0.0,
                "exact_coverage": sum(coverage) / len(coverage) if coverage else 0.0,
                "exact_hits": exact_hits,
                "missed_at_3": missed,
            }
        strategies[name] = per_k
    return {"answerable_questions": len(usable), "strategies": strategies}


def _has_gold_labels(rows: list[dict]) -> bool:
    if not rows:
        return False
    for row in rows:
        if not isinstance(row.get("answer_value"), str) or not row["answer_value"].strip():
            return False
        if not isinstance(row.get("ref_id"), str) or not row["ref_id"].strip():
            return False
    return True


def _make_emit(question_id: str, handle, lock: Lock):
    def emit(event, *args, **kwargs):
        entry = {"timestamp": _now(), "event": event, "question_id": question_id}
        if kwargs:
            entry.update(kwargs)
        with lock:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        return None

    return emit


def parse_question_ids(value) -> set[str] | None:
    """Normalize a question-ID selector.

    Accepts ``None``, an iterable of IDs, a comma-separated string, or a path to
    a text/CSV file containing IDs.  Returns ``None`` when no selector is given.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (set, frozenset)):
        return {str(item).strip() for item in value if str(item).strip()} or None
    if isinstance(value, (list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()} or None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_file():
        raw = path.read_text(encoding="utf-8")
        if path.suffix.casefold() == ".csv":
            items: list[str] = []
            for index, row in enumerate(csv.reader(raw.splitlines())):
                if not row:
                    continue
                first = row[0].strip()
                if index == 0 and first.casefold() in {"id", "question_id", "qid"}:
                    continue
                items.append(first)
        else:
            items = [part for line in raw.splitlines() for part in line.split(",")]
        result = {item.strip().strip("\"'") for item in items if item.strip()}
        return result or None
    result = {part.strip() for part in text.split(",") if part.strip()}
    return result or None


def filter_questions(questions: list[dict], question_ids=None) -> list[dict]:
    """Restrict ``questions`` to ``question_ids`` while preserving input order."""
    ids = parse_question_ids(question_ids)
    if ids is None:
        return list(questions)
    known = {str(question["id"]) for question in questions}
    missing = sorted(ids - known)
    if missing:
        raise ValueError(f"Requested question IDs are not present: {missing[:5]}")
    return [question for question in questions if str(question["id"]) in ids]


def run_doc_routing_subset(
    questions: list[dict],
    metadata: dict,
    *,
    corpus: dict | None = None,
    cards: dict | None = None,
    client=None,
    output_dir: Path,
    docs: Path = ROOT / "data/docs",
    cache_dir: Path = ROOT / "outputs/doc-routing/corpus_cache",
    top_docs: int = 3,
    backend: str = "hkustgz",
    workers: int = 2,
    resume: bool = False,
    question_ids=None,
    gold_rows: list[dict] | None = None,
    questions_path: Path | None = None,
    no_llm_cards: bool = False,
    force_cards: bool = False,
) -> tuple[list[dict], dict]:
    """Run document routing on a (possibly filtered) question subset.

    This is the programmatic entry point used by the online Pure-RAG cascade.
    It builds the markdown corpus and document cards when they are not supplied,
    reuses existing on-disk artifacts, and returns the ordered prediction rows
    together with a JSON-serializable summary.
    """
    questions = filter_questions(questions, question_ids)
    output_dir = Path(output_dir)
    if not questions:
        return [], {
            "rows": 0,
            "answered": 0,
            "abstained": 0,
            "token_usage": {},
            "question_ids": [],
        }

    if corpus is None:
        corpus = build_markdown_corpus(metadata, Path(docs), Path(cache_dir), force=False)
    if cards is None:
        cards = build_document_cards(
            corpus,
            Path(cache_dir),
            client=client,
            use_llm=not no_llm_cards,
            workers=workers,
            force=force_cards,
        )
    if client is None:
        client = ResilientClient(DeepSeek(backend, timeout=180, transport_attempts=4))

    router = HybridDocumentRouter(corpus, cards)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    trace_path = output_dir / "trace.jsonl"
    predictions_path = output_dir / "predictions.csv"
    token_usage_path = output_dir / "token-usage.json"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.json"

    manifest = {
        "method": "doc_routing",
        "questions_file": str(questions_path) if questions_path else None,
        "questions_sha256": hashlib.sha256(
            (questions_path.read_bytes() if questions_path else json.dumps(
                [str(question["id"]) for question in questions], sort_keys=True
            ).encode("utf-8"))
        ).hexdigest(),
        "questions_count": len(questions),
        "top_docs": top_docs,
        "backend": backend,
        "model": getattr(client, "model", None) or backend,
        "corpus_sha256": hashlib.sha256(
            json.dumps(corpus, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "cards_sha256": hashlib.sha256(
            json.dumps(cards, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    completed_rows: dict = {}
    if resume:
        if not config_path.exists():
            raise FileNotFoundError(str(config_path))
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("config.json does not match current manifest")
        if trace_path.exists():
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("event") == "row" and isinstance(entry.get("row"), dict):
                    completed_rows[entry["question_id"]] = entry["row"]
    else:
        if trace_path.exists():
            raise FileExistsError(str(trace_path))
        config_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pending = [question for question in questions if question["id"] not in completed_rows]
    lock = Lock()
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": _now(),
                    "event": "run_resume" if resume else "run_start",
                    "question_id": None,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        handle.flush()

        def _run_one(question):
            emit = _make_emit(question["id"], handle, lock)
            row = solve_question(question, router, client, emit, top_docs=top_docs)
            validate_rows(COLUMNS, [row], [question], metadata)
            emit("row", row=row)
            return question["id"], row

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for qid, row in pool.map(_run_one, pending):
                completed_rows[qid] = row

    ordered = [completed_rows[question["id"]] for question in questions]
    validate_rows(COLUMNS, ordered, questions, metadata)
    atomic_csv(predictions_path, ordered)

    token_usage = summarize_token_usage(trace_path)
    token_usage_path.write_text(
        json.dumps(token_usage, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    answered = sum(
        1 for row in ordered if str(row.get("answer_value", "")).strip() not in {"", BLANK}
    )
    summary = {
        "rows": len(ordered),
        "answered": answered,
        "abstained": len(ordered) - answered,
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "token_usage": token_usage,
        "question_ids": [question["id"] for question in questions],
    }

    if gold_rows and _has_gold_labels(gold_rows):
        gold_by_qid = {row["id"]: row for row in gold_rows}
        ordered_gold = [gold_by_qid.get(question["id"], {}) for question in questions]
        if all(ordered_gold):
            metrics = evaluate(ordered, ordered_gold)
            records_by_qid: dict = {}
            if trace_path.exists():
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    if entry.get("event") == "doc_routing":
                        qid = entry.get("question_id")
                        payload = {
                            key: value
                            for key, value in entry.items()
                            if key not in {"timestamp", "event", "question_id"}
                        }
                        records_by_qid[qid] = {
                            "question_id": qid,
                            "gold": gold_by_qid.get(qid),
                            **payload,
                        }
            routing_records = list(records_by_qid.values())
            metrics["stage1_router_recall"] = evaluate_router_recall(routing_records, ordered_gold)
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary.update(metrics)

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"timestamp": _now(), "event": "run_complete", "question_id": None},
                ensure_ascii=False,
            )
            + "\n"
        )
    return ordered, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--metadata", type=Path, default=ROOT / "data/raw/metadata.csv")
    shared.add_argument("--docs", type=Path, default=ROOT / "data/docs")
    shared.add_argument(
        "--cache-dir", type=Path, default=ROOT / "outputs/doc-routing/corpus_cache"
    )
    shared.add_argument("--backend", choices=("hkustgz", "openrouter"), default="hkustgz")
    shared.add_argument("--workers", type=int, default=2)
    shared.add_argument("--no-llm-cards", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", parents=[shared])
    build_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval-router", parents=[shared])
    eval_parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw/train_QA.csv"
    )
    eval_parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/doc-routing/router-eval"
    )
    eval_parser.add_argument("--top-docs", type=int, default=3)
    eval_parser.add_argument("--limit", type=int)
    eval_parser.add_argument("--force-cards", action="store_true")
    eval_parser.add_argument(
        "--question-ids",
        help="Comma-separated question IDs, or a text/CSV file of IDs",
    )

    run_parser = subparsers.add_parser("run", parents=[shared])
    run_parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw/train_QA.csv"
    )
    run_parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/doc-routing/train"
    )
    run_parser.add_argument("--top-docs", type=int, default=3)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--force-cards", action="store_true")
    run_parser.add_argument(
        "--question-ids",
        help="Comma-separated question IDs, or a text/CSV file of IDs",
    )

    args = parser.parse_args()

    metadata = load_metadata(args.metadata)
    corpus = build_markdown_corpus(
        metadata, args.docs, args.cache_dir, force=getattr(args, "force", False)
    )

    client = None
    if args.command != "build" or not args.no_llm_cards:
        client = ResilientClient(DeepSeek(args.backend, timeout=180, transport_attempts=4))

    cards = build_document_cards(
        corpus,
        args.cache_dir,
        client=client,
        use_llm=not args.no_llm_cards,
        workers=args.workers,
        force=getattr(args, "force", False) or getattr(args, "force_cards", False),
    )

    if args.command == "build":
        print(
            json.dumps(
                {
                    "documents": len(corpus.get("sources") or {}),
                    "pages": sum(
                        int(s.get("pages") or 0)
                        for s in (corpus.get("sources") or {}).values()
                    ),
                    "total_chars": sum(
                        int(s.get("total_chars") or 0)
                        for s in (corpus.get("sources") or {}).values()
                    ),
                    "llm_enriched": sum(
                        1
                        for c in (cards.get("cards") or {}).values()
                        if c.get("llm_enriched")
                    ),
                    "cache_dir": str(args.cache_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    router = HybridDocumentRouter(corpus, cards)
    _, source_rows = read_csv(args.questions)
    active_source = source_rows[: args.limit] if args.limit else source_rows
    selected_ids = parse_question_ids(getattr(args, "question_ids", None))
    if selected_ids is not None:
        present = {row["id"] for row in active_source}
        missing = sorted(selected_ids - present)
        if missing:
            raise ValueError(f"Requested question IDs are not present: {missing[:5]}")
        active_source = [row for row in active_source if row["id"] in selected_ids]
    questions = [
        {
            "id": row["id"],
            "question": row["question"],
            "answer_unit": row.get("answer_unit", BLANK),
        }
        for row in active_source
    ]
    gold_by_qid = {row["id"]: row for row in active_source}

    if args.command == "eval-router":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "router_trace.jsonl"
        lock = Lock()
        routing_records: list[dict] = []

        with trace_path.open("w", encoding="utf-8") as handle:
            def _eval_one(question):
                emit = _make_emit(question["id"], handle, lock)
                info = router.route(question, client, emit, top_docs=args.top_docs)
                emit("row", routed=info.get("final_docs"))
                return question["id"], dict(info or {})

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for qid, info in pool.map(_eval_one, questions):
                    routing_records.append(
                        {"question_id": qid, "gold": gold_by_qid.get(qid), **info}
                    )

        token_usage = summarize_token_usage(trace_path)
        (output_dir / "token-usage.json").write_text(
            json.dumps(token_usage, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {"questions": len(questions), "token_usage": token_usage}
        if _has_gold_labels(active_source):
            metrics = evaluate_router_recall(routing_records, active_source)
            (output_dir / "router_metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary["router_metrics"] = metrics
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        ordered, summary = run_doc_routing_subset(
            questions,
            metadata,
            corpus=corpus,
            cards=cards,
            client=client,
            output_dir=args.output_dir,
            top_docs=args.top_docs,
            backend=args.backend,
            workers=args.workers,
            resume=args.resume,
            question_ids=None,
            gold_rows=active_source,
            questions_path=args.questions,
            no_llm_cards=args.no_llm_cards,
            force_cards=args.force_cards,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

