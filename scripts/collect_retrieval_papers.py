#!/usr/bin/env python3
"""Collect retrieval / agentic-RAG papers relevant to the WattBot 2025 task.

KohakuRAG (the winning solution) does not publish a bibliography, so the seed
papers below are the canonical references for the components it actually
implements: hierarchical indexing, dense + BM25 hybrid retrieval, multi-query
planning, dedup/rerank, multimodal embeddings and answer ensembling.

Usage:
    python3 scripts/collect_retrieval_papers.py
    python3 scripts/collect_retrieval_papers.py --download
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "related"
PDF_DIR = OUT_DIR / "pdfs"
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WattBot2025-retrieval-collector/1.0)"}

SEEDS = {
    "rag-foundations": [
        "2005.11401",  # Retrieval-Augmented Generation for Knowledge-Intensive NLP
        "2002.08909",  # REALM
        "2208.03299",  # Atlas
        "2301.12652",  # REPLUG
        "2302.00083",  # In-Context RALM
    ],
    "dense-sparse-retrieval": [
        "2004.04906",  # Dense Passage Retrieval
        "2107.05720",  # SPLADE
        "2004.12832",  # ColBERT
        "2112.01488",  # ColBERTv2
        "1603.09320",  # HNSW
        "2104.08663",  # BEIR
        "2210.07316",  # MTEB
    ],
    "embeddings-chunking": [
        "2205.13147",  # Matryoshka Representation Learning
        "2409.10173",  # jina-embeddings-v3
        "2409.04701",  # Late Chunking
    ],
    "hierarchical-long-context": [
        "2401.18059",  # RAPTOR
        "2307.03172",  # Lost in the Middle
        "2407.16833",  # RAG vs long-context
    ],
    "query-planning": [
        "2212.10496",  # HyDE
        "2305.14283",  # Rewrite-Retrieve-Read
        "2305.03653",  # Query expansion by prompting
        "2310.06117",  # Step-Back Prompting
    ],
    "agentic-retrieval": [
        "2210.03629",  # ReAct
        "2210.03350",  # Self-Ask
        "2212.10509",  # IRCoT
        "2305.06983",  # FLARE
        "2501.09136",  # Agentic RAG survey
        "2503.09516",  # Search-R1
        "2503.05592",  # R1-Searcher
        "2504.03160",  # DeepResearcher
        "2302.04761",  # Toolformer
    ],
    "self-reflective-rag": [
        "2310.11511",  # Self-RAG
        "2401.15884",  # Corrective RAG
        "2403.14403",  # Adaptive-RAG
    ],
    "graph-rag": [
        "2404.16130",  # From Local to Global (GraphRAG)
        "2410.05779",  # LightRAG
        "2405.14831",  # HippoRAG
    ],
    "reranking": [
        "1901.04085",  # Passage Re-ranking with BERT
        "2003.06713",  # monoT5
        "2304.09542",  # Is ChatGPT Good at Search?
    ],
    "multimodal-document-rag": [
        "2407.01449",  # ColPali
        "2410.10594",  # VisRAG
        "2506.18902",  # jina-embeddings-v4
        "2308.13418",  # Nougat
        "2502.18443",  # olmOCR
    ],
    "rag-evaluation": [
        "2309.15217",  # RAGAS
        "2311.09476",  # ARES
        "2408.08067",  # RAGChecker
        "2401.17043",  # CRUD-RAG
    ],
    "surveys-multihop": [
        "2312.10997",  # RAG survey
        "1809.09600",  # HotpotQA
        "2108.00573",  # MuSiQue
        "2407.08223",  # Speculative RAG
    ],
}

EXTRA_SEEDS = {
    "rag-foundations": ["2402.19473", "2407.08223"],
    "dense-sparse-retrieval": [
        "2112.09118",  # Contriever
        "2212.03533",  # E5
        "2309.07597",  # BGE / C-Pack
        "2308.03281",  # GTE
        "2312.06648",  # Dense X Retrieval (propositions)
        "1702.08734",  # FAISS billion-scale search
        "1908.10396",  # ScaNN
    ],
    "hierarchical-long-context": ["2308.14508", "2404.06654"],
    "embeddings-chunking": ["2412.13663"],  # ModernBERT encoder
    "query-planning": ["2305.04091", "2305.10601", "2402.03620", "2210.02406"],
    "agentic-retrieval": [
        "2307.16789",  # ToolLLM
        "2305.15334",  # Gorilla
        "2308.11432",  # LLM autonomous agents survey
        "2309.07864",  # Rise and potential of LLM agents
        "2310.08560",  # MemGPT
        "2304.03442",  # Generative Agents
        "2303.11366",  # Reflexion
        "2305.16291",  # Voyager
        "2312.04511",  # LLM Compiler
        "2308.03688",  # AgentBench
        "2307.13854",  # WebArena
        "2306.06070",  # Mind2Web
        "2403.10081",  # DRAGIN
    ],
    "graph-rag": ["2408.08921"],
    "reranking": [],
    "multimodal-document-rag": [
        "2411.04952",  # M3DocRAG
        "2401.00908",  # DocLLM
        "2204.08387",  # LayoutLMv3
        "2111.15664",  # Donut
        "2007.00398",  # DocVQA
                "2409.18839",  # MinerU
    ],
    "rag-evaluation": [
        "2309.01431",  # RGB
        "2401.15391",  # MultiHop-RAG
        "2407.11005",  # RAGBench
        "2406.04744",  # CRAG benchmark
    ],
    "surveys-multihop": [],
    "context-compression": ["2310.05736", "2310.04408"],  # LLMLingua, RECOMP
    "routing": ["2406.18665"],  # RouteLLM
}

QUERIES = []

FILTER_RE = re.compile(
    r"\b(rag|retrieval|retriever|retrieved|rerank|re-rank|dense passage|bm25|"
    r"hybrid search|embedding|vector search|query rewrit|query expansion|multi-hop|"
    r"agentic|search agent|knowledge base|context window|chunking|colbert|contriever|"
    r"graphrag|self-rag|corrective rag|in-context retrieval)\b"
)
AI_RE = re.compile(
    r"\b(ai|llms?|large language models?|language models?|machine learning|deep learning|"
    r"neural|transformers?|gpus?|inference|question answering|qa)\b"
)
STRONG = ["retrieval", "rag", "rerank", "re-rank", "embedding", "bm25", "dense", "sparse",
          "query", "agentic", "multi-hop", "chunk", "colbert", "graphrag", "context"]


def _get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            wait = 10 * (attempt + 1)
            print(f"  request failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    return ""


def _clean(text: str, prefix: str) -> str:
    text = " ".join(text.split())
    return text[len(prefix):].strip() if text.startswith(prefix) else text


def fetch_abs(arxiv_id: str) -> dict | None:
    html = _get(f"https://arxiv.org/abs/{arxiv_id}")
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1.title")
    abs_el = soup.select_one("blockquote.abstract")
    if not title_el or not abs_el:
        return None
    authors_el = soup.select_one("div.authors")
    date_el = soup.select_one("div.dateline")
    subj_el = soup.select_one("span.primary-subject")
    year_match = re.search(r"(19|20)\d{2}", date_el.get_text(" ", strip=True) if date_el else "")
    return {
        "arxiv_id": arxiv_id,
        "title": _clean(title_el.get_text(" ", strip=True), "Title:"),
        "abstract": _clean(abs_el.get_text(" ", strip=True), "Abstract:"),
        "authors": _clean(authors_el.get_text(" ", strip=True), "Authors:") if authors_el else "",
        "year": year_match.group(0) if year_match else "",
        "primary_category": subj_el.get_text(" ", strip=True) if subj_el else "",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def fetch_ids(ids: list[str]) -> dict[str, dict]:
    cache_path = OUT_DIR / ".seed_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    out = {i: cache[i] for i in ids if i in cache}
    missing = [i for i in ids if i not in cache]
    for i, arxiv_id in enumerate(missing, 1):
        print(f"  seed {i}/{len(missing)}: {arxiv_id}")
        p = fetch_abs(arxiv_id)
        if p:
            out[arxiv_id] = p
            cache[arxiv_id] = p
        time.sleep(2)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def fetch_query(query: str, max_results: int) -> dict[str, dict]:
    url = "https://arxiv.org/search/?" + urllib.parse.urlencode(
        {"query": query, "searchtype": "all", "abstracts": "show", "order": "",
         "size": max_results if max_results in (25, 50, 100, 200) else 50}
    )
    html = _get(url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for li in soup.select("li.arxiv-result"):
        link = li.select_one("p.list-title a")
        title_el = li.select_one("p.title")
        abs_el = li.select_one("span.abstract-full") or li.select_one("span.abstract-short")
        if not link or not title_el or not abs_el:
            continue
        arxiv_id = re.sub(r"v\d+$", "", link.get("href", "").rsplit("/abs/", 1)[-1])
        if not arxiv_id:
            continue
        date_el = li.select_one("p.is-size-7")
        year_match = re.search(r"(19|20)\d{2}", date_el.get_text(" ", strip=True) if date_el else "")
        cat_el = li.select_one("div.tags span.tag")
        authors_el = li.select_one("p.authors")
        out[arxiv_id] = {
            "arxiv_id": arxiv_id,
            "title": " ".join(title_el.get_text(" ", strip=True).split()),
            "abstract": _clean(abs_el.get_text(" ", strip=True), "Abstract:"),
            "authors": _clean(authors_el.get_text(" ", strip=True), "Authors:") if authors_el else "",
            "year": year_match.group(0) if year_match else "",
            "primary_category": cat_el.get_text(" ", strip=True) if cat_el else "",
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
    return out


def score(title: str, abstract: str) -> int:
    t, a = title.lower(), abstract.lower()
    s = 4 * sum(t.count(k) for k in STRONG)
    s += sum(a.count(k) for k in STRONG)
    s += 6 * len(FILTER_RE.findall(t)) + 2 * len(FILTER_RE.findall(a))
    return s


def download(pdf_url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        req = urllib.request.Request(pdf_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
            out.write(resp.read())
        if dest.read_bytes()[:4] != b"%PDF":
            dest.unlink(missing_ok=True)
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  download failed {dest.stem}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-download", type=int, default=130)
    ap.add_argument("--max-results", type=int, default=50)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    papers: dict[str, dict] = {}
    all_seeds = {k: list(v) for k, v in SEEDS.items()}
    for k, v in EXTRA_SEEDS.items():
        all_seeds.setdefault(k, []).extend(v)
    print(f"Fetching {sum(len(v) for v in all_seeds.values())} seed papers")
    seed_meta = fetch_ids([i for ids in all_seeds.values() for i in ids])
    for topic, ids in all_seeds.items():
        for arxiv_id in ids:
            p = seed_meta.get(arxiv_id)
            if not p:
                print(f"  missing seed {arxiv_id} ({topic})")
                continue
            p = dict(p)
            p["topics"] = {topic}
            p["seed"] = True
            p["score"] = score(p["title"], p["abstract"]) + 40
            papers[arxiv_id] = p

    for i, (topic, query) in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {topic}: {query}")
        results = fetch_query(query, args.max_results)
        for arxiv_id, p in results.items():
            low = (p["title"] + " " + p["abstract"]).lower()
            if not FILTER_RE.search(low) or not AI_RE.search(low):
                continue
            p = dict(p)
            if arxiv_id in papers:
                papers[arxiv_id]["topics"].add(topic)
                papers[arxiv_id]["score"] += score(p["title"], p["abstract"]) // 2
                continue
            p["topics"] = {topic}
            p["seed"] = False
            p["score"] = score(p["title"], p["abstract"])
            papers[arxiv_id] = p
        time.sleep(4)

    rows = sorted(papers.values(), key=lambda p: (p["score"], p["year"]), reverse=True)
    for p in rows:
        p["topic"] = "|".join(sorted(p["topics"]))
        p["pdf_path"] = f"data/related/pdfs/{p['arxiv_id']}.pdf"
    fields = ["arxiv_id", "title", "authors", "year", "topic", "score", "seed",
              "primary_category", "url", "pdf_url", "pdf_path"]
    with (OUT_DIR / "papers.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    seeds = [p for p in rows if p["seed"]]
    rest = [p for p in rows if not p["seed"]]
    shortlist = (seeds + rest)[:130]
    with (OUT_DIR / "shortlist.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(shortlist)

    by_topic: dict[str, list[dict]] = {}
    for p in rows:
        for t in p["topics"]:
            by_topic.setdefault(t, []).append(p)
    with (OUT_DIR / "README.md").open("w", encoding="utf-8") as f:
        f.write("# WattBot2025 retrieval / agentic-RAG papers\n\n")
        f.write(
            "KohakuRAG (WattBot 2025 winner) publishes no formal bibliography. "
            "These are curated canonical references for the components its pipeline "
            "implements (hierarchical indexing, dense + BM25 hybrid retrieval, "
            "multi-query planning, reranking, multimodal embeddings, ensembling).\n\n"
        )
        f.write(f"Collected {len(rows)} papers; shortlist {len(shortlist)}; "
                f"{len(seeds)} seed references.\n\n")
        f.write("## KohakuRAG component -> reference area\n\n")
        f.write("| KohakuRAG component | Paper topic |\n|---|---|\n")
        f.write("| Document -> section -> paragraph -> sentence tree | hierarchical-long-context (RAPTOR) |\n")
        f.write("| Sentence embeddings + vector store | dense-sparse-retrieval, embeddings-chunking |\n")
        f.write("| BM25 hybrid search | dense-sparse-retrieval (BM25/SPLADE/BEIR) |\n")
        f.write("| LLM planner / multi-query | query-planning, agentic-retrieval |\n")
        f.write("| Dedup + frequency/score rerank | reranking |\n")
        f.write("| Jina v3/v4 text + image embeddings | embeddings-chunking, multimodal-document-rag |\n")
        f.write("| Answer generation over context | rag-foundations, self-reflective-rag |\n")
        f.write("| Ensemble voting / validation | rag-evaluation |\n\n")
        for t in sorted(by_topic):
            f.write(f"## {t} ({len(by_topic[t])})\n\n")
            for p in by_topic[t][:15]:
                tag = " (seed)" if p["seed"] else ""
                f.write(f"- [{p['title']}]({p['url']}) ({p['year']}){tag}\n")
            f.write("\n")

    print(f"\nCollected {len(rows)} papers -> {OUT_DIR / 'papers.csv'}")
    for t in sorted(by_topic):
        print(f"  {t}: {len(by_topic[t])}")

    if args.download:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        keep = {p["arxiv_id"] for p in shortlist}
        removed = 0
        for p in PDF_DIR.glob("*.pdf"):
            if p.stem not in keep:
                p.unlink()
                removed += 1
        if removed:
            print(f"Removed {removed} PDFs outside the new shortlist")
        todo = shortlist[: args.max_download]
        ok = 0
        for i, p in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {p['arxiv_id']} {p['title'][:70]}")
            if download(p["pdf_url"], PDF_DIR / f"{p['arxiv_id']}.pdf"):
                ok += 1
            time.sleep(3)
        print(f"Downloaded {ok}/{len(todo)} PDFs to {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
