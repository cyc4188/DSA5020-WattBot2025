"""Build and search a page-aware index of the competition corpus."""

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def split_text(text, max_chars=1300):
    """Keep short paragraphs intact and break oversized ones at word boundaries."""
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        yield remaining[:cut]
        remaining = remaining[cut:].lstrip()
    if remaining:
        yield remaining


def build_index(metadata, docs, output):
    import pymupdf

    sources = {}
    pages = {}
    nodes = []
    for ref_id, entry in metadata.items():
        path = docs / f"{ref_id}.pdf"
        stats = Counter()
        with pymupdf.open(path) as document:
            stats["pages"] = len(document)
            for page in document:
                page_id = f"{ref_id}:p{page.number + 1}"
                blocks = [
                    " ".join(block[4].split())
                    for block in page.get_text("blocks", sort=True)
                    if block[6] == 0 and block[4].strip()
                ]
                pages[page_id] = "\n".join(blocks)
                if not blocks:
                    stats["pages_without_text"] += 1
                for block in blocks:
                    for passage in split_text(block):
                        nodes.append({"id": f"{page_id}:b{len(nodes)}", "page_id": page_id,
                                      "ref_id": ref_id, "page": page.number + 1,
                                      "kind": "text", "text": passage})
                        stats["text_nodes"] += 1
                if not blocks or len(page.get_drawings()) < 3:
                    continue
                try:
                    tables = page.find_tables().tables
                    for number, table in enumerate(tables, 1):
                        rows = [
                            " | ".join(" ".join((cell or "").split()) for cell in row)
                            for row in table.extract()
                        ]
                        rows = [row for row in rows if row.replace("|", "").strip()]
                        if len(rows) < 2:
                            continue
                        stats["tables"] += 1
                        for row_number, row in enumerate(rows[1:], 1):
                            text = f"{rows[0]}\n{row}"
                            for passage in split_text(text):
                                nodes.append({
                                    "id": f"{page_id}:t{number}:r{row_number}:{len(nodes)}",
                                    "page_id": page_id, "ref_id": ref_id,
                                    "page": page.number + 1, "kind": "table_row",
                                    "text": passage,
                                })
                except (RuntimeError, ValueError, TypeError, IndexError) as error:
                    stats["table_parse_errors"] += 1
                    print(f"Table extraction {page_id}: {type(error).__name__}: {error}", flush=True)
        if not stats["text_nodes"]:
            raise ValueError(f"No indexable text in {path}")
        sources[ref_id] = {"title": entry["title"], "url": entry["url"],
                           "sha256": sha256_file(path), "stats": dict(stats)}
        print(f"{ref_id}: {stats['pages']} pages, {stats['text_nodes']} text nodes, "
              f"{stats['tables']} tables", flush=True)
    write_json(output, {"version": 1, "sources": sources, "pages": pages, "nodes": nodes})
    return len(nodes)


def tokens(text):
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"([A-Za-z]{2,})-\s+([a-z]{2,})", r"\1\2", text)
    return re.findall(r"[^\W_]+", text.casefold())


class SearchIndex:
    def __init__(self, path, dense=False):
        self.digest = sha256_file(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") not in (1, 2):
            raise ValueError("Rebuild the index for this version of the pipeline")
        self.version = data["version"]
        self.sources = data["sources"]
        self.pages = data["pages"]
        self.nodes = data["nodes"]
        self.postings = defaultdict(list)
        self.dense = None
        lengths = []
        for i, node in enumerate(self.nodes):
            terms = tokens(node["text"])
            lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self.postings[term].append((i, frequency))
        average = sum(lengths) / len(lengths)
        self.norm = [1.2 * (0.25 + 0.75 * length / average) for length in lengths]
        if dense:
            self.enable_dense(path)

    def enable_dense(self, path):
        import numpy as np
        from fastembed import TextEmbedding

        self.dense = TextEmbedding("BAAI/bge-small-en-v1.5")
        cache = path.with_name(f"{path.stem}.bge-small.{self.digest}.npy")
        if cache.exists():
            self.vectors = np.load(cache, allow_pickle=False)
        else:
            self.vectors = np.asarray(list(self.dense.passage_embed(
                [node["text"] for node in self.nodes]
            )), dtype=np.float32)
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, self.vectors, allow_pickle=False)
            temporary.replace(cache)

    def bm25(self, query, limit):
        scores = defaultdict(float)
        for term in set(tokens(query)):
            postings = self.postings.get(term, [])
            idf = math.log1p((len(self.nodes) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings:
                scores[index] += idf * (frequency * 2.2) / (frequency + self.norm[index])
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def search(self, queries, top_k):
        fused = defaultdict(float)
        hits = defaultdict(list)
        for query in queries:
            rankings = {"bm25": self.bm25(query, top_k * 3)}
            if self.dense is not None:
                vector = next(self.dense.query_embed(query))
                scores = self.vectors @ vector
                rankings["dense"] = [
                    (int(i), float(scores[i])) for i in scores.argsort()[::-1][:top_k * 3]
                ]
            for method, ranking in rankings.items():
                for rank, (index, score) in enumerate(ranking, 1):
                    fused[index] += 1 / (60 + rank)
                    hits[index].append({"query": query, "method": method, "rank": rank,
                                        "score": round(score, 6)})
        ranked = sorted(fused, key=lambda index: (-fused[index], index))
        selected = []
        seen = set()
        for index in ranked:
            node = self.nodes[index]
            identity = (node["page_id"], node["text"].casefold())
            if identity in seen:
                continue
            seen.add(identity)
            selected.append({**node, "title": self.sources[node["ref_id"]]["title"],
                             "score": round(fused[index], 6), "hits": hits[index]})
            if len(selected) == top_k:
                break
        return selected

    def expand(self, selected, max_chars):
        """Include nearby text from the parent page under a per-passage budget."""
        expanded = []
        for node in selected:
            text = node["text"]
            parent = self.pages[node["page_id"]]
            position = parent.find(text)
            if position >= 0:
                spare = max(0, max_chars - len(text))
                start = max(0, position - spare // 2)
                text = parent[start:start + max_chars]
            elif node.get("kind") in ("table_row", "table_block"):
                text = node["text"]
            expanded.append({**node, "matched_text": node["text"], "text": text})
        return expanded
