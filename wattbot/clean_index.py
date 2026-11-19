"""Build a cleaner, table-aware index of the competition corpus."""

import contextlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

from .pipeline import load_metadata
from .retrieval import sha256_file, split_text, write_json

ROOT = Path(__file__).resolve().parents[1]

COMPOUND_PREFIXES = {
    "on", "off", "co", "pre", "non", "sub", "cross", "open", "real", "fine",
    "high", "low", "self", "multi", "single", "double", "zero", "end", "use",
    "all", "well", "full", "mid", "post", "over", "under", "inter", "intra",
    "semi", "micro", "macro", "climate", "carbon", "energy", "water", "power",
    "platform", "application", "model", "data", "life", "time", "cost", "state",
    "green", "cloud", "server", "hardware", "software", "resource", "scale",
    "bit", "step", "level", "long", "short", "large", "small", "top", "first",
    "second", "third",
}

CAPTION_START = re.compile(r"^(?:Table|TABLE)\s+[A-Z0-9]+")
CAPTION_SEARCH = re.compile(r"(?:Table|TABLE)\s+[A-Z0-9]+")
MARKDOWN_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|){2,}$")
YEAR = re.compile(r"^(?:19|20)\d{2}$")
NUMBER_CELL = re.compile(
    r"^[<>~≈]?\s*[$€£]?\s*[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:%|‰)?$"
)
NUMBER_TOKEN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
UNIT_PARENTHETICAL = re.compile(
    r"\([^)]*(?:million|billion|thousand|USD|EUR|GBP|kWh|MWh|GWh|TWh|"
    r"CO2|CO₂|mL|L/|per|/|%)[^)]*\)",
    flags=re.IGNORECASE,
)
DISPLACED_DECIMAL = re.compile(r"\d\s+\.(?:\s|\||$)")
LIGATURES = re.compile(r"[\ufb00-\ufb06]")
BROKEN_HYPHEN = re.compile(r"[A-Za-z\ufb00-\ufb06]{2,}-\s+[a-z]{2,}")


def _repair_hyphen(left, right):
    if left.casefold() in COMPOUND_PREFIXES:
        return f"{left}-{right}"
    return f"{left}{right}"


def clean_text(text):
    """Normalize PDF text and repair line-wrap hyphenation."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(
        r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})",
        lambda match: _repair_hyphen(match.group(1), match.group(2)),
        text,
    )
    text = re.sub(
        r"([A-Za-z]{2,})-[ \t]+([a-z]{2,})",
        lambda match: _repair_hyphen(match.group(1), match.group(2)),
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def normalize_tokens_text(text):
    """Normalize text so BM25 query and passage tokens agree."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return re.sub(
        r"([A-Za-z]{2,})-\s+([a-z]{2,})",
        lambda match: _repair_hyphen(match.group(1), match.group(2)),
        text,
    )


def _word_center_inside(word, bbox):
    x0, y0, x1, y1 = bbox
    center_x = (word[0] + word[2]) / 2
    center_y = (word[1] + word[3]) / 2
    return x0 <= center_x <= x1 and y0 <= center_y <= y1


def _group_visual_lines(words):
    lines = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        height = word[3] - word[1]
        placed = False
        for line in lines:
            overlap = min(word[3], line["y1"]) - max(word[1], line["y0"])
            minimum = min(height, line["y1"] - line["y0"])
            if minimum > 0 and overlap > 0.3 * minimum:
                line["words"].append(word)
                line["y0"] = min(line["y0"], word[1])
                line["y1"] = max(line["y1"], word[3])
                placed = True
                break
        if not placed:
            lines.append({"y0": word[1], "y1": word[3], "words": [word]})
    lines.sort(key=lambda line: ((line["y0"] + line["y1"]) / 2, line["y0"]))
    return [
        [word[4] for word in sorted(line["words"], key=lambda item: item[0])]
        for line in lines
    ]


def _cell_text(words, bbox):
    matched = [word for word in words if _word_center_inside(word, bbox)]
    lines = _group_visual_lines(matched)
    return clean_text("\n".join(" ".join(line) for line in lines if line))


def _drop_empty_columns(rows):
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep = [column for column in range(width)
            if any(row[column].strip() for row in padded)]
    return [[row[column] for column in keep] for row in padded]


def _drop_empty_rows(rows):
    return [row for row in rows if any(cell.strip() for cell in row)]


def _is_bogus_table(rows):
    if len(rows) < 2 or (rows and len(rows[0]) < 2):
        return True
    if len(rows[0]) <= 2 and sum(
        1 for r in rows
        if re.match(r"^\[[^\]]{2,15}\]", r[0].strip())
        or "arxiv:" in " ".join(r).casefold()
    ) / len(rows) >= 0.5:
        return True
    prose = 0
    for row in rows:
        populated = [cell for cell in row if cell.strip()]
        if len(populated) == 1 and len(populated[0]) > 220:
            prose += 1
    return prose / len(rows) > 0.7


def _nearby_caption(page, bbox, max_distance=90):
    candidates = []
    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue
        text = clean_text(block[4])
        if not CAPTION_START.match(text):
            continue
        bx0, by0, bx1, by1 = block[:4]
        tx0, ty0, tx1, ty1 = bbox
        if by1 <= ty0:
            distance = ty0 - by1
        elif by0 >= ty1:
            distance = by0 - ty1
        else:
            distance = 0
        if distance <= max_distance:
            candidates.append((distance, text))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1] if candidates else ""


def _primary_tables(page, words):
    tables = []
    for table in page.find_tables().tables:
        raw = [[_cell_text(words, cell) if cell else "" for cell in row.cells]
               for row in table.rows]
        rows = _drop_empty_rows(_drop_empty_columns(raw))
        if _is_bogus_table(rows):
            continue
        tables.append({"caption": _nearby_caption(page, table.bbox),
                       "rows": rows, "borderless": False})
    return tables


def _horizontal_drawings(page):
    count = 0
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is not None and rect.width >= 20 and rect.height <= max(2, rect.width * 0.1):
            count += 1
    return count


def _lines_with_multiple_numbers(page):
    count = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if len(NUMBER_TOKEN.findall(text)) >= 2:
                count += 1
    return count


def _has_table_indicators(page):
    if "table" in page.get_text().casefold():
        return True
    if _horizontal_drawings(page) >= 2:
        return True
    return _lines_with_multiple_numbers(page) >= 4


@contextlib.contextmanager
def _suppress_rapidocr_stdout():
    """Hide RapidOCR chatter while preserving unrelated stdout output."""
    import sys
    temporary = None
    try:
        sys.stdout.flush()
        saved = os.dup(1)
        temporary = tempfile.TemporaryFile(mode="w+b")
        os.dup2(temporary.fileno(), 1)
    except (OSError, AttributeError):
        if temporary is not None:
            temporary.close()
        yield
        return
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        temporary.seek(0)
        captured = temporary.read().decode("utf-8", "replace")
        temporary.close()
        for line in captured.splitlines():
            if re.search(r"rapidocr|document parser messages|install for much improved",
                         line, flags=re.IGNORECASE):
                continue
            print(line)


def _page_markdown(doc, page, cache):
    if cache is None:
        cache = {}
    if page.number in cache:
        return cache[page.number]
    import pymupdf4llm

    with _suppress_rapidocr_stdout():
        chunks = pymupdf4llm.to_markdown(doc, pages=[page.number], page_chunks=True)
    if isinstance(chunks, str):
        markdown = chunks
    else:
        markdown = "\n".join(
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in chunks
        )
    cache[page.number] = markdown
    return markdown


def _is_pipe_row(line):
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _split_pipe_row(line):
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", stripped)]


def _clean_markdown_cell(cell):
    cell = re.sub(r"<br\s*/?>", "\n", cell, flags=re.IGNORECASE)
    cell = re.sub(r"</?sup>", "", cell, flags=re.IGNORECASE)
    cell = re.sub(r"</?sub>", "", cell, flags=re.IGNORECASE)
    cell = re.sub(r"\*\*(.*?)\*\*", r"\1", cell)
    cell = cell.replace("**", "").replace("`", "")
    cell = re.sub(r"(?<!\w)_(.*?)_(?!\w)", r"\1", cell)
    return clean_text(cell)


def _markdown_caption(lines, start, end):
    seen = 0
    for index in range(start - 1, -1, -1):
        line = lines[index].strip()
        if not line:
            continue
        seen += 1
        if CAPTION_SEARCH.search(line):
            line = re.sub(r"</?(?:sup|sub)\b[^>]*>?", "", line, flags=re.IGNORECASE)
            return clean_text(re.sub(r"^[#>*_\s]+|[#>*_\s]+$", "", line))
        if seen >= 3:
            break
    seen = 0
    for index in range(end, len(lines)):
        line = lines[index].strip()
        if not line:
            continue
        seen += 1
        if CAPTION_SEARCH.search(line):
            line = re.sub(r"</?(?:sup|sub)\b[^>]*>?", "", line, flags=re.IGNORECASE)
            return clean_text(re.sub(r"^[#>*_\s]+|[#>*_\s]+$", "", line))
        if seen >= 3:
            break
    return ""


def _parse_markdown_tables(markdown):
    lines = markdown.splitlines()
    tables = []
    index = 0
    while index < len(lines):
        if (MARKDOWN_SEPARATOR.match(lines[index].strip())
                and index > 0 and _is_pipe_row(lines[index - 1])):
            start = index - 1
            end = index + 1
            raw = [[_clean_markdown_cell(cell)
                    for cell in _split_pipe_row(lines[start])]]
            while end < len(lines) and _is_pipe_row(lines[end]):
                raw.append([_clean_markdown_cell(cell)
                            for cell in _split_pipe_row(lines[end])])
                end += 1
            width = max(len(row) for row in raw)
            rows = [row + [""] * (width - len(row)) for row in raw]
            rows = _drop_empty_rows(_drop_empty_columns(rows))
            if not _is_bogus_table(rows):
                tables.append({"caption": _markdown_caption(lines, start, end),
                               "rows": rows, "borderless": True})
            index = end
        else:
            index += 1
    return tables


def _table_signature(rows):
    return tuple(tuple(cell.strip() for cell in row) for row in rows)


def _secondary_tables(doc, page, cache):
    if not _has_table_indicators(page):
        return []
    return _parse_markdown_tables(_page_markdown(doc, page, cache))


def _is_data_number(cell):
    value = cell.strip()
    if not value:
        return False
    return bool(NUMBER_CELL.match(value))


def _row_all_years(row):
    values = [cell.strip() for cell in row if cell.strip()]
    return (len(values) >= 2
            and sum(1 for value in values if YEAR.match(value)) >= 2
            and not any(_is_data_number(value) and not YEAR.match(value)
                        for value in values))


def _row_has_unit(row):
    return any(UNIT_PARENTHETICAL.search(cell) for cell in row)


def _detect_header_rows(rows):
    header_count = 1
    for number in range(1, min(3, len(rows) - 1)):
        row = rows[number]
        values = [cell for cell in row if cell.strip()]
        if not values or _is_section_row(row):
            break
        non_year_numbers = any(_is_data_number(cell) and not YEAR.match(cell.strip())
                               for cell in values)
        year_count = sum(1 for cell in values if YEAR.match(cell.strip()))
        if non_year_numbers or (year_count == 1 and len(values) <= 3):
            break
        previous_populated = len([cell for cell in rows[number - 1] if cell.strip()])
        has_gap = (any(not cell.strip() for cell in row)
                   and previous_populated > len(values))
        has_number = any(_is_data_number(cell) for cell in values)
        has_parenthetical = any("(" in cell and ")" in cell for cell in values)
        later_has_number = any(
            any(_is_data_number(cell) for cell in later)
            for later in rows[number + 1:]
        )
        if (_row_all_years(row) or _row_has_unit(row)
                or (has_gap and not has_number)
                or (not has_number and has_parenthetical and number == 1)
                or (not has_number and later_has_number and number == 1)):
            header_count += 1
        else:
            break
    return rows[:header_count], rows[header_count:]


def _is_section_row(row):
    values = [cell for cell in row if cell.strip()]
    if not values:
        return False
    if len(values) == 1:
        return True
    empty_fraction = sum(1 for cell in row if not cell.strip()) / len(row)
    return empty_fraction >= 0.6 and not any(_is_data_number(cell) for cell in values)


def _format_row(row):
    cells = [cell.strip() for cell in row]
    while cells and not cells[-1]:
        cells.pop()
    return " | ".join(cells)


def _truncate(text, limit=240):
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text.rfind(" ", 0, limit - 3)
    cut = cut if cut > limit // 2 else limit - 3
    return text[:cut].rstrip() + "..."


def _markdown_row(row, columns):
    cells = [cell.strip() for cell in row]
    cells = (cells + [""] * columns)[:columns]
    return "| " + " | ".join(cells) + " |"


def _chunk_block(prefix_lines, rows, max_chars=1300):
    prefix = "\n".join(prefix_lines)
    chunks = []
    current = prefix
    for row in rows:
        if not row.strip():
            continue
        addition = ("\n" if current else "") + row
        if len(current) + len(addition) > max_chars and current != prefix:
            chunks.append(current)
            current = prefix
            addition = ("\n" if current else "") + row
        if len(current) + len(addition) <= max_chars:
            current += addition
            continue
        budget = max(1, max_chars - len(current) - 1)
        pieces = list(split_text(row, budget))
        for number, piece in enumerate(pieces):
            if number == 0 and len(current) + 1 + len(piece) <= max_chars:
                current += ("\n" if current else "") + piece
                continue
            if current.strip():
                chunks.append(current)
            current = (prefix + "\n" + piece) if prefix else piece
    if current.strip():
        chunks.append(current)
    return chunks


def extract_page_tables(doc, page, page_md_cache=None):
    """Return cleaned tables on a page as dicts with caption/rows/borderless."""
    tables = []
    if len(page.get_drawings()) >= 3:
        tables = _primary_tables(page, page.get_text("words"))
    if not tables:
        tables = _secondary_tables(doc, page, page_md_cache)
    return tables


def build_clean_index(metadata, docs, output):
    import pymupdf

    sources = {}
    pages = {}
    nodes = []
    for ref_id, entry in metadata.items():
        path = docs / f"{ref_id}.pdf"
        stats = Counter()
        for key in ("pages", "pages_without_text", "text_nodes", "tables",
                    "table_rows", "table_blocks", "borderless_tables",
                    "table_parse_errors"):
            stats[key] = 0
        with pymupdf.open(path) as document:
            stats["pages"] = len(document)
            page_md_cache = {}
            for page in document:
                page_id = f"{ref_id}:p{page.number + 1}"
                blocks = [
                    clean_text(block[4])
                    for block in page.get_text("blocks", sort=True)
                    if block[6] == 0 and block[4].strip()
                ]
                blocks = [block for block in blocks if block]
                pages[page_id] = "\n".join(blocks)
                if not blocks:
                    stats["pages_without_text"] += 1
                for block in blocks:
                    for passage in split_text(block):
                        nodes.append({
                            "id": f"{page_id}:b{len(nodes)}",
                            "page_id": page_id, "ref_id": ref_id,
                            "page": page.number + 1, "kind": "text",
                            "text": passage,
                        })
                        stats["text_nodes"] += 1
                try:
                    tables = extract_page_tables(document, page, page_md_cache)
                except (RuntimeError, ValueError, TypeError, IndexError,
                        AttributeError, OSError) as error:
                    stats["table_parse_errors"] += 1
                    print(f"Table extraction {page_id}: {type(error).__name__}: {error}",
                          flush=True)
                    tables = []
                seen_tables = set()
                for number, table in enumerate(tables, 1):
                    signature = _table_signature(table["rows"])
                    if signature in seen_tables:
                        continue
                    seen_tables.add(signature)
                    if table["borderless"]:
                        stats["borderless_tables"] += 1
                    stats["tables"] += 1
                    header_rows, body_rows = _detect_header_rows(table["rows"])
                    header_lines = [line for line in map(_format_row, header_rows) if line]
                    caption = _truncate(table.get("caption", ""))
                    entries = []
                    active_section = ""
                    for row in body_rows:
                        if _is_section_row(row):
                            active_section = _format_row(row)
                            continue
                        entries.append((active_section, row))
                    for row_number, (section, row) in enumerate(entries, 1):
                        parts = []
                        if caption:
                            parts.append(caption)
                        parts.extend(header_lines)
                        if section:
                            parts.append(section)
                        parts.append(_format_row(row))
                        passage = "\n".join(part for part in parts if part)
                        nodes.append({
                            "id": f"{page_id}:t{number}:r{row_number}:{len(nodes)}",
                            "page_id": page_id, "ref_id": ref_id,
                            "page": page.number + 1, "kind": "table_row",
                            "text": passage,
                        })
                        stats["table_rows"] += 1
                    columns = max([len(row) for row in table["rows"]] + [2])
                    separator = "| " + " | ".join(["---"] * columns) + " |"
                    block_rows = []
                    for row in body_rows:
                        block_rows.append(_markdown_row(row, columns))
                    prefix = ([caption] if caption else []) + header_lines + [separator]
                    for block_number, chunk in enumerate(
                            _chunk_block(prefix, block_rows), 1):
                        nodes.append({
                            "id": f"{page_id}:t{number}:blk{block_number}:{len(nodes)}",
                            "page_id": page_id, "ref_id": ref_id,
                            "page": page.number + 1, "kind": "table_block",
                            "text": chunk,
                        })
                        stats["table_blocks"] += 1
        if not stats["text_nodes"]:
            raise ValueError(f"No indexable text in {path}")
        sources[ref_id] = {"title": entry["title"], "url": entry["url"],
                           "sha256": sha256_file(path), "stats": dict(stats)}
        print(f"{ref_id}: {stats['pages']} pages, {stats['text_nodes']} text nodes, "
              f"{stats['tables']} tables", flush=True)
    index = {"version": 2, "sources": sources, "pages": pages, "nodes": nodes}
    write_json(output, index)
    return index


def _table_row_body(text):
    lines = text.splitlines()
    if lines and CAPTION_START.match(lines[0].strip()):
        return "\n".join(lines[1:])
    return text


def audit_index(index_path):
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    nodes = data.get("nodes", [])
    text_nodes = [node for node in nodes if node.get("kind") == "text"]
    table_rows = [node for node in nodes if node.get("kind") == "table_row"]
    table_blocks = [node for node in nodes if node.get("kind") == "table_block"]
    stats_tables = 0
    sources_with_tables = 0
    for source in data.get("sources", {}).values():
        count = int((source.get("stats") or {}).get("tables", 0))
        stats_tables += count
        sources_with_tables += 1 if count else 0
    if not stats_tables:
        prefixes = set()
        for node in table_rows + table_blocks:
            match = re.search(r":t(\d+)(?::|$)", node.get("id", ""))
            if match:
                prefixes.add((node.get("page_id"), match.group(1)))
        stats_tables = len(prefixes)
        sources_with_tables = len({
            str(page).split(":p", 1)[0] for page, _ in prefixes
        })
    return {
        "version": data.get("version"),
        "total_nodes": len(nodes),
        "text_nodes": len(text_nodes),
        "table_rows": len(table_rows),
        "table_blocks": len(table_blocks),
        "tables": stats_tables,
        "sources_with_tables": sources_with_tables,
        "displaced_decimal_rows": sum(
            1 for node in table_rows
            if DISPLACED_DECIMAL.search(_table_row_body(node.get("text", "")))
        ),
        "raw_displaced_decimal_rows": sum(
            1 for node in table_rows if DISPLACED_DECIMAL.search(node.get("text", ""))
        ),
        "ligature_text_nodes": sum(
            1 for node in text_nodes if LIGATURES.search(node.get("text", ""))
        ),
        "broken_hyphen_text_nodes": sum(
            1 for node in text_nodes if BROKEN_HYPHEN.search(node.get("text", ""))
        ),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/metadata.csv")
    parser.add_argument("--docs", type=Path, default=ROOT / "data/docs")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/parser-fix/wattbot-index-v2.json")
    parser.add_argument("--compare-v1", type=Path,
                        help="Print a before/after audit against this version 1 index")
    args = parser.parse_args()
    metadata = load_metadata(args.metadata)
    index = build_clean_index(metadata, args.docs, args.output)
    summary = audit_index(args.output)
    if args.compare_v1:
        summary = {"v1": audit_index(args.compare_v1), "v2": summary}
    else:
        summary["output"] = str(args.output)
        summary["nodes_written"] = len(index["nodes"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
