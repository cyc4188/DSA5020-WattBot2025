import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from wattbot.multihop_rag import (
    ContextualSearchIndex,
    benchmark_retrieval,
    clean_text,
    clean_tokens,
    normalize_submission_fields,
    parse_page_chunks,
    prune_candidate_citations,
    resolve_candidate,
    safe_eval_decimal,
    solve_multihop,
)
from scripts.make_submission import BLANK, COLUMNS
from wattbot.pipeline import validate_rows


def build_index(tmpdir):
    sources = {
        "docA": {"title": "Title A", "url": "https://a.org"},
        "docB": {"title": "Title B", "url": "https://b.org"},
    }
    pages = {
        "docA:p1": "Alpha intro about solar capacity and efficiency.",
        "docA:p2": "Alpha continuation with more capacity detail.",
        "docB:p1": "Beta baseline results for comparison.",
    }
    nodes = [
        {
            "id": "docA:p1:n0",
            "page_id": "docA:p1",
            "ref_id": "docA",
            "page": 1,
            "kind": "text",
            "section": "Intro",
            "text": "Alpha intro about solar capacity and efficiency.",
            "context_prefix": "[docA | Title A | p.1 | Intro]",
            "search_text": "[docA | Title A | p.1 | Intro]\nAlpha intro about solar capacity and efficiency.",
        },
        {
            "id": "docA:p1:n1",
            "page_id": "docA:p1",
            "ref_id": "docA",
            "page": 1,
            "kind": "table",
            "section": "Intro",
            "text": "Year | Capacity\n2020 | 100",
            "context_prefix": "[docA | Title A | p.1 | Intro]",
            "search_text": "[docA | Title A | p.1 | Intro]\nYear | Capacity\n2020 | 100",
        },
        {
            "id": "docA:p2:n0",
            "page_id": "docA:p2",
            "ref_id": "docA",
            "page": 2,
            "kind": "text",
            "section": "Body",
            "text": "Alpha continuation with more capacity detail.",
            "context_prefix": "[docA | Title A | p.2 | Body]",
            "search_text": "[docA | Title A | p.2 | Body]\nAlpha continuation with more capacity detail.",
        },
        {
            "id": "docB:p1:n0",
            "page_id": "docB:p1",
            "ref_id": "docB",
            "page": 1,
            "kind": "text",
            "section": "Results",
            "text": "Beta baseline results for comparison.",
            "context_prefix": "[docB | Title B | p.1 | Results]",
            "search_text": "[docB | Title B | p.1 | Results]\nBeta baseline results for comparison.",
        },
    ]
    payload = {"version": 2, "sources": sources, "pages": pages, "nodes": nodes}
    path = Path(tmpdir) / "index.json"
    path.write_text(json.dumps(payload))
    return str(path)


class CleanTextTests(unittest.TestCase):
    def test_nfkc_ligature(self):
        self.assertEqual(clean_text("e\ufb03cient"), "efficient")

    def test_line_wrap_hyphen_repair(self):
        self.assertEqual(clean_text("multi-\nhop"), "multihop")

    def test_short_prefix_preserved(self):
        self.assertEqual(clean_text("co- operation"), "co- operation")

    def test_br_in_table_becomes_space(self):
        out = clean_text("a | b<br>1 | 2")
        self.assertIn("a | b 1 | 2", out)

    def test_br_in_prose_becomes_newline(self):
        out = clean_text("First sentence.<br>Second sentence here.")
        self.assertIn("\n", out)


class CleanTokensTests(unittest.TestCase):
    def test_compound_and_decimal_preservation(self):
        tokens = clean_tokens("6.1b 5.439 gpt-3")
        self.assertIn("6.1b", tokens)
        self.assertIn("5.439", tokens)
        self.assertIn("gpt-3", tokens)

    def test_stopword_filtering(self):
        tokens = clean_tokens("the and of capacity")
        self.assertNotIn("the", tokens)
        self.assertNotIn("and", tokens)
        self.assertIn("capac", tokens)

    def test_no_stem(self):
        tokens = clean_tokens("capacities", stem=False)
        self.assertIn("capacities", tokens)


class ParsePageChunksTests(unittest.TestCase):
    def test_markdown_section_tracking(self):
        raw = "# Intro\nSome prose.\n# Methods\nMore prose."
        chunks, section = parse_page_chunks("docA", "Title A", 1, raw, "")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section"], "Intro")
        self.assertEqual(chunks[1]["section"], "Methods")
        self.assertEqual(section, "Methods")

    def test_table_extraction_multirow_unit_header(self):
        raw = (
            "## Data\n"
            "| Year | (million $) |\n"
            "| --- | --- |\n"
            "| 2020 | 5 |\n"
            "| 2021 | 6 |\n"
            "Table 1: Revenue.\n"
            "Prose after.\n"
        )
        chunks, _ = parse_page_chunks("docA", "Title A", 1, raw, "")
        table_chunks = [c for c in chunks if c.get("kind") == "table"]
        self.assertTrue(table_chunks)
        table_text = table_chunks[0]["text"]
        self.assertIn("2020", table_text)
        self.assertIn("2021", table_text)
        self.assertIn("Table 1: Revenue.", table_text)

    def test_caption_not_duplicated_in_prose(self):
        raw = (
            "## Data\n"
            "| A | B |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
            "Table 2: Numbers.\n"
            "Prose after.\n"
        )
        chunks, _ = parse_page_chunks("docA", "Title A", 1, raw, "")
        prose = " ".join(c["text"] for c in chunks if c.get("kind") != "table")
        self.assertNotIn("Table 2: Numbers.", prose)


class SafeEvalDecimalTests(unittest.TestCase):
    def test_exact_division(self):
        result = safe_eval_decimal("103500 / 10649")
        self.assertEqual(result, Decimal("103500") / Decimal("10649"))

    def test_unary_minus(self):
        self.assertEqual(safe_eval_decimal("-5 + 2"), Decimal("-3"))

    def test_reject_abs_call(self):
        self.assertIsNone(safe_eval_decimal("abs(-1)"))

    def test_reject_power(self):
        self.assertIsNone(safe_eval_decimal("2**3"))

    def test_reject_boolean(self):
        self.assertIsNone(safe_eval_decimal("True"))

    def test_division_by_zero(self):
        self.assertIsNone(safe_eval_decimal("1 / 0"))


class NormalizeSubmissionFieldsTests(unittest.TestCase):
    def test_million_liters_scaling(self):
        row = normalize_submission_fields(
            {"id": "q1", "question": "Water?", "answer_unit": "liters"},
            {"answer_value": "5.439", "answer_unit": "million liters"},
        )
        self.assertEqual(row["answer_value"], "5439000")
        self.assertEqual(row["answer_unit"], "liters")

    def test_million_usd_scaling(self):
        row = normalize_submission_fields(
            {"id": "q1", "question": "Cost?", "answer_unit": "USD"},
            {"answer_value": "2.51", "answer_unit": "million $"},
        )
        self.assertEqual(row["answer_value"], "2510000")
        self.assertEqual(row["answer_unit"], "USD")

    def test_list_million_usd_scaling(self):
        row = normalize_submission_fields(
            {"id": "q1", "question": "Cost?", "answer_unit": "USD"},
            {"answer_value": "[1.84,3.17]", "answer_unit": "million $"},
        )
        self.assertEqual(row["answer_value"], "[1840000,3170000]")
        self.assertEqual(row["answer_unit"], "USD")

    def test_si_mwh_to_kwh(self):
        row = normalize_submission_fields(
            {"id": "q1", "question": "Energy?", "answer_unit": "kWh"},
            {"answer_value": "103.5", "answer_unit": "MWh"},
        )
        self.assertEqual(row["answer_value"], "103500")
        self.assertEqual(row["answer_unit"], "kWh")

    def test_boolean_conversion(self):
        q = {"id": "q1", "question": "True or False: solar works?", "answer_unit": BLANK}
        self.assertEqual(
            normalize_submission_fields(q, {"answer_value": "True", "answer_unit": BLANK})[
                "answer_value"
            ],
            "1",
        )
        self.assertEqual(
            normalize_submission_fields(q, {"answer_value": "False", "answer_unit": BLANK})[
                "answer_value"
            ],
            "0",
        )


class ContextualSearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = ContextualSearchIndex(
            build_index(self.tmp.name), use_dense=False, use_rerank=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_bm25(self):
        results = self.index.bm25("solar capacity", limit=5)
        self.assertTrue(results)
        self.assertEqual(results[0], 0)

    def test_lexical_alignment(self):
        score = self.index.lexical_alignment("solar capacity", "solar capacity here")
        self.assertGreater(score, 0)

    def test_search_multihop_slot_quota(self):
        fact_slots = [
            {"slot": "alpha", "queries": ["alpha solar capacity"]},
            {"slot": "beta", "queries": ["beta baseline results"]},
        ]
        out = self.index.search_multihop(
            "compare alpha and beta",
            ["alpha", "beta"],
            fact_slots,
            top_k=4,
            mode="bm25",
        )
        self.assertTrue(any("alpha" in r["matched_slots"] for r in out))
        self.assertTrue(any("beta" in r["matched_slots"] for r in out))

    def test_expand_preserves_table(self):
        node = {
            "id": "docA:p1:n1",
            "text": "Year | Capacity\n2020 | 100",
        }
        expanded = self.index.expand([node], max_chars=2000)
        self.assertIn("2020 | 100", expanded[0]["text"])

    def test_adjacent_page_nodes(self):
        found = self.index.adjacent_page_nodes(
            ["docA:p1:n0"],
            exclude_ids=None,
            max_nodes=6,
        )
        ids = [n["id"] if isinstance(n, dict) else n for n in found]
        self.assertIn("docA:p2:n0", ids)

    def test_benchmark_retrieval(self):
        questions = [
            {"question": "solar capacity", "ref_id": "docA"},
            {"question": "beta baseline", "ref_id": "docB"},
        ]
        report = benchmark_retrieval(self.index, questions, top_k=3)
        self.assertEqual(report["ctx_bm25"]["questions"], 2)
        self.assertEqual(report["ctx_bm25"]["recall@5"], 1.0)


def _blank_row(qid):
    return {**{col: BLANK for col in COLUMNS}, "id": qid}


class ReplyClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, stage, system, data, emit):
        self.calls.append((stage, data))
        if self.replies:
            return self.replies.pop(0)
        return {"answerable": False}


def _emit(event, **fields):
    return None


class SolveMultihopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = ContextualSearchIndex(
            build_index(self.tmp.name), use_dense=False, use_rerank=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_hop_passes_validation(self):
        question = {"id": "q1", "question": "solar capacity?", "answer_unit": "MW"}
        client = ReplyClient([
            {"queries": ["solar capacity"], "fact_slots": []},
            {
                "answerable": True,
                "answer": "100 MW",
                "answer_value": "100",
                "answer_unit": "MW",
                "explanation": "found",
                "evidence": [{"node_id": "docA:p1:n1", "quote": "2020 | 100"}],
            },
            {"supported": True, "error_type": "none", "reason": "ok"},
        ])
        row = solve_multihop(
            question, self.index, client, _emit, top_k=3, retrieval_mode="bm25"
        )
        validate_rows(COLUMNS, [row], [question], self.index.sources)

    def test_iterative_ircot_followup(self):
        question = {
            "id": "q2",
            "question": "compare alpha and beta baseline?",
            "answer_unit": "MW",
        }
        round1 = {
            "answerable": False,
            "missing_facts": ["beta baseline"],
            "reason": "missing beta",
        }
        round2 = {
            "answerable": True,
            "answer": "42 MW",
            "answer_value": "42",
            "answer_unit": "MW",
            "explanation": "found",
            "evidence": [{"node_id": "docB:p1:n0", "quote": "Beta baseline"}],
        }
        verify_ok = {"supported": True, "error_type": "none", "reason": "ok"}
        client = ReplyClient([
            {
                "queries": ["alpha solar capacity"],
                "fact_slots": [{"slot": "alpha", "queries": ["alpha solar capacity"]}],
            },
            round1,
            round2,
            verify_ok,
        ])
        row = solve_multihop(
            question,
            self.index,
            client,
            _emit,
            top_k=3,
            max_rounds=3,
            retrieval_mode="bm25",
        )
        self.assertEqual(row["ref_id"], '["docB"]')

    def test_over_citation_pruning(self):
        srcs = {
            "docA": {"title": "Title A", "url": "https://a.org"},
            "docB": {"title": "Title B", "url": "https://b.org"},
        }
        row = _blank_row("q3")
        row.update({
            "question": "q?",
            "answer": "1 MW",
            "answer_value": "1",
            "answer_unit": "MW",
            "ref_id": '["docA"]',
            "ref_url": "https://a.org",
            "supporting_materials": "c1",
            "explanation": "ok",
        })
        cited = [
            {"ref_id": "docA", "page": 1, "node_id": "docA:p1:n0", "quote": "q a"},
            {"ref_id": "docB", "page": 1, "node_id": "docB:p1:n0", "quote": "q b"},
        ]
        pruned, kept = prune_candidate_citations(
            row, cited, ["docA:p1:n0"], srcs
        )
        self.assertEqual(pruned["ref_id"], '["docA"]')
        self.assertEqual(len(kept), 1)
        validate_rows(COLUMNS, [pruned], [{"id": "q3", "question": "q?"}], srcs)

    def test_calculation_mismatch_retry(self):
        question = {"id": "q4", "question": "compute 100+1?", "answer_unit": "MW"}
        round1 = {
            "answerable": True,
            "answer": "999 MW",
            "answer_value": "999",
            "answer_unit": "MW",
            "explanation": "computed",
            "calculation": "100 + 1",
            "evidence": [{"node_id": "docA:p1:n1", "quote": "2020 | 100"}],
        }
        round2 = {
            "answerable": True,
            "answer": "101 MW",
            "answer_value": "101",
            "answer_unit": "MW",
            "explanation": "corrected",
            "calculation": "100 + 1",
            "evidence": [{"node_id": "docA:p1:n1", "quote": "2020 | 100"}],
        }
        verify_ok = {"supported": True, "error_type": "none", "reason": "ok"}
        client = ReplyClient([
            {"queries": ["alpha solar capacity"], "fact_slots": []},
            round1,
            round2,
            verify_ok,
        ])
        row = solve_multihop(
            question,
            self.index,
            client,
            _emit,
            top_k=3,
            max_rounds=3,
            retrieval_mode="bm25",
        )
        self.assertEqual(row["answer_value"], "101")


if __name__ == "__main__":
    unittest.main()
