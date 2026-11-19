import json
import tempfile
import unittest
from pathlib import Path

from wattbot.baselines.full_context import FullContext
from wattbot.citation_recovery import match_quote_in_text, recover_citation, recover_evidence
from wattbot.clean_index import (
    _detect_header_rows,
    _is_bogus_table,
    _is_section_row,
    _parse_markdown_tables,
    clean_text,
)
from wattbot.pipeline import BLANK, submission_fields
from wattbot.retrieval import SearchIndex, tokens


class CleanTextTest(unittest.TestCase):
    def test_unicode_ligatures(self):
        self.assertEqual(clean_text("ef\ufb01ciency"), "efficiency")
        self.assertEqual(clean_text("\ufb02ow"), "flow")

    def test_soft_hyphens_removed(self):
        self.assertEqual(clean_text("hy\u00adphen\u00adated"), "hyphenated")

    def test_line_wrap_hyphenation_is_joined(self):
        self.assertEqual(clean_text("ef\ufb01-\nciency"), "efficiency")
        self.assertEqual(clean_text("applica- tion"), "application")

    def test_compound_prefixes_keep_hyphen(self):
        self.assertEqual(clean_text("climate-\nrelated"), "climate-related")
        self.assertEqual(clean_text("green-\nhouse"), "green-house")

    def test_whitespace_collapsed(self):
        self.assertEqual(clean_text("  a\n\t b  "), "a b")


class RetrievalTest(unittest.TestCase):
    def test_tokens_normalize_ligatures(self):
        self.assertEqual(tokens("ef\ufb01ciency"), ["efficiency"])

    def test_tokens_repair_broken_hyphens(self):
        self.assertEqual(tokens("applica- tion"), ["application"])
        self.assertEqual(tokens("climate-\nrelated"), ["climaterelated"])

    def _write_index(self, directory, version):
        path = Path(directory) / f"index-v{version}.json"
        data = {
            "version": version,
            "sources": {"A": {"title": "Doc A", "url": "https://example.org/a"}},
            "pages": {"A:p1": "High applica- tion efficiency"},
            "nodes": [
                {
                    "id": "A:p1:b0",
                    "page_id": "A:p1",
                    "ref_id": "A",
                    "page": 1,
                    "kind": "text",
                    "text": "High applica- tion efficiency",
                }
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_search_index_loads_version_one(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SearchIndex(self._write_index(directory, 1))
            self.assertEqual(index.version, 1)
            hits = index.bm25("application", 5)
            self.assertEqual([hit[0] for hit in hits], [0])
            self.assertGreater(hits[0][1], 0)
            results = index.search(["application"], 1)
            self.assertEqual(results[0]["ref_id"], "A")
            self.assertEqual(results[0]["title"], "Doc A")

    def test_search_index_loads_version_two(self):
        with tempfile.TemporaryDirectory() as directory:
            index = SearchIndex(self._write_index(directory, 2))
            self.assertEqual(index.version, 2)
            self.assertTrue(index.bm25("efficiency", 5))

    def test_search_index_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SearchIndex(self._write_index(directory, 99))


class TableHelperTest(unittest.TestCase):
    def test_detect_multi_level_header_with_units(self):
        rows = [
            ["Category", "Revenue", "Cost"],
            ["", "(million $)", "(million $)"],
            ["Air travel", "10", "20"],
        ]
        header, body = _detect_header_rows(rows)
        self.assertEqual(header, rows[:2])
        self.assertEqual(body, [["Air travel", "10", "20"]])

    def test_detect_year_header_not_single_year_data_row(self):
        rows = [
            ["Category", "2021", "2022", "2023"],
            ["Air travel", "1984"],
        ]
        header, body = _detect_header_rows(rows)
        self.assertEqual(header, [["Category", "2021", "2022", "2023"]])
        self.assertEqual(body, [["Air travel", "1984"]])

    def test_detect_parenthetical_data_row_is_not_header(self):
        rows = [
            ["Category", "2021", "2022"],
            ["Intensity (g/$)", "122.8", "102.7"],
            ["Other", "1", "2"],
        ]
        header, body = _detect_header_rows(rows)
        self.assertEqual(header, [["Category", "2021", "2022"]])
        self.assertEqual(body[0], ["Intensity (g/$)", "122.8", "102.7"])

    def test_is_section_row(self):
        self.assertTrue(_is_section_row(["Section title", "", ""]))
        self.assertTrue(_is_section_row(["Heading", "", "", "", "note"]))
        self.assertFalse(_is_section_row(["Air travel", "10", "20"]))

    def test_is_bogus_table_single_column(self):
        self.assertTrue(_is_bogus_table([["Only"], ["one"]]))

    def test_is_bogus_table_bibliography(self):
        rows = [
            ["Ref", "Title"],
            ["[Arm10]", "Some paper arXiv:1234.5678"],
            ["[Ber09]", "Another bibliography entry"],
        ]
        self.assertTrue(_is_bogus_table(rows))

    def test_is_bogus_table_accepts_real_table(self):
        self.assertFalse(_is_bogus_table([["Year", "Value"], ["2020", "10"]]))

    def test_parse_markdown_tables_strips_formatting(self):
        markdown = "\n".join([
            "**Table 3: Energy use**",
            "| **Year** | Energy<br>Use | Value |",
            "| --- | --- | --- |",
            "| 2020 | `kWh` | 10<sup>3</sup> |",
        ])
        tables = _parse_markdown_tables(markdown)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]["caption"], "Table 3: Energy use")
        self.assertEqual(
            tables[0]["rows"],
            [["Year", "Energy Use", "Value"], ["2020", "kWh", "103"]],
        )


class MatchQuoteTest(unittest.TestCase):
    def test_exact_match(self):
        text, method = match_quote_in_text("energy use", "Total energy use in 2020")
        self.assertEqual(method, "exact")
        self.assertEqual(text, "energy use")

    def test_token_span_repairs_broken_hyphen(self):
        text, method = match_quote_in_text(
            "The application works well.", "The applica- tion works well."
        )
        self.assertEqual(method, "token_span")
        self.assertIn("applica- tion", text)

    def test_token_span_handles_curly_quotes_and_ligatures(self):
        page = "The system\u2019s ef\ufb01-\nciency improved."
        text, method = match_quote_in_text(
            "The system's efficiency improved.", page
        )
        self.assertEqual(method, "token_span")
        self.assertIn("ef\ufb01-\nciency", text)

    def test_token_span_handles_multiplication_sign(self):
        text, method = match_quote_in_text(
            "The area is 3\u00d74 metres.", "The area is 3 \u00d7 4 metres."
        )
        self.assertEqual(method, "token_span")

    def test_ellipsis_segments(self):
        page = (
            "The GShard-600B model reached 24.1 percent while the smaller "
            "model reached 4.3 percent."
        )
        text, method = match_quote_in_text("GShard-600B ... 24.1 ... 4.3", page)
        self.assertEqual(method, "ellipsis_segments")
        self.assertIn("GShard-600B", text)
        self.assertIn("24.1", text)
        self.assertIn("4.3", text)

    def test_token_lcs_accepts_one_word_insertion(self):
        page = "the quick brown lazy fox jumps over the dog"
        text, method = match_quote_in_text("the quick brown fox jumps over", page)
        self.assertEqual(method, "token_lcs")
        self.assertIn("lazy", text)

    def test_token_lcs_rejects_wrong_number(self):
        page = "the model used 32.7 units of power"
        self.assertEqual(
            match_quote_in_text("the model used 24.1 units of power", page),
            (None, None),
        )

    def test_token_lcs_rejects_unrelated_text(self):
        page = "completely unrelated sentence about something else entirely"
        self.assertEqual(
            match_quote_in_text("the model used 24.1 units of power", page),
            (None, None),
        )


class RecoverCitationTest(unittest.TestCase):
    def setUp(self):
        self.pages = {
            ("A", 1): "climate related energy policy text",
            ("A", 2): "efficiency improved significantly",
            ("A", 5): "unique finding about solar panels here",
        }

    def test_same_page(self):
        result = recover_citation("A", 2, "efficiency improved significantly",
                                  self.pages, {})
        self.assertEqual(result["scope"], "same_page")
        self.assertEqual(result["page"], 2)

    def test_adjacent_page(self):
        result = recover_citation("A", 2, "climate related energy policy text",
                                  self.pages, {})
        self.assertEqual(result["scope"], "adjacent_page")
        self.assertEqual(result["page"], 1)

    def test_same_doc(self):
        result = recover_citation("A", 2, "unique finding about solar panels here",
                                  self.pages, {})
        self.assertEqual(result["scope"], "same_doc")
        self.assertEqual(result["page"], 5)

    def test_corpus_unique(self):
        pages = {("B", 3): "the global temperature rose by 1.2 degrees"}
        result = recover_citation(
            "A", 2, "the global temperature rose by 1.2 degrees", pages, {}
        )
        self.assertEqual(result["scope"], "corpus_unique")
        self.assertEqual(result["ref_id"], "B")
        self.assertEqual(result["page"], 3)

    def test_ambiguous_short_quote_rejected(self):
        pages = {
            ("B", 1): "energy use is rising",
            ("C", 1): "energy use is falling",
        }
        self.assertIsNone(recover_citation("A", 1, "energy use", pages, {}))

    def test_ambiguous_long_quote_rejected(self):
        pages = {
            ("B", 1): "the global temperature rose by 1.2 degrees",
            ("C", 1): "the global temperature rose by 1.2 degrees",
        }
        self.assertIsNone(recover_citation(
            "A", 2, "the global temperature rose by 1.2 degrees", pages, {}
        ))


class FullContextTest(unittest.TestCase):
    def _contexts(self, directory):
        root = Path(directory)
        metadata = root / "metadata.csv"
        metadata.write_text(
            "id,title,url\nA,Doc A,https://example.org/a\n", encoding="latin-1"
        )
        page_text = "The system achieved high ef\ufb01-\nciency in 2020."
        data = {
            "version": 2,
            "sources": {"A": {"title": "Doc A", "url": "https://example.org/a",
                              "stats": {"pages": 1}}},
            "pages": {"A:p1": page_text},
            "nodes": [{
                "id": "A:p1:b0", "page_id": "A:p1", "ref_id": "A", "page": 1,
                "kind": "text", "text": page_text,
            }],
        }
        index = root / "index.json"
        index.write_text(json.dumps(data), encoding="utf-8")
        strict = FullContext(index, metadata, recover_citations=False)
        recovery = FullContext(index, metadata, recover_citations=True)
        return strict, recovery

    def test_strict_mode_requires_verbatim_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            strict, _ = self._contexts(directory)
            events = []
            reply = {"evidence": [{"ref_id": "A", "page": 1,
                                   "quote": "high efficiency"}]}
            result = strict.citations(reply, lambda event, **fields: events.append(event))
            self.assertEqual(result["ref_id"], BLANK)
            self.assertEqual(result["supporting_materials"], BLANK)
            self.assertIn("citation_dropped", events)

    def test_recovery_mode_repairs_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            _, recovery = self._contexts(directory)
            events = []
            reply = {"evidence": [{"ref_id": "A", "page": 1,
                                   "quote": "high efficiency"}]}
            result = recovery.citations(reply, lambda event, **fields: events.append(event))
            self.assertIn("A", result["ref_id"])
            self.assertIn("[A, p. 1]", result["supporting_materials"])
            self.assertIn("citation_recovered", events)

    def test_strict_mode_keeps_exact_quote(self):
        with tempfile.TemporaryDirectory() as directory:
            strict, _ = self._contexts(directory)
            reply = {"evidence": [{"ref_id": "A", "page": 1,
                                   "quote": "system achieved"}]}
            result = strict.citations(reply, lambda event, **fields: None)
            self.assertIn("A", result["ref_id"])
            self.assertIn("[A, p. 1] system achieved", result["supporting_materials"])

    def test_recover_evidence_matches_full_context(self):
        with tempfile.TemporaryDirectory() as directory:
            _, recovery = self._contexts(directory)
            reply = {"evidence": [{"ref_id": "A", "page": 1,
                                   "quote": "high efficiency"}]}
            result = recover_evidence(reply, recovery.pages, recovery.sources)
            self.assertIn("A", result["ref_id"])


class SubmissionFieldsTest(unittest.TestCase):
    def test_million_liters_to_liters(self):
        question = {"id": "q1", "question": "How much water in million liters?",
                    "answer_unit": "liters"}
        row = {"answer_value": "5.439", "answer_unit": "million L"}
        result = submission_fields(question, row)
        self.assertEqual(result["answer_value"], "5439000")
        self.assertEqual(result["answer_unit"], "liters")

    def test_thousands_of_grams_range(self):
        question = {"id": "q2", "question": "How much in thousands of grams?",
                    "answer_unit": "grams"}
        row = {"answer_value": "[1,2]", "answer_unit": "thousands of grams"}
        result = submission_fields(question, row)
        self.assertEqual(result["answer_value"], "[1000,2000]")
        self.assertEqual(result["answer_unit"], "grams")

    def test_million_dollars_to_usd(self):
        question = {"id": "q3", "question": "How much in USD?",
                    "answer_unit": "USD"}
        row = {"answer_value": "2.51", "answer_unit": "million $"}
        result = submission_fields(question, row)
        self.assertEqual(result["answer_value"], "2510000")
        self.assertEqual(result["answer_unit"], "USD")


if __name__ == "__main__":
    unittest.main()
