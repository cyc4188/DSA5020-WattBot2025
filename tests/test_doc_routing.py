import unittest

from wattbot import doc_routing as dr


def _source(ref_id="chen2024", year="2024", aliases=None):
    return {
        "title": "A Test Paper",
        "url": "https://example.org/" + ref_id,
        "year": year,
        "authors": "Chen et al.",
        "year_aliases": aliases or [year],
    }


def _card(ref_id="chen2024", year="2024", aliases=None):
    return {
        "ref_id": ref_id,
        "title": "A Test Paper",
        "authors": "Chen et al.",
        "year_aliases": aliases or [year],
        "pages": 2,
        "summary": "summary",
        "models": [],
        "hardware_and_infra": [],
        "datasets_and_benchmarks": [],
        "metrics_and_topics": [],
        "reference_baselines": [],
        "table_figure_titles": [],
    }


def _corpus(ref_ids=("chen2024",), pages=None):
    sources = {r: _source(r) for r in ref_ids}
    docs = {r: pages or [
        {"page": 1, "text": "Page one about energy efficiency kWh."},
        {"page": 2, "text": "Page two about cooling loads."},
    ] for r in ref_ids}
    return {"version": 1, "sources": sources, "docs": docs}


def _cards(ref_ids=("chen2024",)):
    return {"version": 1, "cards": {r: _card(r) for r in ref_ids}}


class FakeClient:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def complete(self, stage, system, data, emit):
        self.calls.append(stage)
        response = self.responses[stage]
        if isinstance(response, list):
            return response.pop(0)
        return response


def _emit(*args, **kwargs):
    return None


class TestCleanMarkdownPage(unittest.TestCase):
    def test_clean_markdown_page_strips_noise(self):
        raw = "# Title\n\n\n\nBody text   with spaces  \n\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
        cleaned = dr.clean_markdown_page(raw)
        self.assertIsInstance(cleaned, str)
        self.assertIn("Body text", cleaned)
        self.assertNotIn("\n\n\n", cleaned)


class TestRoutingTokens(unittest.TestCase):
    def test_routing_tokens_returns_list_of_strings(self):
        tokens = dr.routing_tokens("Energy efficiency of GPU clusters in 2024!")
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))
        self.assertTrue(any(t.lower() == "energy" for t in tokens))

    def test_routing_tokens_preserves_decimal_identifiers(self):
        tokens = dr.routing_tokens("6.1B-parameter")
        self.assertIn("6.1b", tokens)
        tokens = dr.routing_tokens("model 3.1 and 3.1b")
        self.assertIn("3.1", tokens)
        self.assertIn("3.1b", tokens)


class TestExtractHeadingsAndCaptions(unittest.TestCase):
    def test_extract_headings_and_captions(self):
        pages = [
            {"page": 1, "text": "# Introduction\nSome body.\n"},
            {"page": 2, "text": "## Method\nFigure 1: Energy usage of the cluster.\n"},
        ]
        headings, captions = dr.extract_headings_and_captions(pages)
        self.assertIsInstance(headings, list)
        self.assertIsInstance(captions, list)
        joined_h = " ".join(headings)
        self.assertIn("Introduction", joined_h)
        self.assertTrue(any("Figure 1" in c for c in captions))


class TestExtractYearAliases(unittest.TestCase):
    def test_extract_year_aliases(self):
        aliases = dr.extract_year_aliases(
            "chen2024",
            {"year": "2024", "citation": "Chen (2024)"},
            "Published in 2024 by Chen et al.",
            None,
        )
        self.assertIsInstance(aliases, list)
        self.assertIn("2024", aliases)


class TestResilientClient(unittest.TestCase):
    def test_retries_on_429(self):
        class Flaky:
            def __init__(self):
                self.n = 0

            def complete(self, stage, system, data, emit):
                self.n += 1
                if self.n < 3:
                    raise RuntimeError("HTTP 429")
                return {"ok": True}

        client = dr.ResilientClient(Flaky(), max_retries=5, backoff_base=0.0)
        result = client.complete("route", "sys", {}, _emit)
        self.assertEqual(result, {"ok": True})


class TestHybridDocumentRouter(unittest.TestCase):
    def setUp(self):
        corpus = _corpus()
        cards = _cards()
        self.router = dr.HybridDocumentRouter(corpus, cards)

    def test_select_pages_for_doc(self):
        pages = self.router.select_pages_for_doc(
            "chen2024", ["energy efficiency"], max_full_pages=35,
            max_full_chars=130_000, page_budget=22,
        )
        self.assertIsInstance(pages, list)
        self.assertTrue(all(isinstance(p, int) for p in pages))
        self.assertIn(1, pages)

    def test_build_context(self):
        ctx, supplied, meta = self.router.build_context(
            ["chen2024"], ["energy efficiency"],
        )
        self.assertIsInstance(ctx, str)
        self.assertIn("chen2024", supplied)
        self.assertIsInstance(supplied["chen2024"], dict)
        self.assertIsInstance(meta, dict)

    def test_route_returns_expected_keys(self):
        client = FakeClient({
            "route": {
                "needs_multiple_docs": False,
                "reasoning": "single doc",
                "selected_docs": [
                    {"ref_id": "chen2024", "role": "primary", "reason": "relevant"}
                ],
                "search_keywords": ["energy efficiency"],
            }
        })
        question = {"id": "q1", "question": "Energy efficiency?", "answer_unit": "kWh"}
        result = self.router.route(question, client, _emit, top_docs=3)
        for key in (
            "bm25_top_docs", "llm_docs", "final_docs",
            "needs_multiple_docs", "search_keywords", "bm25_ranked",
        ):
            self.assertIn(key, result)
        self.assertIn("chen2024", result["final_docs"])
        self.assertEqual(result["needs_multiple_docs"], False)


class TestVerifyAndRecoverCitations(unittest.TestCase):
    def test_verify_and_recover_citations(self):
        reply = {"evidence": [{"ref_id": "chen2024", "page": 1, "quote": "page 1 text"}]}
        supplied = {"chen2024": {1: "page 1 text", 2: "page 2 text"}}
        sources = {"chen2024": {"url": "https://example.org/chen2024"}}
        row, verified = dr.verify_and_recover_citations(
            reply, supplied, sources, _emit,
        )
        self.assertIsInstance(row, dict)
        self.assertEqual(row["ref_id"], "['chen2024']")
        self.assertEqual(row["ref_url"], "['https://example.org/chen2024']")
        self.assertIn("supporting_materials", row)
        self.assertIsInstance(verified, list)


class TestNormalizeDocRoutingFields(unittest.TestCase):
    def test_normalize_doc_routing_fields(self):
        question = {"id": "q1", "question": "How much water?", "answer_unit": "liters"}
        row = {
            "answer": "5.439M",
            "answer_value": "5.439",
            "answer_unit": "million liters",
            "ref_id": "['li2025b']",
            "ref_url": "['https://example.org/li2025b']",
            "supporting_materials": "[li2025b, p. 1] quote",
            "explanation": "calc",
        }
        out = dr.normalize_doc_routing_fields(question, row)
        self.assertIsInstance(out, dict)
        for key in row:
            self.assertIn(key, out)


class TestSolveQuestion(unittest.TestCase):
    def test_solve_question_answerable(self):
        router = dr.HybridDocumentRouter(_corpus(), _cards())
        client = FakeClient({
            "route": {
                "needs_multiple_docs": False,
                "reasoning": "one",
                "selected_docs": [
                    {"ref_id": "chen2024", "role": "primary", "reason": "rel"}
                ],
                "search_keywords": ["energy"],
            },
            "read_answer": {
                "answerable": True,
                "answer": "42",
                "answer_value": "42",
                "answer_unit": "kWh",
                "explanation": "from text",
                "evidence": [{"ref_id": "chen2024", "page": 1, "quote": "energy"}],
            },
        })
        question = {"id": "q1", "question": "Energy?", "answer_unit": "kWh"}
        row = dr.solve_question(question, router, client, _emit, top_docs=3)
        self.assertIsInstance(row, dict)
        self.assertIn("answer_value", row)


class TestEvaluateRouterRecall(unittest.TestCase):
    def test_evaluate_router_recall(self):
        routing_records = [
            {
                "question_id": "q1",
                "bm25_top_docs": ["a"],
                "llm_docs": ["a"],
                "final_docs": ["a", "b"],
            }
        ]
        gold_rows = [{"id": "q1", "ref_id": "['a']"}]
        result = dr.evaluate_router_recall(routing_records, gold_rows)
        self.assertIn("answerable_questions", result)
        self.assertIn("strategies", result)
        self.assertIn("hybrid", result["strategies"])
        self.assertIn("top_1", result["strategies"]["hybrid"])


if __name__ == "__main__":
    unittest.main()
