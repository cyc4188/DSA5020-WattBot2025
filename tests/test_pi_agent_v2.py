import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from wattbot.pi_agent_v2 import (
    CleanSearchIndex,
    QuestionToolsV2,
    clean_page_text,
    clean_text_for_search,
    convert_unit_v2,
    eval_math_expression,
    extract_quote_numbers,
    normalize_submission_v2,
    source_quote_v2,
    summarize_trace_usage,
)

BLANK_ROW = {"ref_id": "is_blank", "ref_url": "is_blank",
             "supporting_materials": "is_blank"}


def submit_row(question, value, unit):
    return {"id": question["id"], "question": question["question"], "answer": "answer",
            "answer_value": value, "answer_unit": unit, "explanation": "explanation",
            **BLANK_ROW}


class EvalMathExpressionTest(unittest.TestCase):
    def test_multi_step_decimal_arithmetic(self):
        result, used = eval_math_expression("(a / b) / 24", {"a": Decimal("1000"), "b": Decimal("3")})
        self.assertEqual(result, "13.888888888888888888888888888889")
        self.assertEqual(used, {"a", "b"})
        self.assertEqual(eval_math_expression("cost * 1e6", {"cost": Decimal("1.84")})[0], "1840000")
        self.assertEqual(
            eval_math_expression("(old - new) / old * 100",
                                 {"old": Decimal("10"), "new": Decimal("8")})[0],
            "20",
        )

    def test_named_constants(self):
        expected = {
            "HOURS_PER_DAY": "24",
            "DAYS_PER_YEAR": "365",
            "MILLION": "1000000",
            "LB_PER_KG": "2.20462262185",
            "KG_PER_LB": "0.45359237",
            "L_PER_GAL": "3.785411784",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                result, used = eval_math_expression(name, {})
                self.assertEqual(result, value)
                self.assertEqual(used, set())

    def test_whitelisted_literals_need_no_quotes(self):
        self.assertEqual(eval_math_expression("24 * 365", {})[0], "8760")
        self.assertEqual(eval_math_expression("1e6", {})[0], "1000000")
        self.assertEqual(eval_math_expression("1e9 / 1000", {})[0], "1000000")
        self.assertEqual(eval_math_expression("0.5 + 0.1", {})[0], "0.6")

    def test_full_precision_is_preserved(self):
        result, _ = eval_math_expression("a / b", {"a": Decimal("1"), "b": Decimal("3")})
        self.assertEqual(result, "0.33333333333333333333333333333333")
        self.assertNotEqual(result, "0.33")

    def test_rejects_arbitrary_unquoted_literals(self):
        for expression in ("42.5", "1438", "10**25"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_math_expression(expression, {})

    def test_rejects_unsafe_syntax(self):
        expressions = [
            "__import__('os')",
            "eval('1')",
            "open('secret')",
            "round(1)",
            "x.y",
            "x[0]",
            "lambda value: value",
            "[item for item in data]",
            "True",
            "False",
            "'text'",
            "1 / 0",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    eval_math_expression(expression, {})


class TextCleaningTest(unittest.TestCase):
    def test_clean_text_normalizes_ligatures_and_subscripts(self):
        self.assertEqual(clean_text_for_search("efﬁciency with CO₂"), "efficiency with CO2")
        self.assertEqual(clean_text_for_search("data cen- ters"), "data centers")
        self.assertEqual(clean_text_for_search("Llama-3.1 model"), "Llama-3.1 model")

    def test_clean_page_text_unwraps_hyphens_but_keeps_tables(self):
        text = "data cen-\nters and Llama-3.1\n| column |\n| --- |\n"
        cleaned = clean_page_text(text)
        self.assertEqual(cleaned, "data centers and Llama-3.1\n| column |\n| --- |\n")

    def test_source_quote_matches_ligatures(self):
        self.assertEqual(source_quote_v2("efﬁciency", "high efficiency"), "efficiency")
        self.assertEqual(source_quote_v2("efﬁciency improved", "efficiency improved"),
                         "efficiency improved")

    def test_source_quote_returns_exact_markdown_spans(self):
        passage = "The **efficiency** improved."
        self.assertEqual(source_quote_v2("efficiency improved", passage), "efficiency** improved")
        table = "| efficiency | improved |"
        self.assertEqual(source_quote_v2("efficiency improved", table), "efficiency | improved")

    def test_source_quote_matches_thousands_commas_and_dehyphenation(self):
        self.assertEqual(source_quote_v2("1438 units", "reached 1,438 units"), "1,438 units")
        self.assertEqual(source_quote_v2("data centers", "data cen- ters"), "data cen- ters")

    def test_extract_quote_numbers_includes_dash_ranges(self):
        self.assertEqual(
            extract_quote_numbers("range 12–15, 1,438 and −2"),
            [Decimal("12"), Decimal("15"), Decimal("1438"), Decimal("-2")],
        )


class ConvertUnitV2Test(unittest.TestCase):
    def test_scaled_units(self):
        self.assertEqual(convert_unit_v2(Decimal("5"), "million liters", "liters"), "5000000")
        self.assertEqual(convert_unit_v2(Decimal("5"), "millions of liters", "liters"), "5000000")
        self.assertEqual(convert_unit_v2(Decimal("1.84"), "million $", "USD"), "1840000")

    def test_dimensional_conversions(self):
        self.assertEqual(convert_unit_v2(Decimal("3"), "lbs", "kg"), "1.36077711")
        self.assertEqual(convert_unit_v2(Decimal("1"), "MWh", "kWh"), "1000")
        self.assertEqual(convert_unit_v2(Decimal("24"), "hours", "days"), "1")
        self.assertEqual(convert_unit_v2(Decimal("1"), "kg", "lbs"),
                         "2.2046226218487758072297380134503")


class NormalizeSubmissionV2Test(unittest.TestCase):
    def test_scaled_units_follow_answer_unit_hint(self):
        cases = [
            ("5", "million liters", "liters", "5000000"),
            ("5", "millions of liters", "liters", "5000000"),
            ("1.84", "million USD", "USD", "1840000"),
        ]
        for value, unit, hint, expected in cases:
            with self.subTest(value=value, unit=unit):
                question = {"id": "q", "question": "How much?", "answer_unit": hint}
                row = normalize_submission_v2(question, submit_row(question, value, unit), [])
                self.assertEqual(row["answer_value"], expected)
                self.assertEqual(row["answer_unit"], hint)

    def test_range_million_usd_converts_to_usd(self):
        question = {"id": "q", "question": "What is the observed range?", "answer_unit": "USD"}
        row = normalize_submission_v2(question, submit_row(question, "[1.84,3.17]", "million USD"), [])
        self.assertEqual(row["answer_value"], "[1840000,3170000]")
        self.assertEqual(row["answer_unit"], "USD")


class QuestionToolsV2Test(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        directory = Path(self.tempdir.name)
        self.passage = "The **efﬁciency** was 42.5 percent and the baseline cost is 1.84 million USD."
        self.second = "Unrelated paper about solvents and catalysts."
        index_path = directory / "index.json"
        index_path.write_text(json.dumps({
            "version": 1,
            "sources": {
                "p1": {"title": "Paper One", "url": "https://example.org/1",
                       "stats": {"pages": 1}},
                "p2": {"title": "Paper Two", "url": "https://example.org/2",
                       "stats": {"pages": 1}},
            },
            "pages": {"p1:p1": self.passage, "p2:p1": self.second},
            "nodes": [
                {"id": "p1:p1:n0", "ref_id": "p1", "page": 1, "page_id": "p1:p1",
                 "kind": "text", "text": self.passage},
                {"id": "p2:p1:n0", "ref_id": "p2", "page": 1, "page_id": "p2:p1",
                 "kind": "text", "text": self.second},
            ],
        }), encoding="utf-8")
        self.index = CleanSearchIndex(index_path, dense=False)

    def make_solver(self, mode="pi", events=None, question=None):
        class Review:
            def complete(self, stage, prompt, args, emit):
                return {"supported": True, "reason": "Supported by the quote."}

        question = question or {"id": "q", "question": "What is the cost?", "answer_unit": "USD"}
        events = events if events is not None else []
        solver = QuestionToolsV2(
            question, self.index, Review(),
            lambda event, **fields: events.append((event, fields)), mode,
        )
        solver.events = events
        return solver

    def evaluate(self, solver, expression, operands=None, unit="USD"):
        return solver.execute("eval_math", {
            "expression": expression, "result_unit": unit,
            "named_operands": operands or [],
        })

    def test_eval_math_grounds_dict_and_list_operands(self):
        solver = self.make_solver()
        solver.execute("search_corpus", {"query": "efficiency cost", "top_k": 2})
        result = self.evaluate(solver, "pct / HOURS_PER_DAY", {
            "pct": {"value": "42.5", "node_id": "p1:p1:n0",
                    "quote": "efﬁciency was 42.5 percent"},
        }, unit="percent")
        self.assertEqual(result["calculation_id"], "c1")
        self.assertEqual(result["result"], "1.7708333333333333333333333333333")
        listed = self.evaluate(solver, "cost * 2", [
            {"name": "cost", "value": "1.84", "node_id": "p1:p1:n0",
             "quote": "baseline cost is 1.84 million USD"},
        ], unit="million USD")
        self.assertEqual(listed["result"], "3.68")

    def test_eval_math_rejects_constant_only_expression(self):
        solver = self.make_solver()
        with self.assertRaisesRegex(ValueError, "grounded operand"):
            self.evaluate(solver, "HOURS_PER_DAY * 2", unit="hours")

    def test_calculation_id_can_be_chained_directly_or_as_operand(self):
        solver = self.make_solver()
        solver.execute("search_corpus", {"query": "cost", "top_k": 2})
        first = self.evaluate(solver, "cost * 2", {
            "cost": {"value": "1.84", "node_id": "p1:p1:n0",
                     "quote": "baseline cost is 1.84 million USD"},
        }, unit="million USD")
        direct = self.evaluate(solver, "c1 / 24", unit="million USD")
        self.assertEqual(direct["result"], "0.15333333333333333333333333333333")
        operand = self.evaluate(solver, "previous / HOURS_PER_DAY", [
            {"name": "previous", "calculation_id": "c1"},
        ], unit="million USD")
        self.assertEqual(operand["result"], direct["result"])
        self.assertNotEqual(first["calculation_id"], operand["calculation_id"])

    def test_search_list_and_read_page_tools(self):
        solver = self.make_solver()
        self.assertEqual([item["ref_id"] for item in solver.execute("list_documents", {})],
                         ["p1", "p2"])
        hits = solver.execute("search_corpus", {"query": "cost", "top_k": 2})
        self.assertEqual(hits[0]["id"], "p1:p1:n0")
        filtered = solver.execute("search_corpus",
                                  {"query": "catalysts", "top_k": 2, "ref_id": "p1"})
        self.assertEqual(filtered, [])
        unfiltered = solver.execute("search_corpus", {"query": "catalysts", "top_k": 2})
        self.assertEqual(unfiltered[0]["id"], "p2:p1:n0")
        page = solver.execute("read_page", {"ref_id": "p1", "page": 1})
        self.assertIn("baseline cost", page["text"])
        with self.assertRaisesRegex(ValueError, "out of range"):
            solver.execute("read_page", {"ref_id": "p1", "page": 9})

    def test_premature_abstention_is_blocked_in_pi_mode(self):
        solver = self.make_solver()
        with self.assertRaisesRegex(ValueError, "Cannot abstain"):
            solver.execute("submit_answer", {"answerable": False, "answer_value": "is_blank",
                                             "answer_unit": "is_blank"})
        self.assertTrue(any(event == "premature_abstention_blocked"
                            for event, _ in solver.events))
        solver.execute("search_corpus", {"query": "cost", "top_k": 2})
        self.assertTrue(solver.execute("submit_answer", {
            "answerable": False, "answer_value": "is_blank", "answer_unit": "is_blank",
        })["accepted"])
        self.assertIsNotNone(solver.row)

    def test_abstention_is_allowed_in_calculator_mode(self):
        solver = self.make_solver(mode="calculator")
        self.assertTrue(solver.execute("submit_answer", {
            "answerable": False, "answer_value": "is_blank", "answer_unit": "is_blank",
        })["accepted"])

    def test_submit_answer_binds_calculation_with_unit_conversion(self):
        solver = self.make_solver()
        solver.execute("search_corpus", {"query": "cost", "top_k": 2})
        calculation = self.evaluate(solver, "cost * 2", {
            "cost": {"value": "1.84", "node_id": "p1:p1:n0",
                     "quote": "baseline cost is 1.84 million USD"},
        }, unit="million USD")
        result = solver.execute("submit_answer", {
            "answerable": True, "answer_value": "3680000", "answer_unit": "USD",
            "calculation_id": f" {calculation['calculation_id']} ",
            "explanation": "Double the baseline cost and convert millions to USD.",
        })
        self.assertTrue(result["accepted"])
        self.assertEqual(solver.row["answer_value"], "3680000")
        self.assertEqual(solver.row["answer_unit"], "USD")

    def test_submit_answer_keeps_full_precision_from_calculation(self):
        solver = self.make_solver()
        solver.execute("search_corpus", {"query": "efficiency", "top_k": 2})
        calculation = self.evaluate(solver, "pct / HOURS_PER_DAY", {
            "pct": {"value": "42.5", "node_id": "p1:p1:n0",
                    "quote": "efﬁciency was 42.5 percent"},
        }, unit="percent")
        result = solver.execute("submit_answer", {
            "answerable": True, "answer_value": calculation["result"], "answer_unit": "percent",
            "calculation_id": calculation["calculation_id"], "explanation": "Divide by 24.",
        })
        self.assertTrue(result["accepted"])
        self.assertEqual(solver.row["answer_value"], calculation["result"])

    def test_submit_answer_rejects_disagreeing_calculation_value(self):
        solver = self.make_solver()
        solver.execute("search_corpus", {"query": "cost", "top_k": 2})
        calculation = self.evaluate(solver, "cost * 2", {
            "cost": {"value": "1.84", "node_id": "p1:p1:n0",
                     "quote": "baseline cost is 1.84 million USD"},
        }, unit="million USD")
        with self.assertRaisesRegex(ValueError, "must agree"):
            solver.execute("submit_answer", {
                "answerable": True, "answer_value": "1", "answer_unit": "USD",
                "calculation_id": calculation["calculation_id"], "explanation": "Wrong value.",
            })
        self.assertIsNone(solver.row)

    def test_empty_or_none_calculation_id_is_treated_as_omitted(self):
        for calculation_id in ("", " none "):
            with self.subTest(calculation_id=calculation_id):
                solver = self.make_solver()
                solver.execute("search_corpus", {"query": "cost", "top_k": 2})
                result = solver.execute("submit_answer", {
                    "answerable": True, "answer_value": "1.84", "answer_unit": "million USD",
                    "calculation_id": calculation_id, "explanation": "Direct answer.",
                    "evidence": [{"node_id": "p1:p1:n0",
                                  "quote": "baseline cost is 1.84 million USD"}],
                })
                self.assertTrue(result["accepted"])
                self.assertIsNotNone(solver.row)


class SummarizeTraceUsageTest(unittest.TestCase):
    def test_aggregates_usage_tools_and_premature_abstention(self):
        events = [
            {"event": "llm_request", "stage": "generate", "question_id": "q1"},
            {"event": "llm_response", "stage": "generate", "question_id": "q1",
             "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                       "completion_tokens_details": {"reasoning_tokens": 5},
                       "prompt_tokens_details": {"cached_tokens": 10}},
             "elapsed_ms": 1000},
            {"event": "llm_response", "stage": "verify", "question_id": "q1",
             "usage": {"prompt_tokens": 50, "completion_tokens": 10},
             "elapsed_ms": 2000},
            {"event": "tool_request", "tool": "search_corpus"},
            {"event": "tool_request", "tool": "eval_math"},
            {"event": "tool_result", "tool": "search_corpus"},
            {"event": "premature_abstention_blocked", "question_id": "q1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text("\n".join(json.dumps(event) for event in events) + "\n",
                             encoding="utf-8")
            usage = summarize_trace_usage(trace)
        self.assertEqual(usage["prompt_tokens"], 150)
        self.assertEqual(usage["completion_tokens"], 30)
        self.assertEqual(usage["reasoning_tokens"], 5)
        self.assertEqual(usage["total_tokens"], 180)
        self.assertEqual(usage["tool_calls"]["search_corpus"], 1)
        self.assertEqual(usage["tool_calls"]["eval_math"], 1)
        self.assertEqual(usage["tool_calls"]["submit_answer"], 0)
        self.assertEqual(usage["premature_abstention_blocked"], 1)


if __name__ == "__main__":
    unittest.main()
