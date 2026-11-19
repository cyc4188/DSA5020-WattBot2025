import unittest
from decimal import Decimal
from types import SimpleNamespace

from wattbot.pi_agent import QuestionTools, calculate, normalize_candidate


class PiAgentToolsTest(unittest.TestCase):
    def test_program_calculates_ratios_and_unit_conversions(self):
        self.assertEqual(calculate("percent_change", [Decimal("8"), Decimal("10")]), "25")
        self.assertEqual(calculate("percent_decrease", [Decimal("10"), Decimal("8")]), "20")
        self.assertEqual(calculate("percent_change", [Decimal("10"), Decimal("8")]), "-20")
        with self.assertRaisesRegex(ValueError, "lower or equal"):
            calculate("percent_decrease", [Decimal("8"), Decimal("10")])
        self.assertEqual(calculate("convert", [Decimal("2.5")], "MWh", "kWh"), "2500")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            calculate("convert", [Decimal("2.5")], "kWh", "days")

    def test_operands_must_appear_in_seen_source(self):
        index = SimpleNamespace(pages={"p:p1": "The reading was 2.5."}, nodes=[])
        solver = QuestionTools({}, index, None, lambda *args, **kwargs: None, "pi")
        with self.assertRaisesRegex(ValueError, "Search for the document"):
            solver.execute("read_page", {"ref_id": "p", "page": 1})
        solver.context["p:p1"] = {"text": "The reading was 2.5.", "ref_id": "p"}
        operands = [{"value": "5", "node_id": "p:p1", "quote": "The reading was 2.5."}]
        args = {"operation": "convert", "operands": operands, "from_unit": "MWh", "to_unit": "kWh"}
        with self.assertRaisesRegex(ValueError, "Operand must appear"):
            solver.execute("calculate", args)
        operands[0]["value"] = "2.5"
        self.assertEqual(solver.execute("calculate", args), {"calculation_id": "c1", "result": "2500", "unit": "kWh"})

    def test_normalizes_numeric_interval_spellings(self):
        question = {"question": "What is the observed range?"}
        for value in ("0.02–0.1", "0.02 to 0.1", "0.02-0.1"):
            with self.subTest(value=value):
                row = {"answer_value": value, "answer_unit": "kWh"}
                self.assertEqual(normalize_candidate(question, row, [])["answer_value"], "[0.02,0.1]")

    def test_dataset_descriptor_requires_matching_evidence(self):
        question = {"question": "Which dataset was used?"}
        row = {"answer_value": "Financial Sentiment Analysis dataset", "answer_unit": "is_blank"}
        citations = [{"quote": "Our dataset, Financial Sentiment Analysis [20], has labeled entries."}]
        self.assertEqual(normalize_candidate(question, row, citations)["answer_value"],
                         "Financial Sentiment Analysis")
        self.assertEqual(normalize_candidate(question, row, []), row)
        formal = {**row, "answer_value": "Example Dataset"}
        self.assertEqual(normalize_candidate(question, formal, [{"quote": "Example Dataset"}]), formal)

    def test_decline_magnitude_preserves_signed_change_elsewhere(self):
        row = {"answer_value": "-55.5556", "answer_unit": "percent"}
        decline = {"question": "By what percentage did emissions fall?"}
        change = {"question": "What was the percentage change in emissions?"}
        self.assertEqual(normalize_candidate(decline, row, [])["answer_value"], "55.5556")
        self.assertEqual(normalize_candidate(change, row, []), row)

    def test_answer_value_required_and_calculation_id_must_exist(self):
        solver = QuestionTools({}, None, None, lambda *args, **kwargs: None, "pi")
        args = {"answerable": True, "answer": "1438 lbs", "answer_unit": "lbs",
                "explanation": "The table gives the amount."}
        with self.assertRaisesRegex(ValueError, "answer_value"):
            solver.execute("submit_answer", args)
        with self.assertRaisesRegex(ValueError, "Unknown calculation_id"):
            solver.execute("submit_answer", {**args, "answer_value": "1438", "calculation_id": "1438"})
        with self.assertRaisesRegex(ValueError, "An abstention"):
            solver.execute("submit_answer", {"answerable": False, "answer_value": "1", "answer_unit": "is_blank"})

    def test_duration_calculation_retains_units_through_submission(self):
        class Review:
            def complete(self, stage, prompt, args, emit):
                self.args = args
                return {"supported": True, "reason": "Duration is supported."}

        question = {"id": "example", "question": "Give the wall time in days.", "answer_unit": "days"}
        index = SimpleNamespace(sources={"p": {"url": "https://example.org/source"}})
        client = Review()
        events = []
        solver = QuestionTools(question, index, client,
                               lambda event, **fields: events.append((event, fields)), "pi")
        solver.context["p:p1"] = {"id": "p:p1", "ref_id": "p", "page": 1,
                                 "text": "30,000 GPU hours on 96 GPUs."}
        first = solver.execute("calculate", {
            "operation": "divide", "result_unit": "hours",
            "operands": [{"value": value, "node_id": "p:p1", "quote": "30,000 GPU hours on 96 GPUs."}
                         for value in ("30000", "96")],
        })
        self.assertEqual(first, {"calculation_id": "c1", "result": "312.5", "unit": "hours"})
        with self.assertRaisesRegex(ValueError, "use that as from_unit"):
            solver.execute("calculate", {"operation": "convert", "operands": [{"calculation_id": "c1"}],
                                         "from_unit": "days", "to_unit": "hours"})
        second = solver.execute("calculate", {"operation": "convert", "operands": [{"calculation_id": "c1"}],
                                              "from_unit": "h", "to_unit": "days"})
        self.assertEqual(second["unit"], "days")
        args = {"answerable": True, "answer_unit": "days",
                "reason": "Divide GPU hours by GPUs, then convert hours to days.",
                "calculation_id": "c1", "answer_value": "312.5"}
        with self.assertRaisesRegex(ValueError, "must agree"):
            solver.execute("submit_answer", args)
        self.assertIsNone(solver.row)
        self.assertTrue(solver.execute("submit_answer", {**args, "answer_value": "13.02"})["accepted"])
        self.assertEqual(solver.row["answer_value"], second["result"])
        self.assertEqual(solver.row["answer_unit"], "days")
        self.assertEqual(len(client.args["citations"]), 1)
        self.assertEqual([e[1]["field"] for e in events if e[0] == "submission_field_recovered"],
                         ["explanation", "answer"])

    def test_false_statement_is_answerable_when_evidence_contradicts_it(self):
        class Review:
            def complete(self, stage, prompt, args, emit):
                self.value = args["answer_value"]
                return {"supported": True, "reason": "The quote contradicts the statement."}

        question = {"id": "example", "question": "True or False: The equipment uses air cooling.",
                    "answer_unit": "is_blank"}
        index = SimpleNamespace(sources={"p": {"url": "https://example.org/source"}})
        client = Review()
        solver = QuestionTools(question, index, client, lambda *args, **kwargs: None, "pi")
        quote = "The equipment uses liquid cooling."
        solver.context["p:p1"] = {"id": "p:p1", "ref_id": "p", "page": 1, "text": quote}
        result = solver.execute("submit_answer", {
            "answerable": True, "answer_value": "0", "answer_unit": "is_blank",
            "answer": "False", "explanation": quote,
            "evidence": [{"node_id": "p:p1", "quote": quote}],
        })
        self.assertTrue(result["accepted"])
        self.assertEqual(client.value, "0")
        self.assertEqual(solver.row["answer_value"], "0")
        self.assertEqual(solver.row["ref_id"], "['p']")
