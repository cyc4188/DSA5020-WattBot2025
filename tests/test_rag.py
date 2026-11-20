import unittest
from decimal import Decimal

from wattbot.evaluation import value_correct
from wattbot.pipeline import (
    COLUMNS, abstention, canonical_answer_value, model_value, solve, submission_fields, validate_rows,
)


class FixtureIndex:
    sources = {"paper": {"url": "https://example.org/paper"}}

    def search(self, queries, top_k):
        return [{"id": "paper:p1:b0", "ref_id": "paper", "page": 1, "title": "Paper",
                 "kind": "text", "text": "The system drew 1.25 kWh."}]

    def expand(self, selected, max_chars):
        return selected


class FixtureModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.stages = []

    def complete(self, stage, system, data, emit):
        self.stages.append(stage)
        return next(self.replies)


class RagVerificationTest(unittest.TestCase):
    question = {"id": "q1", "question": "How much energy did the system draw?"}

    def setUp(self):
        self.events = []
        self.index = FixtureIndex()

    def emit(self, event, **fields):
        self.events.append((event, fields))

    @staticmethod
    def answer(quote):
        return {"answerable": True, "answer": "It drew 1.25 kWh.",
                "answer_value": Decimal("1.25"), "answer_unit": "kWh",
                "explanation": "The passage states the energy directly.",
                "evidence": [{"node_id": "paper:p1:b0", "quote": quote}]}

    def test_hallucinated_quote_is_rejected_and_retried(self):
        client = FixtureModel([
            {"queries": ["system energy draw"]}, self.answer("The system drew 12.5 kWh."),
            self.answer("The system drew 1.25 kWh."),
            {"supported": True, "reason": "The value and unit match."},
        ])
        row = solve(self.question, self.index, client, self.emit, 4)
        self.assertEqual(client.stages, ["plan", "answer", "answer", "verify"])
        self.assertIn("invalid_output", [event for event, _ in self.events])
        self.assertEqual(row["answer_value"], "1.25")
        self.assertEqual(row["ref_id"], "['paper']")
        validate_rows(COLUMNS, [row], [self.question], self.index.sources)

    def test_incomplete_abstention_is_rejected(self):
        row = abstention(self.question)
        validate_rows(COLUMNS, [row], [self.question], self.index.sources)
        row["answer_unit"] = "kWh"
        with self.assertRaisesRegex(ValueError, "Incomplete abstention"):
            validate_rows(COLUMNS, [row], [self.question], self.index.sources)

    def test_native_json_answer_values_and_numeric_tolerance(self):
        self.assertEqual(model_value([1, Decimal("2.5")]), "[1,2.5]")
        self.assertEqual(model_value(True), "TRUE")
        self.assertEqual(canonical_answer_value("True or False: It drew 1.25 kWh.", "TRUE"), "1")
        self.assertEqual(canonical_answer_value("True/False: It drew 2 kWh.", "False"), "0")
        self.assertEqual(canonical_answer_value("Which value was returned?", "True"), "True")
        self.assertTrue(value_correct("100.1", "100"))
        self.assertFalse(value_correct("100.2", "100"))
        self.assertTrue(value_correct("0.05", "[0.02,0.1]"))
        with self.assertRaises(ValueError):
            model_value("~2")

    def test_scaled_units_are_converted_to_the_requested_unit(self):
        question = {**self.question, "answer_unit": "parameters"}
        row = {**self.question, "answer_value": "2", "answer_unit": "B parameters"}
        result = submission_fields(question, row)
        self.assertEqual(result["answer_value"], "2000000000")
        self.assertEqual(result["answer_unit"], "parameters")
        question["question"] = "What is its size in B parameters?"
        self.assertEqual(submission_fields(question, row), result)


if __name__ == "__main__":
    unittest.main()
