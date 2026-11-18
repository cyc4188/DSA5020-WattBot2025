import unittest

from wattbot.baselines.direct import solve


class DirectBaselineTest(unittest.TestCase):
    def test_only_question_and_hint_are_sent(self):
        calls = []

        class Client:
            def complete(self, stage, prompt, data, emit):
                calls.append((stage, data))
                return {
                    "answerable": True, "answer": True,
                    "answer_unit": "is_blank", "explanation": "Closed-book response.",
                    "ref_id": "fabricated",
                }

        question = {
            "id": "q1", "question": "True or False: The statement holds.",
            "answer_unit": "is_blank", "answer_value": "0", "ref_id": "gold",
        }
        row = solve(question, Client(), lambda *args, **kwargs: None)
        self.assertEqual(calls, [
            ("direct_answer", {"question": question["question"], "unit_hint": "is_blank"})
        ])
        self.assertEqual(row["answer_value"], "1")
        for field in ("ref_id", "ref_url", "supporting_materials"):
            self.assertEqual(row[field], "is_blank")


if __name__ == "__main__":
    unittest.main()
