import hashlib
import json
import unittest
from types import SimpleNamespace

from wattbot.baselines.direct import solve
from wattbot.baselines.full_context import PROMPT


class FullContextBaselineTest(unittest.TestCase):
    def test_entire_context_reaches_client_and_trace_keeps_hash(self):
        corpus = "DOCUMENT a\nPAGE 1\nFirst source.\nDOCUMENT b\nPAGE 1\nSecond source."
        digest = hashlib.sha256(corpus.encode()).hexdigest()
        context = SimpleNamespace(
            text=corpus, digest=digest, prompt=PROMPT,
            citations=lambda reply, emit: {"ref_id": "['b']"},
        )
        requests = []
        events = []

        class Client:
            def complete(self, stage, system, data, emit):
                requests.append((stage, data.copy()))
                emit("llm_request", messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(data)},
                ])
                return {
                    "answerable": True, "answer": "2", "answer_value": 2,
                    "answer_unit": "items", "explanation": "Count the items.",
                }

        question = {"id": "q1", "question": "How many?", "answer_unit": "items"}
        row = solve(question, Client(), lambda event, **fields: events.append((event, fields)), context)
        self.assertEqual(requests, [("full_context_answer", {
            "corpus": corpus, "question": "How many?", "unit_hint": "items",
        })])
        logged = json.loads(events[0][1]["messages"][1]["content"])
        self.assertEqual(logged, {
            "corpus_sha256": digest, "question": "How many?", "unit_hint": "items",
        })
        self.assertEqual(row["ref_id"], "['b']")
