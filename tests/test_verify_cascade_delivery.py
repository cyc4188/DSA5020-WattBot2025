import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.verify_cascade_delivery import check_trace, main
from wattbot.pipeline import BLANK, COLUMNS
from scripts.make_submission import FALLBACK_ANSWER
from wattbot.retrieval import sha256_file


def answer_row(question, value="1.25"):
    return {
        "id": question["id"],
        "question": question["question"],
        "answer": f"The answer is {value} kWh.",
        "answer_value": value,
        "answer_unit": "kWh",
        "ref_id": "paper",
        "ref_url": "https://example.org/paper",
        "supporting_materials": f"The system drew {value} kWh.",
        "explanation": "The supplied passage states the value.",
    }


def blank_row(question):
    return {
        "id": question["id"],
        "question": question["question"],
        "answer": FALLBACK_ANSWER,
        **{field: BLANK for field in COLUMNS[3:]},
    }


class Fixture:
    def __init__(self, directory, questions, rows):
        self.directory = Path(directory)
        self.questions = questions
        self.rows = rows
        self.questions_path = self.directory / "questions.csv"
        self.metadata_path = self.directory / "metadata.csv"
        self.index = self.directory / "index.json"
        self.visual_index = self.directory / "visual.json"
        self.predictions = self.directory / "predictions.csv"
        self.trace = self.directory / "trace.jsonl"

        with self.questions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("id", "question", "answer_unit"))
            writer.writeheader()
            writer.writerows(questions)
        self.metadata_path.write_text(
            "id,url,title\npaper,https://example.org/paper,Paper\n", encoding="utf-8")
        self.index.write_text('{"version": 2}\n', encoding="utf-8")
        self.visual_index.write_text('{"version": 1}\n', encoding="utf-8")
        with self.predictions.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        self.events = []

    def event(self, name, question_id=None, **fields):
        event = {"event": name, "timestamp": "2026-11-20T00:00:00+00:00"}
        if question_id is not None:
            event["question_id"] = question_id
        event.update(fields)
        return event

    def run_start(self, **overrides):
        event = self.event(
            "run_start",
            mode="cascade",
            questions_sha256=sha256_file(self.questions_path),
            index_sha256=sha256_file(self.index),
            question_count=len(self.questions),
            visual_fallback={"index_sha256": sha256_file(self.visual_index)},
        )
        event.update(overrides)
        return event

    def run_complete(self, **overrides):
        event = self.event(
            "run_complete", rows=len(self.rows),
            output_sha256=sha256_file(self.predictions))
        event.update(overrides)
        return event

    def start(self, question):
        return self.event("question_start", question["id"], question=question["question"])

    def rag(self, question_id, usage=None):
        return [
            self.event("llm_request", question_id, stage="plan", pipeline_stage="rag"),
            self.event("llm_response", question_id, stage="plan", pipeline_stage="rag",
                       usage=(usage if usage is not None else
                              {"prompt_tokens": 10, "completion_tokens": 5,
                               "total_tokens": 15})),
        ]

    def multimodal(self, question_id):
        return [
            self.event("llm_request", question_id, stage="answer",
                       pipeline_stage="multimodal"),
            self.event("llm_response", question_id, stage="answer",
                       pipeline_stage="multimodal",
                       usage={"prompt_tokens": 20, "completion_tokens": 8,
                              "total_tokens": 28}),
        ]

    def route(self, question_id, escalated):
        return self.event("cascade_route", question_id, escalated=escalated,
                          reason="rag_abstained" if escalated else "rag_answered")

    def write_trace(self, events):
        self.trace.write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    def check(self, events):
        self.write_trace(events)
        return check_trace(self.trace, self.predictions, self.questions_path,
                           self.metadata_path, self.index, self.visual_index)


class VerifyCascadeDeliveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.text = {"id": "q1", "question": "How much energy did the text report?",
                     "answer_unit": "kWh"}
        self.figure = {"id": "q2", "question": "How much energy does the figure show?",
                       "answer_unit": "kWh"}

    def fixture(self, questions, rows):
        return Fixture(self.directory, questions, rows)

    def normal_events(self, fixture, questions):
        events = [fixture.run_start()]
        for question in questions:
            events.append(fixture.start(question))
            events.extend(fixture.rag(question["id"]))
            escalated = question["id"] == self.figure["id"]
            events.append(fixture.route(question["id"], escalated))
            if escalated:
                events.extend(fixture.multimodal(question["id"]))
                events.append(fixture.event("cascade_result", question["id"],
                                            selected="multimodal"))
            events.append(fixture.event("row", question["id"],
                                        row=fixture.rows[questions.index(question)]))
        events.append(fixture.run_complete())
        return events

    def test_valid_normal_trace(self):
        questions = [self.text, self.figure]
        fixture = self.fixture(questions, [answer_row(self.text, "1.25"),
                                           answer_row(self.figure, "2")])
        summary = fixture.check(self.normal_events(fixture, questions))
        self.assertEqual(summary["counts"]["rows"], 2)
        self.assertEqual(summary["counts"]["routes"],
                         {"rag_answered": 1, "rag_abstained": 1})
        self.assertEqual(summary["counts"]["usage_missing"], 0)
        self.assertEqual(summary["counts"]["stage_usage"]["rag"]["total_tokens"], 30)
        self.assertEqual(summary["counts"]["stage_usage"]["multimodal"]["total_tokens"], 28)

    def test_valid_interrupted_resumed_trace(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            *fixture.rag(self.text["id"]),
            fixture.event("question_error", self.text["id"], error_type="RuntimeError",
                          detail="interrupted"),
            fixture.event("run_resume", completed_rows=0),
            fixture.start(self.text),
            *fixture.rag(self.text["id"]),
            fixture.route(self.text["id"], False),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        summary = fixture.check(events)
        self.assertEqual(summary["counts"]["attempts"], 2)
        self.assertEqual(summary["counts"]["failed_attempts"], 1)
        self.assertEqual(summary["counts"]["resumes"], 1)

    def test_duplicate_question_start_before_resume_is_rejected(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            fixture.start(self.text),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate question_start"):
            fixture.check(events)

    def test_unlabelled_llm_request_cannot_bypass_route_gate(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            fixture.event("llm_request", self.text["id"], stage="plan"),
            fixture.route(self.text["id"], False),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        with self.assertRaisesRegex(ValueError, "lacks a rag or multimodal pipeline_stage"):
            fixture.check(events)

    def test_all_model_events_require_a_pipeline_stage(self):
        names = ("llm_request", "llm_response", "llm_http_error", "llm_network_error",
                 "llm_response_error")
        for name in names:
            with self.subTest(name=name):
                fixture = self.fixture([self.text], [answer_row(self.text)])
                events = [
                    fixture.run_start(),
                    fixture.start(self.text),
                    fixture.event(name, self.text["id"], stage="plan"),
                ]
                with self.assertRaisesRegex(
                        ValueError, "lacks a rag or multimodal pipeline_stage"):
                    fixture.check(events)

    def test_route_requires_a_rag_request(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            fixture.route(self.text["id"], False),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        with self.assertRaisesRegex(ValueError, "cascade_route without a RAG llm_request"):
            fixture.check(events)

    def test_cascade_result_requires_a_multimodal_request(self):
        fixture = self.fixture([self.text], [blank_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            *fixture.rag(self.text["id"]),
            fixture.route(self.text["id"], True),
            fixture.event("cascade_result", self.text["id"], selected="abstention"),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        with self.assertRaisesRegex(
                ValueError, "cascade_result without a multimodal llm_request"):
            fixture.check(events)

    def test_nonrouted_blank_row_is_rejected(self):
        fixture = self.fixture([self.text], [blank_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            *fixture.rag(self.text["id"]),
            fixture.route(self.text["id"], False),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        with self.assertRaisesRegex(ValueError, "Non-routed row is an abstention"):
            fixture.check(events)

    def test_cascade_result_contradicting_row_is_rejected(self):
        fixture = self.fixture([self.text], [blank_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            *fixture.rag(self.text["id"]),
            fixture.route(self.text["id"], True),
            *fixture.multimodal(self.text["id"]),
            fixture.event("cascade_result", self.text["id"], selected="multimodal"),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        with self.assertRaisesRegex(ValueError, "cascade_result contradicts row"):
            fixture.check(events)

    def test_usage_without_total_tokens_is_counted_missing(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = [
            fixture.run_start(),
            fixture.start(self.text),
            *fixture.rag(self.text["id"], usage={}),
            fixture.route(self.text["id"], False),
            fixture.event("row", self.text["id"], row=fixture.rows[0]),
            fixture.run_complete(),
        ]
        summary = fixture.check(events)
        self.assertEqual(summary["counts"]["usage_missing"], 1)
        self.assertEqual(summary["counts"]["stage_usage"]["rag"]["total_tokens"], 0)

    def test_run_start_hash_mismatch_is_rejected(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = self.normal_events(fixture, [self.text])
        events[0] = fixture.run_start(questions_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "disagree on questions_sha256"):
            fixture.check(events)

    def test_run_complete_prediction_hash_mismatch_is_rejected(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        events = self.normal_events(fixture, [self.text])
        events[-1] = fixture.run_complete(output_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "output_sha256 differs from the predictions"):
            fixture.check(events)

    def test_output_must_not_resolve_to_an_input_path(self):
        fixture = self.fixture([self.text], [answer_row(self.text)])
        inputs = {
            "questions": fixture.questions_path,
            "metadata": fixture.metadata_path,
            "index": fixture.index,
            "visual index": fixture.visual_index,
            "predictions": fixture.predictions,
            "trace": fixture.trace,
        }
        for label, path in inputs.items():
            with self.subTest(input=label):
                argv = [
                    "verify_cascade_delivery.py",
                    "--questions", str(fixture.questions_path),
                    "--metadata", str(fixture.metadata_path),
                    "--index", str(fixture.index),
                    "--visual-index", str(fixture.visual_index),
                    "--predictions", str(fixture.predictions),
                    "--trace", str(fixture.trace),
                    "--output", str(path),
                ]
                with patch("sys.argv", argv), self.assertRaisesRegex(
                        ValueError, "output must not overwrite"):
                    main()


if __name__ == "__main__":
    unittest.main()
