import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_wattbot_runs import (
    ManifestError,
    _reported_usage,
    align_predictions,
    build_report,
    load_questions,
    token_report,
)
from scripts.make_submission import FALLBACK_ANSWER
from wattbot.pipeline import BLANK, COLUMNS, EVIDENCE_FIELDS

DOCS = {"doc1": "https://example.org/doc1", "doc2": "https://example.org/doc2"}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path):
    with Path(path).open("w", encoding="latin-1", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "url"])
        writer.writeheader()
        writer.writerows({"id": key, "url": url} for key, url in DOCS.items())


def question(qid, value, refs):
    return {
        "id": qid, "question": f"Question {qid}?", "answer": f"Gold {qid}",
        "answer_value": value,
        "answer_unit": BLANK if value == BLANK else "kWh",
        "ref_id": repr(list(refs)) if refs else BLANK,
        "ref_url": repr([DOCS[ref] for ref in refs]) if refs else BLANK,
        "supporting_materials": f"[{refs[0]}, p. 1] Gold text." if refs else BLANK,
        "explanation": "Gold explanation." if refs else BLANK,
    }


def prediction(qid, value, refs=()):
    return {
        "id": qid, "question": f"Question {qid}?", "answer": f"Answer {qid}",
        "answer_value": value, "answer_unit": "kWh",
        "ref_id": repr(list(refs)) if refs else BLANK,
        "ref_url": repr([DOCS[ref] for ref in refs]) if refs else BLANK,
        "supporting_materials": f"[{refs[0]}, p. 1] Value." if refs else BLANK,
        "explanation": "Evidence explanation." if refs else BLANK,
    }


def blank_prediction(qid):
    return {
        "id": qid, "question": f"Question {qid}?", "answer": FALLBACK_ANSWER,
        **dict.fromkeys(EVIDENCE_FIELDS, BLANK),
        "explanation": BLANK,
    }


def write_trace(path, token_map):
    records = [
        {"question_id": qid, "event": "llm_response",
         "usage": {"total_tokens": total, "prompt_tokens": total - 1,
                   "completion_tokens": 1}}
        for qid, total in token_map.items()
    ]
    Path(path).write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


class CompareRunsAlignmentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.questions_path = self.directory / "questions.csv"
        write_csv(self.questions_path, [
            question("q1", "10", ("doc1",)),
            question("q2", "20", ("doc2",)),
        ])
        self.questions = load_questions(self.questions_path)

    def assert_alignment_error(self, rows, message):
        path = self.directory / "predictions.csv"
        write_csv(path, rows)
        with self.assertRaisesRegex(ManifestError, message):
            align_predictions(path, self.questions)

    def test_rejects_extra_duplicate_missing_and_identity_mismatch(self):
        self.assert_alignment_error([
            prediction("q1", "10", ("doc1",)),
            prediction("q2", "20", ("doc2",)),
            prediction("q3", "30", ("doc1",)),
        ], "extra question ID")
        self.assert_alignment_error([
            prediction("q1", "10", ("doc1",)),
            prediction("q1", "20", ("doc2",)),
        ], "duplicate question ID")
        self.assert_alignment_error([
            prediction("q1", "10", ("doc1",)),
        ], "missing question IDs")
        mismatch = prediction("q1", "10", ("doc1",))
        mismatch["question"] = "A different question?"
        self.assert_alignment_error([
            mismatch, prediction("q2", "20", ("doc2",)),
        ], "identity mismatch")

    def test_aligns_rows_by_id_not_file_order(self):
        path = self.directory / "predictions.csv"
        write_csv(path, [
            prediction("q2", "20", ("doc2",)),
            prediction("q1", "10", ("doc1",)),
        ])
        aligned = align_predictions(path, self.questions)
        self.assertEqual([row["id"] for row in aligned], ["q1", "q2"])


class CompareRunsReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.questions_path = self.directory / "questions.csv"
        self.metadata_path = self.directory / "metadata.csv"
        write_csv(self.questions_path, [
            question("q1", "10", ("doc1",)),
            question("q2", "20", ("doc2",)),
            question("q3", BLANK, ()),
        ])
        write_metadata(self.metadata_path)
        self.questions = load_questions(self.questions_path)

        self.clean_csv = self.directory / "clean.csv"
        self.clean_trace = self.directory / "clean.jsonl"
        write_csv(self.clean_csv, [
            prediction("q1", "10", ("doc1",)),
            blank_prediction("q2"),
            blank_prediction("q3"),
        ])
        write_trace(self.clean_trace, {"q1": 10, "q2": 10, "q3": 10})

        self.fallback_csv = self.directory / "fallback.csv"
        self.fallback_trace = self.directory / "fallback.jsonl"
        write_csv(self.fallback_csv, [
            prediction("q1", "99", ("doc1",)),
            prediction("q2", "20", ("doc2",)),
            blank_prediction("q3"),
        ])
        write_trace(self.fallback_trace, {"q1": 100, "q2": 100, "q3": 100})

        self.incomplete_csv = self.directory / "incomplete.csv"
        write_csv(self.incomplete_csv, [
            prediction("q1", "10"),
            prediction("q2", "20"),
            blank_prediction("q3"),
        ])

        self.manifest_path = self.directory / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "schema_version": 1,
            "questions": str(self.questions_path),
            "metadata": str(self.metadata_path),
            "runs": [
                {"id": "clean", "label": "Clean", "role": "candidate",
                 "predictions": str(self.clean_csv),
                 "trace": str(self.clean_trace)},
                {"id": "fallback", "label": "Fallback", "role": "fallback",
                 "predictions": str(self.fallback_csv),
                 "trace": str(self.fallback_trace)},
                {"id": "incomplete", "label": "Incomplete", "role": "baseline",
                 "predictions": str(self.incomplete_csv)},
            ],
            "pairings": [{
                "name": "clean -> fallback",
                "tier1": "clean",
                "fallback": "fallback",
                "live_reference": "clean",
            }],
            "exclusions": [{"id": "partial", "reason": "test exclusion"}],
        }), encoding="utf-8")

    def test_report_scores_citations_tokens_pareto_and_cascade(self):
        report = build_report(self.manifest_path)
        self.assertEqual(report["gold_questions"], 3)
        self.assertEqual(len(report["runs"]), 3)
        self.assertEqual([run["id"] for run in
                          report["baselines_incomplete_citations"]], ["incomplete"])

        clean = next(run for run in report["runs"] if run["id"] == "clean")
        self.assertEqual(clean["metrics"]["value"], 2 / 3)
        self.assertEqual(clean["metrics"]["abstention"], 2 / 3)
        self.assertTrue(clean["metrics"]["citation_complete"])
        self.assertEqual(clean["tokens"]["token_source"], "per_question_trace")
        self.assertEqual(clean["tokens"]["mean_tokens_per_question"], 10.0)

        incomplete = next(run for run in report["runs"]
                          if run["id"] == "incomplete")
        self.assertFalse(incomplete["metrics"]["citation_complete"])
        self.assertFalse(incomplete["comparability"]["pareto_eligible"])

        pareto_ids = {entry["id"] for entry in report["pareto_entries"]}
        self.assertEqual(pareto_ids, {"clean", "fallback", "clean -> fallback"})
        self.assertTrue(any(entry["pareto_optimal"]
                            for entry in report["pareto_entries"]))

        cascade = report["simulated_cascades"][0]
        self.assertEqual(cascade["escalated"], 2)
        self.assertEqual(cascade["resolved_by_fallback"], 1)
        self.assertEqual(cascade["fallback_abstained"], 1)
        self.assertEqual(cascade["metrics"]["value"], 1.0)
        self.assertEqual(cascade["tokens"]["total_tokens"], 230)
        self.assertEqual(cascade["tokens"]["token_source"],
                         "per_question_trace_replay")
        self.assertEqual(cascade["measurement"], "historical_replay")
        self.assertIn("live_reference", cascade)
        self.assertEqual(cascade["live_reference"]["run"], "clean")

        self.assertTrue(report["complementary_errors"])
        self.assertTrue(report["unique_correct"])
        self.assertEqual(report["exclusions"][0]["id"], "partial")

    def test_reported_usage_and_mismatch(self):
        usage_path = self.directory / "usage.json"
        usage_path.write_text(json.dumps({
            "questions": 3, "total_tokens": 90,
            "mean_total_tokens_per_question": 30.0,
        }), encoding="utf-8")
        usage = _reported_usage(usage_path, 3)
        self.assertTrue(usage["available"])
        self.assertEqual(usage["token_source"], "reported_aggregate")
        self.assertEqual(usage["mean_tokens_per_question"], 30.0)

        mismatch = _reported_usage(usage_path, 41)
        self.assertFalse(mismatch["available"])
        self.assertIn("covers 3", mismatch["reason"])

        no_trace = token_report(
            {"id": "x", "predictions": str(self.clean_csv),
             "usage": str(usage_path)},
            ["q1", "q2", "q3"], self.directory)
        self.assertEqual(no_trace["token_source"], "reported_aggregate")
        self.assertFalse(no_trace["per_question_available"])

    def test_nested_reported_usage_is_found(self):
        usage_path = self.directory / "nested-usage.json"
        usage_path.write_text(json.dumps({
            str(self.clean_trace): {
                "questions": 3, "total_tokens": 90,
                "mean_total_tokens_per_question": 30.0,
                "by_stage": {"answer": {"total_tokens": 90}},
            }
        }), encoding="utf-8")
        usage = _reported_usage(usage_path, 3)
        self.assertTrue(usage["available"])
        self.assertEqual(usage["total_tokens"], 90)


if __name__ == "__main__":
    unittest.main()
