import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_visual_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_visual_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8")


class CohortTest(unittest.TestCase):
    def test_derive_abstention_ids_is_ordered_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            write_jsonl(trace, [
                {"event": "cascade_route", "question_id": "q2",
                 "escalated": True, "reason": "rag_abstained"},
                {"event": "cascade_route", "question_id": "q1",
                 "escalated": False, "reason": "rag_answered"},
                {"event": "cascade_route", "question_id": "q3",
                 "escalated": True, "reason": "rag_abstained"},
                {"event": "cascade_route", "question_id": "q2",
                 "escalated": True, "reason": "rag_abstained"},
            ])
            self.assertEqual(ablation.derive_abstention_ids(trace), ["q2", "q3"])

    def test_question_subset_has_no_labels(self):
        source = [{
            "id": "q1", "question": "How much?", "answer_unit": "kWh",
            "answer_value": "1.5", "ref_id": "['paper']", "ref_url": "url",
            "supporting_materials": "quote", "explanation": "why",
        }]
        questions, labels = ablation.build_cohort(source, ["q1"])
        self.assertEqual(set(questions[0]), set(ablation.QUESTION_COLUMNS))
        self.assertNotIn("answer_value", questions[0])
        self.assertEqual(labels[0]["answer_value"], "1.5")


class ArmStateTest(unittest.TestCase):
    def make_arm(self, directory, expected, rows=1):
        write_jsonl(directory / "trace.jsonl", [])
        (directory / "config.json").write_text(
            json.dumps(expected), encoding="utf-8")
        with (directory / "predictions.csv").open(
                "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id"])
            for index in range(rows):
                writer.writerow([f"q{index}"])

    def test_complete_arm_requires_matching_config(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            arm.mkdir()
            expected = {"mode": "text-only", "workers": 1}
            self.make_arm(arm, expected)
            self.assertEqual(
                ablation.arm_status(arm, expected, 1)[0], "complete")
            self.assertEqual(
                ablation.arm_status(arm, {"mode": "other"}, 1)[0],
                "config_mismatch")

    def test_incomplete_arm_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            arm.mkdir()
            (arm / "trace.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ablation.prepare_arm_dir(arm, {"mode": "text-only"}, 1, False)
            self.assertTrue((arm / "trace.jsonl").is_file())

    def test_complete_arm_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            arm.mkdir()
            expected = {"mode": "text-only"}
            self.make_arm(arm, expected)
            self.assertEqual(
                ablation.prepare_arm_dir(arm, expected, 1, False), "reuse")


class ScoringTest(unittest.TestCase):
    def test_paired_direction_is_visual_minus_text(self):
        text = {"per_question": [
            {"id": "q1", "value": 0, "citation": 0, "abstention": 1,
             "weighted": 0.1},
            {"id": "q2", "value": 1, "citation": 1, "abstention": 1,
             "weighted": 1.0},
        ]}
        visual = {"per_question": [
            {"id": "q1", "value": 1, "citation": 1, "abstention": 1,
             "weighted": 1.0},
            {"id": "q2", "value": 0, "citation": 1, "abstention": 1,
             "weighted": 0.4},
        ]}
        paired = ablation.paired_changes(text, visual)
        self.assertAlmostEqual(paired["summary"]["delta_weighted"], 0.15)
        self.assertAlmostEqual(paired["summary"]["delta_value"], 0.0)
        self.assertEqual(paired["summary"]["improved"], 1)
        self.assertEqual(paired["summary"]["worsened"], 1)
        self.assertEqual(paired["summary"]["n"], 2)
        self.assertIn("caveat", paired["summary"])


if __name__ == "__main__":
    unittest.main()
