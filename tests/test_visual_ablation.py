import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_visual_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_visual_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8")


def question_rows(ids):
    return [{"id": question_id, "question": f"Question {question_id}",
             "answer_unit": ablation.BLANK} for question_id in ids]


def abstention_row(question):
    return {
        "id": question["id"],
        "question": question["question"],
        "answer": ablation.FALLBACK_ANSWER,
        "answer_value": ablation.BLANK,
        "answer_unit": ablation.BLANK,
        "ref_id": ablation.BLANK,
        "ref_url": ablation.BLANK,
        "supporting_materials": ablation.BLANK,
        "explanation": ablation.BLANK,
    }


def build_arm(directory, manifest, questions, expected, *, rows=None,
              trace=None, write_manifest=True):
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else [abstention_row(q) for q in questions]
    ablation.write_rows(directory / "predictions.csv",
                        ablation.COLUMNS, rows)
    ablation.write_json(directory / "config.json", expected)
    if write_manifest:
        ablation.write_json(directory / "input_manifest.json", manifest)
    if trace is None:
        trace = [{
            "event": "run_complete",
            "output_sha256": ablation.sha256_file(
                directory / "predictions.csv"),
        }]
    write_jsonl(directory / "trace.jsonl", trace)
    return rows


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

    def test_malformed_trace_line_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            trace.write_text('{"event": "row"}\nnot json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                ablation.read_jsonl(trace)


class ManifestTest(unittest.TestCase):
    def test_input_manifest_binds_digests_model_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("source.csv", "trace.jsonl", "index.json",
                         "metadata.csv", "catalog.json"):
                paths[name] = root / name
                paths[name].write_text(f"{name} contents", encoding="utf-8")
            subset_sha = hashlib.sha256(b"subset bytes").hexdigest()
            args = SimpleNamespace(
                questions_source=paths["source.csv"],
                trace=paths["trace.jsonl"],
                index=paths["index.json"],
                metadata=paths["metadata.csv"],
                catalog=paths["catalog.json"],
                questions=root / "questions.csv",
                model="model-name",
                provider="provider-name",
                top_k=10,
                max_images=3,
                dpi=180,
            )
            manifest = ablation.build_input_manifest(args, subset_sha)
            for key, name in (("source_questions", "source.csv"),
                              ("source_trace", "trace.jsonl"),
                              ("index", "index.json"),
                              ("metadata", "metadata.csv"),
                              ("catalog", "catalog.json")):
                self.assertEqual(
                    manifest[key]["sha256"],
                    hashlib.sha256(paths[name].read_bytes()).hexdigest())
            self.assertEqual(manifest["subset"]["sha256"], subset_sha)
            self.assertEqual(manifest["model"],
                             {"vlm_model": "model-name",
                              "provider": "provider-name"})
            self.assertEqual(manifest["settings"]["workers"], 1)
            self.assertFalse(manifest["settings"]["hybrid_retrieval"])


class OutputDirectoryTest(unittest.TestCase):
    def test_legacy_non_empty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "scores.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ablation.check_output_manifest(output)

    def test_fresh_and_matching_manifest_outputs_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / "fresh"
            self.assertIsNone(ablation.check_output_manifest(fresh))
            manifest = {"schema": 1}
            ablation.write_fresh_manifest(fresh, manifest)
            self.assertEqual(
                ablation.check_output_manifest(fresh), manifest)


class ArmStateTest(unittest.TestCase):
    def make(self, directory, *, manifest=None, questions=None,
             expected=None, **kwargs):
        manifest = manifest if manifest is not None else {"schema": 1}
        questions = questions if questions is not None else question_rows(["q0"])
        expected = expected if expected is not None else {"mode": "text-only"}
        arm = Path(directory) / "arm"
        build_arm(arm, manifest, questions, expected, **kwargs)
        return arm, manifest, questions, expected

    def test_complete_arm_requires_matching_config_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            arm, manifest, questions, expected = self.make(directory)
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, manifest, questions, {})[0], "complete")
            self.assertEqual(
                ablation.arm_status(
                    arm, {"mode": "other"}, manifest, questions, {})[0],
                "config_mismatch")
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, {"schema": 2}, questions, {})[0],
                "manifest_mismatch")

    def test_legacy_arm_without_manifest_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            arm, manifest, questions, expected = self.make(
                directory, write_manifest=False)
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, manifest, questions, {})[0], "incomplete")

    def test_incomplete_arm_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            arm.mkdir()
            (arm / "trace.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ablation.prepare_arm_dir(
                    arm, {"mode": "text-only"}, {"schema": 1},
                    question_rows(["q0"]), {}, False)
            self.assertTrue((arm / "trace.jsonl").is_file())

    def test_complete_arm_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            arm, manifest, questions, expected = self.make(directory)
            self.assertEqual(
                ablation.prepare_arm_dir(
                    arm, expected, manifest, questions, {}, False), "reuse")

    def test_new_arm_writes_manifest_before_run(self):
        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            manifest = {"schema": 1}
            self.assertEqual(
                ablation.prepare_arm_dir(
                    arm, {"mode": "text-only"}, manifest,
                    question_rows(["q0"]), {}, False), "run")
            self.assertEqual(
                json.loads((arm / "input_manifest.json").read_text(
                    encoding="utf-8")), manifest)

    def test_reuse_rejects_wrong_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = question_rows(["q0", "q1"])
            arm, manifest, _, expected = self.make(
                directory, questions=questions,
                rows=[abstention_row(q) for q in reversed(questions)])
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, manifest, questions, {})[0], "invalid_rows")

    def test_reuse_rejects_output_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            arm, manifest, questions, expected = self.make(
                directory, trace=[{"event": "run_complete",
                                   "output_sha256": "0" * 64}])
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, manifest, questions, {})[0],
                "output_hash_mismatch")

    def test_reuse_requires_completed_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            arm, manifest, questions, expected = self.make(
                directory, trace=[{"event": "row"}])
            self.assertEqual(
                ablation.arm_status(
                    arm, expected, manifest, questions, {})[0], "incomplete")


class HelpTest(unittest.TestCase):
    def test_help_does_not_require_local_benchmark_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--help"],
                cwd=directory, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--trace", result.stdout)


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
