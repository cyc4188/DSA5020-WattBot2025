import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wattbot import doc_routing as dr
from wattbot.pipeline import BLANK, COLUMNS, abstention, read_csv, validate_rows
from wattbot.pure_rag_cascade import (
    StageResult,
    arbitrate_escalated,
    consensus_gate,
    run_cascade,
)

DOCS = {
    "doc1": "https://example.org/doc1",
    "doc2": "https://example.org/doc2",
    "doc3": "https://example.org/doc3",
}
METADATA = {key: {"id": key, "url": url} for key, url in DOCS.items()}


def question(qid, unit="kWh"):
    return {"id": qid, "question": f"Question {qid}?", "answer_unit": unit}


def make_row(qid, value, refs=("doc1",), unit="kWh"):
    refs = list(refs)
    return {
        "id": qid,
        "question": f"Question {qid}?",
        "answer": f"The value is {value}.",
        "answer_value": value,
        "answer_unit": unit,
        "ref_id": repr(refs),
        "ref_url": repr([DOCS[ref] for ref in refs]),
        "supporting_materials": "\n".join(f"[{ref}, p. 1] Value {value}." for ref in refs),
        "explanation": "The passage states the value directly.",
    }


def blank_row(qid):
    return abstention(question(qid))


class RecordingSolver:
    def __init__(self, rows, tokens=None):
        self.rows = rows
        self.tokens = tokens or {}
        self.calls = []

    def __call__(self, questions, stage_dir):
        self.calls.append([item["id"] for item in questions])
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        return StageResult(
            rows={item["id"]: self.rows[item["id"]] for item in questions},
            tokens={item["id"]: self.tokens.get(item["id"], 0) for item in questions},
        )


class ExplodingSolver:
    def __init__(self):
        self.calls = []

    def __call__(self, questions, stage_dir):
        self.calls.append([item["id"] for item in questions])
        raise AssertionError("Tier-2 must not be invoked")


class ConsensusGateTest(unittest.TestCase):
    def test_agreement_is_accepted_and_disagreement_escalates(self):
        q = question("q1")
        accepted, reasons = consensus_gate(q, make_row("q1", "100"), make_row("q1", "100.05"),
                                           METADATA)
        self.assertFalse(accepted)
        self.assertEqual(reasons, [])

        accepted, reasons = consensus_gate(q, make_row("q1", "100"), make_row("q1", "200"),
                                           METADATA)
        self.assertTrue(accepted)
        self.assertEqual(reasons, ["disagreement"])

    def test_abstention_and_missing_citation_escalate(self):
        q = question("q1")
        accepted, reasons = consensus_gate(q, blank_row("q1"), make_row("q1", "100"), METADATA)
        self.assertTrue(accepted)
        self.assertIn("abstain", reasons)

        uncited = make_row("q1", "100", refs=("doc1",))
        uncited["ref_url"] = BLANK
        accepted, reasons = consensus_gate(q, uncited, make_row("q1", "100"), METADATA)
        self.assertTrue(accepted)
        self.assertIn("missing_citation", reasons)


class ArbitrationTest(unittest.TestCase):
    def test_tier2_resolves_disagreement(self):
        q = question("q1")
        row, resolution = arbitrate_escalated(
            q, make_row("q1", "100"), make_row("q1", "200"), make_row("q1", "300"), METADATA)
        self.assertEqual(resolution, "escalated_tier2")
        self.assertEqual(row["answer_value"], "300")

    def test_tier2_corroborates_one_tier1_path(self):
        q = question("q1")
        row, resolution = arbitrate_escalated(
            q, make_row("q1", "100"), make_row("q1", "200"), make_row("q1", "100"), METADATA)
        self.assertEqual(resolution, "escalated_consensus")
        self.assertEqual(row["answer_value"], "100")

    def test_chart_abstention_falls_back_to_multimodal(self):
        q = question("q1")
        row, resolution = arbitrate_escalated(
            q, blank_row("q1"), make_row("q1", "5"), blank_row("q1"), METADATA)
        self.assertEqual(resolution, "escalated_fallback_multimodal")
        self.assertEqual(row["answer_value"], "5")

    def test_all_abstain(self):
        q = question("q1")
        row, resolution = arbitrate_escalated(
            q, blank_row("q1"), blank_row("q1"), blank_row("q1"), METADATA)
        self.assertEqual(resolution, "escalated_abstained")
        self.assertEqual(row["answer_value"], BLANK)

    def test_trace_metadata_weighting_selects_best_member(self):
        q = question("q1")
        row_a = make_row("q1", "100", refs=("doc1",))
        row_b = make_row("q1", "200", refs=("doc2",))
        row_t2 = make_row("q1", "100", refs=("doc3",))
        weighted_trace = {
            "has_calculation_binding": True,
            "verification_accepted": True,
            "verification_rejected_count": 0,
            "invalid_output_count": 0,
        }

        weighted_row, weighted_resolution = arbitrate_escalated(
            q, row_a, row_b, row_t2, METADATA,
            trace_a={}, trace_b={}, trace_t2=weighted_trace)
        plain_row, plain_resolution = arbitrate_escalated(
            q, row_a, row_b, row_t2, METADATA)

        self.assertEqual(weighted_resolution, "escalated_consensus")
        self.assertEqual(plain_resolution, "escalated_consensus")
        # The calculation-bound, verified Tier-2 candidate becomes the cluster
        # representative and supplies the citation.
        self.assertEqual(weighted_row["ref_id"], repr(["doc3"]))
        self.assertNotEqual(plain_row["ref_id"], repr(["doc3"]))

    def test_default_thinking_is_enabled(self):
        from wattbot.pure_rag_cascade import PiAgentOptions, build_parser
        self.assertTrue(PiAgentOptions(
            metadata=Path("m.csv"), index=Path("i.json"), docs_dir=Path("docs")).thinking)
        self.assertTrue(build_parser().parse_args([]).thinking)
        self.assertFalse(build_parser().parse_args(["--thinking", "false"]).thinking)


class CascadeOrchestrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name) / "cascade"

    def test_agreed_questions_never_reach_tier2(self):
        questions = [question("q1"), question("q2")]
        solver_a = RecordingSolver(
            {"q1": make_row("q1", "100"), "q2": make_row("q2", "100")},
            {"q1": 10, "q2": 20},
        )
        solver_b = RecordingSolver(
            {"q1": make_row("q1", "100"), "q2": make_row("q2", "200")},
            {"q1": 5, "q2": 6},
        )
        solver_t2 = RecordingSolver({"q2": make_row("q2", "100")}, {"q2": 100})

        rows, diagnostics = run_cascade(
            questions, METADATA,
            tier1_a_solver=solver_a,
            tier1_b_solver=solver_b,
            tier2_solver=solver_t2,
            output_dir=self.output_dir,
        )

        self.assertEqual(solver_t2.calls, [["q2"]])
        self.assertEqual(diagnostics["escalated_question_ids"], ["q2"])
        per_question = {item["id"]: item for item in diagnostics["per_question"]}
        self.assertEqual(per_question["q1"]["resolution"], "tier1_accepted")
        self.assertEqual(per_question["q2"]["resolution"], "escalated_consensus")
        self.assertEqual(per_question["q1"]["tokens"]["tier2_doc_routing"], 0)
        self.assertEqual(per_question["q1"]["tokens"]["total"], 15)
        self.assertEqual(per_question["q2"]["tokens"]["tier2_doc_routing"], 100)
        self.assertEqual(per_question["q2"]["tokens"]["total"], 126)
        self.assertEqual(diagnostics["summary"]["total_tokens"], 141)
        self.assertEqual(diagnostics["summary"]["tier2_actual_tokens"], 100)

        columns, written = read_csv(self.output_dir / "predictions.csv")
        validate_rows(columns, written, questions, METADATA)
        self.assertEqual([row["answer_value"] for row in written], ["100", "100"])

    def test_zero_escalation_skips_tier2(self):
        questions = [question("q1"), question("q2")]
        rows = {"q1": make_row("q1", "1"), "q2": make_row("q2", "2")}
        solver_a = RecordingSolver(rows, {"q1": 10, "q2": 10})
        solver_b = RecordingSolver(rows, {"q1": 4, "q2": 4})
        solver_t2 = ExplodingSolver()

        _, diagnostics = run_cascade(
            questions, METADATA,
            tier1_a_solver=solver_a,
            tier1_b_solver=solver_b,
            tier2_solver=solver_t2,
            output_dir=self.output_dir,
        )

        self.assertEqual(solver_t2.calls, [])
        summary = diagnostics["summary"]
        self.assertFalse(summary["tier2_invoked"])
        self.assertEqual(summary["escalation_count"], 0)
        self.assertEqual(summary["tier2_actual_tokens"], 0)
        self.assertEqual(summary["token_savings_vs_full_tier2_pct"], 100.0)
        self.assertEqual(summary["resolution_counts"], {"tier1_accepted": 2})
        skipped = json.loads(
            (self.output_dir / "tier2_doc_routing" / "_stage_result.json").read_text())
        self.assertTrue(skipped["skipped"])

    def test_three_way_resolution_paths(self):
        questions = [question("resolve"), question("corroborate"),
                     question("chart"), question("abstain")]
        solver_a = RecordingSolver({
            "resolve": make_row("resolve", "100"),
            "corroborate": make_row("corroborate", "100"),
            "chart": blank_row("chart"),
            "abstain": blank_row("abstain"),
        })
        solver_b = RecordingSolver({
            "resolve": make_row("resolve", "200"),
            "corroborate": make_row("corroborate", "200"),
            "chart": make_row("chart", "5"),
            "abstain": blank_row("abstain"),
        })
        solver_t2 = RecordingSolver({
            "resolve": make_row("resolve", "300"),
            "corroborate": make_row("corroborate", "100"),
            "chart": blank_row("chart"),
            "abstain": blank_row("abstain"),
        })

        _, diagnostics = run_cascade(
            questions, METADATA,
            tier1_a_solver=solver_a,
            tier1_b_solver=solver_b,
            tier2_solver=solver_t2,
            output_dir=self.output_dir,
        )

        self.assertEqual(solver_t2.calls, [["resolve", "corroborate", "chart", "abstain"]])
        per_question = {item["id"]: item for item in diagnostics["per_question"]}
        self.assertEqual(per_question["resolve"]["resolution"], "escalated_tier2")
        self.assertEqual(per_question["corroborate"]["resolution"], "escalated_consensus")
        self.assertEqual(per_question["chart"]["resolution"], "escalated_fallback_multimodal")
        self.assertEqual(per_question["abstain"]["resolution"], "escalated_abstained")
        self.assertEqual(per_question["chart"]["reasons"], ["abstain"])

    def test_resume_reuses_stage_artifacts(self):
        questions = [question("q1")]
        solver_a = RecordingSolver({"q1": make_row("q1", "100")}, {"q1": 10})
        solver_b = RecordingSolver({"q1": make_row("q1", "100")}, {"q1": 5})
        solver_t2 = ExplodingSolver()

        run_cascade(
            questions, METADATA,
            tier1_a_solver=solver_a,
            tier1_b_solver=solver_b,
            tier2_solver=solver_t2,
            output_dir=self.output_dir,
        )
        calls_before = (len(solver_a.calls), len(solver_b.calls))
        run_cascade(
            questions, METADATA,
            tier1_a_solver=solver_a,
            tier1_b_solver=solver_b,
            tier2_solver=solver_t2,
            output_dir=self.output_dir,
            resume=True,
        )
        self.assertEqual((len(solver_a.calls), len(solver_b.calls)), calls_before)


class DocRoutingSubsetTest(unittest.TestCase):
    def test_parse_question_ids_from_text_csv_and_inline(self):
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "ids.txt"
            text_path.write_text("q1\nq3\n", encoding="utf-8")
            csv_path = Path(directory) / "ids.csv"
            csv_path.write_text("id,note\nq2,x\nq4,y\n", encoding="utf-8")
            self.assertEqual(dr.parse_question_ids(str(text_path)), {"q1", "q3"})
            self.assertEqual(dr.parse_question_ids(str(csv_path)), {"q2", "q4"})
        self.assertEqual(dr.parse_question_ids("q1, q2"), {"q1", "q2"})
        self.assertIsNone(dr.parse_question_ids(None))

    def test_subset_filters_questions_and_writes_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "doc-routing"
            questions = [question("q1"), question("q2"), question("q3")]
            captured = []

            def fake_solve(question, router, client, emit, top_docs=3):
                captured.append(question["id"])
                return abstention(question)

            with patch.object(dr, "solve_question", fake_solve):
                rows, summary = dr.run_doc_routing_subset(
                    questions,
                    METADATA,
                    corpus={"sources": {}, "docs": {}},
                    cards={"cards": {}},
                    client=object(),
                    output_dir=output_dir,
                    question_ids=["q3", "q1"],
                    workers=1,
                )

            self.assertEqual(captured, ["q1", "q3"])
            self.assertEqual([row["id"] for row in rows], ["q1", "q3"])
            self.assertEqual(summary["question_ids"], ["q1", "q3"])
            columns, written = read_csv(output_dir / "predictions.csv")
            validate_rows(columns, written, [questions[0], questions[2]], METADATA)

    def test_unknown_question_id_raises(self):
        with self.assertRaises(ValueError):
            dr.run_doc_routing_subset(
                [question("q1")],
                METADATA,
                corpus={"sources": {}, "docs": {}},
                cards={"cards": {}},
                client=object(),
                output_dir=Path(tempfile.gettempdir()) / "unused-doc-routing",
                question_ids=["missing"],
            )


if __name__ == "__main__":
    unittest.main()
