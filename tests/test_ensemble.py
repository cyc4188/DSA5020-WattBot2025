import csv
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from wattbot.ensemble import (
    CampusModelClient, CorpusPageLookup, RunSpec, cascaded_route, cluster_candidates,
    compute_pareto_frontier, compute_unlabeled_diagnostics, consensus_ensemble,
    format_pareto_markdown, is_blank_row, load_trace_metadata, parse_run_spec,
    canonicalize_interval, reconcile_citations, recover_row_citations, trace_token_summary,
    values_equivalent,
)
from wattbot.pipeline import BLANK, COLUMNS, EVIDENCE_FIELDS, validate_rows

DOCS = {"doc1": "https://example.org/doc1", "doc2": "https://example.org/doc2",
        "doc3": "https://example.org/doc3"}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(directory):
    path = Path(directory) / "metadata.csv"
    with path.open("w", encoding="latin-1", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "url"])
        writer.writeheader()
        writer.writerows({"id": key, "url": url} for key, url in DOCS.items())
    return path


def write_trace(path, records):
    Path(path).write_text("".join(
        (record if isinstance(record, str) else json.dumps(record)) + "\n"
        for record in records), encoding="utf-8")
    return Path(path)


def make_row(qid, value, refs=("doc1",), unit="kWh"):
    refs = list(refs)
    return {
        "id": qid, "question": f"Question {qid}?", "answer": f"The value is {value}.",
        "answer_value": value, "answer_unit": unit,
        "ref_id": repr(refs) if refs else BLANK,
        "ref_url": repr([DOCS[r] for r in refs]) if refs else BLANK,
        "supporting_materials": "\n".join(f"[{r}, p. 1] Value {value}." for r in refs)
                                 if refs else BLANK,
        "explanation": "The passage states the value directly." if refs else BLANK,
    }


def blank_row(qid):
    return {"id": qid, "question": f"Question {qid}?", "answer": "Cannot answer.",
            **dict.fromkeys(EVIDENCE_FIELDS, BLANK)}


def make_run(directory, name, rows, trace=None, weight=1.0):
    csv_path = Path(directory) / f"{name}.csv"
    write_csv(csv_path, rows)
    trace_path = write_trace(Path(directory) / f"{name}.jsonl", trace) if trace else None
    return RunSpec(name, csv_path, trace_path, weight)


def token_events(qid, total):
    return [{"question_id": qid, "event": "llm_response",
             "usage": {"total_tokens": total, "prompt_tokens": total - 1,
                       "completion_tokens": 1}}]


class TestEquivalenceAndClustering(unittest.TestCase):
    def test_values_equivalent_numeric_tolerance_boundaries(self):
        cases = [("100", "100.1", True), ("100", "100.2", False),
                 ("55.6", "55.55555555555556", True), ("0", "0.0", True),
                 ("0.18", "0.1801", True), ("0.18", "0.181", False)]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                self.assertIs(values_equivalent(left, right), expected)

    def test_values_equivalent_ranges_categorical_and_blank(self):
        self.assertTrue(values_equivalent("[0.02,0.1]", "[0.02001,0.10005]"))
        self.assertFalse(values_equivalent("[0.02,0.1]", "[0.03,0.1]"))
        self.assertTrue(values_equivalent("Water consumption", "water  consumption"))
        self.assertTrue(values_equivalent(BLANK, BLANK))
        self.assertFalse(values_equivalent(BLANK, "100"))
        self.assertTrue(values_equivalent("0.02 - 0.1", "[0.02,0.1]"))
        self.assertTrue(values_equivalent("80 to 90", "[80,90]"))
        self.assertEqual(canonicalize_interval("80-90"), "[80,90]")
        self.assertEqual(canonicalize_interval("2024-2020"), "2024-2020")

    def test_cluster_candidates_complete_linkage_prevents_drift(self):
        candidates = [
            {"answer_value": "100.0", "score": 3.0, "run_index": 0, "valid_cited": True},
            {"answer_value": "100.08", "score": 2.0, "run_index": 1, "valid_cited": True},
            {"answer_value": "100.16", "score": 1.0, "run_index": 2, "valid_cited": True},
        ]
        clusters = cluster_candidates(candidates)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters[0]["representative_value"], "100.0")

    def test_cluster_candidates_orders_by_weight_then_cited_count(self):
        candidates = [
            {"answer_value": "1", "score": 1.0, "run_index": 0, "valid_cited": True},
            {"answer_value": "2", "score": 2.5, "run_index": 1, "valid_cited": False},
            {"answer_value": "3", "score": 2.5, "run_index": 2, "valid_cited": True},
        ]
        clusters = cluster_candidates(candidates)
        self.assertEqual([c["representative_value"] for c in clusters], ["3", "2", "1"])
        self.assertEqual(clusters[0]["cited_count"], 1)


class TestTraceAndCitationRecovery(unittest.TestCase):
    def test_trace_metadata_aggregates_all_event_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            write_trace(trace, [
                token_events("q1", 10)[0], token_events("q1", 10)[0],
                {"question_id": "q1", "event": "verification", "decision": "rejected",
                 "attempt": 1},
                {"question_id": "q1", "event": "verification",
                 "decision": "abstained_invalid_output", "attempt": 2},
                {"question_id": "q1", "event": "verification", "decision": "accepted",
                 "attempt": 3},
                {"question_id": "q1", "event": "calculation_binding"},
                {"question_id": "q1", "event": "citation_dropped", "count": 2},
                {"question_id": "q1", "event": "invalid_output", "count": 1},
                {"question_id": "q1", "event": "llm_response",
                 "content": json.dumps({"evidence": [{"quote": "quoted"}]})},
                "not-json",
            ])
            traces = load_trace_metadata(trace)
            meta = traces["q1"]
            self.assertEqual(meta["llm_calls"], 3)
            self.assertEqual(meta["total_tokens"], 20)
            self.assertTrue(meta["verification_accepted"])
            self.assertEqual(meta["verification_rejected_count"], 1)
            self.assertEqual(meta["verification_abstained_count"], 1)
            self.assertEqual(meta["verification_attempts"], 3)
            self.assertTrue(meta["has_calculation_binding"])
            self.assertEqual(meta["citation_dropped_count"], 2)
            self.assertEqual(meta["invalid_output_count"], 1)
            self.assertEqual(meta["raw_evidence"], [{"quote": "quoted"}])
            summary = trace_token_summary(trace)
            self.assertEqual(summary["questions"], 1)
            self.assertEqual(summary["mean_tokens_per_question"], 20.0)
            self.assertEqual(summary["per_question_tokens"], {"q1": 20})

    def test_recover_citations_hyphen_ellipsis_and_refuses_fabrication(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.json"
            index.write_text(json.dumps({
                "pages": {"doc1:p1": ("The energy consum-\nption rate was 1.25 kWh in total. "
                                      "Alpha beta gamma delta. The measured power was "
                                      "55.6 watts in the final run.")},
                "sources": {"doc1": {"stats": {"pages": 1}}},
            }), encoding="utf-8")
            corpus = CorpusPageLookup(index)
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            row = make_row("q1", "1.25", refs=())
            evidence = [{"ref_id": "doc1", "page": 1,
                         "quote": "The energy consum- ption rate was 1.25 kWh"}]
            recovered, ok = recover_row_citations(row, evidence, corpus, metadata)
            self.assertTrue(ok)
            self.assertEqual(recovered["ref_id"], "['doc1']")
            self.assertEqual(recovered["ref_url"], "['https://example.org/doc1']")
            self.assertIn("1.25 kWh", recovered["supporting_materials"])

            ellipsis, ok = recover_row_citations(
                make_row("q2", "55.6", refs=()),
                [{"ref_id": "doc1", "page": 1,
                  "quote": "Alpha beta gamma delta... The measured power was 55.6 watts"}],
                corpus, metadata)
            self.assertTrue(ok)
            self.assertIn("55.6", ellipsis["supporting_materials"])

            unknown, ok = recover_row_citations(
                make_row("q3", "1.25", refs=()),
                [{"ref_id": "docX", "page": 1, "quote": "The energy consumption rate"}],
                corpus, metadata)
            self.assertFalse(ok)
            self.assertEqual(unknown["ref_id"], BLANK)

            absent, ok = recover_row_citations(
                make_row("q4", "1.25", refs=()),
                [{"ref_id": "doc1", "page": 1, "quote": "zebra quantum telescope"}],
                corpus, metadata)
            self.assertFalse(ok)
            self.assertEqual(absent["ref_id"], BLANK)


class TestConsensusEnsemble(unittest.TestCase):
    question = {"id": "q1", "question": "Question q1?", "answer_unit": "kWh"}

    def test_weighted_voting_uses_calculation_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            plain = make_run(directory, "plain", [make_row("q1", "100")],
                             token_events("q1", 3))
            boosted = make_run(directory, "boosted", [make_row("q1", "200")], [
                {"question_id": "q1", "event": "verification", "decision": "accepted"},
                {"question_id": "q1", "event": "calculation_binding"},
            ], weight=1.0)
            rows, diag = consensus_ensemble([self.question], [plain, boosted], metadata)
            self.assertEqual(rows[0]["answer_value"], "200")
            self.assertEqual(diag["per_question"][0]["decision"], "consensus_answered")

    def test_uncited_row_corroborates_but_never_wins_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            runs = [
                make_run(directory, "cited", [make_row("q1", "100"),
                                              make_row("q2", "100", refs=())]),
                make_run(directory, "uncited", [make_row("q1", "100", refs=()),
                                                make_row("q2", "300", refs=())],
                         weight=2.0),
                make_run(directory, "blank1", [blank_row("q1"), blank_row("q2")]),
                make_run(directory, "blank2", [blank_row("q1"), blank_row("q2")]),
            ]
            questions = [self.question, {**self.question, "id": "q2",
                                        "question": "Question q2?"}]
            runs = [RunSpec(r.name, r.csv_path, r.trace_path, r.weight) for r in runs]
            rows, diag = consensus_ensemble(questions, runs, metadata)
            self.assertEqual(rows[0]["answer_value"], "100")
            self.assertEqual(diag["per_question"][0]["decision"], "consensus_answered")
            self.assertTrue(is_blank_row(rows[1]))
            self.assertEqual(diag["per_question"][1]["decision"],
                             "abstained_no_valid_cluster")

    def test_calibrated_quorum_and_blank_abstention(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            corroborated = [
                make_run(directory, "a", [make_row("q1", "100")]),
                make_run(directory, "b", [make_row("q1", "100")]),
                make_run(directory, "c", [blank_row("q1")]),
                make_run(directory, "d", [blank_row("q1")]),
            ]
            rows, diag = consensus_ensemble([self.question], corroborated, metadata,
                                            quorum_alpha=0.75)
            self.assertEqual(rows[0]["answer_value"], "100")
            singleton = [
                make_run(directory, "e", [make_row("q1", "100")]),
                make_run(directory, "f", [blank_row("q1")]),
                make_run(directory, "g", [blank_row("q1")]),
            ]
            rows, diag = consensus_ensemble([self.question], singleton, metadata,
                                            singleton_alpha=1.0)
            self.assertTrue(is_blank_row(rows[0]))
            self.assertEqual(diag["per_question"][0]["decision"], "abstained_quorum")
            blanks = [make_run(directory, "h", [blank_row("q1")]),
                      make_run(directory, "i", [blank_row("q1")])]
            rows, _ = consensus_ensemble([self.question], blanks, metadata)
            self.assertTrue(all(rows[0][field] == BLANK for field in EVIDENCE_FIELDS))
            validate_rows(COLUMNS, rows, [self.question], metadata)

    def test_reconcile_citation_strategies(self):
        metadata = {key: {"url": url} for key, url in DOCS.items()}
        members = [
            {"score": 3.0, "run_index": 0, "row": make_row("q", "1", ("doc1", "doc2"))},
            {"score": 2.0, "run_index": 1, "row": make_row("q", "1", ("doc1", "doc2"))},
            {"score": 1.0, "run_index": 2, "row": make_row("q", "1", ("doc3",))},
        ]
        kept, urls, _ = reconcile_citations(members, metadata, "majority")
        self.assertEqual(kept, ["doc1", "doc2"])
        self.assertEqual(urls, [DOCS["doc1"], DOCS["doc2"]])
        members = [
            {"score": 3.0, "run_index": 0, "row": make_row("q", "1", ("doc1", "doc2"))},
            {"score": 2.0, "run_index": 1, "row": make_row("q", "1", ("doc1",))},
        ]
        self.assertEqual(reconcile_citations(members, metadata, "intersection")[0],
                         ["doc1"])
        self.assertEqual(reconcile_citations(members, metadata, "best")[0],
                         ["doc1", "doc2"])


class TestCascadedRouter(unittest.TestCase):
    def test_escalation_tokens_resolutions_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            qids = ["q1", "q2", "q3", "q4", "q5"]
            questions = [{"id": qid, "question": f"Question {qid}?",
                          "answer_unit": "kWh"} for qid in qids]
            t1_rows = [make_row("q1", "100"), blank_row("q2"),
                       make_row("q3", "100", refs=()), make_row("q4", "100"),
                       make_row("q5", "100")]
            aux_rows = [make_row("q1", "100"), blank_row("q2"), make_row("q3", "100"),
                        make_row("q4", "100"), make_row("q5", "200")]
            t2_rows = [make_row("q1", "999", ("doc2",)), make_row("q2", "300", ("doc2",)),
                       make_row("q3", "200", ("doc2",)), make_row("q4", "300", ("doc2",)),
                       make_row("q5", "100", ("doc2",))]
            t1_trace = [event for qid in qids for event in token_events(qid, 10)]
            t1_trace.append({"question_id": "q4", "event": "verification",
                             "decision": "rejected"})
            aux_trace = [event for qid in qids for event in token_events(qid, 5)]
            t2_trace = [event for qid in qids for event in token_events(qid, 100)]
            tier1 = make_run(directory, "t1", t1_rows, t1_trace)
            aux = make_run(directory, "aux", aux_rows, aux_trace)
            tier2 = make_run(directory, "t2", t2_rows, t2_trace)
            rows, diag = cascaded_route(questions, tier1, tier2, metadata,
                                        tier1_aux=aux, escalate_on_verif_reject=1)
            self.assertEqual([q["escalated"] for q in diag["per_question"]],
                             [False, True, True, True, True])
            self.assertEqual(diag["per_question"][0]["tokens"], 15)
            self.assertEqual(diag["per_question"][1]["tokens"], 115)
            self.assertEqual(diag["per_question"][0]["resolution"], "tier1_accepted")
            self.assertEqual(diag["per_question"][1]["resolution"], "escalated_tier2")
            self.assertEqual(diag["per_question"][3]["reasons"], ["verif_reject"])
            self.assertEqual(diag["per_question"][4]["reasons"], ["disagreement"])
            self.assertIn("missing_citation", diag["per_question"][2]["reasons"])
            self.assertEqual(diag["summary"]["trigger_counts"]["abstain"], 1)
            self.assertEqual(diag["summary"]["escalation_count"], 4)
            self.assertEqual(diag["summary"]["escalation_rate"], 0.8)
            self.assertEqual(diag["summary"]["total_tokens"], 475)
            self.assertEqual(diag["summary"]["tier2_full_tokens"], 500)
            self.assertEqual(diag["summary"]["token_savings_vs_tier2_pct"], 5.0)
            validate_rows(COLUMNS, rows, questions, metadata)
            self.assertEqual(rows[1]["answer_value"], "300")


class TestParetoAndDiagnosticsAndClient(unittest.TestCase):
    def test_pareto_frontier_and_markdown(self):
        entries = [
            {"system": "A", "mean_tokens_per_question": 100, "weighted": 0.5},
            {"system": "B", "mean_tokens_per_question": 50, "weighted": 0.3},
            {"system": "C", "mean_tokens_per_question": 150, "weighted": 0.7},
            {"system": "D", "mean_tokens_per_question": 100, "weighted": 0.4},
        ]
        result = compute_pareto_frontier(entries)
        self.assertEqual([e["system"] for e in result], ["B", "A", "D", "C"])
        self.assertEqual({e["system"]: e["pareto_optimal"] for e in result},
                         {"A": True, "B": True, "C": True, "D": False})
        markdown = format_pareto_markdown(result)
        lines = markdown.splitlines()
        self.assertTrue(lines[0].startswith("| System |"))
        self.assertEqual(lines[1], "| --- | --- | --- | --- | --- | --- | --- | --- |")
        self.assertIn("| B |", lines[2])
        self.assertTrue(lines[2].endswith("| ✓ |"))

    def test_unlabeled_diagnostics_pairwise_and_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = {key: {"url": url} for key, url in DOCS.items()}
            questions = [{"id": qid, "question": f"Question {qid}?",
                          "answer_unit": "kWh"} for qid in ("q1", "q2", "q3")]
            run_a = make_run(directory, "a", [make_row("q1", "100"),
                                              make_row("q2", "200"), blank_row("q3")])
            run_b = make_run(directory, "b", [make_row("q1", "100"),
                                              blank_row("q2"), blank_row("q3")])
            diag = compute_unlabeled_diagnostics(questions, [run_a, run_b], metadata)
            pairwise = diag["pairwise"][0]
            self.assertEqual(pairwise["overall_value_agreement_rate"], 0.6667)
            self.assertEqual(pairwise["jointly_answered_count"], 1)
            self.assertEqual(pairwise["jointly_answered_agreement_rate"], 1.0)
            self.assertEqual(pairwise["abstention_agreement_rate"], 0.6667)
            self.assertEqual(pairwise["mean_citation_jaccard"], 0.6667)
            self.assertEqual(diag["consensus"], {"all_blank": 1,
                                                 "unanimous_non_blank": 1,
                                                 "majority_cluster": 0,
                                                 "no_majority_split": 1})

    def test_client_parses_fenced_json_with_decimal_floats(self):
        parsed = CampusModelClient._parse_content(
            '```json\n{"answer_value": 1.25, "count": 2}\n```')
        self.assertEqual(parsed["answer_value"], Decimal("1.25"))
        self.assertEqual(parsed["count"], 2)

    def test_parse_run_spec_variants(self):
        spec = parse_run_spec("name=csv")
        self.assertEqual((spec.name, spec.csv_path, spec.trace_path, spec.weight),
                         ("name", Path("csv"), None, 1.0))
        spec = parse_run_spec("name=csv::trace")
        self.assertEqual((spec.trace_path, spec.weight), (Path("trace"), 1.0))
        spec = parse_run_spec("name=csv::trace::1.5")
        self.assertEqual((spec.trace_path, spec.weight), (Path("trace"), 1.5))
        spec = parse_run_spec("name=csv::::2.0")
        self.assertEqual((spec.trace_path, spec.weight), (None, 2.0))


if __name__ == "__main__":
    unittest.main()
