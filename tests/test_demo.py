"""Tests for the WattBot interactive demo (``demo/app.py``).

The suite is split into two parts:

* a fully synthetic fixture under a temporary directory, so the demo tests
  always pass on a fresh checkout that has no ``outputs/`` or ``data/docs/``;
* a real-output smoke test that is skipped when the saved cascade artefacts are
  not present.
"""

import csv
import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

from demo.app import BLANK, DemoData, create_server

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "id", "question", "answer", "answer_value", "answer_unit", "ref_id",
    "ref_url", "supporting_materials", "explanation",
]


def write_csv(path, columns, rows, encoding="utf-8-sig"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_pdf(path, text="WattBot synthetic page"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - older PyMuPDF installs
        import fitz as pymupdf
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text, fontsize=18)
    document.save(path)
    document.close()


def synthetic_row(qid, question, value="42", unit="kWh", refs=("doc1",), urls=None):
    urls = urls or [f"https://example.org/{ref}" for ref in refs]
    return {
        "id": qid,
        "question": question,
        "answer": f"The answer is {value} {unit}.",
        "answer_value": value,
        "answer_unit": unit,
        "ref_id": repr(list(refs)),
        "ref_url": repr(list(urls)),
        "supporting_materials": f"[{refs[0]}, p. 1] The value is {value} {unit}.",
        "explanation": "Synthetic explanation for the demo test.",
    }


class SyntheticFixture:
    """Build a miniature project tree that mirrors the real output layout."""

    def __init__(self, root):
        self.root = Path(root)
        self._write_inputs()
        self._write_outputs()
        self._write_traces()
        self._write_pdf()

    def _write_inputs(self):
        questions = [
            {"id": "qtest", "question": "How many kWh were used in the chart?", "answer_unit": "kWh"},
            {"id": "qvis", "question": "What percentage is shown in the figure?", "answer_unit": "percent"},
        ]
        write_csv(self.root / "data/raw/test_Q.csv", COLUMNS, [
            {**{column: "" for column in COLUMNS}, **question} for question in questions
        ])
        write_csv(self.root / "data/raw/train_QA.csv", COLUMNS, [])
        write_csv(
            self.root / "data/raw/metadata.csv",
            ["id", "url", "title"],
            [
                {"id": "doc1", "url": "https://example.org/doc1", "title": "Doc One"},
                {"id": "doc2", "url": "https://example.org/doc2", "title": "Doc Two"},
            ],
            encoding="latin-1",
        )

    def _write_outputs(self):
        base = self.root / "outputs/multi-paradigm-ensemble"
        final_rows = [
            synthetic_row("qtest", "How many kWh were used in the chart?"),
            synthetic_row("qvis", "What percentage is shown in the figure?",
                          value="12", unit="percent"),
        ]
        write_csv(base / "rag-cascade-pion-mm-doc.csv", COLUMNS, final_rows)
        write_json(base / "rag-cascade-pion-mm-doc.diag.json", {
            "summary": {
                "questions": 2, "answered": 2, "abstained": 0,
                "mean_tokens_per_question": 1234.5,
                "escalation_count": 1, "escalation_rate": 0.5,
                "trigger_counts": {"disagreement": 1, "abstain": 0},
                "resolution_counts": {"tier1_accepted": 1, "escalated_consensus": 1},
            },
            "per_question": [
                {"id": "qtest", "escalated": True, "reasons": ["disagreement"],
                 "resolution": "escalated_consensus", "tokens": 1500},
                {"id": "qvis", "escalated": False, "reasons": [],
                 "resolution": "tier1_accepted", "tokens": 900},
            ],
        })

    def _write_traces(self):
        candidate = synthetic_row("qtest", "How many kWh were used in the chart?")
        write_jsonl(
            self.root / "outputs/pi-v2-sandbox/pi-v2-thinking-on/test/trace.jsonl",
            [
                {"event": "run_start"},
                {"event": "llm_response", "question_id": "qtest", "stage": "pi_agent",
                 "usage": {"total_tokens": 100,
                           "completion_tokens_details": {"reasoning_tokens": 10}}},
                {"event": "tool_request", "question_id": "qtest", "tool": "eval_math",
                 "tool_id": "t1",
                 "arguments": {
                     "expression": "a / b",
                     "named_operands": {
                         "a": {"value": "10", "quote": "ten"},
                         "b": {"value": "2", "quote": "two"},
                     },
                     "result_unit": "ratio",
                 }},
                {"event": "tool_result", "question_id": "qtest", "tool": "eval_math",
                 "tool_id": "t1", "ok": True,
                 "data": {"result": "5", "unit": "ratio"}},
                {"event": "candidate", "question_id": "qtest", "row": candidate,
                 "citations": [{"ref_id": "doc1", "page": 1, "quote": "The value is 42 kWh."}]},
                {"event": "verification", "question_id": "qtest", "supported": True,
                 "decision": "accepted", "reason": "Synthetic verification passed."},
                {"event": "agent_done", "question_id": "qtest", "stop_reason": "toolUse"},
            ],
        )
        write_jsonl(
            self.root / "outputs/multimodal/test-mode2/trace.jsonl",
            [
                {"event": "run_start"},
                {"event": "visual_escalation", "question_id": "qtest", "escalate": True,
                 "reason": "text_abstained"},
                {"event": "visual_candidates", "question_id": "qtest", "candidates": [
                    {"ref_id": "doc1", "page": 1, "reasons": ["figure_page"]}
                ]},
                {"event": "visual_candidate", "question_id": "qtest", "row": candidate,
                 "citations": [{"ref_id": "doc1", "page": 1,
                                "quote": "Chart label reads 42 kWh."}]},
                {"event": "multimodal_decision", "question_id": "qtest",
                 "decision": "visual_accepted", "reason": "text_abstained"},
                {"event": "llm_response", "question_id": "qtest",
                 "usage": {"total_tokens": 200}},
            ],
        )
        write_jsonl(
            self.root / "outputs/doc-routing/test/trace.jsonl",
            [
                {"event": "run_start"},
                {"event": "doc_routing", "question_id": "qtest",
                 "bm25_top_docs": ["doc1", "doc2"], "llm_docs": ["doc1"],
                 "final_docs": ["doc1", "doc2"], "needs_multiple_docs": True,
                 "search_keywords": ["chart", "kWh"],
                 "reasoning": "Synthetic router reasoning."},
                {"event": "decision", "question_id": "qtest", "decision": "answered"},
                {"event": "row", "question_id": "qtest", "row": candidate},
                {"event": "llm_response", "question_id": "qtest",
                 "usage": {"total_tokens": 300}},
            ],
        )

    def _write_pdf(self):
        try:
            make_pdf(self.root / "data/docs/doc1.pdf")
        except ImportError:  # pragma: no cover
            pass


class DemoServerCase(unittest.TestCase):
    root = None

    @classmethod
    def setUpClass(cls):
        cls.server = create_server(cls.root, "127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=30)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        payload = response.read()
        status = response.status
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        return status, content_type, payload

    def json(self, path):
        status, _, payload = self.request("GET", path)
        self.assertEqual(status, 200, payload[:300])
        return json.loads(payload)


class TestSyntheticDemo(DemoServerCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tempdir.name)
        SyntheticFixture(cls.root)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.tempdir.cleanup()

    def test_index_serves_html(self):
        status, content_type, payload = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        text = payload.decode("utf-8")
        self.assertIn("WattBot", text)
        self.assertIn("Trace Explorer", text)
        self.assertIn("/api/overview", text)

    def test_overview_shape_and_kpis(self):
        overview = self.json("/api/overview")
        self.assertIn("kpis", overview)
        self.assertIn("counts", overview)
        self.assertIn("picks", overview)
        self.assertEqual(overview["kpis"]["public"], 0.895)
        self.assertEqual(overview["kpis"]["private"], 0.872)
        self.assertEqual(overview["counts"]["total"], 2)
        self.assertEqual(overview["counts"]["escalated"], 1)
        self.assertTrue(overview["datasets"]["cascade"])
        self.assertTrue(overview["datasets"]["pi"])
        self.assertTrue(overview["datasets"]["multimodal"])
        self.assertTrue(overview["datasets"]["doc_routing"])

    def test_questions_expose_filter_metadata(self):
        payload = self.json("/api/questions")
        questions = payload["questions"]
        self.assertEqual(len(questions), 2)
        by_id = {item["id"]: item for item in questions}
        self.assertTrue(by_id["qtest"]["escalated"])
        self.assertTrue(by_id["qtest"]["flags"]["disagreement"])
        self.assertTrue(by_id["qtest"]["flags"]["ast"])
        self.assertTrue(by_id["qtest"]["flags"]["visual"])
        self.assertFalse(by_id["qtest"]["flags"]["abstained"])
        self.assertTrue(by_id["qvis"]["flags"]["fast_path"])
        self.assertIn("unit_hint", by_id["qtest"])

    def test_question_breakdown_has_three_stages(self):
        detail = self.json("/api/question/qtest")
        self.assertEqual(detail["id"], "qtest")
        self.assertEqual(detail["final"]["answer_value"], "42")
        self.assertEqual(detail["final"]["ref_id"], ["doc1"])
        self.assertEqual(detail["cascade"]["resolution"], "escalated_consensus")
        self.assertTrue(detail["tier1a"]["available"])
        self.assertEqual(detail["tier1a"]["tools"][0]["tool"], "eval_math")
        self.assertTrue(detail["tier1a"]["calculations"])
        self.assertEqual(detail["tier1a"]["calculations"][0]["expression"], "a / b")
        self.assertEqual(detail["tier1a"]["reasoning_tokens"], 10)
        self.assertTrue(detail["tier1b"]["vision_triggered"])
        self.assertTrue(detail["tier1b"]["pages"])
        self.assertEqual(detail["tier1b"]["pages"][0]["ref_id"], "doc1")
        self.assertEqual(detail["tier2"]["router"]["final_docs"], ["doc1", "doc2"])
        self.assertEqual(detail["tier2"]["decision"]["decision"], "answered")
        self.assertEqual(detail["tokens"]["cascade"], 1500)
        self.assertEqual(detail["tokens"]["full_context"], 689804)

    def test_unknown_question_is_404(self):
        status, _, _ = self.request("GET", "/api/question/does-not-exist")
        self.assertEqual(status, 404)

    def test_page_image_returns_png(self):
        status, content_type, payload = self.request(
            "GET", "/api/page-image?ref_id=doc1&page=1")
        if status == 404 and not (self.root / "data/docs/doc1.pdf").is_file():
            self.skipTest("PyMuPDF is not available to create the synthetic PDF")
        self.assertEqual(status, 200, payload[:200])
        self.assertIn("image/png", content_type)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_page_image_rejects_bad_input(self):
        cases = [
            "/api/page-image?ref_id=..%2Fsecret&page=1",
            "/api/page-image?ref_id=doc1&page=0",
            "/api/page-image?ref_id=doc1&page=not-a-number",
        ]
        for path in cases:
            with self.subTest(path=path):
                status, _, _ = self.request("GET", path)
                self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/api/page-image?ref_id=missing&page=1")
        self.assertEqual(status, 404)
        status, _, _ = self.request("GET", "/api/page-image?ref_id=doc1&page=999")
        self.assertEqual(status, 404)

    def test_ask_rejects_empty_and_invalid_body(self):
        status, _, payload = self.request(
            "POST", "/api/ask", body=json.dumps({"question": ""}),
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)
        self.assertFalse(json.loads(payload)["ok"])
        status, _, _ = self.request(
            "POST", "/api/ask", body="not-json",
            headers={"Content-Type": "application/json"})
        self.assertEqual(status, 400)

    def test_ask_collection_survives_missing_live_configuration(self):
        # Empty questions are rejected before any heavy live import.
        data = DemoData(self.root)
        payload, status = data.ask("", "text")
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])


@unittest.skipUnless(
    (ROOT / "outputs/multi-paradigm-ensemble/rag-cascade-pion-mm-doc.diag.json").is_file(),
    "saved cascade outputs are not present",
)
class TestRealOutputsDemo(DemoServerCase):
    root = ROOT

    def test_real_showcase_question(self):
        overview = self.json("/api/overview")
        qids = {item["id"] for item in self.json("/api/questions")["questions"]}
        qid = "q015" if "q015" in qids else next(iter(qids))
        detail = self.json(f"/api/question/{qid}")
        self.assertEqual(detail["id"], qid)
        self.assertIn("final", detail)
        self.assertIn("tier1a", detail)
        self.assertIn("tier1b", detail)
        self.assertIn("tier2", detail)
        self.assertGreaterEqual(detail["tokens"]["full_context"], 1)
        self.assertTrue(qids)
        self.assertGreaterEqual(overview["counts"]["total"], len(qids))

    def test_real_page_image_if_docs_are_present(self):
        pdf = ROOT / "data/docs/han2024.pdf"
        if not pdf.is_file():
            self.skipTest("competition PDFs are not downloaded")
        status, content_type, payload = self.request(
            "GET", "/api/page-image?ref_id=han2024&page=7")
        self.assertEqual(status, 200)
        self.assertIn("image/png", content_type)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
