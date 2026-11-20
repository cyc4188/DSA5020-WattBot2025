import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf

from wattbot.__main__ import main
from wattbot.pipeline import COLUMNS, EVIDENCE_FIELDS, load_metadata, read_csv, validate_rows
from wattbot.retrieval import sha256_file


TEXT_QUESTION = "How much energy did the system draw?"
FIGURE_QUESTION = "How much energy does Figure 1 show?"
MISSING_QUESTION = "How much energy does Figure 1 show for the other system?"
TEXT_QUOTE = "The system drew 1.25 kWh."
FIGURE_QUOTE = "Figure 1. Energy use: 2 kWh."


class TextClient:
    backend = "hkustgz"
    model = "fixture"

    def complete(self, stage, system, data, emit):
        emit("llm_request", stage=stage)
        if stage == "plan":
            return {"queries": ["energy"]}
        if stage == "verify":
            return {"supported": True, "reason": "The text gives the value."}
        if data["question"] == TEXT_QUESTION:
            return {"answerable": True, "answer": "1.25 kWh", "answer_value": "1.25",
                    "answer_unit": "kWh", "explanation": "The text gives the value.",
                    "evidence": [{"node_id": "paper:p1:b0", "quote": TEXT_QUOTE}]}
        return {"answerable": False}


class VisionClient:
    def __init__(self):
        self.unavailable = False

    def complete(self, stage, system, data, emit):
        emit("llm_request", stage=stage)
        return {"queries": ["energy"]} if stage == "plan" else {"answerable": False}

    def complete_vision(self, stage, system, content_parts, emit):
        if not any(part["type"] == "image_url" for part in content_parts):
            raise AssertionError("Expected a rendered PDF page")
        emit("llm_request", stage=stage)
        if self.unavailable:
            raise RuntimeError("Vision provider unavailable")
        if stage == "visual_verify":
            return {"supported": True, "reason": "The figure gives the value."}
        if json.loads(content_parts[0]["text"])["question"] != FIGURE_QUESTION:
            return {"answerable": False}
        return {"answerable": True, "answer": "2 kWh", "answer_value": "2",
                "answer_unit": "kWh", "explanation": "The figure gives the value.",
                "evidence": [{"page_id": "paper:p1", "visual_element": "Figure 1",
                              "quote": FIGURE_QUOTE}]}


class CascadeCliTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.output = directory / "predictions.csv"
        self.trace = directory / "trace.jsonl"
        self.questions = directory / "questions.csv"
        self.metadata = directory / "metadata.csv"
        self.questions.write_text(f"id,question,answer_unit\n"
                                  f"text,{TEXT_QUESTION},kWh\n"
                                  f"figure,{FIGURE_QUESTION},kWh\n"
                                  f"missing,{MISSING_QUESTION},kWh\n")
        self.metadata.write_text("id,url,title\npaper,https://example.org/paper,Paper\n")
        source = directory / "paper.pdf"
        page_text = f"{TEXT_QUOTE}\n{FIGURE_QUOTE}"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), page_text)
            pdf.save(source)
        index = {
            "sources": {"paper": {"url": "https://example.org/paper", "title": "Paper",
                                   "sha256": sha256_file(source)}},
            "pages": {"paper:p1": page_text},
            "nodes": [{"id": "paper:p1:b0", "page_id": "paper:p1", "ref_id": "paper",
                       "page": 1, "kind": "text", "text": page_text}],
        }
        text_index = directory / "clean.json"
        visual_index = directory / "visual.json"
        text_index.write_text(json.dumps({**index, "version": 2}))
        visual_index.write_text(json.dumps({**index, "version": 1}))
        self.argv = ["wattbot", "--metadata", str(self.metadata), "cascade",
                     "--questions", str(self.questions), "--index", str(text_index),
                     "--visual-index", str(visual_index), "--docs", str(directory),
                     "--output", str(self.output), "--trace", str(self.trace)]
        self.vision = VisionClient()

    def run_cascade(self, *options):
        with (patch("sys.argv", [*self.argv, *options]),
              patch("wattbot.__main__.DeepSeek", return_value=TextClient()),
              patch("wattbot.cascade.VisionClient", return_value=self.vision) as vision,
              patch("builtins.print")):
            main()
            return vision.call_count

    def trace_events(self):
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_visual_fallback_only_runs_for_rag_abstentions(self):
        self.assertEqual(self.run_cascade("--workers", "2"), 1)
        columns, rows = read_csv(self.output)
        _, questions = read_csv(self.questions)
        validate_rows(columns, rows, questions, load_metadata(self.metadata))
        self.assertEqual(columns, COLUMNS)
        self.assertEqual([row["answer_value"] for row in rows], ["1.25", "2", "is_blank"])
        self.assertTrue(all(rows[-1][field] == "is_blank" for field in EVIDENCE_FIELDS))
        events = self.trace_events()
        routed = {row["question_id"] for row in events
                  if row["event"] == "cascade_route" and row["escalated"]}
        visual = {row["question_id"] for row in events
                  if row.get("pipeline_stage") == "multimodal" and row["event"] == "llm_request"}
        self.assertEqual(routed, {"figure", "missing"})
        self.assertEqual(visual, routed)

    def test_fallback_failure_is_resumable(self):
        self.vision.unavailable = True
        with self.assertRaisesRegex(RuntimeError, "Vision provider unavailable"):
            self.run_cascade("--limit", "2")
        self.assertFalse(self.output.exists())
        self.vision.unavailable = False
        self.run_cascade("--limit", "2", "--resume")
        _, rows = read_csv(self.output)
        self.assertEqual([row["answer_value"] for row in rows], ["1.25", "2"])
        started = [row["question_id"] for row in self.trace_events()
                   if row["event"] == "question_start"]
        self.assertEqual(started, ["text", "figure", "figure"])
