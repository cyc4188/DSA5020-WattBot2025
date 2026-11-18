"""Summarize provider-reported token usage from a WattBot JSONL trace."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def summarize(path):
    requests = 0
    responses = 0
    rows = set()
    stages = defaultdict(Counter)
    total = Counter()
    latencies = []
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "row":
                rows.add(event["question_id"])
            elif event["event"] == "llm_request":
                requests += 1
            elif event["event"] == "llm_response":
                responses += 1
                usage = event.get("usage") or {}
                if type(event.get("elapsed_ms")) is int:
                    latencies.append(event["elapsed_ms"])
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                if type(cached) is int:
                    total["cached_prompt_tokens"] += cached
                    total["responses_reporting_cache"] += 1
                if not all(type(usage.get(field)) is int for field in fields):
                    continue
                stage = stages[event["stage"]]
                stage.update({field: usage[field] for field in fields})
                stage["responses"] += 1
                total.update({field: usage[field] for field in fields})
                total["responses"] += 1
    return {
        "questions": len(rows), "requests": requests, "responses": responses,
        "responses_missing_usage": responses - total["responses"],
        **{field: total[field] for field in fields},
        "cached_prompt_tokens": total["cached_prompt_tokens"],
        "responses_reporting_cache": total["responses_reporting_cache"],
        "median_response_seconds": round(median(latencies) / 1000, 2) if latencies else None,
        "mean_total_tokens_per_question": round(total["total_tokens"] / len(rows), 2) if rows else None,
        "by_stage": {stage: dict(values) for stage, values in sorted(stages.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    print(json.dumps({str(path): summarize(path) for path in parser.parse_args().trace}, indent=2))
