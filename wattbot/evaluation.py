"""Compute local WattBot 2025 diagnostic scores."""

from decimal import Decimal

from .pipeline import BLANK, EVIDENCE_FIELDS, NUMBER, RANGE, compact, parse_list
import re


def value_correct(predicted, gold):
    actual, expected = compact(predicted), compact(gold)
    if bounds := RANGE.fullmatch(expected):
        target = tuple(map(Decimal, bounds.groups()))
        if match := RANGE.fullmatch(actual):
            return tuple(map(Decimal, match.groups())) == target
        return bool(re.fullmatch(NUMBER, actual)) and target[0] <= Decimal(actual) <= target[1]
    if re.fullmatch(NUMBER, expected):
        if not re.fullmatch(NUMBER, actual):
            return False
        expected_number = Decimal(expected)
        return abs(Decimal(actual) - expected_number) <= abs(expected_number) * Decimal("0.001")
    return actual.casefold() == expected.casefold()


def evaluate(predictions, gold):
    details = []
    for prediction, truth in zip(predictions, gold, strict=True):
        refs = set(parse_list(prediction["ref_id"]))
        expected = set(parse_list(truth["ref_id"]))
        union = refs | expected
        gold_blank = truth["answer_value"] == BLANK
        predicted_blank = all(prediction[field] == BLANK for field in EVIDENCE_FIELDS)
        value = int(value_correct(prediction["answer_value"], truth["answer_value"]))
        citation = len(refs & expected) / len(union) if union else 1.0
        abstention = int(gold_blank == predicted_blank)
        details.append({
            "id": truth["id"], "value": value, "citation": citation,
            "abstention": abstention, "gold_blank": gold_blank, "predicted_blank": predicted_blank,
            "weighted": 0.75 * value + 0.15 * citation + 0.10 * abstention,
        })
    return {
        "questions": len(details),
        **{key: sum(row[key] for row in details) / len(details)
           for key in ("value", "citation", "abstention", "weighted")},
        "gold_blank": sum(row["gold_blank"] for row in details),
        "correct_blank": sum(row["gold_blank"] and row["predicted_blank"] for row in details),
        "false_blank": sum(not row["gold_blank"] and row["predicted_blank"] for row in details),
        "per_question": details,
    }
