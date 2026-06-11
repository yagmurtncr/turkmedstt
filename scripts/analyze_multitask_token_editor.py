"""Analyze task metrics, regressions, and threshold profiles for a multi-head editor."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_multitask_token_editor import PUNCT, apply
from scripts.multitask_token_editor_model import MultiTaskTokenEditor
from scripts.speech_shared import char_error_rate, clean_text, word_error_rate

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_PATTERN = re.compile(r"[,.;:!?…]")
MARKS = [",", ".", "?", "!", ";", ":", "…"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/readable_asr_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/turkish-asr-multitask-editor"))
    parser.add_argument("--output", type=Path, default=Path("reports/multitask_token_editor_v3_analysis.json"))
    parser.add_argument("--regressions-output", type=Path, default=Path("audits/multitask_token_editor_v3_regressions.csv"))
    parser.add_argument("--task-type", default="asr_readability_projection")
    return parser.parse_args()


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def words_and_marks(text: str) -> tuple[list[str], list[str]]:
    matches = list(WORD_PATTERN.finditer(text))
    words = [match.group(0) for match in matches]
    marks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found = PUNCTUATION_PATTERN.search(text[match.end():end])
        marks.append(found.group(0) if found else "NONE")
    return words, marks


def add_edit(values: Counter, prefix: str, source: str, target: str, prediction: str) -> None:
    needed = source != target
    predicted = prediction != source
    correct = prediction == target
    values[f"{prefix}_tp"] += predicted and needed and correct
    values[f"{prefix}_fp"] += predicted and not (needed and correct)
    values[f"{prefix}_fn"] += needed and not (predicted and correct)


def add_boolean_edit(values: Counter, prefix: str, needed: bool, predicted: bool, correct: bool) -> None:
    values[f"{prefix}_tp"] += predicted and needed and correct
    values[f"{prefix}_fp"] += predicted and not (needed and correct)
    values[f"{prefix}_fn"] += needed and not (predicted and correct)


def summarize(values: Counter) -> dict[str, object]:
    rows = values["rows"]
    return {
        "rows": rows,
        "changed": values["changed"],
        "improved": values["improved"],
        "worsened": values["worsened"],
        "correct_input_corrupted": values["correct_input_corrupted"],
        "baseline_wer": values["baseline_wer"] / rows if rows else 0.0,
        "prediction_wer": values["prediction_wer"] / rows if rows else 0.0,
        "baseline_cer": values["baseline_cer"] / rows if rows else 0.0,
        "prediction_cer": values["prediction_cer"] / rows if rows else 0.0,
        "tasks": {
            name: prf(values[f"{prefix}_tp"], values[f"{prefix}_fp"], values[f"{prefix}_fn"])
            for name, prefix in (
                ("any_edit", "any"),
                ("lexical", "lexical"),
                ("casing", "casing"),
                ("punctuation", "punctuation"),
            )
        },
        "punctuation_by_mark": {
            mark: prf(values[f"mark_{mark}_tp"], values[f"mark_{mark}_fp"], values[f"mark_{mark}_fn"])
            for mark in MARKS
        },
    }


def evaluate(rows, predictions, metadata, thresholds, collect_regressions: bool = False):
    totals = Counter()
    regressions = []
    by_source: dict[str, Counter] = {}
    by_asr_model: dict[str, Counter] = {}
    for row, token_predictions in zip(rows, predictions):
        source, target = clean_text(row["input_text"]), clean_text(row["target_text"])
        result = apply(source, token_predictions, metadata, thresholds)
        baseline_wer = word_error_rate(target, source)
        result_wer = word_error_rate(target, result)
        buckets = (
            totals,
            by_source.setdefault(row["source"], Counter()),
            by_asr_model.setdefault(row["asr_model"], Counter()),
        )
        for values in buckets:
            values["rows"] += 1
            values["changed"] += source != result
            values["improved"] += result_wer < baseline_wer
            values["worsened"] += result_wer > baseline_wer
            values["correct_input_corrupted"] += source == target and result != target
            values["baseline_wer"] += baseline_wer
            values["prediction_wer"] += result_wer
            values["baseline_cer"] += char_error_rate(target, source)
            values["prediction_cer"] += char_error_rate(target, result)

        source_words, source_marks = words_and_marks(source)
        target_words, target_marks = words_and_marks(target)
        result_words, result_marks = words_and_marks(result)
        if len(source_words) == len(target_words) == len(result_words):
            for sw, tw, pw, sm, tm, pm in zip(source_words, target_words, result_words, source_marks, target_marks, result_marks):
                for values in buckets:
                    add_edit(values, "lexical", sw.lower(), tw.lower(), pw.lower())
                    add_edit(values, "punctuation", sm, tm, pm)
                    add_boolean_edit(
                        values,
                        "casing",
                        sw.lower() == tw.lower() and sw != tw,
                        sw.lower() == pw.lower() and sw != pw,
                        pw == tw,
                    )
                    needed_any = sw != tw or sm != tm
                    predicted_any = sw != pw or sm != pm
                    correct_any = pw == tw and pm == tm
                    values["any_tp"] += predicted_any and needed_any and correct_any
                    values["any_fp"] += predicted_any and not (needed_any and correct_any)
                    values["any_fn"] += needed_any and not (predicted_any and correct_any)
                    for mark in MARKS:
                        needed_mark = tm == mark and sm != tm
                        predicted_mark = pm == mark and sm != pm
                        correct_mark = pm == tm
                        add_boolean_edit(values, f"mark_{mark}", needed_mark, predicted_mark, correct_mark)
        if collect_regressions and result_wer > baseline_wer:
            regressions.append({
                "id": row["id"],
                "source": row["source"],
                "asr_model": row["asr_model"],
                "raw_asr": source,
                "prediction": result,
                "target": target,
                "baseline_wer": baseline_wer,
                "prediction_wer": result_wer,
                "wer_delta": result_wer - baseline_wer,
            })
    return {
        "thresholds": thresholds,
        "overall": summarize(totals),
        "by_source": {name: summarize(values) for name, values in sorted(by_source.items())},
        "by_asr_model": {name: summarize(values) for name, values in sorted(by_asr_model.items())},
    }, regressions


def compact(result: dict) -> dict:
    overall = result["overall"]
    return {
        **result["thresholds"],
        **{key: overall[key] for key in ("changed", "improved", "worsened", "correct_input_corrupted", "prediction_wer", "prediction_cer")},
    }


def evaluate_sweep(rows, predictions, metadata, thresholds) -> dict:
    totals = Counter()
    for row, token_predictions in zip(rows, predictions):
        source, target = clean_text(row["input_text"]), clean_text(row["target_text"])
        result = apply(source, token_predictions, metadata, thresholds)
        baseline_wer = word_error_rate(target, source)
        result_wer = word_error_rate(target, result)
        totals["changed"] += source != result
        totals["improved"] += result_wer < baseline_wer
        totals["worsened"] += result_wer > baseline_wer
        totals["correct_input_corrupted"] += source == target and result != target
        totals["prediction_wer"] += result_wer
        totals["prediction_cer"] += char_error_rate(target, result)
    return {
        **thresholds,
        **{key: totals[key] for key in ("changed", "improved", "worsened", "correct_input_corrupted")},
        "prediction_wer": totals["prediction_wer"] / len(rows),
        "prediction_cer": totals["prediction_cer"] / len(rows),
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.model / "multitask_metadata.json").read_text(encoding="utf-8"))
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["task_type"] == args.task_type and row["split"] in {"validation", "test"}
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = MultiTaskTokenEditor.from_pretrained(args.model)
    model.eval()
    predictions = []
    for start in range(0, len(rows), 32):
        batch_words = [WORD_PATTERN.findall(row["input_text"]) for row in rows[start:start + 32]]
        encoded = tokenizer(batch_words, is_split_into_words=True, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.inference_mode():
            output = model(**encoded)
        probabilities = [logits.softmax(-1) for logits in (output.case_logits, output.punct_logits, output.edit_logits)]
        for batch_index, words in enumerate(batch_words):
            word_ids, item = encoded.word_ids(batch_index), []
            for word_index in range(len(words)):
                positions = [i for i, value in enumerate(word_ids) if value == word_index]
                values = [probability[batch_index, positions[-1]] for probability in probabilities]
                indexes = [int(value.argmax()) for value in values]
                item.append((
                    metadata["case_labels"][indexes[0]], float(values[0][indexes[0]]),
                    metadata["punct_labels"][indexes[1]], float(values[1][indexes[1]]),
                    metadata["edit_labels"][indexes[2]], float(values[2][indexes[2]]),
                ))
            predictions.append(item)

    validation_indexes = [i for i, row in enumerate(rows) if row["split"] == "validation"]
    test_indexes = [i for i, row in enumerate(rows) if row["split"] == "test"]
    validation_rows = [rows[i] for i in validation_indexes]
    validation_predictions = [predictions[i] for i in validation_indexes]
    test_rows = [rows[i] for i in test_indexes]
    test_predictions = [predictions[i] for i in test_indexes]

    grid = list(itertools.product(
        (0.75, 0.80, 0.85, 0.90, 0.95),
        (0.80, 0.85, 0.90, 0.95, 0.98),
        (0.90, 0.95, 0.98, 0.99, 0.995),
    ))
    validation_sweep = []
    for case, punct, edit in grid:
        thresholds = {"case": case, "punct": punct, "edit": edit}
        validation_sweep.append(evaluate_sweep(validation_rows, validation_predictions, metadata, thresholds))

    profiles = {}
    rules = {
        "safe": lambda item: item["correct_input_corrupted"] == 0 and item["worsened"] == 0,
        "balanced": lambda item: item["correct_input_corrupted"] == 0 and item["worsened"] <= max(2, item["improved"] * 0.03),
        "aggressive": lambda item: item["correct_input_corrupted"] == 0 and item["worsened"] <= max(5, item["improved"] * 0.10),
    }
    for name, rule in rules.items():
        eligible = [item for item in validation_sweep if rule(item)]
        selected = min(eligible or validation_sweep, key=lambda item: (item["prediction_wer"], item["prediction_cer"], item["worsened"]))
        thresholds = {key: selected[key] for key in ("case", "punct", "edit")}
        validation_result, _ = evaluate(validation_rows, validation_predictions, metadata, thresholds)
        test_result, _ = evaluate(test_rows, test_predictions, metadata, thresholds)
        profiles[name] = {"validation": validation_result, "test": test_result}

    fixed_thresholds = {"case": 0.8, "punct": 0.9, "edit": 0.95}
    fixed_validation, _ = evaluate(validation_rows, validation_predictions, metadata, fixed_thresholds)
    fixed_test, regressions = evaluate(test_rows, test_predictions, metadata, fixed_thresholds, True)
    report = {
        "model": str(args.model),
        "fixed_v3": {"validation": fixed_validation, "test": fixed_test},
        "validation_selected_profiles": profiles,
        "validation_threshold_sweep": sorted(validation_sweep, key=lambda item: (item["prediction_wer"], item["worsened"]))[:50],
        "regression_count": len(regressions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.regressions_output.parent.mkdir(parents=True, exist_ok=True)
    with args.regressions_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regressions[0]) if regressions else ["id"])
        writer.writeheader()
        writer.writerows(regressions)
    print(json.dumps({
        "fixed_v3": compact(fixed_test),
        "profiles": {name: compact(value["test"]) for name, value in profiles.items()},
        "regressions": len(regressions),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
