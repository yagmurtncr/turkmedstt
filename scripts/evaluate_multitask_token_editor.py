"""Blind-evaluate the single-encoder multi-head Turkish ASR editor."""

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

from scripts.multitask_token_editor_model import MultiTaskTokenEditor
from scripts.speech_shared import char_error_rate, clean_text, word_error_rate

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_PATTERN = re.compile(r"[,.;:!?…]")
PUNCT = {"NONE": "", "PERIOD": ".", "COMMA": ",", "QUESTION": "?", "EXCLAMATION": "!", "COLON": ":", "SEMICOLON": ";", "ELLIPSIS": "…"}
EXPLICIT_PUNCT = {f"SET_{label}": mark for label, mark in PUNCT.items() if label != "NONE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/readable_asr_dataset.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/turkish-asr-multitask-editor"))
    parser.add_argument("--output", type=Path, default=Path("reports/multitask_token_editor_v2_evaluation.json"))
    parser.add_argument("--task-type", default="asr_readability_projection")
    return parser.parse_args()


def apply(text, prediction, metadata, thresholds):
    matches, output, cursor = list(WORD_PATTERN.finditer(text)), [], 0
    for index, match in enumerate(matches):
        output.append(text[cursor:match.start()])
        word = match.group(0)
        case_label, case_score, punct_label, punct_score, edit_label, edit_score = prediction[index]
        if edit_label != "KEEP" and edit_score >= thresholds["edit"]:
            replacement = metadata["replacement_map"].get(word.lower())
            if replacement:
                word = replacement[:1].upper() + replacement[1:] if word[:1].isupper() else replacement
        if case_score >= thresholds["case"]:
            if case_label == "UPPER":
                word = word[:1].upper() + word[1:]
            elif case_label == "LOWER":
                word = word[:1].lower() + word[1:]
        output.append(word)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        gap = text[match.end():end]
        if punct_score >= thresholds["punct"]:
            if punct_label == "REMOVE":
                gap = PUNCTUATION_PATTERN.sub("", gap)
            elif punct_label in EXPLICIT_PUNCT:
                gap = EXPLICIT_PUNCT[punct_label] + PUNCTUATION_PATTERN.sub("", gap)
            elif punct_label not in {"NONE", "KEEP"}:
                gap = PUNCTUATION_PATTERN.sub("", gap)
                gap = PUNCT[punct_label] + gap
        output.append(gap)
        cursor = end
    return clean_text("".join(output))


def evaluate(rows, predictions, metadata, thresholds):
    totals = Counter()
    for row, prediction in zip(rows, predictions):
        source, target = clean_text(row["input_text"]), clean_text(row["target_text"])
        result = apply(source, prediction, metadata, thresholds)
        baseline_wer, result_wer = word_error_rate(target, source), word_error_rate(target, result)
        totals["rows"] += 1
        totals["changed"] += source != result
        totals["improved"] += result_wer < baseline_wer
        totals["worsened"] += result_wer > baseline_wer
        totals["correct_input_corrupted"] += source == target and result != target
        totals["baseline_wer"] += baseline_wer
        totals["prediction_wer"] += result_wer
        totals["baseline_cer"] += char_error_rate(target, source)
        totals["prediction_cer"] += char_error_rate(target, result)
    count = totals["rows"]
    return dict(thresholds) | {key: totals[key] for key in ("rows", "changed", "improved", "worsened", "correct_input_corrupted")} | {
        "baseline_wer": totals["baseline_wer"] / count, "prediction_wer": totals["prediction_wer"] / count,
        "baseline_cer": totals["baseline_cer"] / count, "prediction_cer": totals["prediction_cer"] / count,
    }


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.model / "multitask_metadata.json").read_text(encoding="utf-8"))
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["task_type"] == args.task_type and row["split"] in {"validation", "test"}]
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
    validation_rows, validation_predictions = [rows[i] for i in validation_indexes], [predictions[i] for i in validation_indexes]
    candidates = [
        evaluate(validation_rows, validation_predictions, metadata, {"case": case, "punct": punct, "edit": edit})
        for case, punct, edit in itertools.product((0.80, 0.90, 0.95, 0.98), (0.80, 0.90, 0.95, 0.98), (0.90, 0.95, 0.98, 0.99))
    ]
    eligible = [item for item in candidates if item["correct_input_corrupted"] == 0 and item["worsened"] <= max(2, item["improved"] * 0.1)]
    best = min(eligible or candidates, key=lambda item: (item["prediction_wer"], item["prediction_cer"], item["worsened"]))
    test = evaluate([rows[i] for i in test_indexes], [predictions[i] for i in test_indexes], metadata, {key: best[key] for key in ("case", "punct", "edit")})
    report = {"model": str(args.model), "selected_thresholds": {key: best[key] for key in ("case", "punct", "edit")}, "validation": best, "test": test, "eligible_candidates": len(eligible)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
