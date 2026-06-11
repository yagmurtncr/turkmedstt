"""Build token labels for one multi-head Turkish ASR postprocessor."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_PATTERN = re.compile(r"[,.;:!?…]")
PUNCT_LABELS = ["NONE", "PERIOD", "COMMA", "QUESTION", "EXCLAMATION", "COLON", "SEMICOLON", "ELLIPSIS"]
MARK_TO_LABEL = {"": "NONE", ".": "PERIOD", ",": "COMMA", "?": "QUESTION", "!": "EXCLAMATION", ":": "COLON", ";": "SEMICOLON", "…": "ELLIPSIS"}
CASE_LABELS = ["KEEP", "UPPER", "LOWER"]
EDIT_LABELS = ["KEEP", "REPLACE"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("postprocessing/data/processed/readable_asr_dataset_augmented.csv"))
    parser.add_argument("--output", type=Path, default=Path("postprocessing/data/processed/multitask-token-edits.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("postprocessing/models/multitask-token-edit-metadata.json"))
    parser.add_argument("--report", type=Path, default=Path("postprocessing/reports/multitask_token_edit_dataset.json"))
    parser.add_argument("--min-replacement-count", type=int, default=20)
    return parser.parse_args()


def info(text: str) -> tuple[list[str], list[str]]:
    matches = list(WORD_PATTERN.finditer(text))
    words, punctuation = [match.group(0) for match in matches], []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        found = PUNCTUATION_PATTERN.search(text[match.end():end])
        punctuation.append(found.group(0) if found else "")
    return words, punctuation


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    replacement_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row["split"] != "train":
            continue
        source, _ = info(row["input_text"])
        target, _ = info(row["target_text"])
        if len(source) != len(target):
            continue
        for left, right in zip(source, target):
            if left.lower() != right.lower():
                replacement_counts[left.lower()][right.lower()] += 1
    replacement_map = {}
    for source, targets in replacement_counts.items():
        target, count = targets.most_common(1)[0]
        if count >= args.min_replacement_count and count / sum(targets.values()) >= 0.80:
            replacement_map[source] = target

    output, stats = [], Counter()
    for row in rows:
        source, source_punct = info(row["input_text"])
        target, target_punct = info(row["target_text"])
        if not source or len(source) != len(target):
            stats["unaligned_rows"] += 1
            continue
        case_labels, punct_labels, edit_labels, replacement_targets = [], [], [], []
        for left, right, left_mark, right_mark in zip(source, target, source_punct, target_punct):
            if left == right:
                case = "KEEP"
            elif left.lower() == right.lower():
                case = "UPPER" if right[:1].isupper() else "LOWER"
            else:
                case = "KEEP"
            candidate = replacement_map.get(left.lower())
            edit = "REPLACE" if candidate == right.lower() and left.lower() != right.lower() else "KEEP"
            case_labels.append(case)
            punct_labels.append(MARK_TO_LABEL[right_mark] if left_mark != right_mark else "NONE")
            edit_labels.append(edit)
            replacement_targets.append(candidate if edit == "REPLACE" else "")
            stats[f"case:{case}"] += 1
            stats[f"punct:{punct_labels[-1]}"] += 1
            stats[f"edit:{edit}"] += 1
        output.append({
            "id": row["id"], "words": source, "case_labels": case_labels,
            "punct_labels": punct_labels, "edit_labels": edit_labels,
            "replacement_targets": replacement_targets, "split": row["split"],
        })
        stats["rows"] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "case_labels": CASE_LABELS,
        "punct_labels": PUNCT_LABELS,
        "edit_labels": EDIT_LABELS,
        "replacement_map": replacement_map,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"rows": stats["rows"], "unaligned_rows": stats["unaligned_rows"], "replacement_map_size": len(replacement_map), "stats": dict(stats)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
