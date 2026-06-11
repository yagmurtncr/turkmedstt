"""Build sentence-isolated synthetic medical post-processing pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

PUNCT = re.compile(r"[,.;:!?…]")
TURKISH_FOLD = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
FIELDS = ["id", "input_text", "target_text", "reference_text", "task_type", "source", "asr_model", "license", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("postprocessing/data/medical_corpus_v3_text_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("postprocessing/data/processed/medical_postprocessor_synthetic.csv"))
    return parser.parse_args()


def split(row: dict[str, str]) -> str:
    if row["split"] == "train":
        return "train"
    value = int(hashlib.sha1(row["source_sentence_id"].encode()).hexdigest()[:8], 16)
    return "validation" if value % 2 == 0 else "test"


def strip_punctuation(text: str) -> str:
    return re.sub(r"\s+", " ", PUNCT.sub("", text)).strip()


def corrupt_term(text: str, terms: str, fold: bool = False) -> str:
    result = text
    for term in filter(None, (item.strip() for item in terms.split("|"))):
        replacement = term.translate(TURKISH_FOLD) if fold else (term[:-1] if len(term) > 4 else term)
        result = re.sub(re.escape(term), replacement, result, flags=re.IGNORECASE)
    return result


def main() -> None:
    args = parse_args()
    unique = {}
    with args.input.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            unique.setdefault(row["source_sentence_id"], row)
    output = []
    for row in unique.values():
        target = row["text"].strip()
        variants = [
            ("presentation", strip_punctuation(target.lower())),
            ("diacritic", strip_punctuation(target.translate(TURKISH_FOLD).lower())),
            ("medical_term_diacritic", strip_punctuation(corrupt_term(target, row["term_targets"], True).lower())),
            ("medical_term_typo", strip_punctuation(corrupt_term(target, row["term_targets"], False).lower())),
        ]
        for name, source in variants:
            if source == target:
                continue
            output.append({
                "id": f"{row['source_sentence_id']}:{name}",
                "input_text": source,
                "target_text": target,
                "reference_text": target,
                "task_type": "medical_synthetic_projection",
                "source": "medical_corpus_v3_text_synthetic",
                "asr_model": f"synthetic/{name}",
                "license": row["license"],
                "split": split(row),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    counts = {name: sum(row["split"] == name for row in output) for name in ("train", "validation", "test")}
    print({"unique_sentences": len(unique), "pairs": len(output), "splits": counts})


if __name__ == "__main__":
    main()
