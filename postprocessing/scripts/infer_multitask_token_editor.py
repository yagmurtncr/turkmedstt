"""Run the single-checkpoint multi-head Turkish ASR postprocessor."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

try:
    from postprocessing.scripts.multitask_token_editor_model import MultiTaskTokenEditor
except ModuleNotFoundError:
    from multitask_token_editor_model import MultiTaskTokenEditor

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_PATTERN = re.compile(r"[,.;:!?…]")
PUNCT = {"NONE": "", "PERIOD": ".", "COMMA": ",", "QUESTION": "?", "EXCLAMATION": "!", "COLON": ":", "SEMICOLON": ";", "ELLIPSIS": "…"}
EXPLICIT_PUNCT = {f"SET_{label}": mark for label, mark in PUNCT.items() if label != "NONE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=Path("postprocessing/models/turkish-asr-multitask-editor"))
    parser.add_argument("--case-threshold", type=float, default=0.75)
    parser.add_argument("--punct-threshold", type=float, default=0.90)
    parser.add_argument("--edit-threshold", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads((args.model / "multitask_metadata.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = MultiTaskTokenEditor.from_pretrained(args.model)
    model.eval()
    words = WORD_PATTERN.findall(args.text)
    encoded = tokenizer([words], is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        output = model(**encoded)
    probabilities = [logits.softmax(-1)[0] for logits in (output.case_logits, output.punct_logits, output.edit_logits)]
    word_ids = encoded.word_ids(0)
    predictions = []
    for word_index in range(len(words)):
        positions = [index for index, value in enumerate(word_ids) if value == word_index]
        values = [probability[positions[-1]] for probability in probabilities]
        indexes = [int(value.argmax()) for value in values]
        predictions.append({
            "case": metadata["case_labels"][indexes[0]], "case_score": float(values[0][indexes[0]]),
            "punct": metadata["punct_labels"][indexes[1]], "punct_score": float(values[1][indexes[1]]),
            "edit": metadata["edit_labels"][indexes[2]], "edit_score": float(values[2][indexes[2]]),
        })
    matches, result, cursor = list(WORD_PATTERN.finditer(args.text)), [], 0
    for index, match in enumerate(matches):
        result.append(args.text[cursor:match.start()])
        word, prediction = match.group(0), predictions[index]
        if prediction["edit"] != "KEEP" and prediction["edit_score"] >= args.edit_threshold:
            replacement = metadata["replacement_map"].get(word.lower())
            if replacement:
                word = replacement[:1].upper() + replacement[1:] if word[:1].isupper() else replacement
        if prediction["case_score"] >= args.case_threshold:
            if prediction["case"] == "UPPER":
                word = word[:1].upper() + word[1:]
            elif prediction["case"] == "LOWER":
                word = word[:1].lower() + word[1:]
        result.append(word)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(args.text)
        gap = args.text[match.end():end]
        if prediction["punct_score"] >= args.punct_threshold:
            if prediction["punct"] == "REMOVE":
                gap = PUNCTUATION_PATTERN.sub("", gap)
            elif prediction["punct"] in EXPLICIT_PUNCT:
                gap = EXPLICIT_PUNCT[prediction["punct"]] + PUNCTUATION_PATTERN.sub("", gap)
            elif prediction["punct"] not in {"NONE", "KEEP"}:
                gap = PUNCTUATION_PATTERN.sub("", gap)
                gap = PUNCT[prediction["punct"]] + gap
        result.append(gap)
        cursor = end
    text = re.sub(r"\s+", " ", "".join(result)).strip()
    print(json.dumps({"text": text, "changed": text != args.text, "predictions": predictions}, ensure_ascii=False))


if __name__ == "__main__":
    main()
