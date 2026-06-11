"""Run a downloaded TurkMedSTT ASR readability post-processor."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
PUNCTUATION_PATTERN = re.compile(r"[,.;:!?…]")
PUNCT = {
    "NONE": "",
    "PERIOD": ".",
    "COMMA": ",",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
    "ELLIPSIS": "…",
}
EXPLICIT_PUNCT = {f"SET_{label}": mark for label, mark in PUNCT.items() if label != "NONE"}


class TurkishASRPostProcessor:
    def __init__(self, model_path: str | Path, profile: str | None = None):
        self.model_path = Path(model_path)
        self.metadata = json.loads((self.model_path / "multitask_metadata.json").read_text(encoding="utf-8"))
        profiles = json.loads((self.model_path / "deployment_profiles.json").read_text(encoding="utf-8"))
        self.profile = profiles[profile or profiles["default"]]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.eval()

    def __call__(self, text: str) -> str:
        words = WORD_PATTERN.findall(text)
        encoded = self.tokenizer(
            [words],
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = self.model(**encoded)
        probabilities = [
            logits.softmax(-1)[0]
            for logits in (output.case_logits, output.punct_logits, output.edit_logits)
        ]
        word_ids = encoded.word_ids(0)
        predictions = []
        for word_index in range(len(words)):
            positions = [index for index, value in enumerate(word_ids) if value == word_index]
            values = [probability[positions[-1]] for probability in probabilities]
            indexes = [int(value.argmax()) for value in values]
            predictions.append((
                self.metadata["case_labels"][indexes[0]],
                float(values[0][indexes[0]]),
                self.metadata["punct_labels"][indexes[1]],
                float(values[1][indexes[1]]),
                self.metadata["edit_labels"][indexes[2]],
                float(values[2][indexes[2]]),
            ))
        return self._apply(text, predictions)

    def _apply(self, text, predictions):
        matches, result, cursor = list(WORD_PATTERN.finditer(text)), [], 0
        for index, match in enumerate(matches):
            result.append(text[cursor:match.start()])
            word = match.group(0)
            case_label, case_score, punct_label, punct_score, edit_label, edit_score = predictions[index]
            if edit_label != "KEEP" and edit_score >= self.profile["edit_threshold"]:
                replacement = self.metadata["replacement_map"].get(word.lower())
                if replacement:
                    word = replacement[:1].upper() + replacement[1:] if word[:1].isupper() else replacement
            if case_score >= self.profile["case_threshold"]:
                if case_label == "UPPER":
                    word = word[:1].upper() + word[1:]
                elif case_label == "LOWER":
                    word = word[:1].lower() + word[1:]
            result.append(word)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            gap = text[match.end():end]
            if punct_score >= self.profile["punct_threshold"]:
                if punct_label == "REMOVE":
                    gap = PUNCTUATION_PATTERN.sub("", gap)
                elif punct_label in EXPLICIT_PUNCT:
                    gap = EXPLICIT_PUNCT[punct_label] + PUNCTUATION_PATTERN.sub("", gap)
                elif punct_label not in {"NONE", "KEEP"}:
                    gap = PUNCT[punct_label] + PUNCTUATION_PATTERN.sub("", gap)
            result.append(gap)
            cursor = end
        return re.sub(r"\s+", " ", "".join(result)).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", default=".")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    processor = TurkishASRPostProcessor(args.model, args.profile)
    print(processor(args.text))


if __name__ == "__main__":
    main()
