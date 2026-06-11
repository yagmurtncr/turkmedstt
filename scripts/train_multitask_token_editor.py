"""Train one encoder with separate casing, punctuation, and edit heads."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.multitask_token_editor_model import MultiTaskTokenEditor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/multitask-token-edits.jsonl"))
    parser.add_argument("--metadata", type=Path, default=Path("models/multitask-token-edit-metadata.json"))
    parser.add_argument("--encoder", default="ytu-ce-cosmos/turkish-mini-bert-uncased")
    parser.add_argument("--output-dir", type=Path, default=Path("models/turkish-asr-multitask-editor"))
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-train-rows", type=int, default=-1)
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument("--initial-model", type=Path, default=None)
    return parser.parse_args()


class MultiTaskCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = {name: [item.pop(name) for item in features] for name in ("case_labels", "punct_labels", "edit_labels")}
        batch = self.tokenizer.pad(features, return_tensors="pt")
        max_length = batch["input_ids"].shape[1]
        for name, values in labels.items():
            batch[name] = __import__("torch").tensor([value + [-100] * (max_length - len(value)) for value in values])
        return batch


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
    maps = {
        "case_labels": {label: i for i, label in enumerate(metadata["case_labels"])},
        "punct_labels": {label: i for i, label in enumerate(metadata["punct_labels"])},
        "edit_labels": {label: i for i, label in enumerate(metadata["edit_labels"])},
    }
    def tokenize(batch):
        encoded = tokenizer(batch["words"], is_split_into_words=True, truncation=True, max_length=128)
        for name, mapping in maps.items():
            output = []
            for index, word_labels in enumerate(batch[name]):
                previous, aligned = None, []
                for word_id in encoded.word_ids(batch_index=index):
                    aligned.append(-100 if word_id is None or word_id == previous else mapping[word_labels[word_id]])
                    previous = word_id
                output.append(aligned)
            encoded[name] = output
        return encoded
    def dataset(split):
        selected = [row for row in rows if row["split"] == split]
        if split == "train":
            changed, unchanged = [], []
            for row in selected:
                is_changed = (
                    any(x != "KEEP" for x in row["case_labels"])
                    or any(x != "KEEP" for x in row["edit_labels"])
                    or any(x not in {"NONE", "KEEP"} for x in row["punct_labels"])
                )
                (changed if is_changed else unchanged).append(row)
            if args.no_balanced_sampling:
                selected = changed + unchanged
            elif not unchanged:
                selected = changed
            else:
                rng = random.Random(42)
                balanced_unchanged = [
                    unchanged[index % len(unchanged)] for index in range(len(changed))
                ]
                selected = changed + balanced_unchanged
                rng.shuffle(selected)
            if args.max_train_rows > 0:
                selected = selected[: args.max_train_rows]
        data = Dataset.from_list(selected)
        return data.map(tokenize, batched=True, remove_columns=data.column_names)
    model = MultiTaskTokenEditor.from_encoder_pretrained(
        args.encoder,
        num_case_labels=len(metadata["case_labels"]),
        num_punct_labels=len(metadata["punct_labels"]),
        num_edit_labels=len(metadata["edit_labels"]),
    )
    if args.initial_model:
        initial = MultiTaskTokenEditor.from_pretrained(args.initial_model)
        model.encoder.load_state_dict(initial.encoder.state_dict())
        if model.case_head.weight.shape == initial.case_head.weight.shape:
            model.case_head.load_state_dict(initial.case_head.state_dict())
        if model.punct_head.weight.shape == initial.punct_head.weight.shape:
            model.punct_head.load_state_dict(initial.punct_head.state_dict())
        if model.edit_head.weight.shape == initial.edit_head.weight.shape:
            model.edit_head.load_state_dict(initial.edit_head.state_dict())
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.max_steps, num_train_epochs=args.epochs, per_device_train_batch_size=8,
        per_device_eval_batch_size=16, gradient_accumulation_steps=2, learning_rate=2e-5,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=100,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", dataloader_num_workers=0,
        dataloader_pin_memory=False, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset("train"), eval_dataset=dataset("validation"), data_collator=MultiTaskCollator(tokenizer), processing_class=tokenizer)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "multitask_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
