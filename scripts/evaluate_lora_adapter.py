from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Whisper LoRA adapter on a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-dir", default=None,
                        help="LoRA adapter dir. Omit (or 'none') to eval BASE model = M0.")
    parser.add_argument("--base-model", default="openai/whisper-large-v3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", help="Use 'all' to evaluate every row.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    evaluate(args)


def evaluate(args) -> None:
    import soundfile as sf
    import torch
    from librosa import load as librosa_load
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from turkmed_stt.metrics import load_medical_terms, score_pair

    rows = read_manifest(args.manifest)
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    rows = [row for row in rows if row.get("audio_filepath") and row.get("text")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No evaluable rows found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "lora_adapter_eval.csv"
    out_json = output_dir / "lora_adapter_eval_summary.json"
    out_md = output_dir / "LORA_ADAPTER_EVAL.md"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(args.base_model, language="tr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=dtype)
    use_adapter = bool(args.adapter_dir) and str(args.adapter_dir).lower() not in ("none", "base", "")
    if use_adapter:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"Loaded LoRA adapter: {args.adapter_dir}")
    else:
        args.adapter_dir = "BASE_M0"
        print("No adapter -> evaluating BASE model (M0).")
    model.to(device)
    model.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="tr", task="transcribe")
    terms = load_medical_terms(None)

    fieldnames = [
        "model",
        "adapter_dir",
        "audio_filepath",
        "reference",
        "hypothesis",
        "wer",
        "cer",
        "ds_wer",
        "runtime_seconds",
        "audio_seconds",
        "rtf",
        "medical_ref_terms",
        "medical_hit_terms",
    ]
    scores: list[dict] = []
    started_total = time.perf_counter()
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            audio_path = row["audio_filepath"]
            audio, _sr = librosa_load(audio_path, sr=16000, mono=True)
            audio_seconds = audio_duration(audio_path)
            inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device=device, dtype=dtype)

            started = time.perf_counter()
            with torch.inference_mode():
                predicted_ids = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                )
            runtime = time.perf_counter() - started
            hypothesis = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            reference = row.get("text", "")
            score = score_pair(reference, hypothesis, terms)
            record = {
                "model": args.base_model,
                "adapter_dir": args.adapter_dir,
                "audio_filepath": audio_path,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": score.wer,
                "cer": score.cer,
                "ds_wer": score.ds_wer,
                "runtime_seconds": runtime,
                "audio_seconds": audio_seconds,
                "rtf": runtime / audio_seconds if audio_seconds else "",
                "medical_ref_terms": score.medical_ref_terms,
                "medical_hit_terms": score.medical_hit_terms,
            }
            writer.writerow(record)
            scores.append(record)
            print(f"[{index}/{len(rows)}] wer={score.wer:.4f} cer={score.cer:.4f} hyp={hypothesis[:90]}")

    summary = summarize(scores, args, time.perf_counter() - started_total)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(summary), encoding="utf-8")
    print(out_md)


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def audio_duration(path: str | Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(scores: list[dict], args, total_runtime_seconds: float) -> dict:
    ds_values = [float(row["ds_wer"]) for row in scores if row["ds_wer"] not in ("", None)]
    ref_terms = sum(int(row["medical_ref_terms"]) for row in scores)
    hit_terms = sum(int(row["medical_hit_terms"]) for row in scores)
    return {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "manifest": args.manifest,
        "split": args.split,
        "rows": len(scores),
        "mean_wer": mean([float(row["wer"]) for row in scores]),
        "mean_cer": mean([float(row["cer"]) for row in scores]),
        "mean_ds_wer": mean(ds_values) if ds_values else None,
        "mean_rtf": mean([float(row["rtf"]) for row in scores if row["rtf"] != ""]),
        "medical_ref_terms_total": ref_terms,
        "medical_hit_terms_total": hit_terms,
        "medical_term_recall": (hit_terms / ref_terms) if ref_terms else None,
        "total_runtime_seconds": total_runtime_seconds,
    }


def render_markdown(summary: dict) -> str:
    ds_wer = summary["mean_ds_wer"]
    ds_text = "n/a" if ds_wer is None else f"{ds_wer:.4f}"
    recall = summary.get("medical_term_recall")
    recall_text = "n/a" if recall is None else f"{recall:.4f}"
    return "\n".join(
        [
            "# LoRA Adapter Evaluation",
            "",
            f"- Base model: `{summary['base_model']}`",
            f"- Adapter dir: `{summary['adapter_dir']}`",
            f"- Manifest: `{summary['manifest']}`",
            f"- Split: `{summary['split']}`",
            f"- Rows: {summary['rows']}",
            f"- Mean WER: {summary['mean_wer']:.4f}",
            f"- Mean CER: {summary['mean_cer']:.4f}",
            f"- Mean DS-WER (medical): {ds_text}",
            f"- Medical term recall: {recall_text} "
            f"({summary.get('medical_hit_terms_total')}/{summary.get('medical_ref_terms_total')})",
            f"- Mean RTF: {summary['mean_rtf']:.4f}",
            f"- Runtime seconds: {summary['total_runtime_seconds']:.2f}",
            "",
        ]
    )


if __name__ == "__main__":
    main()
