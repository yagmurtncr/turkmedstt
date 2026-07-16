from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "evidence/acosemantic/step_d_per_utterance.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "evidence/acosemantic"
DEFAULT_MODEL = "kubi565/emotion-berturk-turkish"

LABEL_MAP = {
    "LABEL_0": "happy",
    "LABEL_1": "fear",
    "LABEL_2": "anger",
    "LABEL_3": "sadness",
    "LABEL_4": "disgust",
    "LABEL_5": "surprise",
}
EMOTIONS = ["happy", "fear", "anger", "sadness", "disgust", "surprise"]
VALENCE = {
    "happy": 0.80,
    "fear": -0.75,
    "anger": -0.85,
    "sadness": -0.70,
    "disgust": -0.80,
    "surprise": 0.10,
}
AROUSAL = {
    "happy": 0.55,
    "fear": 0.85,
    "anger": 0.85,
    "sadness": 0.45,
    "disgust": 0.70,
    "surprise": 0.75,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute multi-class text-emotion ASCS for ASR ref/hyp pairs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-id", default="", help="Optional suffix for output filenames.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-model", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_path)
    rows = [row for row in rows if str(row.get("is_medical", "0")) in {"0", "", "False", "false"}]
    rows = limit_by_model(rows, args.limit_per_model)

    print(f"Rows: {len(rows)}")
    print(f"Loading text emotion model: {args.model}")
    pipe = load_text_emotion(args.model, args.device)

    for row in rows:
        row["_reference_norm"] = normalize_for_emotion(row.get("reference", ""))
        row["_hypothesis_norm"] = normalize_for_emotion(row.get("hypothesis", ""))

    unique_texts = sorted({row.get("_reference_norm", "") for row in rows} | {row.get("_hypothesis_norm", "") for row in rows})
    unique_texts = [text for text in unique_texts if text.strip()]
    print(f"Unique non-empty texts: {len(unique_texts)}")
    text_emotions = score_texts(pipe, unique_texts, args.batch_size)

    detail_rows = []
    for row in rows:
        ref_text = row.get("reference", "")
        hyp_text = row.get("hypothesis", "")
        ref_norm = row.get("_reference_norm", "")
        hyp_norm = row.get("_hypothesis_norm", "")
        ref = text_emotions.get(ref_norm) or neutral_signal()
        hyp = text_emotions.get(hyp_norm) or neutral_signal()
        metrics = compute_metrics(ref, hyp)
        if ref_norm and ref_norm == hyp_norm:
            metrics = {
                "emotion_js_similarity": 1.0,
                "valence_similarity": 1.0,
                "arousal_similarity": 1.0,
                "ascs_emotion": 1.0,
                "affective_flip": False,
            }
        detail_rows.append(
            {
                "model": row.get("model", ""),
                "audio": row.get("audio", ""),
                "wer": row.get("wer", ""),
                "reference": ref_text,
                "hypothesis": hyp_text,
                "reference_norm": ref_norm,
                "hypothesis_norm": hyp_norm,
                "ref_top_emotion": ref["top_emotion"],
                "ref_top_score": round(ref["top_score"], 6),
                "hyp_top_emotion": hyp["top_emotion"],
                "hyp_top_score": round(hyp["top_score"], 6),
                "ref_valence": round(ref["valence"], 6),
                "hyp_valence": round(hyp["valence"], 6),
                "ref_arousal": round(ref["arousal"], 6),
                "hyp_arousal": round(hyp["arousal"], 6),
                "emotion_js_similarity": round(metrics["emotion_js_similarity"], 6),
                "valence_similarity": round(metrics["valence_similarity"], 6),
                "arousal_similarity": round(metrics["arousal_similarity"], 6),
                "ascs_emotion": round(metrics["ascs_emotion"], 6),
                "affective_flip": int(metrics["affective_flip"]),
                "low_wer_high_impact": int(float(row.get("wer", 0) or 0) <= 0.25 and metrics["ascs_emotion"] < 0.75),
            }
        )

    summary_rows = summarize(detail_rows)
    output_stem = "step_f_ascs_emotion"
    if args.run_id:
        output_stem = f"{output_stem}_{safe_slug(args.run_id)}"
    detail_csv = out_dir / f"{output_stem}_per_utterance.csv"
    summary_csv = out_dir / f"{output_stem}_model_summary.csv"
    report_md = out_dir / f"{output_stem}_report.md"
    write_csv(detail_rows, detail_csv)
    write_csv(summary_rows, summary_csv)
    report_md.write_text(render_report(summary_rows, detail_rows, args.model), encoding="utf-8")

    print(f"detail_csv={detail_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"report_md={report_md}")


def load_text_emotion(model_id: str, device: str):
    import torch
    from transformers import pipeline

    if device == "cuda":
        device_id = 0
    elif device == "cpu":
        device_id = -1
    else:
        device_id = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'cuda:0' if device_id == 0 else 'cpu'}")
    return pipeline("text-classification", model=model_id, device=device_id, top_k=None, truncation=True, max_length=512)


def score_texts(pipe: Any, texts: list[str], batch_size: int) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        outputs = pipe(batch, batch_size=batch_size)
        for text, output in zip(batch, outputs):
            scored[text] = normalize_output(output)
        done = min(start + batch_size, len(texts))
        if done % (batch_size * 10) == 0 or done == len(texts):
            print(f"  emotion {done}/{len(texts)} elapsed={time.time() - t0:.1f}s")
    return scored


def normalize_output(output: Any) -> dict[str, Any]:
    probs = {emotion: 0.0 for emotion in EMOTIONS}
    for item in output:
        label = str(item.get("label", ""))
        emotion = LABEL_MAP.get(label, label.lower())
        if emotion in probs:
            probs[emotion] = float(item.get("score", 0.0))
    total = sum(probs.values())
    if total > 0:
        probs = {key: value / total for key, value in probs.items()}
    top_emotion, top_score = max(probs.items(), key=lambda item: item[1])
    valence = sum(probs[emotion] * VALENCE[emotion] for emotion in EMOTIONS)
    arousal = sum(probs[emotion] * AROUSAL[emotion] for emotion in EMOTIONS)
    return {
        "probs": probs,
        "top_emotion": top_emotion,
        "top_score": top_score,
        "valence": valence,
        "arousal": arousal,
    }


def neutral_signal() -> dict[str, Any]:
    probs = {emotion: 1.0 / len(EMOTIONS) for emotion in EMOTIONS}
    return {
        "probs": probs,
        "top_emotion": "empty_or_neutral",
        "top_score": 1.0 / len(EMOTIONS),
        "valence": sum(probs[emotion] * VALENCE[emotion] for emotion in EMOTIONS),
        "arousal": sum(probs[emotion] * AROUSAL[emotion] for emotion in EMOTIONS),
    }


def compute_metrics(ref: dict[str, Any], hyp: dict[str, Any]) -> dict[str, Any]:
    js_distance = jensen_shannon(ref["probs"], hyp["probs"])
    emotion_js_similarity = 1.0 - js_distance
    valence_similarity = 1.0 - (abs(ref["valence"] - hyp["valence"]) / 2.0)
    arousal_similarity = 1.0 - abs(ref["arousal"] - hyp["arousal"])
    ascs_emotion = (
        0.65 * emotion_js_similarity
        + 0.25 * clamp(valence_similarity)
        + 0.10 * clamp(arousal_similarity)
    )
    affective_flip = (
        (ref["valence"] >= 0.20 and hyp["valence"] <= -0.20)
        or (ref["valence"] <= -0.20 and hyp["valence"] >= 0.20)
    )
    return {
        "emotion_js_similarity": clamp(emotion_js_similarity),
        "valence_similarity": clamp(valence_similarity),
        "arousal_similarity": clamp(arousal_similarity),
        "ascs_emotion": clamp(ascs_emotion),
        "affective_flip": affective_flip,
    }


def jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    p = [left[emotion] for emotion in EMOTIONS]
    q = [right[emotion] for emotion in EMOTIONS]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    divergence = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return clamp(divergence / math.log(2.0))


def kl_divergence(left: list[float], right: list[float]) -> float:
    total = 0.0
    for p, q in zip(left, right):
        if p > 0 and q > 0:
            total += p * math.log(p / q)
    return total


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model"])].append(row)
    summary = []
    for model, model_rows in by_model.items():
        ascs = [float(row["ascs_emotion"]) for row in model_rows]
        wer = [float(row["wer"]) for row in model_rows if row.get("wer") not in ("", None)]
        flips = [int(row["affective_flip"]) for row in model_rows]
        high_impact = [int(row["low_wer_high_impact"]) for row in model_rows]
        summary.append(
            {
                "model": model,
                "n_general": len(model_rows),
                "mean_wer": round(statistics.mean(wer), 4) if wer else "",
                "mean_ascs_emotion": round(statistics.mean(ascs), 4),
                "median_ascs_emotion": round(statistics.median(ascs), 4),
                "low_ascs_emotion_rate_lt_0_75": round(sum(score < 0.75 for score in ascs) / len(ascs), 4),
                "affective_flip_rate": round(sum(flips) / len(flips), 4),
                "low_wer_high_impact_count": sum(high_impact),
                "ascs_emotion_wer_corr": round(correlation(ascs, wer), 4) if len(ascs) == len(wer) and len(ascs) > 2 else "",
            }
        )
    summary.sort(key=lambda row: (float(row["mean_ascs_emotion"]), -float(row["mean_wer"])), reverse=True)
    return summary


def correlation(left: list[float], right: list[float]) -> float:
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / len(left)
    std_left = statistics.stdev(left)
    std_right = statistics.stdev(right)
    if std_left == 0 or std_right == 0:
        return 0.0
    return cov / (std_left * std_right)


def render_report(summary: list[dict[str, Any]], detail_rows: list[dict[str, Any]], model_id: str) -> str:
    high_impact = [row for row in detail_rows if int(row["low_wer_high_impact"]) == 1]
    high_impact.sort(key=lambda row: (float(row["ascs_emotion"]), float(row["wer"])))
    high_impact = dedupe_examples(high_impact)
    lines = [
        "# AcoSemantic Step F - Multi-Class ASCS_emotion",
        "",
        "Scope: normal Turkish ASR outputs only. Medical/TTS rows are excluded.",
        f"Text emotion model: `{model_id}`",
        "",
        "Label mapping from model card: LABEL_0=happy, LABEL_1=fear, LABEL_2=anger, LABEL_3=sadness, LABEL_4=disgust, LABEL_5=surprise.",
        "Model-card license field says more information is needed; therefore this is used as an analysis model, not as redistributed training data.",
        "",
        "ASCS_emotion combines distribution similarity, valence preservation, and arousal preservation:",
        "",
        "`0.65 * JS_similarity + 0.25 * valence_similarity + 0.10 * arousal_similarity`",
        "",
        "## Model Ranking",
        "",
        "| Rank | Model | N | Mean WER | Mean ASCS_emotion | Flip rate | Low ASCS rate | Low-WER high-impact | Corr |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(summary, 1):
        lines.append(
            "| {idx} | `{model}` | {n} | {wer} | {ascs} | {flip} | {low} | {impact} | {corr} |".format(
                idx=idx,
                model=row["model"],
                n=row["n_general"],
                wer=row["mean_wer"],
                ascs=row["mean_ascs_emotion"],
                flip=row["affective_flip_rate"],
                low=row["low_ascs_emotion_rate_lt_0_75"],
                impact=row["low_wer_high_impact_count"],
                corr=row["ascs_emotion_wer_corr"],
            )
        )
    lines.extend(
        [
            "",
            "## Lowest-WER High-Impact Examples",
            "",
            "| Model | WER | ASCS_emotion | Ref emotion | Hyp emotion | Reference | Hypothesis |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    for row in high_impact[:25]:
        lines.append(
            "| `{model}` | {wer} | {ascs} | {ref_e} | {hyp_e} | {ref} | {hyp} |".format(
                model=escape_md(row["model"]),
                wer=row["wer"],
                ascs=row["ascs_emotion"],
                ref_e=row["ref_top_emotion"],
                hyp_e=row["hyp_top_emotion"],
                ref=escape_md(short(row["reference"])),
                hyp=escape_md(short(row["hypothesis"])),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "ASCS_emotion extends ASCS_text beyond positive/negative sentiment. It flags cases where the ASR hypothesis preserves words well enough to have low WER, but changes the inferred emotion distribution or valence. This gives the thesis a second AcoSemantic metric that is independent from acoustic SER and measurable on existing normal Turkish ASR benchmark outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def limit_by_model(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    if limit <= 0:
        return rows
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row.get("model", "")].append(row)
    limited = []
    for model_rows in by_model.values():
        limited.extend(model_rows[:limit])
    return limited


def dedupe_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        key = (
            row.get("model", ""),
            row.get("reference_norm", ""),
            row.get("hypothesis_norm", ""),
            row.get("ref_top_emotion", ""),
            row.get("hyp_top_emotion", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def normalize_for_emotion(value: str) -> str:
    text = str(value or "").casefold()
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\sçğıöşüâîû]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()
    return text


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def short(value: str, n: int = 110) -> str:
    value = " ".join(str(value or "").split())
    return value if len(value) <= n else value[: n - 3] + "..."


def escape_md(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
