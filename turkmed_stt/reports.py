from __future__ import annotations

import csv
from pathlib import Path


def summarize_benchmark(csv_path: str | Path) -> dict:
    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"count": 0}
    numeric = ["wer", "cer", "ds_wer", "rtf"]
    summary = {"count": len(rows), "model": rows[0].get("model"), "backend": rows[0].get("backend")}
    for key in numeric:
        values = [float(row[key]) for row in rows if row.get(key) not in {"", "None", None}]
        summary[f"mean_{key}"] = sum(values) / len(values) if values else None
    return summary


def render_run_markdown(run: dict) -> str:
    result = run.get("transcription", {})
    score = run.get("score", {})
    lines = [
        "# TurkMed STT Run",
        "",
        f"- Model: `{result.get('model_name')}`",
        f"- Backend: `{result.get('backend')}`",
        f"- Runtime: `{result.get('runtime_seconds')}` seconds",
        f"- RTF: `{result.get('rtf')}`",
        f"- WER: `{score.get('wer')}`",
        f"- CER: `{score.get('cer')}`",
        f"- DS-WER: `{score.get('ds_wer')}`",
        "",
        "## Transcript",
        "",
        result.get("text", ""),
    ]
    return "\n".join(lines)
