from __future__ import annotations

import csv
import json
from pathlib import Path

from .manifest import read_manifest
from .metrics import load_medical_terms, score_pair
from .stt_core import transcribe_audio


def run_benchmark(
    manifest_csv: str | Path,
    output_csv: str | Path,
    *,
    backend: str,
    model_name: str,
    language: str = "tr",
    medical_terms_path: str | Path | None = None,
    limit: int | None = None,
    work_dir: str | Path = "runs/audio",
) -> Path:
    rows = read_manifest(manifest_csv)
    if limit:
        rows = rows[:limit]
    terms = load_medical_terms(medical_terms_path)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "backend",
        "audio_filepath",
        "reference",
        "hypothesis",
        "wer",
        "cer",
        "ds_wer",
        "rtf",
        "runtime_seconds",
        "audio_seconds",
        "medical_ref_terms",
        "medical_hit_terms",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result = transcribe_audio(
                row["audio_filepath"],
                backend=backend,
                model_name=model_name,
                language=language,
                work_dir=work_dir,
                prompt="Bu bir Türkçe tıbbi konuşma kaydıdır. Hipertansiyon, diyabet, antibiyotik, radyoloji.",
            )
            score = score_pair(row.get("text", ""), result.text, terms)
            writer.writerow(
                {
                    "model": model_name,
                    "backend": backend,
                    "audio_filepath": row["audio_filepath"],
                    "reference": row.get("text", ""),
                    "hypothesis": result.text,
                    "wer": score.wer,
                    "cer": score.cer,
                    "ds_wer": score.ds_wer,
                    "rtf": result.rtf,
                    "runtime_seconds": result.runtime_seconds,
                    "audio_seconds": result.audio_seconds,
                    "medical_ref_terms": score.medical_ref_terms,
                    "medical_hit_terms": score.medical_hit_terms,
                }
            )
            _write_run_json(out_path.parent, result.to_dict(), score.to_dict())
    return out_path


def _write_run_json(output_dir: Path, result: dict, score: dict) -> None:
    runs_dir = output_dir / "runs_json"
    runs_dir.mkdir(exist_ok=True)
    index = len(list(runs_dir.glob("run_*.json"))) + 1
    (runs_dir / f"run_{index:04d}.json").write_text(
        json.dumps({"transcription": result, "score": score}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
