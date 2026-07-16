from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import wav_duration_seconds

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


@dataclass
class ManifestRow:
    audio_filepath: str
    text: str = ""
    duration: float | None = None
    sampling_rate: int = 16000
    speaker_id: str = "unknown"
    source: str = "local"
    domain: str = "general"
    icd10: str = ""
    synthetic: bool = False
    split: str = "train"
    quality: str = "unchecked"


def discover_audio_files(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def build_manifest_from_folder(
    root: str | Path,
    output_csv: str | Path,
    *,
    default_split: str = "train",
    domain: str = "general",
    source: str = "local",
) -> Path:
    rows = []
    for audio in discover_audio_files(root):
        duration = wav_duration_seconds(audio) if audio.suffix.lower() == ".wav" else None
        transcript = _find_sidecar_transcript(audio)
        rows.append(
            ManifestRow(
                audio_filepath=str(audio),
                text=transcript,
                duration=duration,
                source=source,
                domain=domain,
                split=default_split,
                quality="unchecked" if transcript else "missing_text",
            )
        )
    return write_manifest(rows, output_csv)


def write_manifest(rows: list[ManifestRow], output_csv: str | Path) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRow("").__dict__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(path: str | Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = read_manifest(path)
    required = {"audio_filepath", "text", "split", "synthetic"}
    if not rows:
        errors.append("Manifest is empty.")
        return errors, warnings
    missing = required - set(rows[0])
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    for index, row in enumerate(rows, 2):
        audio = row.get("audio_filepath", "")
        if not audio:
            errors.append(f"Line {index}: audio_filepath is empty.")
        elif not Path(audio).exists():
            warnings.append(f"Line {index}: audio file not found locally: {audio}")
        if not row.get("text", "").strip():
            warnings.append(f"Line {index}: transcript text is empty.")
        if row.get("split") not in {"train", "dev", "validation", "test"}:
            warnings.append(f"Line {index}: unusual split value: {row.get('split')}")
    return errors, warnings


def env_dataset_root() -> str | None:
    return os.getenv("TURKISH_STT_DATA_ROOT")


def _find_sidecar_transcript(audio_path: Path) -> str:
    candidates = [
        audio_path.with_suffix(".txt"),
        audio_path.parent / f"{audio_path.stem}.lab",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return ""
