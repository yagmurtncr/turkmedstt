from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


def ensure_wav_16k_mono(input_path: str | Path, output_dir: str | Path) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}_16k.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if source.suffix.lower() == ".wav":
            return source
        raise RuntimeError("ffmpeg bulunamadı; wav dışı dosya normalize edilemiyor.")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-vn",
        str(target),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return target


def wav_duration_seconds(path: str | Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            return frames / float(rate)
    except Exception:
        return None
