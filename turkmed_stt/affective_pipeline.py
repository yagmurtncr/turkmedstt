from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .ascs import (
    AcousticSignal,
    ASCSResult,
    SentimentSignal,
    acoustic_signal,
    score_signals,
    sentiment_signal,
)
from .audio import ensure_wav_16k_mono
from .stt_core import transcribe_audio

DEFAULT_SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
DEFAULT_ACOUSTIC_MODEL = "SeaBenSea/hubert-large-turkish-speech-emotion-recognition"
FALLBACK_ACOUSTIC_MODEL = "dynann/emotion-speech-recognition"


@dataclass(slots=True)
class AffectiveResult:
    audio_filepath: str
    transcript: str
    asr_model: str
    sentiment_model: str
    acoustic_model: str
    sentiment: SentimentSignal
    acoustic: AcousticSignal
    ascs: ASCSResult

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sentiment"] = self.sentiment.to_dict()
        data["acoustic"] = self.acoustic.to_dict()
        data["ascs"] = self.ascs.to_dict()
        return data


def run_affective_pipeline(
    audio_path: str | Path,
    *,
    transcript: str | None = None,
    asr_backend: str = "openai-whisper",
    asr_model: str = "openai/whisper-small",
    sentiment_model: str = DEFAULT_SENTIMENT_MODEL,
    acoustic_model: str = DEFAULT_ACOUSTIC_MODEL,
    language: str = "tr",
    work_dir: str | Path = "runs/affective_audio",
) -> AffectiveResult:
    audio_path = Path(audio_path)
    if transcript is None:
        asr_result = transcribe_audio(
            audio_path,
            backend=asr_backend,
            model_name=asr_model,
            language=language,
            work_dir=work_dir,
        )
        transcript = asr_result.text

    sentiment = analyze_sentiment(transcript, sentiment_model)
    acoustic = analyze_acoustic_emotion(audio_path, acoustic_model, work_dir=work_dir)
    ascs_result = score_signals(sentiment, acoustic)
    return AffectiveResult(
        audio_filepath=str(audio_path),
        transcript=transcript,
        asr_model=asr_model,
        sentiment_model=sentiment_model,
        acoustic_model=acoustic_model,
        sentiment=sentiment,
        acoustic=acoustic,
        ascs=ascs_result,
    )


def analyze_sentiment(text: str, model_name: str = DEFAULT_SENTIMENT_MODEL) -> SentimentSignal:
    try:
        from transformers import pipeline

        classifier = pipeline("text-classification", model=model_name, top_k=None)
        result = classifier(text, truncation=True)
        candidates = result[0] if result and isinstance(result[0], list) else result
        if not candidates:
            raise RuntimeError("empty sentiment result")
        top = max(candidates, key=lambda item: float(item.get("score", 0.0)))
        signal = sentiment_signal(str(top.get("label", "unknown")), float(top.get("score", 0.0)))
        signal.raw = result
        return signal
    except Exception as exc:
        signal = _heuristic_sentiment(text)
        signal.raw = {"fallback": "lexicon", "error": repr(exc)}
        return signal


def analyze_acoustic_emotion(
    audio_path: str | Path,
    model_name: str = DEFAULT_ACOUSTIC_MODEL,
    *,
    work_dir: str | Path = "runs/affective_audio",
) -> AcousticSignal:
    try:
        import soundfile as sf
        from transformers import pipeline

        wav_path = ensure_wav_16k_mono(audio_path, work_dir)
        audio, sampling_rate = sf.read(str(wav_path))
        classifier = pipeline("audio-classification", model=model_name, top_k=5)
        predictions = classifier({"array": audio, "sampling_rate": sampling_rate})
        if not predictions:
            raise RuntimeError("empty acoustic result")
        top = max(predictions, key=lambda item: float(item.get("score", 0.0)))
        return acoustic_signal(str(top.get("label", "unknown")), float(top.get("score", 0.0)), predictions)
    except Exception as primary_exc:
        if model_name != FALLBACK_ACOUSTIC_MODEL:
            try:
                return analyze_acoustic_emotion(audio_path, FALLBACK_ACOUSTIC_MODEL, work_dir=work_dir)
            except Exception:
                pass
        return acoustic_signal("unknown", 0.0, {"fallback": "unknown", "error": repr(primary_exc)})


def _heuristic_sentiment(text: str) -> SentimentSignal:
    normalized = (text or "").lower()
    positive = ("iyi", "olumlu", "rahat", "mutlu", "harika", "duzeldi", "düzeldi")
    negative = ("kotu", "kötü", "agri", "ağrı", "endise", "endişe", "korku", "ates", "ateş")
    positive_hits = sum(1 for term in positive if term in normalized)
    negative_hits = sum(1 for term in negative if term in normalized)
    p_positive = 0.5 + (0.15 * positive_hits) - (0.15 * negative_hits)
    return sentiment_signal("heuristic", p_positive)
