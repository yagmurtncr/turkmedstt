from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .audio import ensure_wav_16k_mono, wav_duration_seconds


_MODEL_CACHE: dict[tuple[str, str], Any] = {}


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    backend: str
    model_name: str
    language: str
    text: str
    segments: list[TranscriptionSegment]
    audio_seconds: float | None
    runtime_seconds: float
    rtf: float | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["segments"] = [asdict(segment) for segment in self.segments]
        return data


def transcribe_audio(
    audio_path: str | Path,
    *,
    backend: str = "openai-whisper",
    model_name: str = "small",
    language: str = "tr",
    work_dir: str | Path = "runs/audio",
    prompt: str | None = None,
) -> TranscriptionResult:
    normalized = ensure_wav_16k_mono(audio_path, work_dir)
    audio_seconds = wav_duration_seconds(normalized)
    started = time.perf_counter()
    if backend == "openai-whisper":
        result = _transcribe_openai_whisper(normalized, model_name, language, prompt)
    elif backend in {"transformers", "transformers_ctc", "qwen3_asr"}:
        result = _transcribe_transformers(normalized, model_name, language, prompt, backend)
    elif backend == "faster-whisper":
        result = _transcribe_faster_whisper(normalized, model_name, language, prompt)
    elif backend == "qwen-asr":
        result = _transcribe_qwen_asr(normalized, model_name, language)
    elif backend == "transformers_pipeline":
        result = _transcribe_pipeline(normalized, model_name, language)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    runtime = time.perf_counter() - started
    segments = [
        TranscriptionSegment(
            start=float(item.get("start", 0.0)),
            end=float(item.get("end", 0.0)),
            text=str(item.get("text", "")).strip(),
        )
        for item in result.get("segments", [])
        if str(item.get("text", "")).strip()
    ]
    text = str(result.get("text", "")).strip()
    if not text:
        text = " ".join(segment.text for segment in segments).strip()
    return TranscriptionResult(
        backend=backend,
        model_name=model_name,
        language=language,
        text=text,
        segments=segments,
        audio_seconds=audio_seconds,
        runtime_seconds=runtime,
        rtf=(runtime / audio_seconds) if audio_seconds else None,
    )


def _transcribe_openai_whisper(path: Path, model_name: str, language: str, prompt: str | None) -> dict:
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper kurulu değil. requirements.txt kurulumunu yap.") from exc
    cache_key = ("openai-whisper", _openai_whisper_name(model_name))
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = whisper.load_model(cache_key[1])
        _MODEL_CACHE[cache_key] = model
    kwargs: dict[str, Any] = {"language": language}
    if prompt:
        kwargs["initial_prompt"] = prompt
    return model.transcribe(str(path), **kwargs)


def _load_ctc_processor(model_name: str):
    """Load CTC processor with greedy-only fallback when kenlm/pyctcdecode unavailable."""
    from transformers import AutoProcessor, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
    try:
        return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    except Exception as primary_exc:
        # Wav2Vec2ProcessorWithLM requires kenlm; fall back to plain Wav2Vec2Processor (greedy)
        try:
            fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            from transformers import Wav2Vec2CTCTokenizer
            tok = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
            return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
        except Exception:
            raise primary_exc


def _transcribe_transformers(path: Path, model_name: str, language: str, prompt: str | None, backend: str) -> dict:
    try:
        import torch
        import soundfile as sf
        from transformers import AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers/torch kurulu değil. requirements.txt kurulumunu yap.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio, sampling_rate = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    cache_key = (backend, model_name)
    cached = _MODEL_CACHE.get(cache_key)
    if backend == "transformers_ctc":
        if cached is None:
            processor = _load_ctc_processor(model_name)
            model = AutoModelForCTC.from_pretrained(model_name, trust_remote_code=True).to(device)
            cached = (processor, model)
            _MODEL_CACHE[cache_key] = cached
        processor, model = cached
        # MMS models need target_lang set before feature extraction
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_target_lang"):
            processor.tokenizer.set_target_lang("tur")
        inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        # Use tokenizer directly if processor.batch_decode unavailable
        if hasattr(processor, "batch_decode"):
            text = processor.batch_decode(predicted_ids)[0]
        elif hasattr(processor, "tokenizer"):
            text = processor.tokenizer.batch_decode(predicted_ids)[0]
        else:
            text = processor.decode(predicted_ids[0])
        return {"text": text, "segments": []}
    if cached is None:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        cached = (processor, model)
        _MODEL_CACHE[cache_key] = cached
    processor, model = cached
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.get("input_features")
    if input_features is None:
        input_features = inputs.get("input_values")
    input_features = input_features.to(device)
    if device == "cuda" and input_features.dtype == torch.float32:
        input_features = input_features.to(torch.float16)
    generate_kwargs: dict[str, Any] = {}
    if "whisper" in model_name.lower() and hasattr(processor, "get_decoder_prompt_ids"):
        generate_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.inference_mode():
        try:
            predicted_ids = model.generate(input_features, **generate_kwargs)
        except ValueError as exc:
            if "forced_decoder_ids" not in str(exc):
                raise
            predicted_ids = model.generate(input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return {"text": text, "segments": []}


def _transcribe_faster_whisper(path: Path, model_name: str, language: str, prompt: str | None) -> dict:
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper kurulu değil. requirements.txt kurulumunu yap.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    cache_key = ("faster-whisper", model_name, device, compute_type)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _MODEL_CACHE[cache_key] = model
    kwargs: dict[str, Any] = {"language": language}
    if prompt:
        kwargs["initial_prompt"] = prompt
    segments_iter, _info = model.transcribe(str(path), **kwargs)
    segments = []
    text_parts = []
    for segment in segments_iter:
        text = segment.text.strip()
        text_parts.append(text)
        segments.append({"start": segment.start, "end": segment.end, "text": text})
    return {"text": " ".join(text_parts).strip(), "segments": segments}


def _transcribe_pipeline(path: Path, model_name: str, language: str) -> dict:
    """Generic HF pipeline backend for models with custom/unknown processor classes."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise RuntimeError("transformers kurulu değil.") from exc
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    cache_key = ("transformers_pipeline", model_name)
    pipe = _MODEL_CACHE.get(cache_key)
    if pipe is None:
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            torch_dtype=dtype,
            trust_remote_code=True,
            chunk_length_s=30,
        )
        _MODEL_CACHE[cache_key] = pipe
    generate_kwargs: dict[str, Any] = {}
    if hasattr(pipe.model.config, "forced_decoder_ids"):
        generate_kwargs["language"] = language
        generate_kwargs["task"] = "transcribe"
    try:
        out = pipe(str(path), generate_kwargs=generate_kwargs) if generate_kwargs else pipe(str(path))
    except Exception:
        out = pipe(str(path))
    return {"text": (out.get("text") or "").strip(), "segments": []}


def _transcribe_qwen_asr(path: Path, model_name: str, language: str) -> dict:
    """Qwen3-ASR via the official qwen-asr package; falls back to transformers pipeline."""
    try:
        from qwen_asr import QwenASR
        cache_key = ("qwen-asr", model_name)
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = QwenASR(model_name)
            _MODEL_CACHE[cache_key] = model
        result = model.transcribe(str(path), language=language)
        if isinstance(result, str):
            return {"text": result.strip(), "segments": []}
        return {"text": str(result.get("text", "")).strip(), "segments": []}
    except ImportError:
        pass
    try:
        import torch
        import soundfile as sf
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
    except ImportError as exc:
        raise RuntimeError("qwen-asr and transformers are both unavailable.") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = ("qwen-asr-hf", model_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is None:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)
        cached = (processor, model)
        _MODEL_CACHE[cache_key] = cached
    processor, model = cached
    audio, sr = sf.read(str(path))
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = (inputs.get("input_features") or inputs.get("input_values")).to(device)
    if device == "cuda" and input_features.dtype == torch.float32:
        input_features = input_features.to(torch.float16)
    with torch.inference_mode():
        predicted_ids = model.generate(input_features, language=language, task="transcribe")
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return {"text": text.strip(), "segments": []}


def _openai_whisper_name(model_name: str) -> str:
    aliases = {
        "openai/whisper-tiny": "tiny",
        "openai/whisper-base": "base",
        "openai/whisper-small": "small",
        "openai/whisper-medium": "medium",
        "openai/whisper-large": "large",
        "openai/whisper-large-v2": "large-v2",
        "openai/whisper-large-v3": "large-v3",
    }
    return aliases.get(model_name, model_name)
