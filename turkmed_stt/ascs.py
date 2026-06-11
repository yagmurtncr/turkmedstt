from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


NEGATIVE_HIGH_AROUSAL = {
    "angry",
    "anger",
    "fear",
    "fearful",
    "disgust",
    "disgusted",
    "frustrated",
    "stress",
    "stressed",
    "tense",
    "kizgin",
    "kızgın",
    "ofke",
    "öfke",
    "korku",
    "tiksinti",
}
NEGATIVE_MID_AROUSAL = {"sad", "sadness", "uzgun", "üzgün", "huzun", "hüzün"}
POSITIVE_MID_AROUSAL = {"happy", "happiness", "joy", "joyful", "mutlu", "neseli", "neşeli"}
LOW_AROUSAL = {"calm", "neutral", "relaxed", "normal", "sakin", "notr", "nötr"}


@dataclass(slots=True)
class SentimentSignal:
    label: str
    p_positive: float
    semantic_valence: float
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AcousticSignal:
    label: str
    confidence: float
    valence: float
    arousal: float
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ASCSResult:
    ascs: float
    semantic_valence: float
    acoustic_valence: float
    arousal: float
    divergence: float
    flag: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


def sentiment_signal(label: str, score: float) -> SentimentSignal:
    """Convert a binary sentiment classifier output into [-1, 1] valence."""
    normalized = normalize_label(label)
    score = clamp(score)
    if any(token in normalized for token in ("positive", "pos", "happy", "joy", "1", "olumlu")):
        p_positive = score
    elif any(token in normalized for token in ("negative", "neg", "sad", "anger", "0", "olumsuz")):
        p_positive = 1.0 - score
    else:
        p_positive = score
    semantic_valence = (2.0 * clamp(p_positive)) - 1.0
    return SentimentSignal(label=label, p_positive=clamp(p_positive), semantic_valence=semantic_valence)


def acoustic_signal(label: str, confidence: float = 1.0, raw: Any = None) -> AcousticSignal:
    """Map SER labels to continuous acoustic valence and arousal."""
    normalized = normalize_label(label)
    confidence = clamp(confidence)
    tokens = set(normalized.split()) | {normalized}

    if tokens & NEGATIVE_HIGH_AROUSAL or any(token in normalized for token in NEGATIVE_HIGH_AROUSAL):
        valence, arousal = -0.85, 0.85
    elif tokens & NEGATIVE_MID_AROUSAL or any(token in normalized for token in NEGATIVE_MID_AROUSAL):
        valence, arousal = -0.60, 0.55
    elif tokens & POSITIVE_MID_AROUSAL or any(token in normalized for token in POSITIVE_MID_AROUSAL):
        valence, arousal = 0.70, 0.55
    elif tokens & LOW_AROUSAL or any(token in normalized for token in LOW_AROUSAL):
        valence, arousal = 0.0, 0.20
    else:
        valence, arousal = 0.0, 0.50

    return AcousticSignal(
        label=label,
        confidence=confidence,
        valence=valence * confidence,
        arousal=clamp(arousal * (0.50 + 0.50 * confidence)),
        raw=raw,
    )


def compute_ascs(
    semantic_valence: float,
    acoustic_valence: float,
    arousal: float,
    *,
    threshold: float = 0.65,
    arousal_threshold: float = 0.50,
) -> ASCSResult:
    """Compute Acoustic-Semantic Consistency Score.

    ASCS is 1.0 when text polarity and acoustic valence agree. It decreases as
    valence diverges, with high arousal increasing the penalty.
    """
    semantic_valence = min(max(float(semantic_valence), -1.0), 1.0)
    acoustic_valence = min(max(float(acoustic_valence), -1.0), 1.0)
    arousal = clamp(arousal)
    divergence = abs(semantic_valence - acoustic_valence) / 2.0
    arousal_weight = 0.50 + (0.50 * arousal)
    score = clamp(1.0 - (divergence * arousal_weight))
    flag = score < threshold and arousal > arousal_threshold
    reason = "high_arousal_valence_mismatch" if flag else "consistent_or_low_arousal"
    return ASCSResult(
        ascs=score,
        semantic_valence=semantic_valence,
        acoustic_valence=acoustic_valence,
        arousal=arousal,
        divergence=divergence,
        flag=flag,
        reason=reason,
    )


def score_signals(sentiment: SentimentSignal, acoustic: AcousticSignal) -> ASCSResult:
    return compute_ascs(sentiment.semantic_valence, acoustic.valence, acoustic.arousal)


def normalize_label(label: str) -> str:
    return str(label or "").strip().lower().replace("_", " ").replace("-", " ")
