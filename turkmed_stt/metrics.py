from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .normalization import (
    contains_medical_term,
    normalize_turkish_text,
    tokenize_chars,
    tokenize_words,
)

DEFAULT_MEDICAL_TERMS = [
    "hipertansiyon",
    "diyabet",
    "miyokard infarktüsü",
    "anjiyografi",
    "ekokardiyografi",
    "metformin",
    "insülin",
    "atorvastatin",
    "amlodipin",
    "antibiyotik",
    "pnömoni",
    "bronşit",
    "astım",
    "koledokolitiazis",
    "glomerülonefrit",
    "hemoglobin",
    "hba1c",
    "kreatinin",
    "trombosit",
    "radyoloji",
    "patoloji",
]


@dataclass
class Score:
    wer: float
    cer: float
    ds_wer: float | None
    ref_words: int
    hyp_words: int
    medical_ref_terms: int
    medical_hit_terms: int

    def to_dict(self) -> dict:
        return asdict(self)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, 1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, 1):
            substitution = previous[j - 1] + (0 if ref_item == hyp_item else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize_words(reference)
    hyp_tokens = tokenize_words(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def cer(reference: str, hypothesis: str) -> float:
    ref_tokens = tokenize_chars(reference)
    hyp_tokens = tokenize_chars(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def ds_wer(reference: str, hypothesis: str, medical_terms: list[str]) -> tuple[float | None, int, int]:
    ref_terms = [term for term in medical_terms if contains_medical_term(reference, term)]
    if not ref_terms:
        return None, 0, 0
    hits = sum(1 for term in ref_terms if contains_medical_term(hypothesis, term))
    return 1.0 - (hits / len(ref_terms)), len(ref_terms), hits


def score_pair(reference: str, hypothesis: str, medical_terms: list[str] | None = None) -> Score:
    terms = medical_terms or DEFAULT_MEDICAL_TERMS
    domain_wer, term_count, hit_count = ds_wer(reference, hypothesis, terms)
    return Score(
        wer=wer(reference, hypothesis),
        cer=cer(reference, hypothesis),
        ds_wer=domain_wer,
        ref_words=len(tokenize_words(reference)),
        hyp_words=len(tokenize_words(hypothesis)),
        medical_ref_terms=term_count,
        medical_hit_terms=hit_count,
    )


def load_medical_terms(path: str | Path | None) -> list[str]:
    if not path:
        return DEFAULT_MEDICAL_TERMS
    term_path = Path(path)
    if not term_path.exists():
        raise FileNotFoundError(f"Medical terms file not found: {term_path}")
    terms = [
        normalize_turkish_text(line)
        for line in term_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return sorted(set(terms)) or DEFAULT_MEDICAL_TERMS
