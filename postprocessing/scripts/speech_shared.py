"""Shared helpers for the speech-transcript benchmark."""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from postprocessing.scripts.source_specs import MANIFEST_COLUMNS

FINAL_COLUMNS = [
    "id",
    "audio_id",
    "raw_asr",
    "readable_text",
    "wer",
    "cer",
    "error_type",
    "operation",
    "domain",
    "asr_model",
    "confidence",
    "source",
    "source_ref",
    "license",
    "split",
]

ALLOWED_ERROR_TYPES = {
    "clean",
    "diacritics",
    "punctuation",
    "casing",
    "spacing",
    "number_normalization",
    "grammar",
    "multiple",
}

ALLOWED_OPERATIONS = {
    "identity",
    "diacritics_restoration",
    "punctuation_restoration",
    "casing_correction",
    "spacing_correction",
    "number_normalization",
    "grammar_cleanup",
    "readability_improvement",
}

ALLOWED_SPLITS = {"train", "validation", "test"}
WHISPER_MODEL_NAME = "openai/whisper-base"

TURKISH_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "İ": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }
)

TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
DIGIT_SEQUENCE_PATTERN = re.compile(r"\b\d{3,}\b")
NUMBER_WORD_YEAR_PATTERN = re.compile(r"\bbin\s+dokuz\s+yuz\s+yirmi\s+uc\b", flags=re.IGNORECASE)

TURKISH_COMMON_WORDS = {
    "ve",
    "bir",
    "bu",
    "şu",
    "o",
    "için",
    "ile",
    "ama",
    "gibi",
    "çok",
    "daha",
    "da",
    "de",
    "mi",
    "mı",
    "mu",
    "mü",
    "ya",
    "ile",
    "olarak",
    "olan",
    "çünkü",
    "veya",
    "ancak",
    "fakat",
    "sonra",
    "önce",
    "kadar",
    "göre",
    "her",
    "bütün",
    "yani",
    "böyle",
    "şekilde",
    "şimdi",
    "bugün",
    "yarın",
    "şey",
    "şeyler",
    "değil",
    "var",
    "yok",
    "olan",
    "oldu",
    "olur",
    "oluyor",
    "yapmak",
    "ediyor",
    "eden",
    "edenler",
    "bana",
    "sana",
    "ona",
    "biz",
    "siz",
    "onlar",
    "çalışma",
    "metin",
    "konuşma",
    "dil",
}

MANIFEST_REQUIRED = set(MANIFEST_COLUMNS)


@dataclass(frozen=True)
class FinalRow:
    id: int
    audio_id: str
    raw_asr: str
    readable_text: str
    wer: float
    cer: float
    error_type: str
    operation: str
    domain: str
    asr_model: str
    confidence: float
    source: str
    source_ref: str
    license: str
    split: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": str(self.id),
            "audio_id": self.audio_id,
            "raw_asr": self.raw_asr,
            "readable_text": self.readable_text,
            "wer": f"{self.wer:.4f}",
            "cer": f"{self.cer:.4f}",
            "error_type": self.error_type,
            "operation": self.operation,
            "domain": self.domain,
            "asr_model": self.asr_model,
            "confidence": f"{self.confidence:.4f}",
            "source": self.source,
            "source_ref": self.source_ref,
            "license": self.license,
            "split": self.split,
        }


def clean_text(text: str) -> str:
    normalized = text.replace("\u00a0", " ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def normalize_for_match(text: str) -> str:
    lowered = clean_text(text).lower()
    return lowered.translate(TURKISH_TRANSLATION)


def remove_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def remove_punctuation(text: str) -> str:
    text = PUNCTUATION_PATTERN.sub(" ", text)
    return clean_text(text)


def tokenize_words(text: str) -> list[str]:
    return [token for token in clean_text(text).split() if token]


def tokenize_chars(text: str) -> list[str]:
    return list(clean_text(text))


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current_row = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current_row.append(
                min(
                    previous_row[j] + 1,
                    current_row[j - 1] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_tokens = tokenize_words(reference)
    hypothesis_tokens = tokenize_words(hypothesis)
    if not reference_tokens:
        return float(len(hypothesis_tokens) > 0)
    return _edit_distance(reference_tokens, hypothesis_tokens) / len(reference_tokens)


def char_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = tokenize_chars(reference)
    hypothesis_chars = tokenize_chars(hypothesis)
    if not reference_chars:
        return float(len(hypothesis_chars) > 0)
    return _edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)


def wer_bucket(wer_value: float) -> str:
    if wer_value <= 0.10:
        return "near_clean"
    if wer_value <= 0.30:
        return "minor_errors"
    if wer_value <= 0.80:
        return "moderate_errors"
    return "major_errors"


def looks_sensitive(text: str) -> bool:
    return bool(EMAIL_PATTERN.search(text) or URL_PATTERN.search(text) or DIGIT_SEQUENCE_PATTERN.search(text))


def looks_turkish(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return False

    tokens = [token.lower() for token in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", cleaned)]
    if not tokens:
        return False

    common_hits = sum(token in TURKISH_COMMON_WORDS for token in tokens)
    turkish_specific_chars = sum(character in TURKISH_CHARS for character in cleaned)
    alpha_chars = sum(character.isalpha() for character in cleaned)

    if common_hits >= 1:
        return True
    if alpha_chars and turkish_specific_chars / alpha_chars >= 0.01:
        return True
    if len(tokens) >= 3 and common_hits >= 1:
        return True
    return False


def classify_error_type(raw_asr: str, readable_text: str) -> tuple[str, str]:
    raw = clean_text(raw_asr)
    readable = clean_text(readable_text)

    if raw == readable:
        return "clean", "identity"

    if normalize_for_match(raw).replace(" ", "") == normalize_for_match(readable).replace(" ", ""):
        return "spacing", "spacing_correction"

    if remove_diacritics(normalize_for_match(raw)) == remove_diacritics(normalize_for_match(readable)):
        return "diacritics", "diacritics_restoration"

    if remove_punctuation(normalize_for_match(raw)) == remove_punctuation(normalize_for_match(readable)):
        return "punctuation", "punctuation_restoration"

    if normalize_for_match(raw) == normalize_for_match(readable):
        return "casing", "casing_correction"

    raw_digits = bool(DIGIT_SEQUENCE_PATTERN.search(raw))
    readable_digits = bool(DIGIT_SEQUENCE_PATTERN.search(readable))
    if raw_digits != readable_digits or NUMBER_WORD_YEAR_PATTERN.search(readable.lower()) or NUMBER_WORD_YEAR_PATTERN.search(raw.lower()):
        return "number_normalization", "number_normalization"

    return "multiple", "readability_improvement"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter if sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [{key: value or "" for key, value in row.items()} for row in reader]


def write_csv_rows(path: Path, rows: Iterable[dict[str, str]], *, fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or FINAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in (fieldnames or FINAL_COLUMNS)})


def validate_final_row(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    required = ["audio_id", "raw_asr", "readable_text", "error_type", "operation", "domain", "asr_model", "source", "source_ref", "license", "split"]
    for field in required:
        if not clean_text(row.get(field, "")):
            issues.append(f"empty {field}")
    if not row.get("asr_model", ""):
        issues.append("empty asr_model")
    if row.get("error_type", "").lower() not in ALLOWED_ERROR_TYPES:
        issues.append("invalid error_type")
    if row.get("operation", "").lower() not in ALLOWED_OPERATIONS:
        issues.append("invalid operation")
    if row.get("split", "").lower() not in ALLOWED_SPLITS:
        issues.append("invalid split")
    if not looks_turkish(row.get("readable_text", "")):
        issues.append("readable_text is not Turkish")
    try:
        wer_value = float(row.get("wer", ""))
        cer_value = float(row.get("cer", ""))
        if wer_value < 0:
            issues.append("negative wer")
        if cer_value < 0:
            issues.append("negative cer")
    except ValueError:
        issues.append("invalid wer/cer")
    return issues
