from __future__ import annotations

import re
import unicodedata

_PUNCT_RE = re.compile(r"[^\w\sçğıöşüÇĞİÖŞÜ'/-]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def normalize_turkish_text(text: str, *, keep_apostrophe: bool = False) -> str:
    """Normalize Turkish ASR text before WER/CER scoring."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = text.replace("â", "a").replace("î", "i").replace("û", "u")
    if not keep_apostrophe:
        text = text.replace("'", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("-", " ").replace("/", " ")
    return _SPACE_RE.sub(" ", text).strip()


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_turkish_text(text)
    return normalized.split() if normalized else []


def tokenize_chars(text: str) -> list[str]:
    normalized = normalize_turkish_text(text)
    return list(normalized.replace(" ", ""))


def contains_medical_term(text: str, term: str) -> bool:
    normalized_text = normalize_turkish_text(text)
    normalized_term = normalize_turkish_text(term)
    if not normalized_term:
        return False
    return bool(re.search(rf"(^|\s){re.escape(normalized_term)}($|\s)", normalized_text))
