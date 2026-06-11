"""Cleanliness gate for the medical corpus v3 text manifest.

This is the guard that the medv2 contamination taught us to build. It FAILS
(exit 1) if any of these hold:
  * spoken text contains boilerplate / disclaimer / scenario tags / English
    area names (the exact failure mode of medv2);
  * a sentence (source_sentence_id) leaks across train and val splits;
  * an estimated clip duration exceeds the Whisper 30s window;
  * any of the 21 DEFAULT_MEDICAL_TERMS is missing from the corpus.

Run after build_medical_corpus_v3.py and before synthesis.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from turkmed_stt.metrics import DEFAULT_MEDICAL_TERMS  # noqa: E402

# Patterns that must NEVER appear in spoken text (medv2 failure mode).
FORBIDDEN_PATTERNS = [
    r"senaryo", r"scenario", r"\btts\b", r"sentetik", r"sentezlen",
    r"disclaimer", r"ses kayd[ıi]", r"egitim amac", r"eğitim amaç",
    r"varyant", r"bu ifade", r"bu metin", r"yapay zek",
]
# English clinical-area names that leaked as labels in medv2.
ENGLISH_AREA_WORDS = [
    "cardiology", "pulmonology", "endocrinology", "neurology", "gastroenterology",
    "nephrology", "oncology", "hematology", "emergency", "radiology", "pathology",
    "laboratory", "pharmacology", "pediatrics", "obstetrics", "gynecology",
    "urology", "psychiatry", "dermatology", "otolaryngology", "orthopedics",
    "rheumatology", "ophthalmology", "infectious",
]
MAX_DURATION_S = 30.0


def _low(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/medical_corpus_v3/medical_corpus_v3_text_manifest.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8-sig")))
    errors: list[str] = []
    warnings: list[str] = []

    forbidden_re = re.compile("|".join(FORBIDDEN_PATTERNS))
    english_re = re.compile(r"(?<![a-z])(" + "|".join(ENGLISH_AREA_WORDS) + r")(?![a-z])")

    # ---- 1. forbidden content in text ----
    n_forbidden = 0
    for r in rows:
        low = _low(r["text"])
        m = forbidden_re.search(low)
        if m:
            n_forbidden += 1
            if n_forbidden <= 10:
                errors.append(f"FORBIDDEN '{m.group()}' in {r['segment_id']}: {r['text'][:70]}")
        m2 = english_re.search(low)
        if m2:
            n_forbidden += 1
            if n_forbidden <= 10:
                errors.append(f"ENGLISH-AREA '{m2.group()}' in {r['segment_id']}: {r['text'][:70]}")
    if n_forbidden:
        errors.append(f"TOTAL forbidden/english-area hits: {n_forbidden}")

    # ---- 2. split leakage (sentence-based) ----
    splits_by_sent: dict[str, set] = defaultdict(set)
    for r in rows:
        splits_by_sent[r["source_sentence_id"]].add(r["split"])
    leaked = [s for s, sp in splits_by_sent.items() if len(sp) > 1]
    if leaked:
        errors.append(f"SPLIT LEAKAGE: {len(leaked)} sentences in multiple splits, e.g. {leaked[:5]}")

    # ---- 3. duration window ----
    too_long = [r["segment_id"] for r in rows if float(r["estimated_duration"]) > MAX_DURATION_S]
    if too_long:
        errors.append(f"DURATION >30s: {len(too_long)} clips, e.g. {too_long[:5]}")

    # ---- 4. recall-term coverage ----
    # Surface-form aliases: "HbA1c" is an abbreviation Turkish TTS mispronounces,
    # so the corpus uses the spoken form "hemoglobin a1c". Treat them equivalent.
    TERM_ALIASES = {"hba1c": ["hba1c", "hemoglobin a1c"]}
    all_text = " || ".join(_low(r["text"]) for r in rows)
    missing_terms = []
    for t in DEFAULT_MEDICAL_TERMS:
        forms = TERM_ALIASES.get(_low(t), [_low(t)])
        if not any(f in all_text for f in forms):
            missing_terms.append(t)
    if missing_terms:
        errors.append(f"MISSING recall terms: {missing_terms}")

    # ---- 5. structural sanity (warnings) ----
    for col in ["icd10", "term_targets", "tts_voice", "speaking_rate", "clinical_area", "segment_type"]:
        n_blank = sum(1 for r in rows if not str(r.get(col, "")).strip())
        if col == "term_targets":
            # ok for some sentences to have no matched term
            if n_blank:
                warnings.append(f"{n_blank} renditions with empty term_targets (allowed)")
        elif n_blank:
            errors.append(f"EMPTY {col}: {n_blank} rows")

    # ---- report ----
    uniq = len({r["source_sentence_id"] for r in rows})
    area_c = Counter(r["clinical_area"] for r in rows)
    type_c = Counter(r["segment_type"] for r in rows)
    split_c = Counter(r["split"] for r in rows)
    voice_c = Counter(r["tts_voice"] for r in rows)
    total_min = sum(float(r["estimated_duration"]) for r in rows) / 60.0

    print("=" * 64)
    print("MEDICAL CORPUS v3 — VALIDATION REPORT")
    print("=" * 64)
    print(f"renditions: {len(rows)} | unique sentences: {uniq}")
    print(f"total audio (est): {total_min:.1f} min (~{total_min/60:.2f} h)")
    print(f"splits: {dict(split_c)}")
    print(f"voices: {dict(voice_c)}")
    print(f"segment types: {dict(type_c)}")
    print(f"areas ({len(area_c)}): {dict(sorted(area_c.items(), key=lambda x:-x[1]))}")
    print(f"recall terms covered: {len(DEFAULT_MEDICAL_TERMS)-len(missing_terms)}/{len(DEFAULT_MEDICAL_TERMS)}")
    print("-" * 64)
    for w in warnings:
        print(f"WARN: {w}")
    if errors:
        print("-" * 64)
        for e in errors:
            print(f"FAIL: {e}")
        print("=" * 64)
        print("RESULT: FAILED")
        return 1
    print("=" * 64)
    print("RESULT: PASSED — corpus is clean and ready for synthesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
