"""Build the clean medical corpus v3 text manifest from sentence_bank.txt.

This replaces the contaminated medv2 pipeline. Key guarantees:
  * NO boilerplate / disclaimer / scenario tags are ever added to spoken text.
  * Volume comes from honest acoustic augmentation (multi-voice + slight rate
    variation), NOT from text inflation.
  * term_targets are auto-derived from a Turkish medical lexicon (superset of
    metrics.DEFAULT_MEDICAL_TERMS) so the recall metric stays consistent.
  * Train/val split is sentence-based and area-stratified -> zero leakage
    (all acoustic renditions of one sentence land in the same split).

The output text manifest is consumed by scripts/synthesize_medical_corpus_v3.py
which reads tts_voice + speaking_rate per row directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------
# Voices: 4 Google Chirp3-HD Turkish voices for acoustic diversity.
# --------------------------------------------------------------------------
VOICES = [
    {"speaker_id": "tr_v3_aoede", "voice": "tr-TR-Chirp3-HD-Aoede", "gender": "female"},
    {"speaker_id": "tr_v3_kore", "voice": "tr-TR-Chirp3-HD-Kore", "gender": "female"},
    {"speaker_id": "tr_v3_charon", "voice": "tr-TR-Chirp3-HD-Charon", "gender": "male"},
    {"speaker_id": "tr_v3_orus", "voice": "tr-TR-Chirp3-HD-Orus", "gender": "male"},
]

# Slight speaking-rate variation per rendition (deterministic, cycled).
RATE_GRID = [0.92, 0.97, 1.00, 1.05]

# --------------------------------------------------------------------------
# ICD-10 chapter / representative range per clinical area.
# --------------------------------------------------------------------------
AREA_ICD10 = {
    "cardiology": "I00-I99",
    "pulmonology": "J00-J99",
    "endocrinology": "E00-E89",
    "infectious_disease": "A00-B99",
    "neurology": "G00-G99",
    "gastroenterology": "K00-K95",
    "nephrology": "N00-N29",
    "oncology_hematology": "C00-D89",
    "emergency": "R00-R99",
    "radiology": "Z00-Z13",
    "pathology_laboratory": "R70-R97",
    "pharmacology": "Y40-Y59",
    "pediatrics": "P00-P96",
    "obstetrics_gynecology": "O00-O9A",
    "urology": "N30-N53",
    "psychiatry": "F00-F99",
    "dermatology": "L00-L99",
    "otolaryngology": "H60-H95",
    "orthopedics_rheumatology": "M00-M99",
    "ophthalmology": "H00-H59",
}

# --------------------------------------------------------------------------
# Turkish medical term lexicon (superset of DEFAULT_MEDICAL_TERMS).
# Multi-word terms first so longer matches win. Matched terms become
# term_targets for each sentence (lowercased substring match, TR-aware).
# --------------------------------------------------------------------------
TERM_LEXICON = [
    # multi-word
    "miyokard infarktüsü", "akut böbrek hasarı", "kronik böbrek hastalığı",
    "nefrotik sendrom", "idrar yolu enfeksiyonu", "üriner sistem taşı",
    "kalp kapağı", "kan basıncı", "kan şekeri", "tip iki diyabet",
    "tiroid fonksiyon", "hemoglobin a1c", "fizik tedavi", "romatoid artrit", "akut batın",
    "panik bozukluk", "lenf bezi", "göz içi basıncı", "diyabetik retinopati",
    "yaşa bağlı makula dejenerasyonu", "akciğer grafisi", "manyetik rezonans",
    "bilgisayarlı tomografi", "ince iğne aspirasyon", "kemik iliği",
    "tam kan sayımı", "alfa bloker", "folik asit", "seçici serotonin geri alım inhibitörü",
    "orta kulak iltihabı", "alerjik rinit", "kontakt dermatit", "kronik bronşit",
    "nefes darlığı", "göğüs ağrısı", "kan kültürü", "idrar kültürü",
    # cardiology / endo
    "hipertansiyon", "diyabet", "anjiyografi", "ekokardiyografi", "amlodipin",
    "atorvastatin", "metformin", "insülin", "troponin", "aritmi", "elektrokardiyografi",
    "kolesterol", "ateroskleroz", "anjina", "aort", "diseksiyon",
    # pulmo / infx
    "pnömoni", "bronşit", "astım", "antibiyotik", "amoksisilin", "vankomisin",
    "ateş", "öksürük", "inhaler", "spirometri", "tüberküloz", "sinüzit",
    # labs
    "hemoglobin", "kreatinin", "trombosit", "lökositoz", "anemi",
    "piyüri", "bakteriüri", "proteinüri", "hematüri", "biyokimya",
    # neuro
    "inme", "epilepsi", "migren", "parkinson", "nöbet", "felç", "baş dönmesi",
    # gastro
    "koledokolitiazis", "safra", "sarılık", "gastrit", "reflü", "endoskopi",
    "kolonoskopi", "karaciğer", "siroz", "pankreatit",
    # nephro / uro
    "glomerülonefrit", "böbrek", "diyaliz", "prostat", "psa", "mesane",
    # onko / heme
    "tümör", "lenfoma", "lösemi", "kemoterapi", "radyoterapi", "metastaz",
    "biyopsi", "malign", "benign", "immünohistokimya",
    # radyoloji / patoloji
    "radyoloji", "patoloji", "ultrasonografi", "mamografi", "sintigrafi",
    "histopatoloji", "sitoloji", "kontrast",
    # pharma
    "varfarin", "digoksin", "kolşisin", "prednizolon", "kortizon", "levotiroksin",
    "metotreksat", "tramadol", "antikoagülan", "antihistaminik", "kortikosteroid",
    "statin", "antifungal", "retinoid", "bifosfonat",
    # psych
    "depresyon", "anksiyete", "psikoterapi",
    # derm
    "egzama", "psoriazis", "ürtiker", "akne", "sedef",
    # kbb / göz
    "otit", "tonsillit", "odyometri", "glokom", "katarakt", "üveit", "konjonktivit",
    "blefarit", "septum deviasyonu",
    # ortho / romat
    "osteoartrit", "osteoporoz", "kırık", "menisküs", "disk herniasyonu", "gut",
    "kireçlenme", "bel fıtığı",
    # obgyn / peds
    "gebelik", "ultrason", "smear", "kolposkopi", "over kisti", "aşı",
    # emergency
    "anaflaksi", "adrenalin", "travma", "triyaj", "zehirlenme",
]


def _tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def derive_term_targets(text: str) -> list[str]:
    low = _tr_lower(text)
    hits: list[str] = []
    for term in TERM_LEXICON:
        t = _tr_lower(term)
        # word-ish boundary: term surrounded by non-letter (covers TR suffixes)
        if re.search(r"(?<![a-zçğıöşü])" + re.escape(t), low):
            if term not in hits:
                hits.append(term)
    return hits


def parse_bank(path: Path) -> list[dict]:
    area = None
    seg_type = None
    rows: list[dict] = []
    sid = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.rstrip("\n")
        if s.startswith("#") or not s.strip():
            continue
        if s.startswith("@AREA"):
            area = s.split(None, 1)[1].strip()
        elif s.startswith("@TYPE"):
            seg_type = s.split(None, 1)[1].strip()
        elif s.startswith("- "):
            text = s[2:].strip()
            sid += 1
            rows.append({
                "source_sentence_id": f"medv3s_{sid:04d}",
                "clinical_area": area,
                "segment_type": seg_type,
                "text": text,
            })
    return rows


def estimate_duration(text: str, rate: float) -> float:
    words = len(text.split())
    base = words / 2.1 + 0.7  # ~2.1 words/sec + ~0.7s lead/trail
    return round(base / rate, 3)


def assign_splits(sentences: list[dict], val_frac: float, seed: int) -> dict[str, str]:
    """Sentence-based, area-stratified split. Returns sent_id -> split."""
    rng = random.Random(seed)
    by_area: dict[str, list[str]] = {}
    for s in sentences:
        by_area.setdefault(s["clinical_area"], []).append(s["source_sentence_id"])
    split_of: dict[str, str] = {}
    for area, ids in by_area.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_frac))
        val_set = set(ids[:n_val])
        for sid in ids:
            split_of[sid] = "val" if sid in val_set else "train"
    return split_of


def main() -> None:
    ap = argparse.ArgumentParser(description="Build clean medical corpus v3 text manifest.")
    ap.add_argument("--bank", default="data/medical_corpus_v3/sentence_bank.txt")
    ap.add_argument("--out-manifest", default="data/medical_corpus_v3/medical_corpus_v3_text_manifest.csv")
    ap.add_argument("--out-stats", default="data/medical_corpus_v3/medical_corpus_v3_stats.json")
    ap.add_argument("--voices-per-sentence", type=int, default=4,
                    help="How many of the 4 voices to render each sentence with.")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260530)
    args = ap.parse_args()

    bank_path = Path(args.bank)
    sentences = parse_bank(bank_path)
    n_voices = max(1, min(args.voices_per_sentence, len(VOICES)))

    split_of = assign_splits(sentences, args.val_frac, args.seed)

    out_rows: list[dict] = []
    seg_n = 0
    for s_idx, s in enumerate(sentences):
        terms = derive_term_targets(s["text"])
        icd10 = AREA_ICD10.get(s["clinical_area"], "")
        # rotate which voices each sentence uses so all 4 stay balanced
        voice_order = [VOICES[(s_idx + k) % len(VOICES)] for k in range(n_voices)]
        for v_idx, v in enumerate(voice_order):
            rate = RATE_GRID[(s_idx + v_idx) % len(RATE_GRID)]
            seg_n += 1
            out_rows.append({
                "segment_id": f"medv3_{seg_n:05d}",
                "audio_filepath": "",
                "text": s["text"],
                "estimated_duration": estimate_duration(s["text"], rate),
                "sampling_rate": 16000,
                "speaker_id": v["speaker_id"],
                "source": "medical_corpus_v3_synthetic",
                "domain": "medical",
                "icd10": icd10,
                "synthetic": "true",
                "split": split_of[s["source_sentence_id"]],
                "quality": "text_only",
                "clinical_area": s["clinical_area"],
                "segment_type": s["segment_type"],
                "term_targets": "|".join(terms),
                "source_sentence_id": s["source_sentence_id"],
                "tts_provider": "google",
                "tts_voice": v["voice"],
                "speaking_rate": rate,
                "license": "cc-by-4.0",
            })

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # ---- stats ----
    from collections import Counter
    area_c = Counter(s["clinical_area"] for s in sentences)
    type_c = Counter(s["segment_type"] for s in sentences)
    split_c = Counter(r["split"] for r in out_rows)
    voice_c = Counter(r["tts_voice"] for r in out_rows)
    covered_terms = set()
    sent_with_terms = 0
    for s in sentences:
        t = derive_term_targets(s["text"])
        if t:
            sent_with_terms += 1
        covered_terms.update(t)
    total_audio_min = sum(float(r["estimated_duration"]) for r in out_rows) / 60.0
    uniq_min = sum(estimate_duration(s["text"], 1.0) for s in sentences) / 60.0

    # which of the 21 recall terms are present? (HbA1c -> spoken form alias)
    from turkmed_stt.metrics import DEFAULT_MEDICAL_TERMS
    TERM_ALIASES = {"hba1c": {"hba1c", "hemoglobin a1c"}}
    all_low = " || ".join(_tr_lower(s["text"]) for s in sentences)

    def _present(term: str) -> bool:
        for form in TERM_ALIASES.get(_tr_lower(term), {_tr_lower(term)}):
            if form in covered_terms or form in all_low:
                return True
        return False

    recall_present = [t for t in DEFAULT_MEDICAL_TERMS if _present(t)]
    recall_missing = [t for t in DEFAULT_MEDICAL_TERMS if not _present(t)]

    stats = {
        "unique_sentences": len(sentences),
        "total_renditions": len(out_rows),
        "voices_per_sentence": n_voices,
        "unique_content_minutes": round(uniq_min, 1),
        "total_audio_minutes_est": round(total_audio_min, 1),
        "total_audio_hours_est": round(total_audio_min / 60.0, 2),
        "areas": dict(sorted(area_c.items(), key=lambda x: -x[1])),
        "segment_types": dict(type_c),
        "split_renditions": dict(split_c),
        "voice_renditions": dict(voice_c),
        "distinct_terms_covered": len(covered_terms),
        "sentences_with_>=1_term": sent_with_terms,
        "recall_terms_present": recall_present,
        "recall_terms_missing": recall_missing,
    }
    Path(args.out_stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(out_rows)} renditions for {len(sentences)} sentences -> {out_manifest}")
    print(f"  unique content ~{uniq_min:.1f} min | total audio ~{total_audio_min:.1f} min "
          f"(~{total_audio_min/60:.2f} h)")
    print(f"  splits: {dict(split_c)} | voices: {n_voices}")
    print(f"  distinct terms covered: {len(covered_terms)} | recall terms present: "
          f"{len(recall_present)}/{len(DEFAULT_MEDICAL_TERMS)} missing={recall_missing}")
    print(f"  stats -> {args.out_stats}")


if __name__ == "__main__":
    main()
