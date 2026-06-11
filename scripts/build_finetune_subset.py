#!/usr/bin/env python3
"""
build_finetune_subset.py — FT-1 (FINETUNE_PLAN.md §2)

combined_manifest.csv'den ~140h DENGELI alt küme seçer + eval setine karşı
metin-dedup (sızıntı kontrolü) uygular ve iki manifest üretir:

  - train_140h_manifest.csv       (tam ölçek, Faz-2)
  - train_30h_pilot_manifest.csv  (140h'in alt kümesi, Faz-1 pilot)

Karar (FINETUNE_PLAN.md):
  ISSAI  -> tavan ~65h  (CV seviyesine indirilir; baskınlık kırılır)
  CV     -> tamamı ~65h (en çeşitli kaynak, 1.670 konuşmacı)
  OpenSLR-> tamamı ~10h (küçük, hepsi alınır)
  Toplam -> ~140h dengeli (eval dağılımına benzer)

Sızıntı: eval cümleleri normalize edilip eğitimden ÇIKARILIR. ISSAI/OpenSLR'de
speaker_id etiketi olmadığından metin-dedup asıl güvencedir.

NOT: Bu script hiçbir ses dosyasına dokunmaz, sadece manifest üretir.
Çalıştırma (lokalde):
  python scripts/build_finetune_subset.py
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Türkçe-duyarlı normalizasyon (dedup için; eval metni zaten lowercase/noktasız)
# -----------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_tr(text: str) -> str:
    """İ/I Türkçe katlaması + lowercase + noktalama at + boşluk sıkıştır."""
    if text is None:
        return ""
    t = text.replace("İ", "i").replace("I", "ı")
    t = t.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# Kaynak başına saat tavanı (None = tamamını al)
SOURCE_HOUR_CAP = {
    "issai": 65.0,
    "commonvoice_tr": None,
    "openslr_tr": None,
}


def load_eval_texts(eval_manifest: Path) -> set[str]:
    seen: set[str] = set()
    with eval_manifest.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seen.add(normalize_tr(row.get("text", "")))
    seen.discard("")
    return seen


def subsample_by_hours(rows: list[dict], cap_hours: float | None, seed: int) -> list[dict]:
    """Süreye göre cap_hours'a kadar rastgele (seed) alt küme. cap None -> hepsi."""
    if cap_hours is None:
        return list(rows)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    out, acc = [], 0.0
    cap_sec = cap_hours * 3600.0
    for r in shuffled:
        if acc >= cap_sec:
            break
        out.append(r)
        acc += float(r["duration_sec"])
    return out


def hours(rows: list[dict]) -> float:
    return sum(float(r["duration_sec"]) for r in rows) / 3600.0


def write_manifest(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO / "data/cleaned/combined_manifest.csv"))
    ap.add_argument("--eval-manifest",
                    default=str(REPO / "evidence/benchmark_eval_pack/benchmark_eval_manifest.csv"))
    ap.add_argument("--out-dir", default=str(REPO / "data/cleaned/finetune_subsets"))
    ap.add_argument("--issai-hours", type=float, default=65.0)
    ap.add_argument("--pilot-hours", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    eval_manifest = Path(args.eval_manifest)
    out_dir = Path(args.out_dir)

    if not manifest.exists():
        print(f"[HATA] manifest yok: {manifest}", file=sys.stderr)
        return 1

    SOURCE_HOUR_CAP["issai"] = args.issai_hours

    # 1) eval metinleri (sızıntı kontrolü)
    eval_texts = load_eval_texts(eval_manifest) if eval_manifest.exists() else set()
    print(f"[i] eval'de {len(eval_texts)} benzersiz normalize cümle (dedup hedefi)")

    # 2) combined manifest oku + dedup + kaynağa göre grupla
    fieldnames: list[str] = []
    by_source: dict[str, list[dict]] = defaultdict(list)
    removed = Counter()
    total_in = 0
    with manifest.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        for row in r:
            total_in += 1
            src = row.get("source", "?")
            if normalize_tr(row.get("text", "")) in eval_texts:
                removed[src] += 1
                continue
            by_source[src].append(row)
    print(f"[i] manifest satırı: {total_in}, sızıntı-dedup ile atılan: {sum(removed.values())} {dict(removed)}")

    # 3) kaynak başına dengeli alt-örnekleme -> 140h
    full: list[dict] = []
    print("\n=== 140h DENGELI SEÇIM ===")
    for src in sorted(by_source):
        cap = SOURCE_HOUR_CAP.get(src)  # tanımsız kaynak -> hepsi
        sel = subsample_by_hours(by_source[src], cap, args.seed)
        full.extend(sel)
        cap_txt = f"tavan {cap}h" if cap is not None else "tamamı"
        print(f"  {src:16s}: {len(sel):6d} klip, {hours(sel):6.1f} h ({cap_txt})")
    print(f"  {'TOPLAM':16s}: {len(full):6d} klip, {hours(full):6.1f} h")

    # 4) pilot ~30h: 140h içinden kaynak oranını koruyarak (süre bazlı)
    total_h = hours(full)
    frac = min(1.0, args.pilot_hours / total_h) if total_h > 0 else 1.0
    pilot: list[dict] = []
    pilot_by_src: dict[str, list[dict]] = defaultdict(list)
    for row in full:
        pilot_by_src[row.get("source", "?")].append(row)
    print(f"\n=== {args.pilot_hours:.0f}h PILOT (oran={frac:.3f}) ===")
    for src in sorted(pilot_by_src):
        target = hours(pilot_by_src[src]) * frac
        sel = subsample_by_hours(pilot_by_src[src], target, args.seed + 1)
        pilot.extend(sel)
        print(f"  {src:16s}: {len(sel):6d} klip, {hours(sel):6.1f} h")
    print(f"  {'TOPLAM':16s}: {len(pilot):6d} klip, {hours(pilot):6.1f} h")

    # 5) yaz
    out_140 = out_dir / "train_140h_manifest.csv"
    out_30 = out_dir / "train_30h_pilot_manifest.csv"
    write_manifest(full, out_140, fieldnames)
    write_manifest(pilot, out_30, fieldnames)
    print(f"\n[OK] yazıldı:\n  {out_140}\n  {out_30}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
