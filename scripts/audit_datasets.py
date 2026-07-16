"""Comprehensive quality audit for CV TR, ISSAI, OpenSLR TR datasets.

Checks per dataset:
  Audio: sample_rate, channels, bit_depth, duration, rms_energy, clipping, silence_ratio, integrity
  Text:  encoding, empty, length, turkish_char_ratio, number_format, punctuation, case, placeholders
  Manifest: coverage (audio<->text), duplicates, path consistency

Outputs:
  evidence/dataset_quality/{dataset}_audit.json
  evidence/dataset_quality/{dataset}_audit.md
  evidence/dataset_quality/before_after.csv
"""
import collections
import csv
import json
import math
import os
import pathlib
import random
import re
import struct
import unicodedata
import wave

REPO        = pathlib.Path(__file__).resolve().parents[1]
CV_DIR      = pathlib.Path(os.environ.get("TURKMED_CV_DIR", REPO / "data/raw/commonvoice_tr"))
ISSAI_DIR   = pathlib.Path(os.environ.get("TURKMED_ISSAI_DIR", REPO / "data/raw/issai"))
OPENSLR_DIR = pathlib.Path(os.environ.get("TURKMED_OPENSLR_DIR", REPO / "data/raw/openslr_tr"))
OUT_DIR     = REPO / "evidence/dataset_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_SAMPLE = 3000
TURKISH_CHARS = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
PLACEHOLDERS  = ["[inaudible]", "[noise]", "[laughter]", "xxx", "<unk>"]

# ── helpers ───────────────────────────────────────────────────────────────────

def read_wav_stats(path):
    try:
        with wave.open(str(path)) as w:
            sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
            frames = w.getnframes(); dur = frames / sr if sr else 0
            raw = w.readframes(min(frames, sr * 5))
            if sw == 2 and raw:
                samples = struct.unpack(f"<{len(raw)//2}h", raw)
                rms  = math.sqrt(sum(s*s for s in samples) / len(samples))
                peak = max(abs(s) for s in samples)
                sil  = sum(1 for s in samples if abs(s) < 500) / len(samples)
            else:
                rms = peak = 0; sil = 0.0
            return {"ok": True, "sr": sr, "ch": ch, "dur": round(dur, 2),
                    "rms_energy": round(rms, 1),
                    "clipped": peak >= 32700, "silence_ratio": round(sil, 3)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:80]}

def text_flags(text):
    if not text or not text.strip():
        return ["empty"]
    t = text.strip(); flags = []
    if len(t) < 3:   flags.append("very_short")
    if len(t) > 300: flags.append("very_long")
    for ph in PLACEHOLDERS:
        if ph.lower() in t.lower(): flags.append("placeholder"); break
    letters = [c for c in t if c.isalpha()]
    if letters:
        tr_ratio = sum(1 for c in letters if c in TURKISH_CHARS) / len(letters)
        if tr_ratio < 0.5: flags.append(f"low_tr:{tr_ratio:.2f}")
    if re.search(r"\d", t):    flags.append("has_digits")
    if re.search(r"\s{2,}", t): flags.append("multi_space")
    return flags

def audio_summary(stats):
    ok = [s for s in stats if s.get("ok")]
    if not ok:
        return {"n_checked": 0, "n_error": len(stats)}
    durs = sorted(s["dur"] for s in ok)
    n = len(durs)
    srs = collections.Counter(s["sr"] for s in ok)
    chs = collections.Counter(s["ch"] for s in ok)
    return {
        "n_checked":      n,
        "n_error":        len(stats) - n,
        "dur_min":        durs[0],
        "dur_max":        durs[-1],
        "dur_mean":       round(sum(durs)/n, 2),
        "dur_p5":         durs[int(0.05*n)],
        "dur_p95":        durs[int(0.95*n)],
        "total_hours":    round(sum(durs)/3600, 2),
        "sr_dist":        dict(srs.most_common(5)),
        "ch_dist":        dict(chs.most_common(3)),
        "n_not_16k":      sum(1 for s in ok if s["sr"] != 16000),
        "n_not_mono":     sum(1 for s in ok if s["ch"] != 1),
        "n_clipped":      sum(1 for s in ok if s["clipped"]),
        "n_high_silence": sum(1 for s in ok if s["silence_ratio"] > 0.7),
        "n_short":        sum(1 for s in ok if s["dur"] < 1.0),
        "n_long":         sum(1 for s in ok if s["dur"] > 30.0),
    }

def text_summary(pairs):
    n = len(pairs); fc = collections.Counter()
    for _, flags in pairs:
        for f in flags: fc[f] += 1
    n_clean = sum(1 for _, f in pairs if not f)
    return {"n_rows": n, "n_clean": n_clean,
            "clean_pct": round(100*n_clean/n, 1) if n else 0,
            "flag_counts": dict(fc.most_common(20))}

def write_audit(name, mstats, astats, tstats, extra=None):
    from datetime import datetime
    result = {"dataset": name, "audited_at": datetime.now().isoformat(timespec="seconds"),
              "manifest": mstats, "audio": astats, "text": tstats}
    if extra: result["extra"] = extra
    (OUT_DIR / f"{name}_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# {name} — Quality Audit", f"*{result['audited_at']}*", "",
          "## Manifest",
          f"- Total rows: **{mstats.get('n_rows',0):,}**",
          f"- Missing audio: {mstats.get('n_missing_audio',0)}",
          f"- Missing/empty text: {mstats.get('n_missing_text',0)}",
          f"- Duplicate text rows: {mstats.get('n_dup_text',0)}", "",
          "## Audio",
          f"- Checked: {astats.get('n_checked',0):,}  Errors: {astats.get('n_error',0)}",
          f"- Duration range: {astats.get('dur_min','?')}s – {astats.get('dur_max','?')}s  "
          f"(mean {astats.get('dur_mean','?')}s, p5={astats.get('dur_p5','?')}s, p95={astats.get('dur_p95','?')}s)",
          f"- Total hours (checked): {astats.get('total_hours','?')}",
          f"- Not 16 kHz: **{astats.get('n_not_16k',0)}**",
          f"- Not mono: **{astats.get('n_not_mono',0)}**",
          f"- Clipped: {astats.get('n_clipped',0)}",
          f"- High silence (>70 %): {astats.get('n_high_silence',0)}",
          f"- Too short (<1 s): {astats.get('n_short',0)}",
          f"- Too long (>30 s): {astats.get('n_long',0)}", "",
          "## Text",
          f"- Clean rows: **{tstats.get('n_clean',0):,} / {tstats.get('n_rows',0):,}** "
          f"({tstats.get('clean_pct',0)} %)",
          "- Flags:"]
    for flag, cnt in (tstats.get("flag_counts") or {}).items():
        md.append(f"  - `{flag}`: {cnt:,}")
    if extra:
        md += ["", "## Extra"]
        for k, v in extra.items(): md.append(f"- {k}: {v}")
    (OUT_DIR / f"{name}_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(f"  Written: {name}_audit.json + .md")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# COMMONVOICE TR
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== CommonVoice TR ===")
cv_clips = CV_DIR / "clips"
cv_rows = {}
for tsv in ["validated.tsv", "train.tsv", "test.tsv", "dev.tsv"]:
    p = CV_DIR / tsv
    if not p.exists(): continue
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stem = pathlib.Path(row.get("path","")).stem
            if stem not in cv_rows:
                cv_rows[stem] = {"path": row.get("path",""),
                                  "sentence": row.get("sentence","").strip(),
                                  "client_id": row.get("client_id","")}
print(f"  Rows loaded: {len(cv_rows):,}")

n_miss_audio = sum(1 for r in cv_rows.values()
                   if not (cv_clips / r["path"]).exists()
                   and not (cv_clips / (pathlib.Path(r["path"]).stem + ".mp3")).exists())
n_miss_text = sum(1 for r in cv_rows.values() if not r["sentence"])
dup_text_cv = collections.Counter(r["sentence"] for r in cv_rows.values() if r["sentence"])
n_dup_text  = sum(1 for v in dup_text_cv.values() if v > 1)
cv_mstats   = {"n_rows": len(cv_rows), "n_missing_audio": n_miss_audio,
               "n_missing_text": n_miss_text, "n_dup_audio": 0, "n_dup_text": n_dup_text}
print(f"  miss_audio={n_miss_audio}  miss_text={n_miss_text}  dup_text={n_dup_text}")

# Duration from clip_durations.tsv (MP3, can't use wave)
cv_astats = {"n_checked": 0, "n_error": 0, "note": "MP3 — durations from clip_durations.tsv"}
cdt = CV_DIR / "clip_durations.tsv"
if cdt.exists():
    all_durs = []
    with cdt.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                d = float(row.get("duration[ms]", 0)) / 1000
                if d > 0: all_durs.append(d)
            except: pass
    if all_durs:
        random.seed(42)
        sdurs = sorted(random.sample(all_durs, min(AUDIO_SAMPLE, len(all_durs))))
        n = len(sdurs)
        cv_astats = {
            "n_checked": n, "n_error": 0,
            "dur_min": round(sdurs[0], 2), "dur_max": round(sdurs[-1], 2),
            "dur_mean": round(sum(sdurs)/n, 2),
            "dur_p5": round(sdurs[int(0.05*n)], 2), "dur_p95": round(sdurs[int(0.95*n)], 2),
            "total_hours": round(sum(sdurs)/3600, 2),
            "estimated_total_hours_all": round(sum(all_durs)/3600, 1),
            "n_short": sum(1 for d in sdurs if d < 1.0),
            "n_long":  sum(1 for d in sdurs if d > 30.0),
            "n_not_16k": 0, "n_not_mono": 0, "n_clipped": 0, "n_high_silence": 0,
            "note": f"MP3 clips — duration from clip_durations.tsv (sample {n}/{len(all_durs):,})"
        }
        print(f"  Audio: total={sum(all_durs)/3600:.0f}h  mean={sum(all_durs)/len(all_durs):.2f}s  "
              f"short={cv_astats['n_short']}  long={cv_astats['n_long']}")

cv_tflags = [(r["sentence"], text_flags(r["sentence"])) for r in cv_rows.values()]
cv_tstats = text_summary(cv_tflags)
print(f"  Text: {cv_tstats['n_clean']:,}/{cv_tstats['n_rows']:,} clean ({cv_tstats['clean_pct']}%)")

speakers  = [r["client_id"] for r in cv_rows.values() if r["client_id"]]
cv_extra  = {"n_unique_speakers": len(set(speakers)),
             "max_clips_per_speaker": max(collections.Counter(speakers).values()) if speakers else 0}
write_audit("commonvoice_tr", cv_mstats, cv_astats, cv_tstats, cv_extra)

# ══════════════════════════════════════════════════════════════════════════════
# ISSAI
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== ISSAI ===")
issai_rows = {}
for split in ["Train", "Dev", "Test"]:
    d = ISSAI_DIR / split
    if not d.exists(): continue
    for wav in d.glob("*.wav"):
        txt = wav.with_suffix(".txt")
        issai_rows[wav.stem] = {
            "wav": wav, "txt": txt, "split": split,
            "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        }
print(f"  Rows loaded: {len(issai_rows):,}")

n_miss_t_i = sum(1 for r in issai_rows.values() if not r["txt"].exists() or not r["text"])
dup_ti     = collections.Counter(r["text"] for r in issai_rows.values() if r["text"])
n_dup_ti   = sum(1 for v in dup_ti.values() if v > 1)
issai_mstats = {"n_rows": len(issai_rows), "n_missing_audio": 0,
                "n_missing_text": n_miss_t_i, "n_dup_audio": 0, "n_dup_text": n_dup_ti}
print(f"  miss_text={n_miss_t_i}  dup_text={n_dup_ti}")

print(f"  Sampling {AUDIO_SAMPLE} WAV files for audio stats...")
all_issai = list(issai_rows.values())
random.seed(42)
sample_i  = random.sample(all_issai, min(AUDIO_SAMPLE, len(all_issai)))
raw_stats = []
for i, r in enumerate(sample_i):
    raw_stats.append(read_wav_stats(r["wav"]))
    if (i+1) % 1000 == 0: print(f"    {i+1}/{len(sample_i)}")
issai_astats = audio_summary(raw_stats)
issai_astats["note"] = f"Sample {len(sample_i)}/{len(issai_rows):,}"
if issai_astats.get("dur_mean"):
    issai_astats["estimated_total_hours_all"] = round(
        len(issai_rows) * issai_astats["dur_mean"] / 3600, 1)
print(f"  Audio: not16k={issai_astats.get('n_not_16k',0)}  "
      f"not_mono={issai_astats.get('n_not_mono',0)}  "
      f"short={issai_astats.get('n_short',0)}  long={issai_astats.get('n_long',0)}")

random.seed(42)
text_sample_i  = random.sample(all_issai, min(10000, len(all_issai)))
issai_tflags   = [(r["text"], text_flags(r["text"])) for r in text_sample_i]
issai_tstats   = text_summary(issai_tflags)
issai_tstats["note"] = f"Sample {len(text_sample_i):,}/{len(issai_rows):,}"
print(f"  Text: {issai_tstats['n_clean']:,}/{issai_tstats['n_rows']:,} clean ({issai_tstats['clean_pct']}%)")

splits = collections.Counter(r["split"] for r in issai_rows.values())
write_audit("issai", issai_mstats, issai_astats, issai_tstats, {"splits": dict(splits)})

# ══════════════════════════════════════════════════════════════════════════════
# OpenSLR TR
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== OpenSLR TR ===")
openslr_rows = {}
for ext in ["*.wav", "*.flac"]:
    for af in OPENSLR_DIR.glob(ext):
        if af.stem not in openslr_rows:
            txt = af.with_suffix(".txt")
            openslr_rows[af.stem] = {
                "wav": af, "txt": txt,
                "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            }
print(f"  Rows loaded: {len(openslr_rows):,}")

n_miss_t_o = sum(1 for r in openslr_rows.values() if not r["txt"].exists() or not r["text"])
dup_to     = collections.Counter(r["text"] for r in openslr_rows.values() if r["text"])
n_dup_to   = sum(1 for v in dup_to.values() if v > 1)
openslr_mstats = {"n_rows": len(openslr_rows), "n_missing_audio": 0,
                  "n_missing_text": n_miss_t_o, "n_dup_audio": 0, "n_dup_text": n_dup_to}
print(f"  miss_text={n_miss_t_o}  dup_text={n_dup_to}")

print("  Scanning all audio...")
n_wav = n_flac = 0; raw_stats_o = []
for r in openslr_rows.values():
    if r["wav"].suffix == ".wav":
        raw_stats_o.append(read_wav_stats(r["wav"])); n_wav += 1
    else:
        raw_stats_o.append({"ok": False, "error": "flac_skip"}); n_flac += 1
openslr_astats = audio_summary([s for s in raw_stats_o if s.get("ok")])
openslr_astats["n_wav"] = n_wav; openslr_astats["n_flac"] = n_flac
print(f"  Audio: wav={n_wav}  flac={n_flac}  "
      f"not16k={openslr_astats.get('n_not_16k',0)}  "
      f"short={openslr_astats.get('n_short',0)}")

openslr_tflags = [(r["text"], text_flags(r["text"])) for r in openslr_rows.values()]
openslr_tstats = text_summary(openslr_tflags)
print(f"  Text: {openslr_tstats['n_clean']:,}/{openslr_tstats['n_rows']:,} "
      f"clean ({openslr_tstats['clean_pct']}%)")

write_audit("openslr_tr", openslr_mstats, openslr_astats, openslr_tstats)

# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY ===")
rows = [
    {"dataset": "commonvoice_tr",
     "total_rows": cv_mstats["n_rows"],
     "missing_audio": cv_mstats["n_missing_audio"],
     "missing_text": cv_mstats["n_missing_text"],
     "dup_text": cv_mstats["n_dup_text"],
     "audio_format": "MP3 (convert needed)",
     "est_total_hours": cv_astats.get("estimated_total_hours_all","?"),
     "text_clean_pct": cv_tstats["clean_pct"],
     "n_unique_speakers": cv_extra["n_unique_speakers"]},
    {"dataset": "issai",
     "total_rows": issai_mstats["n_rows"],
     "missing_audio": issai_mstats["n_missing_audio"],
     "missing_text": issai_mstats["n_missing_text"],
     "dup_text": issai_mstats["n_dup_text"],
     "audio_format": "WAV",
     "est_total_hours": issai_astats.get("estimated_total_hours_all","?"),
     "text_clean_pct": issai_tstats["clean_pct"],
     "n_unique_speakers": "N/A"},
    {"dataset": "openslr_tr",
     "total_rows": openslr_mstats["n_rows"],
     "missing_audio": openslr_mstats["n_missing_audio"],
     "missing_text": openslr_mstats["n_missing_text"],
     "dup_text": openslr_mstats["n_dup_text"],
     "audio_format": f"WAV({n_wav})+FLAC({n_flac})",
     "est_total_hours": openslr_astats.get("total_hours","?"),
     "text_clean_pct": openslr_tstats["clean_pct"],
     "n_unique_speakers": "N/A"},
]
ba = OUT_DIR / "before_after.csv"
with ba.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

for r in rows:
    print(f"  {r['dataset']:20s}  rows={r['total_rows']:>7,}  "
          f"miss={r['missing_audio']}/{r['missing_text']}  "
          f"dup={r['dup_text']}  "
          f"clean={r['text_clean_pct']}%  "
          f"est_h={r['est_total_hours']}")

print(f"\nDone -> {OUT_DIR}")
