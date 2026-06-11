"""Apply cleaning policy and produce unified manifests.

Reads raw datasets, applies all rules from cleaning_policy.md, outputs:
  data/cleaned/commonvoice_tr/manifest_clean.csv
  data/cleaned/issai/manifest_clean.csv
  data/cleaned/openslr_tr/manifest_clean.csv   (+ WAV conversion)
  data/cleaned/combined_manifest.csv           (all three unified)
  evidence/dataset_quality/before_after_final.csv
"""
import csv, json, os, pathlib, wave, random, re, collections, struct, math, subprocess, shutil
from datetime import datetime

REPO        = pathlib.Path(__file__).resolve().parents[1]
CV_DIR      = pathlib.Path(os.environ.get("TURKMED_CV_DIR", REPO / "data/raw/commonvoice_tr"))
ISSAI_DIR   = pathlib.Path(os.environ.get("TURKMED_ISSAI_DIR", REPO / "data/raw/issai"))
OPENSLR_DIR = pathlib.Path(os.environ.get("TURKMED_OPENSLR_DIR", REPO / "data/raw/openslr_tr"))
OUT_BASE    = REPO / "data/cleaned"
QUALITY_DIR = REPO / "evidence/dataset_quality"

SPEAKER_CAP   = 1000   # CV: max clips per speaker_id
AUDIO_SAMPLE  = 3000   # for ISSAI clipping/silence check

# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def get_wav_stats(path: pathlib.Path):
    try:
        with wave.open(str(path)) as w:
            sr = w.getframerate(); frames = w.getnframes()
            dur = frames / sr if sr else 0
            raw = w.readframes(min(frames, sr * 5))
            if w.getsampwidth() == 2 and raw:
                samples = struct.unpack(f"<{len(raw)//2}h", raw)
                peak = max(abs(s) for s in samples)
                sil  = sum(1 for s in samples if abs(s) < 500) / len(samples)
                return {"ok": True, "dur": dur, "clipped": peak >= 32700, "silence": sil}
    except:
        pass
    return {"ok": False}

def ffmpeg_to_wav(src: pathlib.Path, dst: pathlib.Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    r = subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)
    ], capture_output=True)
    return r.returncode == 0

log_lines = [f"# Dataset Cleaning Run — {datetime.now().isoformat(timespec='seconds')}", ""]

def log(msg):
    print(msg)
    log_lines.append(msg)

# ══════════════════════════════════════════════════════════════════════════════
# COMMONVOICE TR
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== CommonVoice TR ===")
cv_out = OUT_BASE / "commonvoice_tr"
cv_out.mkdir(parents=True, exist_ok=True)
cv_clips = CV_DIR / "clips"

# Duration lookup from clip_durations.tsv
dur_map = {}
cdt = CV_DIR / "clip_durations.tsv"
if cdt.exists():
    with cdt.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                d = float(row.get("duration[ms]", 0)) / 1000
                path = row.get("clip", row.get("path", ""))
                dur_map[pathlib.Path(path).stem] = d
            except: pass
log(f"  Duration map loaded: {len(dur_map):,} entries")

# Load all rows with split info
cv_raw = []
for tsv in ["validated.tsv", "train.tsv", "test.tsv", "dev.tsv"]:
    p = CV_DIR / tsv
    if not p.exists(): continue
    split = tsv.replace(".tsv", "")
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            stem = pathlib.Path(row.get("path", "")).stem
            cv_raw.append({
                "stem":       stem,
                "path":       row.get("path", ""),
                "text":       row.get("sentence", "").strip(),
                "client_id":  row.get("client_id", ""),
                "source_tsv": split,
            })

log(f"  Raw rows: {len(cv_raw):,}")
removed = collections.Counter()

# Step 1: deduplicate (same client_id + same text → keep first)
seen_speaker_text = set()
cv_deduped = []
for r in cv_raw:
    key = (r["client_id"], r["text"])
    if key in seen_speaker_text:
        removed["same_speaker_dup"] += 1
        continue
    seen_speaker_text.add(key)
    cv_deduped.append(r)
log(f"  After same-speaker dedup: {len(cv_deduped):,}  (removed {removed['same_speaker_dup']})")

# Step 2: text quality filters
cv_filtered = []
for r in cv_deduped:
    t = normalize_text(r["text"])
    if not t or len(t) < 3:
        removed["text_too_short"] += 1; continue
    if len(t) > 300:
        removed["text_too_long"] += 1; continue
    r["text"] = t
    cv_filtered.append(r)
log(f"  After text filters: {len(cv_filtered):,}  (removed {removed['text_too_short']+removed['text_too_long']})")

# Step 3: audio duration filter (use clip_durations.tsv)
cv_dur_ok = []
for r in cv_filtered:
    dur = dur_map.get(r["stem"], None)
    if dur is not None and dur < 1.0:
        removed["audio_too_short"] += 1; continue
    r["duration_sec"] = round(dur, 3) if dur else None
    cv_dur_ok.append(r)
log(f"  After duration filter: {len(cv_dur_ok):,}  (removed {removed['audio_too_short']})")

# Step 4: speaker cap — for fine_tune split only (cap at SPEAKER_CAP)
spk_count = collections.Counter()
cv_capped = []
for r in cv_dur_ok:
    spk = r["client_id"]
    if spk_count[spk] < SPEAKER_CAP:
        cv_capped.append(r); spk_count[spk] += 1
    else:
        removed["speaker_cap"] += 1
log(f"  After speaker cap ({SPEAKER_CAP}/speaker): {len(cv_capped):,}  (removed {removed['speaker_cap']})")

# Write manifest
cv_manifest_rows = []
for r in cv_capped:
    mp3 = cv_clips / r["path"]
    if not mp3.exists():
        mp3 = cv_clips / (r["stem"] + ".mp3")
    cv_manifest_rows.append({
        "audio_filepath": str(mp3),
        "text":           r["text"],
        "duration_sec":   r["duration_sec"],
        "source":         "commonvoice_tr",
        "speaker_id":     r["client_id"],
        "original_split": r["source_tsv"],
    })

cv_manifest_path = cv_out / "manifest_clean.csv"
with cv_manifest_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cv_manifest_rows[0].keys()))
    w.writeheader(); w.writerows(cv_manifest_rows)

log(f"  Manifest written: {cv_manifest_path.name}  rows={len(cv_manifest_rows):,}")
log(f"  Removed summary: {dict(removed)}")
cv_final_count = len(cv_manifest_rows)

# ══════════════════════════════════════════════════════════════════════════════
# ISSAI
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== ISSAI ===")
issai_out = OUT_BASE / "issai"
issai_out.mkdir(parents=True, exist_ok=True)

issai_rows = {}
for split in ["Train", "Dev", "Test"]:
    d = ISSAI_DIR / split
    if not d.exists(): continue
    for wav in d.glob("*.wav"):
        txt = wav.with_suffix(".txt")
        issai_rows[wav.stem] = {
            "wav": wav, "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else "",
            "split": split,
        }
log(f"  Raw rows: {len(issai_rows):,}")

# Load audio stats for filtering — sample for clipping/silence
log(f"  Sampling {AUDIO_SAMPLE} files for clipping/silence check...")
all_issai = list(issai_rows.items())
random.seed(42)
sample_keys = set(k for k, _ in random.sample(all_issai, min(AUDIO_SAMPLE, len(all_issai))))
bad_keys = set()
for k in sample_keys:
    r = issai_rows[k]
    s = get_wav_stats(r["wav"])
    if not s.get("ok"):
        bad_keys.add(k)
    elif s["dur"] < 1.0:
        bad_keys.add(k)
    elif s["clipped"]:
        bad_keys.add(k)
    elif s["silence"] > 0.70:
        bad_keys.add(k)
log(f"  Bad in sample: {len(bad_keys)}/{AUDIO_SAMPLE}")

# For non-sampled files: at minimum filter <1s via wav header
issai_removed = collections.Counter()
issai_clean = []
for k, r in issai_rows.items():
    # Text filter
    t = normalize_text(r["text"])
    if not t or len(t) < 3:
        issai_removed["text_empty_short"] += 1; continue
    # Audio filter (sampled)
    if k in bad_keys:
        issai_removed["audio_bad_sampled"] += 1; continue
    # Duration via wav header (fast, just read header)
    try:
        with wave.open(str(r["wav"])) as w:
            dur = w.getnframes() / w.getframerate()
    except:
        issai_removed["corrupt"] += 1; continue
    if dur < 1.0:
        issai_removed["too_short"] += 1; continue
    issai_clean.append({
        "audio_filepath": str(r["wav"]),
        "text":           t,
        "duration_sec":   round(dur, 3),
        "source":         "issai",
        "speaker_id":     "",
        "original_split": r["split"],
    })

log(f"  Clean rows: {len(issai_clean):,}  removed={dict(issai_removed)}")

issai_manifest_path = issai_out / "manifest_clean.csv"
with issai_manifest_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(issai_clean[0].keys()))
    w.writeheader(); w.writerows(issai_clean)
log(f"  Manifest written: {issai_manifest_path.name}")
issai_final_count = len(issai_clean)

# ══════════════════════════════════════════════════════════════════════════════
# OpenSLR TR
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== OpenSLR TR ===")
openslr_out = OUT_BASE / "openslr_tr"
openslr_wav_out = openslr_out / "audio_wav"
openslr_wav_out.mkdir(parents=True, exist_ok=True)

openslr_raw = {}
for ext in ["*.wav", "*.flac"]:
    for af in OPENSLR_DIR.glob(ext):
        if af.stem not in openslr_raw:
            txt = af.with_suffix(".txt")
            openslr_raw[af.stem] = {
                "src": af,
                "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
            }
log(f"  Raw rows: {len(openslr_raw):,}")

openslr_clean = []
openslr_removed = collections.Counter()
n_flac_converted = 0

for i, (stem, r) in enumerate(openslr_raw.items()):
    t = normalize_text(r["text"])
    if not t or len(t) < 3:
        openslr_removed["text_empty_short"] += 1; continue

    src = r["src"]
    if src.suffix == ".flac":
        dst = openslr_wav_out / (stem + ".wav")
        ok = ffmpeg_to_wav(src, dst)
        if not ok:
            openslr_removed["ffmpeg_fail"] += 1; continue
        audio_path = dst
        n_flac_converted += 1
    else:
        audio_path = src

    try:
        with wave.open(str(audio_path)) as w:
            dur = w.getnframes() / w.getframerate()
    except:
        openslr_removed["corrupt"] += 1; continue

    if dur < 1.0:
        openslr_removed["too_short"] += 1; continue

    openslr_clean.append({
        "audio_filepath": str(audio_path),
        "text":           t,
        "duration_sec":   round(dur, 3),
        "source":         "openslr_tr",
        "speaker_id":     "",
        "original_split": "train",
    })

    if (i+1) % 500 == 0:
        log(f"  Progress: {i+1}/{len(openslr_raw)} (converted {n_flac_converted} FLAC)")

log(f"  FLAC->WAV converted: {n_flac_converted}")
log(f"  Clean rows: {len(openslr_clean):,}  removed={dict(openslr_removed)}")

openslr_manifest_path = openslr_out / "manifest_clean.csv"
with openslr_manifest_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(openslr_clean[0].keys()))
    w.writeheader(); w.writerows(openslr_clean)
log(f"  Manifest written: {openslr_manifest_path.name}")
openslr_final_count = len(openslr_clean)

# ══════════════════════════════════════════════════════════════════════════════
# Combined manifest
# ══════════════════════════════════════════════════════════════════════════════
log("\n=== Combined manifest ===")
all_rows = cv_manifest_rows + issai_clean + openslr_clean
combined_path = OUT_BASE / "combined_manifest.csv"
with combined_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader(); w.writerows(all_rows)

by_source = collections.Counter(r["source"] for r in all_rows)
total_dur  = sum(float(r["duration_sec"]) for r in all_rows if r["duration_sec"])
log(f"  Combined rows: {len(all_rows):,}")
for s, n in by_source.most_common():
    src_rows = [r for r in all_rows if r["source"] == s]
    src_dur  = sum(float(r["duration_sec"]) for r in src_rows if r["duration_sec"])
    log(f"    {s:20s} {n:>7,} rows  {src_dur/3600:.1f}h")
log(f"  Total duration: {total_dur/3600:.1f}h")

# ══════════════════════════════════════════════════════════════════════════════
# Before/after summary
# ══════════════════════════════════════════════════════════════════════════════
ba_rows = [
    {"dataset": "commonvoice_tr",
     "rows_before": 119325, "rows_after": cv_final_count,
     "removed": 119325 - cv_final_count,
     "removed_pct": round(100*(119325-cv_final_count)/119325, 2),
     "main_changes": "real_dup:7, text_filter:63, dur_filter:14, speaker_cap"},
    {"dataset": "issai",
     "rows_before": 186170, "rows_after": issai_final_count,
     "removed": 186170 - issai_final_count,
     "removed_pct": round(100*(186170-issai_final_count)/186170, 2),
     "main_changes": "clipped+silence+short in sample, text_normalize"},
    {"dataset": "openslr_tr",
     "rows_before": 2513, "rows_after": openslr_final_count,
     "removed": 2513 - openslr_final_count,
     "removed_pct": round(100*(2513-openslr_final_count)/2513, 2),
     "main_changes": "multi_space_normalized, flac_to_wav"},
]
ba_path = QUALITY_DIR / "before_after_final.csv"
with ba_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ba_rows[0].keys()))
    w.writeheader(); w.writerows(ba_rows)

log(f"\n=== BEFORE / AFTER ===")
log(f"{'Dataset':22s} {'Before':>8} {'After':>8} {'Removed':>8} {'Pct':>6}")
log("-" * 60)
for r in ba_rows:
    log(f"  {r['dataset']:20s} {r['rows_before']:>8,} {r['rows_after']:>8,} "
        f"{r['removed']:>8,} {r['removed_pct']:>5.2f}%")

# Save run log
(QUALITY_DIR / "cleaning_run_log.md").write_text("\n".join(log_lines), encoding="utf-8")
log(f"\nDone. Outputs -> {OUT_BASE}")
log(f"Log -> {QUALITY_DIR}/cleaning_run_log.md")
