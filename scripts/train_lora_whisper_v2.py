"""Whisper-large-v3 LoRA Fine-tune — TurkMedSTT v2.

Data strategy (FINETUNE_PLAN.md — balanced local subset, uploaded as zip):
  - General Turkish: LOCAL balanced subset manifest (train_140h / train_30h_pilot)
    produced by scripts/build_finetune_subset.py. Columns: audio_filepath,text,...
  - Medical Turkish: data/medical_tts_v2_180min/ (724 clips), upsampled (M2 only)

Modes (controlled ablation):
  --mode general          -> M1 (general-only LoRA)
  --mode general+medical  -> M2 (general + medical upsample)

Path remapping: manifest audio_filepath may be a Windows absolute path. On Colab,
pass --audio-root to resolve clips relative to the unzipped audio root (the zip
builder rewrites paths to <source>/<filename> under that root).

LoRA config: r=32, alpha=64, target=q_proj,v_proj
Training: A100, fp16, batch=8 x grad_accum=2 (eff 16), epoch-based + early stop

Outputs:
  <output-dir>/  — adapter checkpoint + training_metadata.json
"""

import argparse
import csv
import json
import pathlib
import random
import time

import torch

REPO = pathlib.Path("/root/turkmedSTT")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="openai/whisper-large-v3")
    p.add_argument("--output-dir",   default=str(REPO / "runs/lora_whisper_large_v3"))
    p.add_argument("--mode",         default="general+medical",
                   choices=["general", "general+medical"],
                   help="general=M1 (genel-only), general+medical=M2")
    p.add_argument("--general-manifest",
                   default=str(REPO / "data/cleaned/finetune_subsets/train_140h_manifest.csv"),
                   help="Balanced subset manifest (audio_filepath,text,...)")
    p.add_argument("--audio-root", default="",
                   help="If set, audio_filepath resolved relative to this root "
                        "(use on Colab after unzip). Empty = use paths as-is.")
    p.add_argument("--general-n",   type=int, default=0,
                   help="Cap general clips (0 = all rows in manifest)")
    p.add_argument("--medical-manifest",
                   default=str(REPO / "data/medical_tts_v2_180min/medical_tts_v2_manifest.csv"))
    p.add_argument("--medical-upsample", type=int, default=3,
                   help="Medical repeat factor (default 3×, M2 only)")
    p.add_argument("--lora-r",       type=int, default=32)
    p.add_argument("--lora-alpha",   type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target",  default="q_proj,v_proj")
    # A100-80GB: per_device=16, grad_accum=1 -> effective batch 16, no accumulation
    # overhead, max step count for a 1-epoch budget (best convergence per FINETUNE_PLAN).
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--grad-accum",   type=int, default=1)
    p.add_argument("--num-epochs",   type=float, default=1.0,
                   help="Epoch-based budget (FINETUNE_PLAN: 1-2 epoch + early stop)")
    p.add_argument("--max-steps",    type=int, default=-1,
                   help="If >0 overrides epochs (fixed-step budget)")
    p.add_argument("--eval-steps",   type=int, default=500)
    p.add_argument("--save-steps",   type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=0,
                   help="0 = keep ALL checkpoints (post-hoc best-checkpoint selection)")
    p.add_argument("--map-workers",  type=int, default=1,
                   help="datasets.map num_proc. 1 = güvenli (Colab'da >1 deadlock yapıyor)")
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Install dependencies ──────────────────────────────────────────────────────
def ensure_deps():
    import subprocess, sys
    pkgs = ["peft>=0.10.0", "evaluate", "jiwer", "soundfile", "librosa"]
    for pkg in pkgs:
        try:
            __import__(pkg.split(">=")[0].split("==")[0].replace("-","_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_deps()


# ── Path resolution ───────────────────────────────────────────────────────────
def resolve_audio_path(fp: str, audio_root: str) -> pathlib.Path:
    """Resolve a manifest audio_filepath, optionally remapping under audio_root.

    Contract with the FT-3 zip builder: the Colab-side manifest stores
    audio_filepath as a RELATIVE path "<source>/<filename>" and clips are
    unzipped under audio_root, so resolution is simply root / fp.

    Fallbacks keep local runs working:
      - audio_root empty  -> path used as-is (absolute Windows/Posix path)
      - fp absolute + root -> try root/<parent>/<name>, then root/<name>
    """
    if not audio_root:
        return pathlib.Path(fp)
    root = pathlib.Path(audio_root)
    win = "\\" in fp
    p = pathlib.PureWindowsPath(fp) if win else pathlib.PurePosixPath(fp)
    if not p.is_absolute():
        cand = root / pathlib.Path(*p.parts)
        if cand.exists():
            return cand
    # absolute path or relative-miss: try 2-level then flat by filename
    cand2 = root / p.parent.name / p.name
    if cand2.exists():
        return cand2
    return root / p.name


# ── Build training dataset ────────────────────────────────────────────────────
def build_training_rows(args):
    """Returns list of {'audio_path': str, 'text': str}."""
    rows = []

    # ── 1. Medical TTS (M2 only) ──────────────────────────────────────────────
    # NOTE: medical audio lives next to its manifest (NOT in the general zip),
    # so it is resolved relative to the manifest dir / CWD — independent of --audio-root.
    if args.mode == "general+medical":
        med_path = pathlib.Path(args.medical_manifest)
        med_base = med_path.parent
        if med_path.exists():
            med_valid = []
            for r in csv.DictReader(med_path.open(encoding="utf-8-sig")):
                fp = r.get("audio_filepath", "").replace("\\", "/")
                p = pathlib.Path(fp)
                if p.is_absolute():
                    full = p
                else:
                    # try CWD-relative, then manifest-relative, then <med_base>/audio/<name>
                    for cand in (pathlib.Path(fp), med_base / fp, med_base / "audio" / p.name):
                        if cand.exists():
                            full = cand
                            break
                    else:
                        full = pathlib.Path(fp)
                text = r.get("text", "").strip()
                if full.exists() and text:
                    med_valid.append({"audio_path": str(full), "text": text})
            print(f"Medical TTS clips: {len(med_valid)}")
            for _ in range(args.medical_upsample):
                rows.extend(med_valid)
            print(f"Medical after {args.medical_upsample}x upsample: "
                  f"{len(med_valid) * args.medical_upsample}")
        else:
            print(f"WARNING: medical manifest not found: {med_path}")
    else:
        print("Mode=general (M1): medical data SKIPPED.")

    # ── 2. General Turkish from LOCAL balanced subset manifest ────────────────
    gen_path = pathlib.Path(args.general_manifest)
    if not gen_path.exists():
        raise FileNotFoundError(f"general manifest not found: {gen_path}")
    gen_valid, missing = [], 0
    for r in csv.DictReader(gen_path.open(encoding="utf-8")):
        fp = r.get("audio_filepath", "")
        full = resolve_audio_path(fp, args.audio_root)
        text = r.get("text", "").strip()
        if not text:
            continue
        if not full.exists():
            missing += 1
            continue
        gen_valid.append({"audio_path": str(full), "text": text})
        if args.general_n and len(gen_valid) >= args.general_n:
            break
    print(f"General clips: {len(gen_valid)} (missing on disk: {missing})")
    rows.extend(gen_valid)

    random.seed(args.seed)
    random.shuffle(rows)
    print(f"\nTotal training rows: {len(rows)}")
    return rows


# ── Main training ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model + processor
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"\nLoading {args.model}...")
    processor = WhisperProcessor.from_pretrained(args.model, language="tr", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        # task_type AYARLAMA: SEQ_2_SEQ_LM, peft'i T5-tarzı sanıp forward'a input_ids
        # enjekte ettiriyor; Whisper input_features kullanır -> "unexpected kwarg input_ids".
        # Whisper LoRA reçetesi generic PeftModel ister (yalnız verilen kwargs'ı forward eder).
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Build data
    train_rows = build_training_rows(args)

    # HuggingFace Dataset
    import numpy as np
    from datasets import Dataset, Audio

    import librosa

    # LAZY feature extraction (set_transform): ham sesi RAM'de BİRİKTİRME.
    # Eski yöntem (tüm klipleri listeye yükle + Dataset.from_list + map) 140h'te
    # ~170GB RAM'e çıkıp OOM-kill (exit 137) yapıyordu. Lazy'de RAM düz kalır:
    # her örnek erişildiğinde tek tek yüklenip mel-feature çıkarılır.
    def _extract(path, text):
        try:
            arr, _ = librosa.load(path, sr=16000, mono=True)
        except Exception:
            arr = np.zeros(16000, dtype=np.float32)  # nadir decode hatası: kısa sessizlik
        feats = processor.feature_extractor(arr, sampling_rate=16000).input_features[0]
        labels = processor.tokenizer(text).input_ids
        return feats, labels

    print(f"\nDataset (lazy feature extraction): {len(train_rows)} satır")
    dataset = Dataset.from_list(train_rows)  # sadece {audio_path, text} — RAM'de hafif

    def _transform(batch):
        feats, labs = [], []
        for p, t in zip(batch["audio_path"], batch["text"]):
            f, l = _extract(p, t)
            feats.append(f)
            labs.append(l)
        return {"input_features": feats, "labels": labs}

    dataset.set_transform(_transform)

    # Data collator
    import torch as _t

    def collate(features):
        input_features = _t.tensor(
            [f["input_features"] for f in features], dtype=_t.float32
        )
        labels = [f["labels"] for f in features]
        # Strip leading <|startoftranscript|>: model.forward re-adds it via
        # shift_tokens_right (decoder_start_token_id). Keeping it causes a
        # double-<sot> off-by-one prompt corruption (HF Whisper FT recipe).
        _sot = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        labels = [l[1:] if l and l[0] == _sot else l for l in labels]
        max_len = max(len(l) for l in labels)
        padded_labels = _t.tensor(
            [l + [-100] * (max_len - len(l)) for l in labels],
            dtype=_t.long
        )
        return {"input_features": input_features, "labels": padded_labels}

    # Trainer
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    # Budget: epoch-based by default; max_steps>0 overrides (FINETUNE_PLAN §3).
    use_steps = args.max_steps and args.max_steps > 0
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=(args.max_steps if use_steps else -1),
        num_train_epochs=(1 if use_steps else args.num_epochs),
        fp16=(device == "cuda"),
        logging_steps=50,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=(None if args.save_total_limit <= 0 else args.save_total_limit),
        predict_with_generate=False,
        report_to=[],
        dataloader_num_workers=4,  # lazy extraction'ı GPU ile örtüştür (DataLoader worker; map num_proc DEĞİL)
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate,
    )

    budget = f"max_steps={args.max_steps}" if use_steps else f"epochs={args.num_epochs}"
    print(f"\n=== Starting LoRA training ===")
    print(f"  Mode:         {args.mode}")
    print(f"  Model:        {args.model}")
    print(f"  LoRA:         r={args.lora_r}, alpha={args.lora_alpha}, "
          f"target={args.lora_target}")
    print(f"  Training rows:{len(dataset)}")
    print(f"  Budget:       {budget}")
    print(f"  Effective BS: {args.batch_size * args.grad_accum}")
    print(f"  Output dir:   {output_dir}")
    t_start = time.time()

    trainer.train()
    elapsed = time.time() - t_start

    # Save
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    meta = {
        "base_model":       args.model,
        "mode":             args.mode,
        "general_manifest": args.general_manifest,
        "medical_upsample": args.medical_upsample if args.mode == "general+medical" else 0,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "lora_target":      args.lora_target,
        "training_rows":    len(dataset),
        "budget":           (f"max_steps={args.max_steps}" if use_steps
                             else f"epochs={args.num_epochs}"),
        "elapsed_minutes":  round(elapsed / 60, 1),
        "output_dir":       str(output_dir),
        "completed_at":     time.strftime("%Y-%m-%d %H:%M"),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== Training complete in {elapsed/60:.1f} min ===")
    print(f"Adapter saved to: {output_dir}")
    print(f"\nNext step — run evaluation:")
    print(f"  python3 scripts/evaluate_lora_adapter.py \\")
    print(f"    --manifest evidence/eval_packs/combined_full_eval/manifest_colab.csv \\")
    print(f"    --base-model {args.model} \\")
    print(f"    --adapter-dir {output_dir} \\")
    print(f"    --output-dir evidence/benchmark_runs/lora_post_general \\")
    print(f"    --split all --limit 200")


if __name__ == "__main__":
    main()
