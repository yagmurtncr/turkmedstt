# -*- coding: utf-8 -*-
"""LoRA adaptörünü whisper-large-v3 base'e merge edip tek, dağıtılabilir model üretir (HF için).
GPU önerilir (CPU'da da çalışır, yavaş). Kullanım:
  python merge_lora.py <adapter_dir> <out_dir>
Örn: python merge_lora.py evidence/finetune_runs/full_r64/adapters/m2_genmed_full_r64 hf_release/merged_m2
Notlar:
- Üretilen klasör tam bir Whisper modeli olur (~3GB safetensors + processor) -> doğrudan HF model repo.
- CPU'da FP16 çıktı için MERGE_DTYPE=fp16 ortam değişkenini kullan.
- transformers/peft uyumu: 4.46.x + peft 0.13.x ile test edilmiştir (5.x'te de merge genelde çalışır)."""
import sys, os, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

ADAP = sys.argv[1]; OUT = sys.argv[2]; BASE = "openai/whisper-large-v3"
os.makedirs(OUT, exist_ok=True)
dtype_name = os.environ.get("MERGE_DTYPE", "auto").lower()
if dtype_name == "fp16":
    dt = torch.float16
elif dtype_name == "fp32":
    dt = torch.float32
elif dtype_name == "auto":
    dt = torch.float16 if torch.cuda.is_available() else torch.float32
else:
    raise ValueError("MERGE_DTYPE auto, fp16 veya fp32 olmalıdır")
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"base yükleniyor ({device_name}, {dt})...")
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=dt)
model = PeftModel.from_pretrained(model, ADAP)
print("merge...")
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
WhisperProcessor.from_pretrained(BASE, language="tr", task="transcribe").save_pretrained(OUT)
print("KAYDEDİLDİ:", OUT)
