# -*- coding: utf-8 -*-
"""TurkMedSTT - Türkçe Tıbbi ASR demo (Hugging Face Space, Gradio)."""
import os

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_ID = os.environ.get(
    "MODEL_ID",
    "turkmedstt/whisper-large-v3-turkish-medical",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEVICE == "cuda" else torch.float32

processor = WhisperProcessor.from_pretrained(MODEL_ID, language="tr", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=DT,
    low_cpu_mem_usage=True,
).to(DEVICE).eval()


def transcribe(audio):
    if audio is None:
        return ""
    arr, _ = librosa.load(audio, sr=16000, mono=True)
    inputs = processor(
        arr,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = inputs.input_features.to(DEVICE, DT)
    attention_mask = inputs.attention_mask.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            language="tr",
            task="transcribe",
            max_new_tokens=256,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Türkçe konuşma (klinik)"),
    outputs=gr.Textbox(label="Transkripsiyon"),
    title="TurkMedSTT - Türkçe Tıbbi Konuşma Tanıma",
    description=(
        "Mikrofon kaydını veya yüklenen sesi 16 kHz mono biçime dönüştürerek "
        "turkmedstt/whisper-large-v3-turkish-medical modeliyle yazıya çevirir. "
        "Konuşmacı ayrımı ve klinik doğrulama yapmaz. Araştırma amaçlıdır; "
        "tıbbi cihaz değildir ve klinik karar için kullanılmaz."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
