from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .benchmark import run_benchmark
from .manifest import validate_manifest
from .metrics import score_pair
from .reports import render_run_markdown
from .stt_core import transcribe_audio


BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = RUNS_DIR / "uploads"
BENCH_DIR = RUNS_DIR / "benchmarks"

RUNS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TurkMed STT Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def models():
    return {
        "recommended": [
            {"backend": "transformers", "model": "openai/whisper-large-v3-turbo"},
            {"backend": "transformers", "model": "openai/whisper-large-v3"},
            {"backend": "openai-whisper", "model": "medium"},
            {"backend": "openai-whisper", "model": "small"},
            {"backend": "transformers", "model": "selimc/whisper-large-v3-turbo-turkish"},
            {"backend": "transformers", "model": "Dokkaemen/whisper-large-v3-turbo-tr-finetuned"},
            {"backend": "faster-whisper", "model": "vincespeed/faster-whisper-large-v3-turbo-turkish"},
            {"backend": "transformers_ctc", "model": "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"},
            {"backend": "qwen3_asr", "model": "Qwen/Qwen3-ASR-0.6B"},
            {"backend": "qwen3_asr", "model": "Qwen/Qwen3-ASR-1.7B"},
        ],
        "default": {"backend": "openai-whisper", "model": "small"},
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    backend: str = Form("openai-whisper"),
    model_name: str = Form("small"),
    reference_text: str = Form(""),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı boş.")
    suffix = Path(file.filename).suffix or ".wav"
    run_id = uuid4().hex
    upload_path = UPLOADS_DIR / f"{run_id}{suffix}"
    upload_path.write_bytes(await file.read())
    try:
        result = transcribe_audio(upload_path, backend=backend, model_name=model_name, work_dir=RUNS_DIR / "audio")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    score = score_pair(reference_text, result.text).to_dict() if reference_text.strip() else None
    payload = {"run_id": run_id, "transcription": result.to_dict(), "score": score}
    _save_run(run_id, payload)
    return payload


@app.post("/api/benchmark")
def benchmark(payload: dict):
    manifest_csv = payload.get("manifest_csv")
    if not manifest_csv:
        raise HTTPException(status_code=400, detail="manifest_csv gerekli.")
    errors, warnings = validate_manifest(manifest_csv)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors, "warnings": warnings})
    backend = payload.get("backend", "openai-whisper")
    model_name = payload.get("model_name", "small")
    out = BENCH_DIR / f"benchmark_{uuid4().hex[:8]}.csv"
    try:
        result_path = run_benchmark(
            manifest_csv,
            out,
            backend=backend,
            model_name=model_name,
            limit=payload.get("limit"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"benchmark_csv": str(result_path), "warnings": warnings}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run bulunamadı.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/export/{export_format}")
def export_run(run_id: str, export_format: str):
    run = get_run(run_id)
    if export_format == "json":
        return FileResponse(RUNS_DIR / f"{run_id}.json", media_type="application/json", filename=f"{run_id}.json")
    if export_format == "md":
        md_path = RUNS_DIR / f"{run_id}.md"
        md_path.write_text(render_run_markdown(run), encoding="utf-8")
        return FileResponse(md_path, media_type="text/markdown", filename=f"{run_id}.md")
    raise HTTPException(status_code=400, detail="Format json veya md olmalı.")


def _save_run(run_id: str, payload: dict) -> None:
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
