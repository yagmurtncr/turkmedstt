"""Prepare clean, upload-ready Hugging Face repositories for accepted releases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "models"
OUTPUT = ROOT / "hf_release"
COMMON_FILES = [
    "config.json",
    "model.safetensors",
    "multitask_metadata.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "deployment_profiles.json",
]
RELEASES = {
    "turkish-asr-readability-postprocessor-v1": {
        "source": WORK / "turkish-asr-multitask-editor-v3-final",
        "card": ROOT / "huggingface" / "README_v1.md",
        "evaluation": ROOT / "reports" / "multitask_token_editor_v3_final_evaluation.json",
    },
    "turkish-medical-asr-readability-postprocessor-v2": {
        "source": WORK / "turkish-asr-postprocessor-v2-medical-synthetic",
        "card": ROOT / "huggingface" / "README_v2_medical.md",
        "evaluation": ROOT / "reports" / "postprocessor_v2_medical_synthetic_evaluation.json",
    },
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, release in RELEASES.items():
        destination = OUTPUT / name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        for filename in COMMON_FILES:
            shutil.copy2(release["source"] / filename, destination / filename)
        shutil.copy2(ROOT / "huggingface" / "modeling_multitask_token_editor.py", destination)
        shutil.copy2(ROOT / "huggingface" / "inference.py", destination)
        shutil.copy2(release["card"], destination / "README.md")
        shutil.copy2(release["evaluation"], destination / "evaluation.json")
        (destination / "requirements.txt").write_text(
            "torch>=2.4.0\ntransformers>=4.48.0\n", encoding="utf-8"
        )
        config_path = destination / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["auto_map"] = {
            "AutoConfig": "modeling_multitask_token_editor.MultiTaskTokenEditorConfig",
            "AutoModel": "modeling_multitask_token_editor.MultiTaskTokenEditor",
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared {destination}")


if __name__ == "__main__":
    main()
