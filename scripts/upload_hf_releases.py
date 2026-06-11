"""Upload prepared releases to TurkMedSTT; dry-run unless --execute is supplied."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("hf_release")
RELEASES = {
    "turkmedstt/turkish-asr-readability-postprocessor-v1":
        ROOT / "turkish-asr-readability-postprocessor-v1",
    "turkmedstt/turkish-medical-asr-readability-postprocessor-v2":
        ROOT / "turkish-medical-asr-readability-postprocessor-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Create repositories and upload files.")
    parser.add_argument("--public", action="store_true", help="Create public repositories instead of private ones.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for repo_id, folder in RELEASES.items():
        if not folder.exists():
            raise SystemExit(f"Missing prepared release: {folder}")
        print(f"{'UPLOAD' if args.execute else 'DRY RUN'}: {folder} -> {repo_id}")
    if not args.execute:
        print("No files uploaded. Run again with --execute after reviewing the release folders.")
        return
    api = HfApi()
    for repo_id, folder in RELEASES.items():
        api.create_repo(repo_id=repo_id, repo_type="model", private=not args.public, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(folder),
            commit_message="Publish initial model release",
        )
        print(f"Uploaded https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
