from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from turkmed_stt.benchmark import run_benchmark
from turkmed_stt.reports import summarize_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TurkMed STT benchmark matrix.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", default="configs/benchmark_models_20.csv")
    parser.add_argument("--output-dir", default="runs/benchmarks/matrix")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / f"matrix_status_{int(time.time())}.csv"
    statuses = []

    with Path(args.models).open("r", newline="", encoding="utf-8-sig") as handle:
        model_rows = []
        for row in csv.DictReader(handle):
            enabled = row.get("enabled", row.get("enabled_for_matrix", "true"))
            if enabled.lower() == "true":
                model_rows.append(row)

    for row in model_rows:
        model_id = row["model_id"]
        backend = row["backend"]
        out_csv = output_dir / f"benchmark_{row['rank']}_{safe_name(model_id)}.csv"
        started = time.perf_counter()
        try:
            run_benchmark(
                args.manifest,
                out_csv,
                backend=backend,
                model_name=model_id,
                limit=args.limit,
            )
            summary = summarize_benchmark(out_csv)
            statuses.append(
                {
                    "rank": row["rank"],
                    "model_id": model_id,
                    "backend": backend,
                    "status": "ok",
                    "seconds": round(time.perf_counter() - started, 2),
                    "output_csv": str(out_csv),
                    "summary": json.dumps(summary, ensure_ascii=False),
                    "error": "",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "rank": row["rank"],
                    "model_id": model_id,
                    "backend": backend,
                    "status": "failed",
                    "seconds": round(time.perf_counter() - started, 2),
                    "output_csv": str(out_csv),
                    "summary": "",
                    "error": repr(exc),
                }
            )
        write_status(status_path, statuses)
        print(statuses[-1])

    print(f"status_csv={status_path}")


def write_status(path: Path, rows: list[dict]) -> None:
    fieldnames = ["rank", "model_id", "backend", "status", "seconds", "output_csv", "summary", "error"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


if __name__ == "__main__":
    main()
