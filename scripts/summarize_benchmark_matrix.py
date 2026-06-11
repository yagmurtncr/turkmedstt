from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a TurkMed STT benchmark matrix run.")
    parser.add_argument("--run-dir", required=True, help="Directory containing matrix_status_*.csv and benchmark_*.csv files.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults to RUN_DIR/benchmark_matrix_summary.md")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output) if args.output else run_dir / "benchmark_matrix_summary.md"

    status_path = latest_status(run_dir)
    status_rows = read_csv(status_path) if status_path else []

    lines: list[str] = [
        "# Benchmark Matrix Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Status file: `{status_path.name if status_path else 'not found'}`",
        "",
        "## Model Results",
        "",
        "| Rank | Model | Backend | Status | Mean WER | Mean CER | Mean DS-WER | Mean RTF | Seconds |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in status_rows:
        summary = parse_summary(row.get("summary", ""))
        lines.append(
            "| {rank} | `{model}` | `{backend}` | {status} | {wer} | {cer} | {dswer} | {rtf} | {seconds} |".format(
                rank=row.get("rank", ""),
                model=row.get("model_id", ""),
                backend=row.get("backend", ""),
                status=row.get("status", ""),
                wer=fmt(summary.get("mean_wer")),
                cer=fmt(summary.get("mean_cer")),
                dswer=fmt(summary.get("mean_ds_wer")),
                rtf=fmt(summary.get("mean_rtf")),
                seconds=row.get("seconds", ""),
            )
        )

    failed = [row for row in status_rows if row.get("status") != "ok"]
    if failed:
        lines.extend(["", "## Failed Models", ""])
        for row in failed:
            lines.append(f"- `{row.get('model_id')}` ({row.get('backend')}): {row.get('error')}")

    for row in status_rows:
        if row.get("status") != "ok":
            continue
        csv_path = Path(row.get("output_csv", ""))
        if not csv_path.is_absolute():
            csv_path = run_dir / csv_path.name
        if not csv_path.exists():
            continue
        lines.extend(render_benchmark_detail(csv_path, int(args.top_k)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


def latest_status(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("matrix_status_*.csv"))
    return paths[-1] if paths else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_summary(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def fmt(value: object) -> str:
    if value in {"", None}:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def source_from_audio(path: str) -> str:
    name = Path(path).name
    if name.startswith("commonvoice_"):
        return "commonvoice"
    if name.startswith("issai_"):
        return "issai"
    if name.startswith("openslr_"):
        return "openslr"
    if name.startswith("medical_pilot_"):
        return "medical_pilot"
    return "unknown"


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in {"", "None", None}]
    return statistics.mean(values) if values else None


def render_benchmark_detail(csv_path: Path, top_k: int) -> list[str]:
    rows = read_csv(csv_path)
    if not rows:
        return []

    model = rows[0].get("model", csv_path.stem)
    backend = rows[0].get("backend", "")
    exact = sum(float(row["wer"]) == 0.0 and float(row["cer"]) == 0.0 for row in rows)
    blanks = sum(not row.get("hypothesis", "").strip() for row in rows)

    lines = [
        "",
        f"## `{model}` ({backend})",
        "",
        f"- Rows: {len(rows)}",
        f"- Exact WER/CER zero rows: {exact}",
        f"- Blank hypotheses: {blanks}",
        f"- Mean WER/CER/RTF: {fmt(mean(rows, 'wer'))} / {fmt(mean(rows, 'cer'))} / {fmt(mean(rows, 'rtf'))}",
        "",
        "| Source | Rows | Mean WER | Mean CER | Mean RTF |",
        "|---|---:|---:|---:|---:|",
    ]

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[source_from_audio(row.get("audio_filepath", ""))].append(row)
    for source, source_rows in sorted(by_source.items()):
        lines.append(
            f"| {source} | {len(source_rows)} | {fmt(mean(source_rows, 'wer'))} | {fmt(mean(source_rows, 'cer'))} | {fmt(mean(source_rows, 'rtf'))} |"
        )

    lines.extend(["", f"### Worst {top_k} By WER", ""])
    for row in sorted(rows, key=lambda item: float(item["wer"]), reverse=True)[:top_k]:
        audio = Path(row.get("audio_filepath", "")).name
        lines.extend(
            [
                f"- `{audio}` WER={fmt(row.get('wer'))} CER={fmt(row.get('cer'))}",
                f"  - REF: {row.get('reference', '')[:220]}",
                f"  - HYP: {row.get('hypothesis', '')[:220]}",
            ]
        )
    return lines


if __name__ == "__main__":
    main()
