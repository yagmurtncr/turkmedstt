from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev

GENERAL_PREFIXES = {
    "commonvoice_tr_": "commonvoice_tr",
    "issai_": "issai",
    "openslr_tr_": "openslr_tr",
}


def source_for(audio_name: str) -> str | None:
    lowered = audio_name.lower()
    for prefix, source in GENERAL_PREFIXES.items():
        if lowered.startswith(prefix):
            return source
    return None


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def prepare_benchmark_rows(input_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    detail_rows: list[dict[str, object]] = []
    for csv_path in sorted(input_dir.glob("benchmark_*.csv")):
        if csv_path.stat().st_size < 1000:
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                audio_id = Path(row["audio_filepath"]).name
                source = source_for(audio_id)
                if source is None:
                    continue
                detail_rows.append(
                    {
                        "model": row["model"],
                        "backend": row["backend"],
                        "audio_id": audio_id,
                        "source": source,
                        "wer": row["wer"],
                        "cer": row["cer"],
                        "rtf": row["rtf"],
                        "runtime_seconds": row["runtime_seconds"],
                        "audio_seconds": row["audio_seconds"],
                    }
                )

    detail_rows.sort(key=lambda row: (str(row["model"]).lower(), str(row["audio_id"])))
    write_csv(
        output_dir / "data" / "per_utterance_metrics.csv",
        [
            "model",
            "backend",
            "audio_id",
            "source",
            "wer",
            "cer",
            "rtf",
            "runtime_seconds",
            "audio_seconds",
        ],
        detail_rows,
    )
    return detail_rows


def summarize_metrics(
    detail_rows: list[dict[str, object]], output_dir: Path
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    by_source: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        key = (str(row["model"]), str(row["backend"]))
        grouped[key].append(row)
        by_source[key + (str(row["source"]),)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (model, backend), rows in grouped.items():
        wers = [as_float(str(row["wer"])) for row in rows]
        cers = [as_float(str(row["cer"])) for row in rows]
        rtfs = [as_float(str(row["rtf"])) for row in rows]
        runtime = [as_float(str(row["runtime_seconds"])) for row in rows]
        audio = [as_float(str(row["audio_seconds"])) for row in rows]
        source_counts = defaultdict(int)
        for row in rows:
            source_counts[str(row["source"])] += 1
        summary_rows.append(
            {
                "model": model,
                "backend": backend,
                "clips": len(rows),
                "mean_wer": rounded(mean(v for v in wers if v is not None)),
                "mean_cer": rounded(mean(v for v in cers if v is not None)),
                "mean_rtf": rounded(mean(v for v in rtfs if v is not None)),
                "runtime_seconds": rounded(sum(v for v in runtime if v is not None)),
                "audio_minutes": rounded(sum(v for v in audio if v is not None) / 60),
                "commonvoice_tr_clips": source_counts["commonvoice_tr"],
                "issai_clips": source_counts["issai"],
                "openslr_tr_clips": source_counts["openslr_tr"],
            }
        )

    summary_rows.sort(key=lambda row: float(str(row["mean_wer"])))
    for rank, row in enumerate(summary_rows, start=1):
        row["rank"] = rank
    write_csv(
        output_dir / "summary" / "leaderboard.csv",
        [
            "rank",
            "model",
            "backend",
            "clips",
            "mean_wer",
            "mean_cer",
            "mean_rtf",
            "runtime_seconds",
            "audio_minutes",
            "commonvoice_tr_clips",
            "issai_clips",
            "openslr_tr_clips",
        ],
        summary_rows,
    )

    source_rows: list[dict[str, object]] = []
    for (model, backend, source), rows in by_source.items():
        source_rows.append(
            {
                "model": model,
                "backend": backend,
                "source": source,
                "clips": len(rows),
                "mean_wer": rounded(mean(float(str(row["wer"])) for row in rows)),
                "mean_cer": rounded(mean(float(str(row["cer"])) for row in rows)),
                "mean_rtf": rounded(mean(float(str(row["rtf"])) for row in rows)),
            }
        )
    source_rows.sort(key=lambda row: (str(row["model"]).lower(), str(row["source"])))
    write_csv(
        output_dir / "summary" / "source_breakdown.csv",
        ["model", "backend", "source", "clips", "mean_wer", "mean_cer", "mean_rtf"],
        source_rows,
    )
    return summary_rows


def prepare_ascs(
    input_path: Path, output_dir: Path
) -> dict[str, dict[str, object]]:
    if not input_path.exists():
        return {}

    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            audio_id = Path(row["audio"]).name
            source = source_for(audio_id)
            if source is None:
                continue
            rows.append(
                {
                    "model": row["model"],
                    "audio_id": audio_id,
                    "source": source,
                    "wer": row["wer"],
                    "ref_sentiment": row["ref_sentiment"],
                    "hyp_sentiment": row["hyp_sentiment"],
                    "ascs_text": row["ascs_text"],
                }
            )
    rows.sort(key=lambda row: (str(row["model"]).lower(), str(row["audio_id"])))
    write_csv(
        output_dir / "data" / "acosemantic_per_utterance.csv",
        [
            "model",
            "audio_id",
            "source",
            "wer",
            "ref_sentiment",
            "hyp_sentiment",
            "ascs_text",
        ],
        rows,
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary = []
    for model, model_rows in grouped.items():
        wers = [float(str(row["wer"])) for row in model_rows]
        ascs_values = [float(str(row["ascs_text"])) for row in model_rows]
        ref_sentiments = [float(str(row["ref_sentiment"])) for row in model_rows]
        hyp_sentiments = [float(str(row["hyp_sentiment"])) for row in model_rows]
        summary.append(
            {
                "model": model,
                "clips": len(model_rows),
                "mean_wer": rounded(mean(wers)),
                "mean_ascs_text": rounded(mean(ascs_values)),
                "std_ascs_text": rounded(pstdev(ascs_values)),
                "ascs_wer_correlation": rounded(pearson_correlation(wers, ascs_values)),
                "mean_ref_sentiment": rounded(mean(ref_sentiments)),
                "mean_hyp_sentiment": rounded(mean(hyp_sentiments)),
                "mean_sentiment_drift": rounded(
                    mean(
                        abs(reference - hypothesis)
                        for reference, hypothesis in zip(
                            ref_sentiments, hyp_sentiments
                        )
                    )
                ),
            }
        )
    summary.sort(key=lambda row: float(str(row["mean_wer"])))
    write_csv(
        output_dir / "summary" / "acosemantic_summary.csv",
        [
            "model",
            "clips",
            "mean_wer",
            "mean_ascs_text",
            "std_ascs_text",
            "ascs_wer_correlation",
            "mean_ref_sentiment",
            "mean_hyp_sentiment",
            "mean_sentiment_drift",
        ],
        summary,
    )
    return {str(row["model"]): row for row in summary}


def write_combined_leaderboard(
    summary_rows: list[dict[str, object]],
    ascs_by_model: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    fields = [
        "rank",
        "model",
        "backend",
        "clips",
        "mean_wer",
        "mean_cer",
        "mean_rtf",
        "mean_ascs_text",
        "std_ascs_text",
        "ascs_wer_correlation",
        "mean_ref_sentiment",
        "mean_hyp_sentiment",
        "mean_sentiment_drift",
        "runtime_seconds",
        "audio_minutes",
        "commonvoice_tr_clips",
        "issai_clips",
        "openslr_tr_clips",
    ]
    for row in summary_rows:
        ascs = ascs_by_model.get(str(row["model"]), {})
        for field in fields:
            if field not in row and field in ascs:
                row[field] = ascs[field]
    write_csv(output_dir / "summary" / "leaderboard.csv", fields, summary_rows)


def render_readme(
    template_path: Path,
    output_path: Path,
    summary_rows: list[dict[str, object]],
) -> None:
    rows = []
    for row in summary_rows:
        rows.append(
            "| {rank} | `{model}` | {backend} | {mean_wer:.4f} | {mean_cer:.4f} | "
            "{mean_rtf:.4f} | {mean_ascs_text:.4f} | {std_ascs_text:.4f} | "
            "{ascs_wer_correlation:.4f} | {mean_sentiment_drift:.4f} |".format(
                rank=row["rank"],
                model=row["model"],
                backend=row["backend"],
                mean_wer=float(str(row["mean_wer"])),
                mean_cer=float(str(row["mean_cer"])),
                mean_rtf=float(str(row["mean_rtf"])),
                mean_ascs_text=float(str(row["mean_ascs_text"])),
                std_ascs_text=float(str(row["std_ascs_text"])),
                ascs_wer_correlation=float(str(row["ascs_wer_correlation"])),
                mean_sentiment_drift=float(str(row["mean_sentiment_drift"])),
            )
        )
    text = template_path.read_text(encoding="utf-8")
    text = text.replace("{{LEADERBOARD_ROWS}}", "\n".join(rows))
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "hf_release"
        / "turkish_asr_benchmark",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    detail_rows = prepare_benchmark_rows(
        project_root / "turkmed_thesis_package" / "03_benchmark_results",
        output_dir,
    )
    summary_rows = summarize_metrics(detail_rows, output_dir)
    ascs_by_model = prepare_ascs(
        project_root
        / "turkmed_thesis_package"
        / "05_analysis"
        / "acosemantic"
        / "step_d_per_utterance.csv",
        output_dir,
    )
    write_combined_leaderboard(summary_rows, ascs_by_model, output_dir)
    render_readme(
        project_root / "hf_release" / "benchmark_README.md",
        output_dir / "README.md",
        summary_rows,
    )
    shutil.copy2(project_root / "hf_release" / "benchmark_LICENSE", output_dir / "LICENSE")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "per_utterance_rows": len(detail_rows),
                "models": len(summary_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
