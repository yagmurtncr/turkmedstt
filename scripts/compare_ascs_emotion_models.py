from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "evidence/acosemantic"
KUBI = IN_DIR / "step_f_ascs_emotion_model_summary.csv"
COLTEKIN = IN_DIR / "step_f_ascs_emotion_coltekin_tremo_model_summary.csv"
OUT_CSV = IN_DIR / "step_f_ascs_emotion_model_comparison.csv"
OUT_MD = IN_DIR / "step_f_ascs_emotion_model_comparison.md"


def main() -> None:
    kubi = keyed(read_csv(KUBI))
    coltekin = keyed(read_csv(COLTEKIN))
    models = sorted(set(kubi) & set(coltekin))
    rows = []
    for model in models:
        a = kubi[model]
        b = coltekin[model]
        rows.append(
            {
                "model": model,
                "kubi_mean_ascs_emotion": a["mean_ascs_emotion"],
                "coltekin_mean_ascs_emotion": b["mean_ascs_emotion"],
                "mean_ascs_delta_abs": round(abs(float(a["mean_ascs_emotion"]) - float(b["mean_ascs_emotion"])), 4),
                "kubi_flip_rate": a["affective_flip_rate"],
                "coltekin_flip_rate": b["affective_flip_rate"],
                "kubi_low_impact": a["low_wer_high_impact_count"],
                "coltekin_low_impact": b["low_wer_high_impact_count"],
                "mean_wer": a["mean_wer"],
            }
        )
    rows.sort(key=lambda row: float(row["kubi_mean_ascs_emotion"]), reverse=True)
    write_csv(rows, OUT_CSV)
    OUT_MD.write_text(render(rows), encoding="utf-8")
    print(f"comparison_csv={OUT_CSV}")
    print(f"comparison_md={OUT_MD}")


def render(rows: list[dict[str, Any]]) -> str:
    kubi_scores = [float(row["kubi_mean_ascs_emotion"]) for row in rows]
    col_scores = [float(row["coltekin_mean_ascs_emotion"]) for row in rows]
    corr = correlation(kubi_scores, col_scores)
    mean_delta = statistics.mean(float(row["mean_ascs_delta_abs"]) for row in rows)
    lines = [
        "# AcoSemantic Step F - Text Emotion Model Comparison",
        "",
        "Compared analysis models:",
        "",
        "- `kubi565/emotion-berturk-turkish`",
        "- `coltekin/berturk-tremo`",
        "",
        "Both are used as analysis models only. Their model-card license metadata is missing/unclear, so this comparison strengthens evaluation robustness but does not create a redistributable trained model claim.",
        "",
        f"Pearson correlation between per-ASR-model ASCS_emotion scores: {corr:.4f}",
        f"Mean absolute ASCS_emotion delta: {mean_delta:.4f}",
        "",
        "## Comparison Table",
        "",
        "| Model | WER | Kubi ASCS | Coltekin ASCS | Abs delta | Kubi flips | Coltekin flips | Kubi low-impact | Coltekin low-impact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| `{model}` | {wer} | {kubi} | {col} | {delta} | {kf} | {cf} | {ki} | {ci} |".format(
                model=row["model"],
                wer=row["mean_wer"],
                kubi=row["kubi_mean_ascs_emotion"],
                col=row["coltekin_mean_ascs_emotion"],
                delta=row["mean_ascs_delta_abs"],
                kf=row["kubi_flip_rate"],
                cf=row["coltekin_flip_rate"],
                ki=row["kubi_low_impact"],
                ci=row["coltekin_low_impact"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If the two analysis models rank ASR systems similarly, ASCS_emotion is less likely to be an artifact of a single text-emotion model. Large deltas should be manually inspected before making strong claims about a specific ASR model.",
            "",
        ]
    )
    return "\n".join(lines)


def correlation(left: list[float], right: list[float]) -> float:
    mean_left = statistics.mean(left)
    mean_right = statistics.mean(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right)) / len(left)
    std_left = statistics.stdev(left)
    std_right = statistics.stdev(right)
    if std_left == 0 or std_right == 0:
        return 0.0
    return cov / (std_left * std_right)


def keyed(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
