# -*- coding: utf-8 -*-
"""V2 gerçek-ses setinde M0-M2 WER/CER farkları için eşleştirilmiş bootstrap."""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "evidence", "finetune_runs", "full_r64", "real_v2")
SPEAKERS = [("k1", "k1"), ("k2", "k2"), ("k4", "k3")]
SEED = 20260610
N_BOOT = 50000


def load_metric(raw_speaker, model, metric):
    path = os.path.join(BASE, f"{raw_speaker}_{model}.csv")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {
            os.path.basename(row["audio_filepath"]).split("_")[0]: float(row[metric])
            for row in csv.DictReader(handle)
        }


def paired_bootstrap(diff, rng):
    samples = diff[rng.integers(0, len(diff), (N_BOOT, len(diff)))].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    p_value = 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))
    return {
        "n": int(len(diff)),
        "delta_m0_minus_m2": float(diff.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_value": float(p_value),
    }


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "bootstrap_repetitions": N_BOOT, "metrics": {}}

    for metric in ("wer", "cer"):
        metric_results = {}
        pooled = []
        for raw_speaker, thesis_speaker in SPEAKERS:
            m0 = load_metric(raw_speaker, "m0", metric)
            m2 = load_metric(raw_speaker, "m2", metric)
            ids = sorted(m0.keys() & m2.keys())
            diff = np.array([m0[item] - m2[item] for item in ids])
            metric_results[thesis_speaker] = paired_bootstrap(diff, rng)
            pooled.extend(diff.tolist())
        metric_results["pooled"] = paired_bootstrap(np.array(pooled), rng)
        results["metrics"][metric] = metric_results

    json_path = os.path.join(BASE, "v2_bootstrap_significance.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    md_path = os.path.join(BASE, "V2_BOOTSTRAP_SIGNIFICANCE.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# V2 Gerçek-Ses Eşleştirilmiş Bootstrap Sonuçları\n\n")
        handle.write(f"Tohum: `{SEED}`; tekrar sayısı: `{N_BOOT}`. Fark M0-M2 olarak hesaplanmıştır.\n\n")
        for metric in ("wer", "cer"):
            handle.write(f"## {metric.upper()}\n\n")
            handle.write("| Kapsam | n | Ortalama fark | %95 GA alt | %95 GA üst | p |\n")
            handle.write("|---|---:|---:|---:|---:|---:|\n")
            for label in ("k1", "k2", "k3", "pooled"):
                row = results["metrics"][metric][label]
                p_text = "<0,0001" if row["p_value"] == 0 else f"{row['p_value']:.4f}".replace(".", ",")
                handle.write(
                    f"| {label} | {row['n']} | {row['delta_m0_minus_m2']:.4f} | "
                    f"{row['ci95_low']:.4f} | {row['ci95_high']:.4f} | {p_text} |\n"
                )
            handle.write("\n")

    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
