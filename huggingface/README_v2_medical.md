---
language:
- tr
license: other
library_name: transformers
pipeline_tag: token-classification
tags:
- turkish
- medical
- asr
- post-processing
- synthetic-data
- punctuation-restoration
- readability
base_model: ytu-ce-cosmos/turkish-mini-bert-uncased
---

# Turkish Medical ASR Readability Post-Processor V2

`turkmedstt/turkish-medical-asr-readability-postprocessor-v2` is a
medical-domain adaptation of the Turkish ASR readability token editor.

**Important:** This model was trained and evaluated using synthetic medical
text corruptions. It has not been validated on real medical ASR because the
referenced medical WAV files were unavailable.

## Intended Use

This release is an experimental domain-adaptation candidate for research,
controlled pilots, and preparation for real medical-ASR validation.

It must not be described as a clinically validated model or used for clinical
decisions without real medical speech evaluation and human review.

## Synthetic Medical Evaluation

The held-out test split contains 136 sentence-isolated synthetic medical
examples.

| Metric | Synthetic corrupted input | V2 Medical |
| --- | ---: | ---: |
| WER | 0.31106 | 0.18352 |
| CER | 0.05491 | 0.03547 |

- Improved / worsened rows: `127 / 0`
- Already-correct inputs corrupted: `0`

Detailed task-level metrics are retained in the project evaluation artifacts.

## General Turkish Retention

On the untouched general Turkish ASR test:

- WER: `0.06179 -> 0.04328`
- Improved / worsened rows: `217 / 5`
- Already-correct inputs corrupted: `0`

The general V1 model remains the recommended model for non-medical Turkish.

## Usage

Clone or download the repository, then run:

```powershell
pip install -r requirements.txt
python inference.py "göğsünüzdeki sıkışma yürürken artıyorsa anjina olabilir"
```

For use outside the synthetic medical scope:

```powershell
python inference.py "örnek bir türkçe asr çıktısı" --profile general_fallback
```

The model requires `trust_remote_code=True` because it uses a custom
multi-head Transformers architecture. Review `modeling_multitask_token_editor.py`
before loading.

## Data Scope

The medical source manifests contain:

- 3,236 records;
- 809 unique medical sentences;
- 20 clinical areas;
- 4 synthetic speaker profiles.

Controlled corruptions include lowercasing, punctuation removal, Turkish
diacritic removal, and selected medical-term spelling corruption.

## Limitations

- No real medical audio or real medical-ASR errors were used for final
  validation.
- Real accents, background noise, spontaneous speech, and genuine medical term
  recognition errors are not represented by the synthetic test.
- Human review is required for medical usage.

## License

The training process used multiple sources with per-row license metadata.
Review source licenses before redistribution or commercial use. This model is
published with `license: other` until a consolidated release license is
formally selected.
