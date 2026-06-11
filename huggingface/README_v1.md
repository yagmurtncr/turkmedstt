---
language:
- tr
license: other
library_name: transformers
pipeline_tag: token-classification
tags:
- turkish
- asr
- post-processing
- punctuation-restoration
- text-normalization
- readability
base_model: ytu-ce-cosmos/turkish-mini-bert-uncased
---

# Turkish ASR Readability Post-Processor V1

`turkmedstt/turkish-asr-readability-postprocessor-v1` is a conservative,
text-only post-processor for Turkish ASR transcripts. It improves casing,
punctuation, and selected high-confidence lexical errors without freely
generating or deleting transcript content.

The model is a single-checkpoint multi-head token editor built on
`ytu-ce-cosmos/turkish-mini-bert-uncased`.

## Intended Use

```text
raw Turkish ASR transcript -> more readable Turkish transcript
```

This model is suitable for improving presentation and readability after ASR.
It is not designed to recover words missing from audio or fully repair severely
corrupted transcripts.

## Evaluation

Evaluation uses an untouched 3,191-row real-ASR readable-target test split.

| Metric | Raw ASR | V1 |
| --- | ---: | ---: |
| WER | 0.06179 | 0.04277 |
| CER | 0.00940 | 0.00631 |

- Improved / worsened rows: `222 / 5`
- Already-correct inputs corrupted: `0`

The model uses conservative confidence thresholds and applies only
high-confidence lexical replacements. Detailed task-level metrics are retained
in the project evaluation artifacts.

## Usage

Clone or download the repository, then run:

```powershell
pip install -r requirements.txt
python inference.py "bugün hava çok güzel dışarı çıkalım"
```

Expected style of output:

```text
Bugün hava çok güzel dışarı çıkalım.
```

Use the lower-risk profile:

```powershell
python inference.py "bugün hava çok güzel dışarı çıkalım" --profile safe
```

The model requires `trust_remote_code=True` because it uses a custom
multi-head Transformers architecture. Review `modeling_multitask_token_editor.py`
before loading.

## Deployment Profiles

- `balanced`: default profile, higher correction coverage.
- `safe`: fewer regressions, lower correction coverage.

See `deployment_profiles.json`.

## Limitations

- It cannot infer content absent from the ASR transcript.
- Lexical correction coverage is low.
- Names, locations, and domain terms may require retrieval or audio-aware
  retranscription.
- Human readability evaluation should be completed before high-stakes use.

## License

The training process used multiple sources with per-row license metadata.
Review source licenses before redistribution or commercial use. This model is
published with `license: other` until a consolidated release license is
formally selected.
