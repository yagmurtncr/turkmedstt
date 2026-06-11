# Turkish ASR Readability Post-Processor

A conservative, text-only post-processor for improving Turkish ASR
transcripts. The model applies casing, punctuation, and selected
high-confidence lexical corrections without freely generating transcript
content.

```text
raw Turkish ASR transcript -> safer, more readable Turkish text
```

## Models

| Release | Scope | WER | Correct inputs corrupted |
| --- | --- | ---: | ---: |
| [V1 General](https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1) | General Turkish ASR | `0.06179 -> 0.04277` | `0` |
| [V2 Medical](https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2) | Medical-domain adaptation | `0.31106 -> 0.18352` | `0` |

V1 General is the recommended release for general Turkish. V2 Medical was
trained and evaluated with controlled medical-text corruptions. It has not yet
been validated on real medical ASR audio and must not be treated as clinically
validated.

## Architecture

The accepted model is a single-checkpoint multi-head token editor built on
[`ytu-ce-cosmos/turkish-mini-bert-uncased`](https://huggingface.co/ytu-ce-cosmos/turkish-mini-bert-uncased).
It predicts:

- casing actions;
- punctuation actions;
- constrained lexical replacements.

This design prioritizes meaning preservation and low regression risk. It
cannot recover words missing from the audio or reliably repair severely
corrupted transcripts.

## Quick Start

Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download either Hugging Face model repository and run its bundled inference
script:

```powershell
python inference.py "bugün hava çok güzel dışarı çıkalım"
```

Expected output style:

```text
Bugün hava çok güzel dışarı çıkalım.
```

## Repository Contents

- `scripts/`: dataset preparation, training, inference, and evaluation code
- `huggingface/`: model-card and custom-inference release templates
- `reports/V1_V2_FINAL_RESULTS.md`: complete Turkish technical report
- `reports/release_manifest.json`: accepted release metrics and checksums
- `docs/HUGGINGFACE_RELEASE.md`: release and validation procedure

Large datasets, audio, model weights, checkpoints, audit queues, and local
runtime artifacts are intentionally excluded from GitHub. Released weights are
available from the Hugging Face model links above.

## Results And Limitations

The full methodology, rejected experiments, evaluation protocol, and
limitations are documented in
[reports/V1_V2_FINAL_RESULTS.md](reports/V1_V2_FINAL_RESULTS.md).

The largest remaining limitation is low coverage for genuine lexical ASR
errors. Improving it requires human-corrected real ASR output/target pairs.

## License Note

The training process used multiple sources with per-row license metadata. A
consolidated release license has not yet been selected. Review source licenses
and model cards before redistribution or commercial use.
