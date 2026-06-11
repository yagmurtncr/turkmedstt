# Hugging Face Release Guide

## Target Organization

`turkmedstt`

The initial private releases are available at:

- https://huggingface.co/turkmedstt/turkish-asr-readability-postprocessor-v1
- https://huggingface.co/turkmedstt/turkish-medical-asr-readability-postprocessor-v2

## Repository Names

| Release | Hugging Face repository |
| --- | --- |
| General Turkish V1 | `turkmedstt/turkish-asr-readability-postprocessor-v1` |
| Medical-domain V2 | `turkmedstt/turkish-medical-asr-readability-postprocessor-v2` |

V2 is named as a medical-domain model, but its model card must clearly state
that training and evaluation used controlled medical-text corruptions and that
the model has not been validated on real medical ASR.

## Prepare Upload Folders

```powershell
python scripts/prepare_hf_releases.py
```

Prepared folders:

```text
hf_release\turkish-asr-readability-postprocessor-v1
hf_release\turkish-medical-asr-readability-postprocessor-v2
```

Each folder contains only:

- final model weights;
- tokenizer files;
- model configuration and deployment profiles;
- custom Transformers model code;
- inference script;
- model card;
- final evaluation JSON;
- runtime requirements.

Training checkpoints, optimizer states, logs, and source datasets are not
included.

## Validate Before Upload

```powershell
python inference.py "bugün hava çok güzel dışarı çıkalım"
```

Run this command inside each prepared folder.

## Upload

First authenticate:

```powershell
huggingface-cli login
```

Preview:

```powershell
python scripts/upload_hf_releases.py
```

Perform the upload:

```powershell
python scripts/upload_hf_releases.py --execute
```

The upload script does not run unless `--execute` is explicitly supplied.

The authenticated token must have write/create permission for the `turkmedstt`
organization. A token scoped only to a personal user namespace will fail with
`403 Forbidden`.

## Required Post-Upload Checks

1. Both model cards render correctly.
2. V2 visibly states that it is synthetic and not clinically validated.
3. `model.safetensors`, `config.json`, custom modeling code, tokenizer files,
   `deployment_profiles.json`, and `inference.py` are present.
4. Download each repository into a clean directory and run inference.
5. Keep the repositories private until the license decision is finalized.

## License Warning

Both cards currently use `license: other`. Source datasets have per-row
licenses, but a consolidated model release license has not yet been formally
selected. Do not publish the repositories publicly until this is reviewed.
