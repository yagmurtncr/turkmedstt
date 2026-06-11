"""Source definitions for the Turkish speech transcript benchmark."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    source: str
    source_ref: str
    license: str
    domain: str
    target_count: int
    dataset_name: str
    dataset_config: str | None
    transcript_column: str
    audio_column: str
    notes: str


SOURCE_SPECS: list[SourceSpec] = [
    SourceSpec(
        source="fleurs_turkish",
        source_ref="google_fleurs",
        license="CC-BY-4.0",
        domain="fleurs",
        target_count=2000,
        dataset_name="google/fleurs",
        dataset_config="tr_tr",
        transcript_column="transcription",
        audio_column="audio",
        notes="First V1 source. Use human reference transcriptions from FLEURS and generate raw_asr from the audio.",
    ),
    SourceSpec(
        source="mediaspeech_turkish",
        source_ref="openslr108_mediaspeech",
        license="CC BY 4.0",
        domain="media_speech",
        target_count=3000,
        dataset_name="zeynepgulhan/mediaspeech-with-cv-tr",
        dataset_config=None,
        transcript_column="sentence",
        audio_column="audio",
        notes="Primary source is OpenSLR SLR108. Fallbacks: emre/Open_SLR108_Turkish_10_hours, then zeynepgulhan/mediaspeech-with-cv-tr parquet.",
    ),
    SourceSpec(
        source="common_voice_turkish",
        source_ref="mozilla_common_voice_tr",
        license="CC0-1.0",
        domain="common_voice",
        target_count=12000,
        dataset_name="mozilla-foundation/common_voice_17_0",
        dataset_config="tr",
        transcript_column="sentence",
        audio_column="audio",
        notes="Primary source is Mozilla Data Collective. Fallback mirrors: fsicoli/common_voice_17_0, ysdede/commonvoice_17_tr_fixed, fcanercan/common_voice_14_tr. If a mirror is used, mark the import as a practical import mirror.",
    ),
]

SOURCE_TARGETS = {spec.source: spec.target_count for spec in SOURCE_SPECS}
SOURCE_BY_NAME = {spec.source: spec for spec in SOURCE_SPECS}

TARGET_SPLITS = {
    "train": 16000,
    "validation": 2000,
    "test": 2000,
}

WER_BUCKET_TARGETS = {
    "near_clean": 0.10,
    "minor_errors": 0.30,
    "moderate_errors": 0.50,
    "major_errors": 0.10,
}

MANIFEST_COLUMNS = [
    "audio_id",
    "audio_path",
    "readable_text",
    "source",
    "source_ref",
    "license",
    "domain",
    "source_split",
]
