"""Hugging Face custom code for the Turkish multi-head ASR token editor."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


class MultiTaskTokenEditorConfig(PretrainedConfig):
    model_type = "multitask_token_editor"

    def __init__(
        self,
        encoder_name="ytu-ce-cosmos/turkish-mini-bert-uncased",
        encoder_config=None,
        num_case_labels=3,
        num_punct_labels=8,
        num_edit_labels=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config
        self.num_case_labels = num_case_labels
        self.num_punct_labels = num_punct_labels
        self.num_edit_labels = num_edit_labels


@dataclass
class MultiTaskTokenEditorOutput(ModelOutput):
    loss: torch.Tensor | None = None
    case_logits: torch.Tensor | None = None
    punct_logits: torch.Tensor | None = None
    edit_logits: torch.Tensor | None = None


class MultiTaskTokenEditor(PreTrainedModel):
    config_class = MultiTaskTokenEditorConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        encoder_config = (
            AutoConfig.for_model(**config.encoder_config)
            if config.encoder_config
            else AutoConfig.from_pretrained(config.encoder_name)
        )
        self.encoder = AutoModel.from_config(encoder_config)
        hidden = encoder_config.hidden_size
        dropout = getattr(encoder_config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout)
        self.case_head = nn.Linear(hidden, config.num_case_labels)
        self.punct_head = nn.Linear(hidden, config.num_punct_labels)
        self.edit_head = nn.Linear(hidden, config.num_edit_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        case_labels=None,
        punct_labels=None,
        edit_labels=None,
        **kwargs,
    ):
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        ).last_hidden_state
        encoded = self.dropout(encoded)
        case_logits = self.case_head(encoded)
        punct_logits = self.punct_head(encoded)
        edit_logits = self.edit_head(encoded)
        loss = None
        if case_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = (
                loss_fn(case_logits.view(-1, self.config.num_case_labels), case_labels.view(-1))
                + loss_fn(punct_logits.view(-1, self.config.num_punct_labels), punct_labels.view(-1))
                + loss_fn(edit_logits.view(-1, self.config.num_edit_labels), edit_labels.view(-1))
            )
        return MultiTaskTokenEditorOutput(
            loss=loss,
            case_logits=case_logits,
            punct_logits=punct_logits,
            edit_logits=edit_logits,
        )
