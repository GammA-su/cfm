from __future__ import annotations

from typing import Optional, Tuple

import inspect
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TinyDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self._warned_truncate = False
        self._warned_bad_ids = False

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + 2, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        encoder_kwargs = {}
        if "enable_nested_tensor" in inspect.signature(nn.TransformerEncoder).parameters:
            encoder_kwargs["enable_nested_tensor"] = False
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, **encoder_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer("_causal_mask", self._build_causal_mask(max_seq_len + 2), persistent=False)

    @staticmethod
    def _build_causal_mask(length: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None and attention_mask.device != input_ids.device:
            attention_mask = attention_mask.to(input_ids.device)
        if prefix_emb is not None and prefix_emb.device != input_ids.device:
            prefix_emb = prefix_emb.to(input_ids.device)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()

        batch_size, seq_len = input_ids.shape
        if prefix_emb is not None:
            if prefix_emb.dim() == 2:
                prefix_emb = prefix_emb.unsqueeze(1)
            prefix_len = prefix_emb.size(1)
        else:
            prefix_len = 0
        max_len = self.pos_emb.num_embeddings
        total_len = seq_len + prefix_len
        if total_len > max_len:
            overflow = total_len - max_len
            input_ids = input_ids[:, overflow:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, overflow:]
            seq_len = input_ids.size(1)
            if not self._warned_truncate:
                logging.getLogger(__name__).warning(
                    "TinyDecoder truncated input from %s to %s tokens to fit max_len=%s",
                    total_len,
                    seq_len + prefix_len,
                    max_len,
                )
                self._warned_truncate = True

        if input_ids.numel():
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= self.vocab_size:
                if not self._warned_bad_ids:
                    logging.getLogger(__name__).warning(
                        "TinyDecoder clamping input_ids to vocab range [0, %s] (saw min=%s max=%s)",
                        self.vocab_size - 1,
                        min_id,
                        max_id,
                    )
                    self._warned_bad_ids = True
                input_ids = input_ids.clamp(min=0, max=self.vocab_size - 1)

        positions = torch.arange(seq_len + prefix_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        token_emb = self.token_emb(input_ids)
        if prefix_emb is not None:
            token_emb = torch.cat([prefix_emb, token_emb], dim=1)

        pos_emb = self.pos_emb(positions)
        hidden = token_emb + pos_emb

        causal_mask = self._causal_mask[: seq_len + prefix_len, : seq_len + prefix_len]
        key_padding_mask = None
        if attention_mask is not None:
            if prefix_len:
                prefix_mask = torch.ones(batch_size, prefix_len, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            key_padding_mask = attention_mask == 0

        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(hidden)
        return hidden, logits


class HfDecoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.config.use_cache = False
        self.emb = self.model.get_input_embeddings()
        self.d_model = self.emb.embedding_dim
        self.lm_head = self.model.get_output_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prefix_emb is not None:
            if prefix_emb.dim() == 2:
                prefix_emb = prefix_emb.unsqueeze(1)
            input_emb = self.emb(input_ids)
            input_emb = torch.cat([prefix_emb, input_emb], dim=1)
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    input_ids.size(0),
                    prefix_emb.size(1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            outputs = self.model(inputs_embeds=input_emb, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        logits = outputs.logits
        return hidden, logits


__all__ = ["TinyDecoder", "HfDecoder"]
