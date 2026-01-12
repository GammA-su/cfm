from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from .tiny_decoder import HfDecoder, TinyDecoder


class CFMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        m: int,
        K: int,
        codebooks: torch.Tensor,
        backbone: str = "tiny",
        hf_model_name: str | None = None,
    ) -> None:
        super().__init__()
        if backbone == "hf":
            if not hf_model_name:
                raise ValueError("hf_model_name must be set when backbone=hf")
            self.backbone = HfDecoder(hf_model_name)
            d_model = self.backbone.d_model
        else:
            self.backbone = TinyDecoder(vocab_size, d_model, n_layers, n_heads, max_seq_len)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.m = m
        self.K = K

        code_dim = codebooks.shape[-1]
        self.addr_heads = nn.ModuleList([nn.Linear(d_model, K) for _ in range(m)])
        self.value_proj = nn.Linear(code_dim, d_model)
        self.verifier = nn.Linear(d_model, 1)

        self.register_buffer("codebooks", codebooks)

    @classmethod
    def from_codebooks(
        cls,
        codebook_path: Path,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
        m: int,
        K: int,
        backbone: str = "tiny",
        hf_model_name: str | None = None,
    ) -> "CFMModel":
        tensors = load_file(str(codebook_path))
        codebooks = tensors["codebooks"]
        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            m=m,
            K=K,
            codebooks=codebooks,
            backbone=backbone,
            hf_model_name=hf_model_name,
        )

    def decode_value(self, codes: torch.Tensor) -> torch.Tensor:
        batch, m = codes.shape
        idx = torch.arange(m, device=codes.device).unsqueeze(0).expand(batch, -1)
        vecs = self.codebooks[idx, codes]
        return vecs.sum(dim=1)

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        hidden, _ = self.backbone(prompt_ids, attention_mask=prompt_mask)
        if prompt_mask is not None:
            lengths = prompt_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            rep = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        else:
            rep = hidden[:, -1]
        addr_logits = [head(rep) for head in self.addr_heads]
        confidence = torch.sigmoid(self.verifier(rep)).squeeze(-1)
        return addr_logits, confidence, rep

    def forward_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        value_vec = self.decode_value(codes)
        value_token = self.value_proj(value_vec)
        hidden, logits = self.backbone(input_ids, attention_mask=attention_mask, prefix_emb=value_token)
        return hidden, logits

    def predict_codes(self, prompt_ids: torch.Tensor, prompt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        addr_logits, _, _ = self.encode_prompt(prompt_ids, prompt_mask)
        codes = torch.stack([logits.argmax(dim=-1) for logits in addr_logits], dim=1)
        return codes

    def generate_answer(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_new_tokens: int,
        eos_id: int,
        bos_id: int,
        sep_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_seq_len and prompt_ids.size(1) > self.max_seq_len:
            prompt_ids = prompt_ids[:, -self.max_seq_len :]
            prompt_mask = prompt_mask[:, -self.max_seq_len :]
        codes = self.predict_codes(prompt_ids, prompt_mask)
        batch_size = prompt_ids.size(0)
        generated = torch.full((batch_size, 1), bos_id, device=prompt_ids.device, dtype=torch.long)
        attention = torch.ones_like(generated)

        for _ in range(max_new_tokens):
            input_ids = torch.cat([prompt_ids, torch.full_like(prompt_ids[:, :1], sep_id), generated], dim=1)
            attn = torch.cat([prompt_mask, torch.ones_like(prompt_ids[:, :1]), attention], dim=1)
            if self.max_seq_len and input_ids.size(1) > self.max_seq_len:
                overflow = input_ids.size(1) - self.max_seq_len
                input_ids = input_ids[:, overflow:]
                attn = attn[:, overflow:]
            _, logits = self.forward_generation(input_ids, attn, codes)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            attention = torch.cat([attention, torch.ones_like(next_token)], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_id):
                break

        return generated, codes


def contrastive_margin_loss(v_pos: torch.Tensor, v_neg: torch.Tensor, margin: float) -> torch.Tensor:
    v_pos = F.normalize(v_pos, dim=-1)
    v_neg = F.normalize(v_neg, dim=-1)
    cosine = (v_pos * v_neg).sum(dim=-1)
    return torch.relu(cosine - margin).mean()


__all__ = ["CFMModel", "contrastive_margin_loss"]
