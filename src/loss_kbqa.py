from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def tuple_scores_from_slot_logits(
    logits: torch.Tensor,
    code_matrix: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")
    if code_matrix.dim() != 2:
        raise ValueError(f"code_matrix must be [K,S], got {tuple(code_matrix.shape)}")
    bsz, slots, _ = logits.shape
    if code_matrix.shape[1] != slots:
        raise ValueError("code_matrix slot count must match logits")
    scores = None
    for slot in range(slots):
        slot_logits = logits[:, slot, :]
        idx = code_matrix[:, slot].unsqueeze(0).expand(bsz, -1)
        gathered = slot_logits.gather(1, idx)
        scores = gathered if scores is None else scores + gathered
    if scores is None:
        return torch.empty((bsz, 0), device=logits.device, dtype=logits.dtype)
    return scores


def tuple_ce_loss(
    logits: torch.Tensor,
    code_matrix: torch.Tensor,
    gold_tuple_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = tuple_scores_from_slot_logits(logits, code_matrix)
    loss = F.cross_entropy(scores, gold_tuple_idx)
    pred_idx = scores.argmax(dim=-1)
    return loss, pred_idx


def candidate_tuple_logprobs(
    slot_logits: torch.Tensor,
    cand_codes: torch.Tensor,
) -> torch.Tensor:
    if slot_logits.dim() != 3:
        raise ValueError(f"slot_logits must be [B,S,V], got {tuple(slot_logits.shape)}")
    if cand_codes.dim() != 3:
        raise ValueError(f"cand_codes must be [B,C,S], got {tuple(cand_codes.shape)}")
    batch, slots, _ = slot_logits.shape
    if cand_codes.shape[0] != batch or cand_codes.shape[2] != slots:
        raise ValueError("cand_codes shape must be [B,C,S] matching slot_logits [B,S,V]")

    logp = F.log_softmax(slot_logits, dim=-1)
    scores = None
    for slot in range(slots):
        slot_logp = logp[:, slot, :]
        idx = cand_codes[:, :, slot]
        gathered = slot_logp.gather(1, idx)
        scores = gathered if scores is None else scores + gathered
    if scores is None:
        return torch.empty((batch, 0), device=slot_logits.device, dtype=slot_logits.dtype)
    return scores


def tuple_ce_loss_candidates(tuple_logprobs: torch.Tensor) -> torch.Tensor:
    if tuple_logprobs.dim() != 2:
        raise ValueError(f"tuple_logprobs must be [B,C], got {tuple(tuple_logprobs.shape)}")
    targets = torch.zeros(tuple_logprobs.size(0), dtype=torch.long, device=tuple_logprobs.device)
    return F.cross_entropy(tuple_logprobs, targets)


__all__ = [
    "tuple_scores_from_slot_logits",
    "tuple_ce_loss",
    "candidate_tuple_logprobs",
    "tuple_ce_loss_candidates",
]
