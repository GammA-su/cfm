from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loss_kbqa import tuple_ce_loss, tuple_scores_from_slot_logits  # noqa: E402


def test_tuple_ce_loss_prefers_correct_tuple() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    code_matrix = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=torch.long,
    )
    gold_idx = torch.tensor([0, 1], dtype=torch.long)
    scores = tuple_scores_from_slot_logits(logits, code_matrix)
    assert scores.argmax(dim=-1).tolist() == [0, 1]
    loss, pred_idx = tuple_ce_loss(logits, code_matrix, gold_idx)
    assert pred_idx.tolist() == [0, 1]
    assert loss.item() >= 0.0
