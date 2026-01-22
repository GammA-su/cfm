from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loss_kbqa import candidate_tuple_logprobs  # noqa: E402


def test_candidate_tuple_logprobs_shapes() -> None:
    slot_logits = torch.randn(2, 3, 5)
    cand_codes = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 0, 2], [2, 1, 3], [3, 2, 4]],
        ],
        dtype=torch.long,
    )
    scores = candidate_tuple_logprobs(slot_logits, cand_codes)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()
