from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codebook import (  # noqa: E402
    build_inverted_index,
    candidate_rows_from_logits,
    constrained_decode_candidates_by_logprobs,
)


def test_candidate_constrained_decode_returns_valid_rows() -> None:
    code_matrix = np.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=np.int64,
    )
    offsets, flat = build_inverted_index(code_matrix, vocab_size=3)
    slot_logits = torch.tensor([[[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]], dtype=torch.float32)
    cand_rows = candidate_rows_from_logits(
        slot_logits,
        offsets,
        flat,
        topk_per_slot=1,
        max_candidates=8,
    )
    pred_rows, pred_codes = constrained_decode_candidates_by_logprobs(slot_logits, code_matrix, cand_rows)
    assert int(pred_rows[0].item()) in set(int(x) for x in cand_rows[0])
    assert np.array_equal(pred_codes[0].cpu().numpy(), code_matrix[int(pred_rows[0].item())])
