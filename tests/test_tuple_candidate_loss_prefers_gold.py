from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from loss_kbqa import candidate_tuple_logprobs, tuple_ce_loss_candidates  # noqa: E402


def test_tuple_candidate_loss_prefers_gold() -> None:
    slot_logits = torch.tensor(
        [
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    cand_codes = torch.tensor(
        [
            [[0, 0], [1, 1], [2, 2]],
        ],
        dtype=torch.long,
    )
    logprobs = candidate_tuple_logprobs(slot_logits, cand_codes)
    loss_good = tuple_ce_loss_candidates(logprobs)

    cand_codes_bad = torch.tensor(
        [
            [[1, 1], [0, 0], [2, 2]],
        ],
        dtype=torch.long,
    )
    logprobs_bad = candidate_tuple_logprobs(slot_logits, cand_codes_bad)
    loss_bad = tuple_ce_loss_candidates(logprobs_bad)

    assert loss_good < loss_bad
