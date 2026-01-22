from __future__ import annotations

from pathlib import Path
import sys
import random

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kbqa_candidates import build_slot0_index, sample_candidates  # noqa: E402


def test_candidate_sampler_deterministic_unique() -> None:
    gold_idx = torch.tensor([2, 5], dtype=torch.long)
    rng = random.Random(123)
    out1 = sample_candidates(gold_idx, 5, 10, rng)
    rng = random.Random(123)
    out2 = sample_candidates(gold_idx, 5, 10, rng)
    assert torch.equal(out1, out2)
    assert torch.all(out1[:, 0] == gold_idx)
    for row in out1.tolist():
        assert len(row) == len(set(row))
        assert row.count(row[0]) == 1


def test_candidate_sampler_hard_neg_slot0() -> None:
    code_matrix = [
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
        [3, 0],
    ]
    slot0_to_rows = build_slot0_index(code_matrix)
    slot0_values = [row[0] for row in code_matrix]
    gold_idx = torch.tensor([0], dtype=torch.long)
    rng = random.Random(0)
    out = sample_candidates(
        gold_idx,
        4,
        len(code_matrix),
        rng,
        hard_neg_slot0=True,
        slot0_to_rows=slot0_to_rows,
        slot0_values=slot0_values,
    )
    row = out[0].tolist()
    assert row[0] == 0
    assert len(row) == len(set(row))
    assert any(idx in {1} for idx in row[1:])
