from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_report_schema_and_values() -> None:
    slot_logits = torch.tensor(
        [
            [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]],
            [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
        ],
        dtype=torch.float32,
    )
    code_matrix = torch.tensor(
        [
            [0, 0],
            [1, 1],
            [2, 2],
        ],
        dtype=torch.long,
    )
    gold_tuple_idx = torch.tensor([1, 2], dtype=torch.long)
    tuple_idx_to_answer = ["2000", "2001", "2002"]
    gold_answers = ["2001", "2002"]

    report = train_cfm._compute_overfit_report(
        slot_logits=slot_logits,
        code_matrix=code_matrix,
        gold_tuple_idx=gold_tuple_idx,
        tuple_idx_to_answer=tuple_idx_to_answer,
        gold_answers=gold_answers,
        run_id="test-run",
        source="overfit-lama-years",
    )

    assert report["code_em"] == 1.0
    assert report["answer_em"] == 1.0
    assert report["orbit_consistency"] == {"count": 0, "rate": None}
    assert report["decode_mode_used"] == "constrained"
    assert report["n_eval"] == 2
    assert report["candidate_size"] == 3
    assert report["run_id"] == "test-run"
    assert report["source"] == "overfit-lama-years"

    negative_margin = report["negative_margin"]
    assert "negative_margin_rate" in negative_margin
    assert "margin_min_mean" in negative_margin
    assert "margin_min_p50" in negative_margin
    assert "margin_min_p05" in negative_margin

    em_breakdown = report["em_breakdown"]
    assert em_breakdown["code_em_constrained"] == 1.0
    assert em_breakdown["answer_em_constrained"] == 1.0
