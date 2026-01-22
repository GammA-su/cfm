from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codebook import build_code_matrix, build_reverse_codebook, constrained_decode_by_logprobs  # noqa: E402
from metrics_kbqa import code_tuple_em, exact_match  # noqa: E402


def test_eval_em_metrics_alignment() -> None:
    df = pd.DataFrame(
        [
            {"answer": "A", "codes": [0, 0]},
            {"answer": "B", "codes": [0, 1]},
            {"answer": "C", "codes": [1, 1]},
        ]
    )
    code_mat = build_code_matrix(df)
    logits = np.array([[[0.0, 2.0], [2.0, 1.5]]], dtype=np.float32)
    pred_codes = constrained_decode_by_logprobs(logits, code_mat, chunk=2)
    reverse = build_reverse_codebook(df)
    pred_answer = reverse.get(tuple(pred_codes[0].tolist()))
    gold_answer = "C"
    gold_codes = tuple([1, 1])
    assert code_tuple_em(tuple(pred_codes[0].tolist()), gold_codes)
    assert exact_match(pred_answer, gold_answer)
