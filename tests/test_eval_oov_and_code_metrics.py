from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402
from codebook import build_code_matrix, build_reverse_codebook, codes_from_answer, constrained_decode_by_logprobs, decode_codes  # noqa: E402


def test_eval_oov_and_code_metrics() -> None:
    df = pd.DataFrame(
        [
            {"answer": "Q1", "codes": [0, 1]},
            {"answer": "Q2", "codes": [1, 0]},
        ]
    )
    reverse = build_reverse_codebook(df)
    gold_codes = codes_from_answer(df, "Q1")
    assert gold_codes is not None

    pred_codes = [2, 2]
    pred_answer = decode_codes(reverse, pred_codes) or "__OOV__"
    pred_oov_count = 1 if pred_answer == "__OOV__" else 0
    tuple_correct, slot_correct, slot_total = eval_lama._code_accuracy(pred_codes, list(gold_codes))

    assert pred_answer == "__OOV__"
    assert pred_oov_count == 1
    assert tuple_correct == 0
    assert slot_correct == 0
    assert slot_total == len(gold_codes)

    code_mat = build_code_matrix(df)
    logits = np.array([[[0.0, 2.0], [2.0, 1.5]]], dtype=np.float32)
    constrained_codes = constrained_decode_by_logprobs(logits, code_mat, chunk=2)
    constrained_pred_oov_count = 0 if decode_codes(reverse, constrained_codes[0].tolist()) else 1
    assert constrained_pred_oov_count == 0
