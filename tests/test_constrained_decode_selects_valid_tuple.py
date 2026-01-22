from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codebook import build_code_matrix, constrained_decode_by_logprobs  # noqa: E402


def test_constrained_decode_selects_valid_tuple() -> None:
    df = __import__("pandas").DataFrame(
        [
            {"answer": "A", "codes": [0, 0]},
            {"answer": "B", "codes": [0, 1]},
            {"answer": "C", "codes": [1, 1]},
        ]
    )
    code_mat = build_code_matrix(df)
    logits = np.array([[[0.0, 2.0], [2.0, 1.5]]], dtype=np.float32)
    pred_codes = constrained_decode_by_logprobs(logits, code_mat, chunk=2)
    assert np.array_equal(pred_codes[0], np.array([1, 1], dtype=np.int64))
