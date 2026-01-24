from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codebook import build_reverse_codebook, constrained_decode_by_logprobs, year_row_indices  # noqa: E402


def test_year_filter_blocks_qid_choice() -> None:
    df = pd.DataFrame(
        {
            "answer": ["1941", "Q123"],
            "codes": [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        }
    )
    code_matrix = np.asarray(df["codes"].tolist(), dtype=np.int64)
    year_rows = year_row_indices(df)
    year_code_matrix = code_matrix[year_rows]
    logits = torch.zeros((1, 6, 2), dtype=torch.float32)
    logits[:, :, 1] = 5.0
    constrained = constrained_decode_by_logprobs(logits.numpy(), year_code_matrix)
    reverse = build_reverse_codebook(df)
    pred = reverse.get(tuple(constrained[0].tolist()))
    assert pred == "1941"
