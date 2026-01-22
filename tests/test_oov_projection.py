from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402
from codebook import build_slot_index  # noqa: E402


def test_oov_projection_selects_nearest() -> None:
    df = pd.DataFrame(
        [
            {"answer": "A", "codes": [0, 0]},
            {"answer": "B", "codes": [0, 1]},
            {"answer": "C", "codes": [1, 1]},
        ]
    )
    slot_index = build_slot_index(df)
    answer_codes = {row["answer"]: row["codes"] for _, row in df.iterrows()}
    projected = eval_lama._project_oov([1, 2], slot_index, answer_codes, slot_conf=[0.9, 0.1])
    assert projected == "C"
