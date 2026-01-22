from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metrics_kbqa import negative_margin_stats  # noqa: E402


def test_metrics_negative_margin() -> None:
    logits = np.array(
        [
            [[0.0, 2.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    gold_codes = np.array([[0, 0]], dtype=np.int64)
    stats = negative_margin_stats(logits, gold_codes)
    assert stats["negative_margin_rate"] == 1.0
