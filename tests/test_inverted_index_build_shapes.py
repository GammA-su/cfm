from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from codebook import build_inverted_index  # noqa: E402


def test_inverted_index_build_shapes() -> None:
    code_matrix = np.array(
        [
            [1, 2],
            [3, 2],
            [1, 4],
            [0, 2],
            [3, 4],
        ],
        dtype=np.int64,
    )
    offsets, flat = build_inverted_index(code_matrix, vocab_size=8)
    assert offsets.shape == (2, 9)
    assert flat.shape[0] == code_matrix.shape[0] * code_matrix.shape[1]

    # postings for slot 0 token 1
    start = offsets[0, 1]
    end = offsets[0, 2]
    rows = set(int(x) for x in flat[start:end])
    expected = {0, 2}
    assert rows == expected

    # postings for slot 1 token 4
    start = offsets[1, 4]
    end = offsets[1, 5]
    rows = set(int(x) for x in flat[start:end])
    expected = {2, 4}
    assert rows == expected
