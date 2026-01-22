from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402


def test_eval_code_space_sanity_raises() -> None:
    with pytest.raises(ValueError, match="out of codebook range"):
        eval_lama._validate_code_space([0, 16], code_vocab_size=16)
