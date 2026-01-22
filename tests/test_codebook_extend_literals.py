from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codebook import ensure_answers_in_codebook, load_answer_codebook  # noqa: E402


def test_codebook_extend_literals(tmp_path: Path) -> None:
    path = tmp_path / "answer_codes.parquet"
    df = pd.DataFrame(
        [
            {"answer": "Q1", "codes": [0, 1]},
            {"answer": "Q2", "codes": [1, 0]},
        ]
    )
    df.to_parquet(path, index=False)

    ensure_answers_in_codebook(path, ["1941", "1982"], code_len=2, code_vocab=8)
    mapping = load_answer_codebook(path)

    assert "1941" in mapping
    assert "1982" in mapping
