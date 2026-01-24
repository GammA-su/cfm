from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402


def test_eval_lama_tokenizer_autopick(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "run"
    ckpt_dir.mkdir()
    tokenizer_path = ckpt_dir / "tokenizer.json"
    tokenizer_path.write_text("{}")
    resolved = eval_lama._resolve_tokenizer_path(ckpt_dir / "model_overfit.pt", None)
    assert resolved == tokenizer_path
