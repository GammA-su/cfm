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


def test_eval_lama_checkpoint_dir_guard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError):
        eval_lama._resolve_checkpoint_path(run_dir)

    (run_dir / "model_overfit.pt").write_text("x")
    resolved = eval_lama._resolve_checkpoint_path(run_dir)
    assert resolved == run_dir / "model_overfit.pt"
