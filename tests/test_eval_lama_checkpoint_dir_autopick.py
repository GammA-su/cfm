from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402


def test_eval_lama_checkpoint_dir_autopick(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    direct = run_dir / "model_overfit.pt"
    direct.write_text("x")
    resolved = eval_lama._resolve_checkpoint_path(run_dir)
    assert resolved == direct

    parent = tmp_path / "parent"
    parent.mkdir()
    old_run = parent / "old"
    new_run = parent / "new"
    old_run.mkdir()
    new_run.mkdir()
    old_ckpt = old_run / "model_overfit.pt"
    new_ckpt = new_run / "model_overfit.pt"
    old_ckpt.write_text("old")
    time.sleep(0.01)
    new_ckpt.write_text("new")
    resolved = eval_lama._resolve_checkpoint_path(parent)
    assert resolved == new_ckpt

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError) as excinfo:
        eval_lama._resolve_checkpoint_path(empty_dir)
    assert "model_overfit.pt" in str(excinfo.value)
