from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ckpt_io import get_ckpt_step, load_checkpoint  # noqa: E402


def test_load_checkpoint_and_step(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"step": 123, "state": {"epoch": 7}}, ckpt_path)

    ckpt = load_checkpoint(ckpt_path)
    assert get_ckpt_step(ckpt) == 123
