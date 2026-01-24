from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_early_exit_uses_acc() -> None:
    streak, should_exit = train_cfm._should_overfit_early_exit(0, 0.03, 0.99, 3)
    assert streak == 0
    assert not should_exit
