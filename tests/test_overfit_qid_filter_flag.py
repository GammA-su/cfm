from __future__ import annotations

from pathlib import Path
import sys

import inspect

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_qid_filter_flag_present() -> None:
    params = inspect.signature(train_cfm.main).parameters
    assert "overfit_lama_qids" in params
