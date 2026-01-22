from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_lama_years_filter_keeps_only_years() -> None:
    answers = ["1941", "foo", "1982", "19a2"]
    filtered = train_cfm._filter_year_answers(answers)
    assert filtered == ["1941", "1982"]
