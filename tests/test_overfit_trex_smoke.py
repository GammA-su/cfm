from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_cfm as train_cfm  # noqa: E402

pytestmark = pytest.mark.slow

if not os.environ.get("RUN_SLOW"):
    pytest.skip("slow test; set RUN_SLOW=1 to enable", allow_module_level=True)


def test_overfit_trex_loader_smoke(tmp_path: Path) -> None:
    records = train_cfm._load_trex_overfit_records(
        limit=8,
        cache_dir=tmp_path / ".cache",
        local_files_only=False,
    )
    assert records
