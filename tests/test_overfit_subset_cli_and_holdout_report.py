from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_subset_cli_and_holdout_report() -> None:
    records = [{"object_label": "1941"} for _ in range(8)]
    train, holdout = train_cfm._split_records_holdout(records, 0.25, seed=0)
    assert len(train) == 6
    assert len(holdout) == 2

    report_train = {"n_eval": len(train), "code_em": 0.0, "answer_em": 0.0, "decode_mode_used": "constrained"}
    report_holdout = {"n_eval": len(holdout), "code_em": 0.0, "answer_em": 0.0, "decode_mode_used": "constrained"}
    combined = train_cfm._compose_overfit_split_report(report_train, report_holdout)
    assert combined["split"]["train"]["n_eval"] == 6
    assert combined["split"]["holdout"]["n_eval"] == 2
    for split in ("train", "holdout"):
        payload = combined["split"][split]
        assert "code_em" in payload
        assert "answer_em" in payload
        assert "decode_mode_used" in payload
