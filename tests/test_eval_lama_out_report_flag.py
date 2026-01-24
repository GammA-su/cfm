from __future__ import annotations

from pathlib import Path
import sys

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.eval_lama as eval_lama  # noqa: E402


def test_eval_lama_out_report_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    out_path = tmp_path / "lama_report.json"
    result = runner.invoke(eval_lama.app, ["--help"])
    assert result.exit_code == 0
    assert "--out-report" in result.output
    assert not out_path.exists()
