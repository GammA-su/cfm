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


def test_eval_tokenizer_flag_help() -> None:
    runner = CliRunner()
    result = runner.invoke(eval_lama.app, ["--help"])
    assert result.exit_code == 0
    assert "--tokenizer" in result.output
