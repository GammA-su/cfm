from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codebook import normalize_lama_answer  # noqa: E402


def test_normalize_lama_answer_qid_uri() -> None:
    assert normalize_lama_answer("ignored", "http://www.wikidata.org/entity/Q23115") == "Q23115"


def test_normalize_lama_answer_literal() -> None:
    assert normalize_lama_answer("1941", "") == "1941"
