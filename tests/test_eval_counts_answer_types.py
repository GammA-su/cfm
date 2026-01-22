from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.eval_lama as eval_lama  # noqa: E402


def test_eval_counts_answer_types() -> None:
    gold_answers = ["Q23115", "1941"]
    counts = eval_lama._answer_type_counts(gold_answers)
    assert counts["qid"] == 1
    assert counts["literal"] == 1

    acc_qid, acc_uri, acc_text = eval_lama._score_answer("Q23115", "Q23115")
    assert acc_qid == 1.0
    assert acc_uri == 1.0
    assert acc_text is None

    acc_lit, acc_uri_lit, acc_text_lit = eval_lama._score_answer("1942", "1941")
    assert acc_lit == 0.0
    assert acc_uri_lit is None
    assert acc_text_lit == 0.0
