from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_gen_targets_nonempty() -> None:
    records = [
        {
            "fact_id": 1,
            "question_orbits": ["Fill in the blank: A relation ____."],
            "object_label": "Q1",
            "hard_negatives": [],
        }
    ]
    answer_codes = {"Q1": [0]}
    examples = train_cfm._build_examples(records, answer_codes)
    with_targets, total, coverage = train_cfm._gen_target_coverage(examples)
    assert total == 1
    assert with_targets == 1
    assert coverage > 0.0
