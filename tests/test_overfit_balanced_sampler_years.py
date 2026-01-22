from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_cfm as train_cfm  # noqa: E402


def test_overfit_balanced_sampler_years() -> None:
    records = [
        {"object_label": "1941"} for _ in range(5)
    ] + [
        {"object_label": "1982"} for _ in range(2)
    ] + [
        {"object_label": "1977"} for _ in range(1)
    ]
    sampled = train_cfm._balanced_sample_records_by_year(records, n=6, seed=0)
    counts = {}
    for rec in sampled:
        year = rec["object_label"]
        counts[year] = counts.get(year, 0) + 1
    # should be roughly balanced across available years
    assert set(counts.keys()) == {"1941", "1982", "1977"}
    assert max(counts.values()) - min(counts.values()) <= 2
