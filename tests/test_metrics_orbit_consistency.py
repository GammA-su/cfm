from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metrics_kbqa import orbit_consistency  # noqa: E402


def test_metrics_orbit_consistency() -> None:
    preds = {
        "o1": ["A", "A"],
        "o2": ["A", "B"],
        "o3": ["__OOV__"],
    }
    stats = orbit_consistency(preds, ignore_oov=True)
    assert stats["count"] == 2
    assert stats["rate"] == 0.5
