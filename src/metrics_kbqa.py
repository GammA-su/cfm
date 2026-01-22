from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split()).lower()


def exact_match(a: str | None, b: str | None) -> bool:
    return _normalize_text(a) == _normalize_text(b)


def code_tuple_em(pred_codes: Tuple[int, ...], gold_codes: Tuple[int, ...]) -> bool:
    return tuple(pred_codes) == tuple(gold_codes)


def slot_em(pred_codes: Tuple[int, ...], gold_codes: Tuple[int, ...]) -> float:
    if not gold_codes:
        return 0.0
    correct = 0
    for idx in range(len(gold_codes)):
        pred_val = pred_codes[idx] if idx < len(pred_codes) else None
        if pred_val == gold_codes[idx]:
            correct += 1
    return correct / len(gold_codes)


def orbit_consistency(
    pred_by_orbit: Dict[str, List[str]],
    *,
    ignore_oov: bool = True,
    oov_token: str = "__OOV__",
) -> Dict[str, float | int | None]:
    consistent = 0
    counted = 0
    for preds in pred_by_orbit.values():
        values = list(preds)
        if ignore_oov:
            values = [v for v in values if v != oov_token]
        if not values:
            continue
        counted += 1
        if len(set(values)) == 1:
            consistent += 1
    rate = consistent / counted if counted > 0 else None
    return {"count": counted, "rate": rate}


def negative_margin_stats(logits: np.ndarray, gold_codes: np.ndarray) -> Dict[str, float]:
    if logits.size == 0 or gold_codes.size == 0:
        return {
            "negative_margin_rate": 0.0,
            "margin_min_mean": 0.0,
            "margin_min_p50": 0.0,
            "margin_min_p05": 0.0,
        }
    logits = np.asarray(logits, dtype=np.float32)
    gold_codes = np.asarray(gold_codes, dtype=np.int64)
    if logits.ndim != 3:
        raise ValueError(f"logits must be [N,S,V], got {logits.shape}")
    if gold_codes.ndim != 2:
        raise ValueError(f"gold_codes must be [N,S], got {gold_codes.shape}")

    n, s, v = logits.shape
    if gold_codes.shape[0] != n or gold_codes.shape[1] != s:
        raise ValueError("gold_codes shape must match logits [N,S]")

    gold_idx = gold_codes[..., None]
    gold_logits = np.take_along_axis(logits, gold_idx, axis=2)[..., 0]

    masked = logits.copy()
    row = np.arange(n)[:, None]
    col = np.arange(s)[None, :]
    masked[row, col, gold_codes] = -np.inf
    max_other = masked.max(axis=2)

    margins = gold_logits - max_other
    min_margins = margins.min(axis=1)

    negative_margin_rate = float(np.mean(min_margins < 0))
    margin_min_mean = float(np.mean(min_margins))
    margin_min_p50 = float(np.percentile(min_margins, 50))
    margin_min_p05 = float(np.percentile(min_margins, 5))

    return {
        "negative_margin_rate": negative_margin_rate,
        "margin_min_mean": margin_min_mean,
        "margin_min_p50": margin_min_p50,
        "margin_min_p05": margin_min_p05,
    }


__all__ = [
    "exact_match",
    "code_tuple_em",
    "slot_em",
    "orbit_consistency",
    "negative_margin_stats",
]
