from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch


def build_slot0_index(code_matrix: Sequence[Sequence[int]]) -> Dict[int, List[int]]:
    index: Dict[int, List[int]] = {}
    for idx, row in enumerate(code_matrix):
        if not row:
            continue
        slot0 = int(row[0])
        index.setdefault(slot0, []).append(idx)
    return index


def sample_candidates(
    gold_tuple_idx: torch.LongTensor,
    num_candidates: int,
    code_matrix_size: int,
    rng: random.Random,
    *,
    hard_neg_slot0: bool = False,
    slot0_to_rows: Optional[Dict[int, List[int]]] = None,
    slot0_values: Optional[Sequence[int]] = None,
) -> torch.LongTensor:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be > 0")
    if code_matrix_size <= 0:
        raise ValueError("code_matrix_size must be > 0")

    gold_list = [int(x) for x in gold_tuple_idx.tolist()]
    batch_size = len(gold_list)
    candidates = torch.empty((batch_size, num_candidates), dtype=torch.long)

    for row_idx, gold in enumerate(gold_list):
        selected: List[int] = [gold]
        used = {gold}
        remaining = num_candidates - 1
        if remaining <= 0:
            candidates[row_idx] = torch.tensor(selected, dtype=torch.long)
            continue

        hard_needed = 0
        if hard_neg_slot0 and slot0_to_rows is not None and slot0_values is not None:
            hard_needed = remaining // 2
            slot0 = slot0_values[gold] if 0 <= gold < len(slot0_values) else None
            if slot0 is not None:
                bucket = [idx for idx in slot0_to_rows.get(int(slot0), []) if idx not in used]
                if bucket:
                    if len(bucket) <= hard_needed:
                        selected.extend(bucket)
                        used.update(bucket)
                    else:
                        picks = rng.sample(bucket, hard_needed)
                        selected.extend(picks)
                        used.update(picks)

        while len(selected) < num_candidates:
            idx = rng.randrange(code_matrix_size)
            if idx in used:
                continue
            selected.append(idx)
            used.add(idx)

        candidates[row_idx] = torch.tensor(selected, dtype=torch.long)

    return candidates


__all__ = ["build_slot0_index", "sample_candidates"]
