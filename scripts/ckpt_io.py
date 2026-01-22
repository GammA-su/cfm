from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_STEP_KEYS = ("step", "global_step", "train_step")
_CONTAINER_KEYS = ("trainer_state", "state", "meta", "trainer", "extra_state", "checkpoint", "metadata")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)


def _coerce_step(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _search_step(mapping: Mapping[str, object], depth: int) -> int | None:
    for key in _STEP_KEYS:
        step = _coerce_step(mapping.get(key))
        if step is not None:
            return step
    if depth <= 0:
        return None
    for key in _CONTAINER_KEYS:
        sub = mapping.get(key)
        if isinstance(sub, Mapping):
            step = _search_step(sub, depth - 1)
            if step is not None:
                return step
    for value in mapping.values():
        if isinstance(value, Mapping):
            step = _search_step(value, depth - 1)
            if step is not None:
                return step
    return None


def get_ckpt_step(ckpt: object) -> int | None:
    if not isinstance(ckpt, Mapping):
        return None
    return _search_step(ckpt, depth=3)


__all__ = ["load_checkpoint", "get_ckpt_step"]
