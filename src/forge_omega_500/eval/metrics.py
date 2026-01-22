from __future__ import annotations

from typing import Dict, List

import numpy as np


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def calibration_curve(confidences: List[float], accuracies: List[float], bins: int) -> Dict[str, List[float]]:
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    bin_acc = []
    bin_conf = []
    bin_count = []

    for i in range(bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences < high)
        if i == bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        if mask.any():
            bin_acc.append(float(accuracies[mask].mean()))
            bin_conf.append(float(confidences[mask].mean()))
            bin_count.append(int(mask.sum()))
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
            bin_count.append(0)

    return {"bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}


__all__ = ["exact_match", "calibration_curve"]
