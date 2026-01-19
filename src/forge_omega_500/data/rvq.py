from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file


def train_rvq(
    embeddings: np.ndarray,
    m: int,
    K: int,
    iters: int,
    seed: int,
    faiss_mod=None,
    use_gpu: bool = False,
    require_gpu: bool = False,
    gpu_search_batch: int | None = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be 2D")
    n, dim = emb.shape
    codebooks = np.zeros((m, K, dim), dtype=np.float32)
    for i in range(m):
        if n >= K:
            idx = rng.choice(n, size=K, replace=False)
            codebooks[i] = emb[idx]
        elif n > 0:
            codebooks[i, :n] = emb
            codebooks[i, n:] = rng.normal(0.0, 1.0, size=(K - n, dim)).astype(np.float32)
        else:
            codebooks[i] = rng.normal(0.0, 1.0, size=(K, dim)).astype(np.float32)
    return codebooks


def assign_codes(
    embeddings: np.ndarray,
    codebooks: np.ndarray,
    faiss_mod=None,
    use_gpu: bool = False,
    require_gpu: bool = False,
    gpu_search_batch: int | None = None,
) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    books = np.asarray(codebooks, dtype=np.float32)
    if emb.ndim != 2 or books.ndim != 3:
        raise ValueError("embeddings must be 2D and codebooks must be 3D")
    n, dim = emb.shape
    m, K, code_dim = books.shape
    if dim != code_dim:
        raise ValueError("embedding dim mismatch")
    codes = np.zeros((n, m), dtype=np.int64)
    batch = 8192
    for i in range(m):
        centroids = books[i]
        for start in range(0, n, batch):
            end = min(start + batch, n)
            chunk = emb[start:end]
            dists = ((chunk[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
            codes[start:end, i] = dists.argmin(axis=1)
    return codes


def save_codebooks(codebooks: np.ndarray, path: Path) -> None:
    tensor = torch.tensor(codebooks, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"codebooks": tensor}, str(path))


def save_answer_codes(answers: List[str], codes: np.ndarray, path: Path) -> None:
    rows = []
    for idx, answer in enumerate(answers):
        rows.append({"answer": answer, "codes": [int(c) for c in codes[idx].tolist()]})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_answer_codes(path: Path) -> Dict[str, List[int]]:
    df = pd.read_parquet(path)
    mapping: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        answer = str(row.get("answer", "")).strip()
        codes = row.get("codes", [])
        if answer:
            mapping[answer] = [int(c) for c in codes]
    return mapping


def save_code_to_label(
    records: Iterable[dict],
    answers: List[str],
    codes: np.ndarray,
    path: Path,
) -> None:
    rows = []
    for idx, answer in enumerate(answers):
        rows.append({"codes": [int(c) for c in codes[idx].tolist()], "label": answer})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_code_to_label(path: Path) -> Dict[Tuple[int, ...], str]:
    df = pd.read_parquet(path)
    mapping: Dict[Tuple[int, ...], str] = {}
    for _, row in df.iterrows():
        codes = row.get("codes", [])
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        try:
            key = tuple(int(c) for c in codes)
        except Exception:
            continue
        if key not in mapping:
            mapping[key] = label
    return mapping


def lookup_code_label(mapping: Dict[Tuple[int, ...], str], codes: List[int]) -> Tuple[str, bool]:
    key = tuple(int(c) for c in codes)
    label = mapping.get(key, "")
    return label, bool(label)


__all__ = [
    "assign_codes",
    "load_answer_codes",
    "load_code_to_label",
    "lookup_code_label",
    "save_answer_codes",
    "save_code_to_label",
    "save_codebooks",
    "train_rvq",
]
