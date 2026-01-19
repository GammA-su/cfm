from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _embed_text(text: str, dim: int) -> np.ndarray:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    seed = int.from_bytes(digest, "big", signed=False)
    rng = np.random.RandomState(seed)
    vec = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def build_answer_embeddings(
    records: Iterable[dict],
    dim: int,
    emb_dir: Path,
) -> Tuple[List[str], np.ndarray]:
    answers: List[str] = []
    seen = set()
    for rec in records:
        label = str(rec.get("object_label", "")).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        answers.append(label)
    embeddings = np.stack([_embed_text(answer, dim) for answer in answers], axis=0)
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "answer_embeddings.npy", embeddings)
    (emb_dir / "answers.txt").write_text("\n".join(answers))
    return answers, embeddings
