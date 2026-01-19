from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import pandas as pd

_WIKIDATA5M_FILES = {
    "transductive": [
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
        "wikidata5m_transductive_test.txt",
    ]
}


def _iter_triples(path: Path) -> Iterable[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) != 3:
                continue
            subject_id, relation_id, object_id = (p.strip() for p in parts)
            if not subject_id or not relation_id or not object_id:
                continue
            yield subject_id, relation_id, object_id, raw


def _stable_hash(seed: int, text: str) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _collect_triples(
    transductive_dir: Path,
    sources: List[str],
    max_triples: int,
    seed: int,
) -> List[dict]:
    triples = []
    heap: list[tuple[int, str]] = []
    for source in sources:
        files = _WIKIDATA5M_FILES.get(source, [])
        for name in files:
            path = transductive_dir / name
            if not path.exists():
                raise FileNotFoundError(f"missing wikidata5m file: {path}")
            for subject_id, relation_id, object_id, raw in _iter_triples(path):
                if max_triples and max_triples > 0:
                    score = _stable_hash(seed, raw)
                    entry = (-score, raw)
                    if len(heap) < max_triples:
                        heap.append(entry)
                        heap.sort(reverse=True)
                    elif entry[0] > heap[0][0]:
                        heap[0] = entry
                        heap.sort(reverse=True)
                else:
                    triples.append((subject_id, relation_id, object_id))
    if max_triples and max_triples > 0:
        selected = [item[1] for item in sorted(heap, reverse=True)]
        for raw in selected:
            subject_id, relation_id, object_id = (p.strip() for p in raw.split("\t"))
            triples.append((subject_id, relation_id, object_id))
    rows = []
    for subject_id, relation_id, object_id in triples:
        rows.append({
            "subject_id": subject_id,
            "subject_label": subject_id,
            "relation_id": relation_id,
            "relation_label": relation_id,
            "object_id_or_value": object_id,
            "object_label": object_id,
        })
    return rows


def ingest_wikidata5m_tar(
    raw_path: Path,
    max_triples: int,
    sources: List[str],
    cache_dir: Path,
    local_files_only: bool,
) -> None:
    transductive_dir = Path("data") / "wikidata5m" / "transductive"
    if not transductive_dir.exists():
        raise FileNotFoundError(
            f"missing wikidata5m directory at {transductive_dir}. "
            "Download and extract wikidata5m_transductive.tar.gz first."
        )
    rows = _collect_triples(transductive_dir, sources, max_triples=max_triples, seed=0)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(raw_path, index=False)


def ingest_wikidata5m(
    raw_path: Path,
    max_triples: int,
    seed: int,
    use_streaming: bool,
    shuffle: bool,
    shuffle_buffer: int,
    local_files_only: bool,
    cache_dir: Path,
) -> None:
    ingest_wikidata5m_tar(
        raw_path=raw_path,
        max_triples=max_triples,
        sources=["transductive"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_triples(raw_path: Path) -> List[dict]:
    if not raw_path.exists():
        raise FileNotFoundError(f"missing wikidata5m parquet at {raw_path}")
    df = pd.read_parquet(raw_path)
    return df.to_dict(orient="records")


__all__ = ["ingest_wikidata5m", "ingest_wikidata5m_tar", "load_triples"]
