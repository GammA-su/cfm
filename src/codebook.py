from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

_QID_RE = re.compile(r"(Q\d+)", re.IGNORECASE)

if TYPE_CHECKING:
    import torch


def normalize_wikidata_qid(s: str | None) -> str | None:
    if not s:
        return None
    match = _QID_RE.search(str(s))
    if not match:
        return None
    return match.group(1).upper()


def normalize_lama_answer(gold: str | None, gold_uri: str | None) -> str:
    qid = normalize_wikidata_qid(gold_uri)
    if qid:
        return qid
    qid = normalize_wikidata_qid(gold)
    if qid:
        return qid
    return str(gold or "").strip()


def load_answer_codebook(path: str | Path) -> Dict[str, List[int]]:
    df = pd.read_parquet(path)
    mapping: Dict[str, List[int]] = {}
    for _, row in df.iterrows():
        answer = str(row.get("answer", "")).strip()
        codes = row.get("codes", [])
        if not answer:
            continue
        mapping[answer] = [int(c) for c in codes]
    return mapping


def build_code_matrix(df: pd.DataFrame, *, codes_col: str = "codes") -> np.ndarray:
    code_vocab_size = code_vocab_size_from_df(df)
    if code_vocab_size <= 0:
        return np.zeros((0, 0), dtype=np.int64)
    rows: List[List[int]] = []
    code_len = None
    for _, row in df.iterrows():
        codes = row.get(codes_col, [])
        try:
            code_list = [int(c) for c in codes]
        except Exception:
            continue
        if code_len is None:
            code_len = len(code_list)
        if len(code_list) != code_len:
            continue
        rows.append(code_list)
    if not rows:
        return np.zeros((0, 0), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def build_inverted_index(
    code_matrix: np.ndarray,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    code_matrix = np.asarray(code_matrix, dtype=np.int64)
    if code_matrix.ndim != 2:
        raise ValueError("code_matrix must be 2D [K,S]")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be > 0")
    num_rows, slots = code_matrix.shape
    offsets = np.zeros((slots, vocab_size + 1), dtype=np.int64)
    flat = np.empty((num_rows * slots,), dtype=np.int32)
    for slot in range(slots):
        codes = code_matrix[:, slot]
        if codes.size == 0:
            continue
        if int(codes.max()) >= vocab_size or int(codes.min()) < 0:
            raise ValueError("code_matrix contains codes outside vocab_size")
        counts = np.bincount(codes, minlength=vocab_size).astype(np.int64)
        base = slot * num_rows
        offsets[slot, 0] = base
        offsets[slot, 1:] = base + np.cumsum(counts)
        order = np.argsort(codes, kind="stable")
        flat[base : base + num_rows] = order.astype(np.int32, copy=False)
    return offsets, flat


def save_inverted_index(path: str | Path, offsets: np.ndarray, flat: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, offsets=offsets, flat=flat)


def load_inverted_index(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path))
    return data["offsets"], data["flat"]


def ensure_inverted_index(
    path: str | Path,
    code_matrix: np.ndarray,
    vocab_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.exists():
        return load_inverted_index(path)
    offsets, flat = build_inverted_index(code_matrix, vocab_size)
    save_inverted_index(path, offsets, flat)
    return offsets, flat


def candidate_rows_from_logits(
    slot_logits: torch.Tensor,
    offsets: np.ndarray,
    flat: np.ndarray,
    topk_per_slot: int,
    max_candidates: int,
    *,
    always_include: torch.LongTensor | None = None,
) -> list[np.ndarray]:
    import torch

    if slot_logits.dim() != 3:
        raise ValueError(f"slot_logits must be [B,S,V], got {tuple(slot_logits.shape)}")
    batch, slots, vocab = slot_logits.shape
    if offsets.shape[0] != slots or offsets.shape[1] != vocab + 1:
        raise ValueError("offsets shape must be [S,V+1] matching logits")
    if topk_per_slot <= 0 or max_candidates <= 0:
        raise ValueError("topk_per_slot and max_candidates must be > 0")
    topk = min(topk_per_slot, vocab)
    topk_ids = torch.topk(slot_logits, k=topk, dim=-1).indices.detach().cpu().numpy()
    flat = np.asarray(flat)
    if flat.size == 0:
        return [np.zeros((0,), dtype=np.int32) for _ in range(batch)]
    code_matrix_size = int(flat.max()) + 1
    results: list[np.ndarray] = []
    for i in range(batch):
        rows: set[int] = set()
        for s in range(slots):
            for tok in topk_ids[i, s]:
                start = int(offsets[s, tok])
                end = int(offsets[s, tok + 1])
                if end > start:
                    rows.update(int(x) for x in flat[start:end])
        if not rows:
            rng = np.random.default_rng(i)
            rows.update(int(x) for x in rng.choice(code_matrix_size, size=min(max_candidates, code_matrix_size), replace=False))
        candidates = np.array(sorted(rows), dtype=np.int32)
        gold = None
        if always_include is not None:
            gold = int(always_include[i].item())
            if gold not in candidates:
                candidates = np.concatenate([np.array([gold], dtype=np.int32), candidates])
            else:
                candidates = np.concatenate(
                    [np.array([gold], dtype=np.int32), candidates[candidates != gold]]
                )
        if candidates.size > max_candidates:
            rng = np.random.default_rng(i)
            if gold is None:
                keep = rng.choice(candidates.size, size=max_candidates, replace=False)
                candidates = np.sort(candidates[keep])
            else:
                rest = candidates[1:]
                if rest.size > max_candidates - 1:
                    pick = rng.choice(rest.size, size=max_candidates - 1, replace=False)
                    rest = rest[pick]
                candidates = np.concatenate([np.array([gold], dtype=np.int32), np.sort(rest)])
        results.append(candidates)
    return results


def constrained_decode_candidates_by_logprobs(
    slot_logits: torch.Tensor,
    code_matrix: np.ndarray,
    cand_rows: list[np.ndarray],
) -> tuple[torch.LongTensor, torch.LongTensor]:
    import torch

    if slot_logits.dim() != 3:
        raise ValueError(f"slot_logits must be [B,S,V], got {tuple(slot_logits.shape)}")
    code_matrix = np.asarray(code_matrix, dtype=np.int64)
    batch, slots, _ = slot_logits.shape
    pred_rows = torch.full((batch,), -1, dtype=torch.long, device=slot_logits.device)
    pred_codes = torch.zeros((batch, slots), dtype=torch.long, device=slot_logits.device)
    logp = torch.log_softmax(slot_logits, dim=-1)
    for i in range(batch):
        rows = cand_rows[i]
        if rows.size == 0:
            pred_rows[i] = 0
            pred_codes[i] = torch.from_numpy(code_matrix[0]).to(device=slot_logits.device, dtype=torch.long)
            continue
        codes = torch.from_numpy(code_matrix[rows]).to(device=slot_logits.device, dtype=torch.long)
        scores = None
        for slot in range(slots):
            slot_logp = logp[i, slot, :]
            gathered = slot_logp.gather(0, codes[:, slot])
            scores = gathered if scores is None else scores + gathered
        best = torch.argmax(scores)
        pred_rows[i] = int(rows[int(best.item())])
        pred_codes[i] = codes[best]
    return pred_rows, pred_codes



def constrained_decode_by_logprobs(
    logits: np.ndarray,
    code_matrix: np.ndarray,
    *,
    chunk: int = 4096,
) -> np.ndarray:
    import torch

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 3:
        raise ValueError(f"logits must be [N,S,V], got {logits.shape}")
    if code_matrix.ndim != 2:
        raise ValueError(f"code_matrix must be [K,S], got {code_matrix.shape}")
    n, s, _ = logits.shape
    if code_matrix.shape[1] != s:
        raise ValueError("code_matrix slot count must match logits")
    if code_matrix.shape[0] == 0:
        return np.zeros((n, s), dtype=np.int64)

    logits_t = torch.from_numpy(logits)
    logp = logits_t - torch.logsumexp(logits_t, dim=-1, keepdim=True)

    best_scores = torch.full((n,), -float("inf"), device=logp.device)
    best_idx = torch.zeros((n,), dtype=torch.long, device=logp.device)
    total = code_matrix.shape[0]
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        chunk_mat = torch.from_numpy(code_matrix[start:end]).to(device=logp.device, dtype=torch.long)
        scores = None
        for slot in range(s):
            slot_logp = logp[:, slot, :]
            idx = chunk_mat[:, slot].unsqueeze(0).expand(n, -1)
            gathered = slot_logp.gather(1, idx)
            scores = gathered if scores is None else scores + gathered
        if scores is None:
            continue
        chunk_best_scores, chunk_best_idx = scores.max(dim=1)
        improved = chunk_best_scores > best_scores
        if improved.any():
            best_scores[improved] = chunk_best_scores[improved]
            best_idx[improved] = chunk_best_idx[improved] + start
    best_idx_cpu = best_idx.cpu().numpy()
    return code_matrix[best_idx_cpu]


def build_reverse_codebook(df: pd.DataFrame) -> Dict[tuple[int, ...], str]:
    mapping: Dict[tuple[int, ...], str] = {}
    for _, row in df.iterrows():
        answer = str(row.get("answer", "")).strip()
        codes = row.get("codes", [])
        if not answer:
            continue
        try:
            key = tuple(int(c) for c in codes)
        except Exception:
            continue
        if key not in mapping:
            mapping[key] = answer
    return mapping


def decode_codes(reverse: Dict[tuple[int, ...], str], codes: Iterable[int] | None) -> str | None:
    if codes is None:
        return None
    try:
        key = tuple(int(c) for c in codes)
    except Exception:
        return None
    return reverse.get(key)


def codes_from_answer(df: pd.DataFrame, answer: str) -> Tuple[int, ...] | None:
    target = str(answer).strip()
    if not target:
        return None
    matches = df[df["answer"] == target]
    if matches.empty:
        return None
    codes = matches.iloc[0].get("codes", [])
    try:
        return tuple(int(c) for c in codes)
    except Exception:
        return None


def build_slot_index(df: pd.DataFrame) -> List[Dict[int, List[str]]]:
    index: List[Dict[int, List[str]]] = []
    for _, row in df.iterrows():
        answer = str(row.get("answer", "")).strip()
        codes = row.get("codes", [])
        if not answer:
            continue
        try:
            code_list = [int(c) for c in codes]
        except Exception:
            continue
        if not index:
            index = [{} for _ in range(len(code_list))]
        if len(code_list) != len(index):
            continue
        for idx, code in enumerate(code_list):
            bucket = index[idx].setdefault(code, [])
            bucket.append(answer)
    return index


def code_vocab_size_from_df(df: pd.DataFrame) -> int:
    max_code = None
    for codes in df.get("codes", []):
        try:
            for code in codes:
                value = int(code)
                if max_code is None or value > max_code:
                    max_code = value
        except Exception:
            continue
    if max_code is None:
        return 0
    return max_code + 1


def _hash_codes(answer: str, code_len: int, code_vocab: int) -> List[int]:
    codes: List[int] = []
    for idx in range(code_len):
        payload = f"{answer}::{idx}".encode("utf-8")
        digest = hashlib.sha1(payload).digest()
        code = int.from_bytes(digest[:4], "little") % code_vocab
        codes.append(int(code))
    return codes


def ensure_answers_in_codebook(
    path: str | Path,
    answers: Iterable[str],
    *,
    code_len: int,
    code_vocab: int,
    method: str = "sha1_mod",
    out_path: str | Path | None = None,
) -> Dict[str, int]:
    if method != "sha1_mod":
        raise ValueError(f"unsupported method: {method}")
    path = Path(path)
    df = pd.read_parquet(path)
    existing = {str(value).strip() for value in df["answer"].tolist() if str(value).strip()}
    answer_list = [str(value).strip() for value in answers if str(value).strip()]
    unique_answers = set(answer_list)
    missing = sorted(unique_answers - existing)
    rows = []
    for answer in missing:
        rows.append({"answer": answer, "codes": _hash_codes(answer, code_len, code_vocab)})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    output_path = Path(out_path) if out_path is not None else path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    updated = {str(value).strip() for value in df["answer"].tolist() if str(value).strip()}
    still_missing = len(unique_answers - updated)
    return {
        "total_answers": len(answer_list),
        "unique_answers": len(unique_answers),
        "added": len(rows),
        "still_missing": still_missing,
    }



__all__ = [
    "normalize_wikidata_qid",
    "normalize_lama_answer",
    "load_answer_codebook",
    "build_code_matrix",
    "build_inverted_index",
    "save_inverted_index",
    "load_inverted_index",
    "ensure_inverted_index",
    "candidate_rows_from_logits",
    "constrained_decode_candidates_by_logprobs",
    "constrained_decode_by_logprobs",
    "build_reverse_codebook",
    "decode_codes",
    "codes_from_answer",
    "build_slot_index",
    "code_vocab_size_from_df",
    "ensure_answers_in_codebook",
]
