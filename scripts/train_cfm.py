from __future__ import annotations

import hashlib
import json
import shutil
import logging
import math
import random
import sys
import tarfile
import uuid
from fnmatch import fnmatch
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from forge_omega_500.runtime import (
    DEFAULT_CPU_THREADS,
    configure_env,
    configure_torch,
    log_runtime,
    resolve_faiss,
    setup_logger,
)

_STARTUP_T0 = time.perf_counter()
_STARTUP_STAGE = "boot"
_STARTUP_LOCK = threading.Lock()


def _startup_log(message: str) -> None:
    elapsed = time.perf_counter() - _STARTUP_T0
    print(f"[startup +{elapsed:.2f}s] {message}", file=sys.stderr, flush=True)


def _set_startup_stage(message: str, log: bool = True) -> None:
    global _STARTUP_STAGE
    with _STARTUP_LOCK:
        _STARTUP_STAGE = message
    if log:
        _startup_log(message)


def _startup_watchdog(interval: float = 30.0) -> None:
    while True:
        time.sleep(interval)
        with _STARTUP_LOCK:
            stage = _STARTUP_STAGE
        if stage == "ready":
            return
        elapsed = time.perf_counter() - _STARTUP_T0
        print(f"[startup +{elapsed:.2f}s] still {stage}", file=sys.stderr, flush=True)


threading.Thread(target=_startup_watchdog, daemon=True).start()

_set_startup_stage("configure_env")
configure_env(DEFAULT_CPU_THREADS)

_set_startup_stage("import numpy")
import numpy as np
_set_startup_stage("import torch")
import torch
_set_startup_stage("import torch.nn.functional")
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
_set_startup_stage("import torch.cuda.amp")
from torch.cuda.amp import autocast, GradScaler
_set_startup_stage("import typer")
import typer
_set_startup_stage("import yaml")
import yaml
from forge_omega_500.data.rvq import load_answer_codes
from forge_omega_500.model.cfm import CFMModel, contrastive_margin_loss
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed
from scripts.ckpt_io import get_ckpt_step, load_checkpoint
from codebook import codes_from_answer, normalize_lama_answer
from loss_kbqa import candidate_tuple_logprobs, tuple_ce_loss, tuple_ce_loss_candidates
from metrics_kbqa import exact_match, negative_margin_stats
from codebook import (
    candidate_rows_from_logits,
    constrained_decode_by_logprobs,
    ensure_inverted_index,
)

logger = setup_logger("train_cfm")

app = typer.Typer(add_completion=False)
_FILE_LOG_READY = False

_ORBIT_ZERO_WARNING_THRESHOLD = 12
_ORBIT_ZERO_WARNING_SHOWN = False
_ORBIT_ZERO_ZERO_STREAK = 0
_FIRST_NONFINITE_COMPONENT_LOGGED = False
_COLLAPSE_TOP1_THRESHOLD = 0.5
_COLLAPSE_UNIQUE_THRESHOLD = 0.05
_COLLAPSE_ENTROPY_THRESHOLD = 0.2
_CONSOLE = None


def _get_console():
    global _CONSOLE
    if _CONSOLE is not None:
        return _CONSOLE
    try:
        _set_startup_stage("import rich.console", log=False)
        from rich.console import Console  # type: ignore

        _CONSOLE = Console()
        _startup_log("import rich.console done")
    except Exception as exc:
        logger.warning("rich_console_unavailable err=%s", exc)

        class _FallbackConsole:
            def print(self, message: str) -> None:
                print(message)

        _CONSOLE = _FallbackConsole()
    return _CONSOLE


def _ensure_file_logger(out_dir: Path) -> None:
    global _FILE_LOG_READY
    if _FILE_LOG_READY:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_cfm.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("file_logging_enabled path=%s", log_path)
    _FILE_LOG_READY = True


def _atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def _save_checkpoint(
    ckpt_dir: Path,
    step: int,
    model: CFMModel,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    epoch_idx: int,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "step": int(step),
        "epoch_idx": int(epoch_idx),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": rng.getstate(),
        "np_state": np.random.get_state(),
        "torch_state": torch.random.get_rng_state(),
        "torch_cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    step_path = ckpt_dir / f"step_{step:06d}.pt"
    _atomic_torch_save(payload, step_path)
    latest_path = ckpt_dir / "latest.pt"
    _atomic_torch_save(payload, latest_path)
    logger.info("checkpoint_saved step=%s path=%s", step, step_path)
    return latest_path


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    return None


def _load_checkpoint(path: Path) -> Dict[str, object]:
    payload = load_checkpoint(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint is not a dict: {type(payload)!r}")
    return payload


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    stats: Dict[str, object] = {}
    if tensor.numel() == 0:
        stats.update(min=None, max=None, mean=None, has_nan=False, has_inf=False)
        return stats
    data = tensor.detach()
    has_nan = bool(torch.isnan(data).any())
    has_inf = bool(torch.isinf(data).any())
    safe = torch.nan_to_num(data.float(), nan=0.0, posinf=1e6, neginf=-1e6)
    stats["min"] = float(safe.min().item())
    stats["max"] = float(safe.max().item())
    stats["mean"] = float(safe.mean().item())
    stats["has_nan"] = has_nan
    stats["has_inf"] = has_inf
    return stats


def _save_nonfinite_debug(
    step: int,
    batch: List[Dict[str, object]],
    code_targets: torch.Tensor,
    orbit_pairs: int,
    addr_logits: List[torch.Tensor],
    gen_logits: torch.Tensor,
    nonfinite_components: List[str],
    component_values: Dict[str, float],
) -> None:
    debug_dir = Path("out") / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    limit = min(len(batch), 4)
    snapshot = {
        "step": step,
        "orbit_pairs": orbit_pairs,
        "nonfinite_components": nonfinite_components,
        "component_values": component_values,
        "prompts": [str(batch[i]["prompt"]) for i in range(limit)],
        "codes": [code_targets[i].cpu().tolist() for i in range(limit)],
        "logits": {
            "addr": [_tensor_stats(logits) for logits in addr_logits],
            "gen": _tensor_stats(gen_logits),
        },
        "timestamp": time.time(),
    }
    path = debug_dir / f"nonfinite_step_{step:06d}.json"
    path.write_text(json.dumps(snapshot, indent=2))


def _compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        norm = param.grad.detach().float().norm(2)
        total += float(norm.item() ** 2)
    return math.sqrt(total) if total > 0 else 0.0


def _is_qid(value: str) -> bool:
    text = value.strip()
    return text.startswith("Q") and text[1:].isdigit()


def _gen_target_coverage(examples: List[Dict[str, object]]) -> Tuple[int, int, float]:
    total = len(examples)
    with_targets = sum(1 for ex in examples if str(ex.get("answer", "")).strip())
    coverage = with_targets / total if total else 0.0
    return with_targets, total, coverage


def _sample_unique_answer_stats(examples: List[Dict[str, object]], sample_size: int = 200) -> Tuple[int, int]:
    sample = examples[:max(0, sample_size)]
    answers = {str(ex.get("answer", "")).strip() for ex in sample if str(ex.get("answer", "")).strip()}
    qids = {value for value in answers if _is_qid(value)}
    return len(answers), len(qids)


def _hash_codes(answer: str, m: int, k: int) -> List[int]:
    codes: List[int] = []
    for idx in range(m):
        payload = f"{answer}::{idx}".encode("utf-8")
        digest = hashlib.md5(payload).digest()
        code = int.from_bytes(digest[:4], "little") % k
        codes.append(int(code))
    return codes


def _build_hashed_answer_codes(answers: List[str], m: int, k: int) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for answer in answers:
        text = str(answer).strip()
        if not text:
            continue
        if text in mapping:
            continue
        mapping[text] = _hash_codes(text, m=m, k=k)
    return mapping


_MASK_TOKENS = ("[MASK]", "<mask>", "MASK")
LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
_GOOGLE_RE_FILES = (
    "Google_RE/date_of_birth_test.jsonl",
    "Google_RE/place_of_birth_test.jsonl",
    "Google_RE/place_of_death_test.jsonl",
)


def _replace_mask_tokens(text: str) -> str:
    value = text
    for token in _MASK_TOKENS:
        value = value.replace(token, "____")
    return value


def _is_year_literal(value: str) -> bool:
    text = str(value).strip()
    return len(text) == 4 and text.isdigit()


def _filter_year_answers(answers: Iterable[str]) -> List[str]:
    return [str(value).strip() for value in answers if _is_year_literal(str(value))]

def _balanced_sample_records_by_year(
    records: List[Dict[str, object]],
    n: int,
    seed: int,
) -> List[Dict[str, object]]:
    if n <= 0:
        return records
    groups: Dict[str, List[Dict[str, object]]] = {}
    for rec in records:
        year = str(rec.get("object_label", "")).strip()
        if not year:
            continue
        groups.setdefault(year, []).append(rec)
    rng = random.Random(seed)
    for items in groups.values():
        rng.shuffle(items)
    selected: List[Dict[str, object]] = []
    active = list(groups.keys())
    while len(selected) < n and active:
        next_active = []
        for year in active:
            bucket = groups.get(year, [])
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_active.append(year)
            if len(selected) >= n:
                break
        active = next_active
    return selected


def _split_records_holdout(
    records: List[Dict[str, object]],
    holdout_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if holdout_frac <= 0.0 or len(records) <= 1:
        return records, []
    n_holdout = int(len(records) * holdout_frac)
    n_holdout = max(1, min(n_holdout, len(records) - 1))
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    holdout = shuffled[:n_holdout]
    train = shuffled[n_holdout:]
    return train, holdout


def _compose_overfit_split_report(report_train: dict, report_holdout: dict) -> dict:
    combined = dict(report_holdout)
    combined["split"] = {"train": report_train, "holdout": report_holdout}
    return combined


def _should_overfit_early_exit(
    acc_streak: int,
    acc: float,
    threshold: float,
    patience: int,
) -> Tuple[int, bool]:
    if acc >= threshold:
        acc_streak += 1
    else:
        acc_streak = 0
    return acc_streak, acc_streak >= patience


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_overfit_tokenizer(tokenizer_path: Path, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "tokenizer.json"
    shutil.copyfile(tokenizer_path, target)
    tokenizer = SimpleTokenizer.load(target)
    return {
        "tokenizer_path_saved": str(target.resolve()),
        "tokenizer_vocab_size": len(tokenizer.vocab),
        "tokenizer_sha256": _sha256_file(target),
    }


def _run_overfit_report_for_examples(
    model: CFMModel,
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    tuple_code_matrix: torch.Tensor,
    tuple_answers: List[str],
    run_id: str,
    source: str,
    answer_filter_used: str,
    gold_year_rate: float,
    gold_qid_rate: float,
    answer_filter_row_count: int,
) -> dict:
    if not examples:
        return {
            "code_em": 0.0,
            "answer_em": 0.0,
            "orbit_consistency": {"count": 0, "rate": None},
            "negative_margin": {"negative_margin_rate": 0.0, "margin_min_mean": 0.0, "margin_min_p50": 0.0, "margin_min_p05": 0.0},
            "answer_filter_used": answer_filter_used,
            "gold_year_rate": gold_year_rate,
            "gold_qid_rate": gold_qid_rate,
            "answer_filter_row_count": answer_filter_row_count,
            "em_breakdown": {
                "answer_em_constrained": 0.0,
                "answer_em_argmax": 0.0,
                "code_em_constrained": 0.0,
                "code_em_argmax": 0.0,
            },
            "decode_mode_used": "constrained",
            "pred_hist_topk": [],
            "gold_hist_topk": [],
            "n_eval": 0,
            "candidate_size": int(tuple_code_matrix.size(0)),
            "run_id": run_id,
            "source": source,
        }
    device = next(model.parameters()).device
    prompts = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in examples]
    prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
    prompt_ids = prompt_ids.to(device)
    prompt_masks = prompt_masks.to(device)
    with torch.no_grad():
        addr_logits, _, _ = model.encode_prompt(prompt_ids, prompt_masks)
    slot_logits = torch.stack([log.float() for log in addr_logits], dim=1)
    gold_tuple_idx = torch.tensor([int(ex.get("tuple_idx", -1)) for ex in examples], device=device)
    gold_answers = [str(ex["answer"]) for ex in examples]
    return _compute_overfit_report(
        slot_logits=slot_logits,
        code_matrix=tuple_code_matrix.to(device),
        gold_tuple_idx=gold_tuple_idx,
        tuple_idx_to_answer=tuple_answers,
        gold_answers=gold_answers,
        run_id=run_id,
        source=source,
        answer_filter_used=answer_filter_used,
        gold_year_rate=gold_year_rate,
        gold_qid_rate=gold_qid_rate,
        answer_filter_row_count=answer_filter_row_count,
    )



def _download_lama_archive(cache_dir: Path, local_files_only: bool) -> Path:
    from datasets import DownloadConfig
    from datasets.download.download_manager import DownloadManager

    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    return Path(dl_manager.download(LAMA_DATA_URL))


def _iter_trex_records(archive_path: Path, max_samples: int) -> Iterable[Dict[str, object]]:
    yielded = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not fnmatch(member.name, "TREx/*"):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            for raw in fileobj:
                data = json.loads(raw.decode("utf-8"))
                obj_label = _first_value(
                    data.get("obj_label") or data.get("object_label") or data.get("obj") or data.get("answer")
                )
                obj_uri = _first_value(data.get("obj_uri") or data.get("object_uri"))
                for evidence in data.get("evidences", []):
                    masked_sentence = _first_value(evidence.get("masked_sentence"))
                    if masked_sentence:
                        yield {
                            "masked_sentence": masked_sentence,
                            "obj_label": obj_label,
                            "obj_uri": obj_uri,
                            "predicate_id": _first_value(data.get("predicate_id") or data.get("relation_id")),
                            "sub_label": _first_value(data.get("sub_label") or data.get("subject_label")),
                        }
                        yielded += 1
                        if max_samples and yielded >= max_samples:
                            return


def _iter_google_re_records(archive_path: Path, max_samples: int) -> Iterable[Dict[str, object]]:
    yielded = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if member.name not in _GOOGLE_RE_FILES:
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            for raw in fileobj:
                data = json.loads(raw.decode("utf-8"))
                yield data
                yielded += 1
                if max_samples and yielded >= max_samples:
                    return


def _first_value(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def _load_trex_overfit_records(
    limit: int,
    cache_dir: Path,
    local_files_only: bool,
) -> List[Dict[str, object]]:
    from datasets import DownloadConfig, load_dataset

    records: List[Dict[str, object]] = []
    try:
        dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
        dataset = load_dataset("facebook/lama", "trex", split="train", download_config=dl_config)
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        iterable: Iterable[Dict[str, object]] = dataset
    except Exception:
        archive_path = _download_lama_archive(cache_dir, local_files_only=local_files_only)
        iterable = _iter_trex_records(archive_path, max_samples=limit)

    for idx, example in enumerate(iterable):
        masked_sentence = _first_value(example.get("masked_sentence") or example.get("masked_sentences"))
        if not masked_sentence:
            continue
        obj_label = _first_value(
            example.get("obj_label") or example.get("object_label") or example.get("obj") or example.get("answer")
        )
        obj_uri = _first_value(example.get("obj_uri") or example.get("object_uri"))
        normalized_answer = normalize_lama_answer(obj_label, obj_uri)
        if not normalized_answer:
            continue
        predicate_id = _first_value(example.get("predicate_id") or example.get("relation_id") or example.get("relation"))
        prompt = "Fill in the blank: " + _replace_mask_tokens(masked_sentence)
        records.append({
            "fact_id": idx,
            "question_orbits": [prompt],
            "object_label": normalized_answer,
            "relation_id": predicate_id or "unknown",
            "relation_label": predicate_id or "",
            "subject_label": _first_value(example.get("sub_label") or example.get("subject_label")),
            "hard_negatives": [],
        })
    return records


def _load_google_re_overfit_records(
    limit: int,
    cache_dir: Path,
    local_files_only: bool,
) -> List[Dict[str, object]]:
    from datasets import DownloadConfig, load_dataset

    records: List[Dict[str, object]] = []
    try:
        dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
        dataset = load_dataset("facebook/lama", "google_re", split="train", download_config=dl_config)
        iterable: Iterable[Dict[str, object]] = dataset
    except Exception:
        archive_path = _download_lama_archive(cache_dir, local_files_only=local_files_only)
        iterable = _iter_google_re_records(archive_path, max_samples=0)

    records.extend(_filter_lama_year_examples(iterable, limit=limit))
    return records


def _filter_lama_year_examples(
    iterable: Iterable[Dict[str, object]],
    limit: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for idx, example in enumerate(iterable):
        masked_sentence = _first_value(example.get("masked_sentence") or example.get("masked_sentences"))
        if not masked_sentence:
            continue
        obj_label = _first_value(
            example.get("obj_label") or example.get("object_label") or example.get("obj") or example.get("answer")
        )
        obj_uri = _first_value(example.get("obj_uri") or example.get("object_uri"))
        normalized_answer = normalize_lama_answer(obj_label, obj_uri)
        if not normalized_answer or not _is_year_literal(normalized_answer):
            continue
        predicate_id = _first_value(example.get("predicate_id") or example.get("relation_id") or example.get("relation"))
        prompt = "Fill in the blank: " + _replace_mask_tokens(masked_sentence)
        records.append({
            "fact_id": idx,
            "question_orbits": [prompt],
            "object_label": normalized_answer,
            "relation_id": predicate_id or "google_re",
            "relation_label": predicate_id or "",
            "subject_label": _first_value(example.get("sub_label") or example.get("subject_label")),
            "hard_negatives": [],
        })
        if limit and len(records) >= limit:
            break
    return records


def _filter_lama_examples(
    iterable: Iterable[Dict[str, object]],
    limit: int,
    *,
    year_only: bool,
    qid_only: bool = False,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for idx, example in enumerate(iterable):
        masked_sentence = _first_value(example.get("masked_sentence") or example.get("masked_sentences"))
        if not masked_sentence:
            continue
        obj_label = _first_value(
            example.get("obj_label") or example.get("object_label") or example.get("obj") or example.get("answer")
        )
        obj_uri = _first_value(example.get("obj_uri") or example.get("object_uri"))
        normalized_answer = normalize_lama_answer(obj_label, obj_uri)
        if not normalized_answer:
            continue
        if year_only and not _is_year_literal(normalized_answer):
            continue
        if qid_only and not _is_qid(normalized_answer):
            continue
        predicate_id = _first_value(example.get("predicate_id") or example.get("relation_id") or example.get("relation"))
        prompt = "Fill in the blank: " + _replace_mask_tokens(masked_sentence)
        records.append({
            "fact_id": idx,
            "question_orbits": [prompt],
            "object_label": normalized_answer,
            "relation_id": predicate_id or "unknown",
            "relation_label": predicate_id or "",
            "subject_label": _first_value(example.get("sub_label") or example.get("subject_label")),
            "hard_negatives": [],
        })
        if limit and len(records) >= limit:
            break
    return records


def _iter_lama_subset_records(
    archive_path: Path,
    subset: str,
    max_samples: int,
) -> Iterable[Dict[str, object]]:
    yielded = 0
    subset_lower = subset.lower()
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if not member.name.lower().startswith(subset_lower):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            for raw in fileobj:
                data = json.loads(raw.decode("utf-8"))
                yield data
                yielded += 1
                if max_samples and yielded >= max_samples:
                    return


def _load_lama_overfit_records(
    subset: str,
    limit: int,
    cache_dir: Path,
    local_files_only: bool,
    *,
    year_only: bool,
    qid_only: bool = False,
) -> List[Dict[str, object]]:
    from datasets import DownloadConfig, load_dataset

    subset = subset.lower()
    if subset == "trex":
        records = _load_trex_overfit_records(limit, cache_dir=cache_dir, local_files_only=local_files_only)
        if year_only:
            records = [rec for rec in records if _is_year_literal(str(rec.get("object_label", "")).strip())]
        if qid_only:
            records = [rec for rec in records if _is_qid(str(rec.get("object_label", "")).strip())]
        return records

    try:
        dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
        dataset = load_dataset("facebook/lama", subset, split="train", download_config=dl_config)
        iterable: Iterable[Dict[str, object]] = dataset
    except Exception:
        archive_path = _download_lama_archive(cache_dir, local_files_only=local_files_only)
        iterable = _iter_lama_subset_records(archive_path, subset, max_samples=0)

    return _filter_lama_examples(iterable, limit=limit, year_only=year_only, qid_only=qid_only)


def _resize_vocab_state(
    model: CFMModel,
    state: Dict[str, torch.Tensor],
    allow_resize: bool,
) -> bool:
    if not allow_resize:
        return False
    resized = False
    key_weights: Dict[str, torch.Tensor] = {}
    if hasattr(model.backbone, "token_emb"):
        key_weights["backbone.token_emb.weight"] = model.backbone.token_emb.weight
    if hasattr(model.backbone, "emb"):
        key_weights["backbone.emb.weight"] = model.backbone.emb.weight
    if hasattr(model.backbone, "lm_head"):
        key_weights["backbone.lm_head.weight"] = model.backbone.lm_head.weight

    for key, current_weight in key_weights.items():
        if key not in state:
            continue
        old_weight = state[key]
        if old_weight.shape == current_weight.shape:
            continue
        old_vocab = int(old_weight.shape[0])
        new_vocab = int(current_weight.shape[0])
        copy_rows = min(old_vocab, new_vocab)
        new_weight = current_weight.detach().cpu().clone()
        if old_weight.dim() == 1 or new_weight.dim() == 1:
            new_weight[:copy_rows] = old_weight[:copy_rows]
        else:
            copy_cols = min(old_weight.shape[1], new_weight.shape[1])
            new_weight[:copy_rows, :copy_cols] = old_weight[:copy_rows, :copy_cols]
            if old_weight.shape[1] != new_weight.shape[1]:
                logger.warning(
                    "vocab_resize_dim_mismatch key=%s old_shape=%s new_shape=%s copied_rows=%s copied_cols=%s",
                    key,
                    tuple(old_weight.shape),
                    tuple(new_weight.shape),
                    copy_rows,
                    copy_cols,
                )
        init_rows = max(new_vocab - copy_rows, 0)
        logger.info(
            "vocab_resize key=%s old_vocab=%s new_vocab=%s copied=%s init=%s",
            key,
            old_vocab,
            new_vocab,
            copy_rows,
            init_rows,
        )
        state[key] = new_weight
        if old_vocab != new_vocab:
            resized = True
    return resized


def _load_factbank(factbank_dir: Path) -> List[Dict[str, object]]:
    import pandas as pd

    df = pd.read_parquet(factbank_dir / "facts.parquet")
    return df.to_dict(orient="records")


def _clean_text(value: object) -> str:
    return str(value).strip()


def _generate_fallback_orbits(
    subject_label: str,
    relation_label: str,
    object_label: str,
    count: int,
    seed: int,
) -> List[str]:
    rng = random.Random(seed)
    subject = subject_label or "Unknown"
    relation = relation_label or "relation"
    base_templates = [
        "Fill in the blank: {subject} {relation} ____.",
        "{subject} {relation} ____.",
        "What is the {relation} of {subject}?",
        "The {relation} of {subject} is ____.",
        "{subject} has {relation} ____.",
    ]
    rng.shuffle(base_templates)
    orbits: List[str] = []
    if count <= 0:
        return orbits
    idx = 0
    while len(orbits) < count:
        tmpl = base_templates[idx % len(base_templates)]
        orbits.append(tmpl.format(subject=subject, relation=relation, object=object_label))
        idx += 1
    return orbits


def _normalize_factbank(
    records: List[Dict[str, object]],
    orbits_per_fact: int,
    seed: int,
) -> Tuple[int, int]:
    filled_labels = 0
    generated_orbits = 0
    for idx, rec in enumerate(records):
        subject_id = _clean_text(rec.get("subject_id", ""))
        relation_id = _clean_text(rec.get("relation_id", ""))
        object_id = _clean_text(rec.get("object_id_or_value", ""))

        subject_label = _clean_text(rec.get("subject_label", ""))
        relation_label = _clean_text(rec.get("relation_label", ""))
        object_label = _clean_text(rec.get("object_label", ""))

        if not subject_label and subject_id:
            rec["subject_label"] = subject_id
            subject_label = subject_id
            filled_labels += 1
        if not relation_label and relation_id:
            rec["relation_label"] = relation_id
            relation_label = relation_id
            filled_labels += 1
        if not object_label and object_id:
            rec["object_label"] = object_id
            object_label = object_id
            filled_labels += 1

        orbits = rec.get("question_orbits")
        cleaned_orbits: List[str] = []
        if isinstance(orbits, list):
            cleaned_orbits = [o for o in (_clean_text(v) for v in orbits) if o]
        if not cleaned_orbits and orbits_per_fact > 0:
            try:
                fact_id = int(rec.get("fact_id", idx))
            except (TypeError, ValueError):
                fact_id = idx
            cleaned_orbits = _generate_fallback_orbits(
                subject_label,
                relation_label,
                object_label,
                orbits_per_fact,
                seed=seed + fact_id,
            )
            generated_orbits += 1
        if cleaned_orbits:
            rec["question_orbits"] = cleaned_orbits
        else:
            rec["question_orbits"] = []
    return filled_labels, generated_orbits


def _resolve_answer(
    rec: Dict[str, object],
    answer_codes: Dict[str, List[int]],
) -> Tuple[Optional[str], Optional[List[int]]]:
    for key in ("object_label", "object_id_or_value"):
        value = _clean_text(rec.get(key, ""))
        if not value:
            continue
        codes = answer_codes.get(value)
        if codes is not None:
            return value, codes
    return None, None


def _build_examples(records: List[Dict[str, object]], answer_codes: Dict[str, List[int]]) -> List[Dict[str, object]]:
    examples = []
    for rec in records:
        answer, codes = _resolve_answer(rec, answer_codes)
        if answer is None or codes is None:
            continue
        for orbit in rec["question_orbits"]:
            examples.append({
                "fact_id": rec["fact_id"],
                "prompt": orbit,
                "answer": answer,
                "codes": codes,
                "negatives": rec["hard_negatives"],
            })
    return examples


def _is_cloze(prompt: str) -> bool:
    return prompt.lstrip().lower().startswith("fill in the blank:")


def _build_fact_index(records: List[Dict[str, object]], answer_codes: Dict[str, List[int]]) -> List[Dict[str, object]]:
    facts = []
    for rec in records:
        answer, codes = _resolve_answer(rec, answer_codes)
        if answer is None or codes is None:
            continue
        orbits = [str(o) for o in rec.get("question_orbits", []) if o]
        if not orbits:
            continue
        cloze_orbits = [o for o in orbits if _is_cloze(o)]
        other_orbits = [o for o in orbits if not _is_cloze(o)]
        relation_id = str(rec.get("relation_id") or rec.get("relation_label") or "unknown")
        facts.append({
            "fact_id": rec["fact_id"],
            "relation_id": relation_id,
            "answer": answer,
            "codes": codes,
            "orbits": orbits,
            "cloze_orbits": cloze_orbits,
            "other_orbits": other_orbits,
            "negatives": rec.get("hard_negatives", []),
            "tuple_idx": rec.get("tuple_idx"),
        })
    return facts


def _build_relation_index(facts: List[Dict[str, object]]) -> Tuple[List[str], Dict[str, List[int]]]:
    relation_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, fact in enumerate(facts):
        relation_to_indices[fact["relation_id"]].append(idx)
    relation_ids = sorted(relation_to_indices.keys())
    return relation_ids, relation_to_indices


def _build_code_to_label(answer_codes: Dict[str, List[int]]) -> Dict[Tuple[int, ...], str]:
    mapping: Dict[Tuple[int, ...], str] = {}
    for answer, codes in sorted(answer_codes.items()):
        key = tuple(int(c) for c in codes)
        if key not in mapping:
            mapping[key] = answer
    return mapping


def _choose_orbit(
    fact: Dict[str, object],
    rng: random.Random,
    cloze_ratio: float,
) -> Tuple[str, bool]:
    cloze_orbits = fact["cloze_orbits"]
    other_orbits = fact["other_orbits"]
    use_cloze = bool(cloze_orbits) and (not other_orbits or rng.random() < cloze_ratio)
    if use_cloze:
        orbit = cloze_orbits[rng.randrange(len(cloze_orbits))]
        return orbit, True
    if other_orbits:
        orbit = other_orbits[rng.randrange(len(other_orbits))]
        return orbit, False
    if cloze_orbits:
        orbit = cloze_orbits[rng.randrange(len(cloze_orbits))]
        return orbit, True
    return "", False


def _sample_fact_batch(
    facts: List[Dict[str, object]],
    relation_ids: List[str],
    relation_to_indices: Dict[str, List[int]],
    rng: random.Random,
    facts_per_batch: int,
    orbits_per_fact: int,
    cloze_ratio: float,
) -> List[Dict[str, object]]:
    batch: List[Dict[str, object]] = []
    for _ in range(facts_per_batch):
        relation_id = relation_ids[rng.randrange(len(relation_ids))]
        indices = relation_to_indices[relation_id]
        fact = facts[indices[rng.randrange(len(indices))]]
        for _ in range(orbits_per_fact):
            prompt, is_cloze = _choose_orbit(fact, rng, cloze_ratio)
            batch.append({
                "fact_id": fact["fact_id"],
                "prompt": prompt,
                "answer": fact["answer"],
                "codes": fact["codes"],
                "negatives": fact["negatives"],
                "relation_id": relation_id,
                "is_cloze": is_cloze,
                "tuple_idx": fact.get("tuple_idx"),
            })
    return batch


def _orbit_consistency_loss(
    addr_logits: List[torch.Tensor],
    fact_ids: List[str],
    is_cloze: List[bool],
    cloze_boost: float,
) -> Tuple[torch.Tensor, int]:
    device = addr_logits[0].device
    dtype = addr_logits[0].dtype
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, fact_id in enumerate(fact_ids):
        groups[fact_id].append(idx)
    loss = torch.tensor(0.0, device=device, dtype=dtype)
    pair_count = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        pair_count += 1
        cloze_fraction = sum(1 for i in indices if is_cloze[i]) / len(indices)
        group_weight = 1.0 + (cloze_boost - 1.0) * cloze_fraction
        for logits in addr_logits:
            group_logits = logits[indices].float()
            log_probs = torch.log_softmax(group_logits, dim=-1)
            probs = torch.softmax(group_logits, dim=-1)
            mean_probs = probs.mean(dim=0, keepdim=True).clamp(min=1e-8)
            loss = loss + group_weight * F.kl_div(log_probs, mean_probs.expand_as(log_probs), reduction="batchmean")
    if pair_count == 0:
        return torch.zeros((), device=device, dtype=dtype), 0
    loss = loss / max(pair_count, 1)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=1e4)
    return loss, pair_count


def _batch_code_entropy(addr_logits: List[torch.Tensor]) -> torch.Tensor:
    entropies = []
    for logits in addr_logits:
        preds = logits.argmax(dim=-1)
        counts = torch.bincount(preds, minlength=logits.size(-1)).float()
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        entropy = entropy / math.log(logits.size(-1))
        entropies.append(entropy)
    return torch.stack(entropies).mean()


def _prepare_batch(
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    tuple_idx_map: Optional[Dict[Tuple[int, ...], int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = []
    prompt_masks = []
    input_ids = []
    attention_masks = []
    labels = []
    code_targets = []
    tuple_targets = []
    for ex in examples:
        prompt = ex["prompt"]
        answer = ex["answer"]
        prompt_tokens = tokenizer.encode(prompt)
        answer_tokens = tokenizer.encode(answer)
        seq = [tokenizer.bos_id] + prompt_tokens + [tokenizer.sep_id] + answer_tokens + [tokenizer.eos_id]

        input_seq = seq[:-1]
        label_seq = [-100] * len(input_seq)
        start = 1 + len(prompt_tokens) + 1
        for i in range(start, len(input_seq)):
            label_seq[i] = seq[i + 1]

        input_ids.append(input_seq)
        labels.append(label_seq)
        prompt_ids.append([tokenizer.bos_id] + prompt_tokens)
        code_targets.append(ex["codes"])
        tuple_idx = ex.get("tuple_idx", -1)
        if (tuple_idx is None or tuple_idx < 0) and tuple_idx_map is not None:
            tuple_idx = tuple_idx_map.get(tuple(int(c) for c in ex["codes"]), -1)
        tuple_targets.append(tuple_idx if tuple_idx is not None else -1)

    input_ids, attention_masks = pad_sequences(input_ids, tokenizer.pad_id, max_len=max_seq_len)
    labels, _ = pad_sequences(labels, -100, max_len=max_seq_len)
    prompt_ids, prompt_masks = pad_sequences(prompt_ids, tokenizer.pad_id, max_len=max_seq_len)
    code_targets = torch.tensor(code_targets, dtype=torch.long)
    tuple_targets = torch.tensor(tuple_targets, dtype=torch.long)

    return input_ids, attention_masks, labels, prompt_ids, prompt_masks, code_targets, tuple_targets


def _compute_orbit_consistency(
    model: CFMModel,
    records: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total = 0
    consistent = 0
    with torch.no_grad():
        for rec in records:
            orbits = rec["question_orbits"]
            prompts = [[tokenizer.bos_id] + tokenizer.encode(o) for o in orbits]
            prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
            prompt_ids = prompt_ids.to(device)
            prompt_masks = prompt_masks.to(device)
            codes = model.predict_codes(prompt_ids, prompt_masks)
            codes = codes.cpu().numpy()
            total += 1
            if np.all(codes == codes[0]):
                consistent += 1
    return consistent / max(total, 1)


def _compute_code_em(
    model: CFMModel,
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    max_samples: int = 200,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    if len(examples) > max_samples:
        examples = examples[:max_samples]
    with torch.no_grad():
        prompts = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in examples]
        prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        pred = model.predict_codes(prompt_ids, prompt_masks).cpu().numpy()
        gold = np.array([ex["codes"] for ex in examples])
        return float((pred == gold).all(axis=1).mean())


def _compute_code_accuracy(
    model: CFMModel,
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    max_samples: int = 200,
) -> Tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    if len(examples) > max_samples:
        examples = examples[:max_samples]
    if not examples:
        return 0.0, 0.0
    with torch.no_grad():
        prompts = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in examples]
        prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        pred = model.predict_codes(prompt_ids, prompt_masks).cpu().numpy()
        gold = np.array([ex["codes"] for ex in examples])
        tuple_acc = float((pred == gold).all(axis=1).mean())
        slot_acc = float((pred == gold).mean())
        return tuple_acc, slot_acc


def _compute_pred_tuple_found_rate(
    model: CFMModel,
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    code_to_label: Dict[Tuple[int, ...], str],
    max_samples: int = 200,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    if len(examples) > max_samples:
        examples = examples[:max_samples]
    if not examples:
        return 0.0
    with torch.no_grad():
        prompts = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in examples]
        prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        pred = model.predict_codes(prompt_ids, prompt_masks).cpu().numpy()
        found = 0
        for row in pred:
            if tuple(int(c) for c in row.tolist()) in code_to_label:
                found += 1
        return found / max(len(examples), 1)


def _compute_answer_em(
    model: CFMModel,
    examples: List[Dict[str, object]],
    tokenizer: SimpleTokenizer,
    max_seq_len: int,
    max_new_tokens: int,
    max_samples: int = 100,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    if len(examples) > max_samples:
        examples = examples[:max_samples]
    with torch.no_grad():
        prompts = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in examples]
        prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        generated, _ = model.generate_answer(
            prompt_ids,
            prompt_masks,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            bos_id=tokenizer.bos_id,
            sep_id=tokenizer.sep_id,
        )
        preds = [tokenizer.decode(seq.tolist()) for seq in generated]
        golds = [ex["answer"] for ex in examples]
        return float(np.mean([p.strip().lower() == g.strip().lower() for p, g in zip(preds, golds)]))
def _compute_overfit_report(
    *,
    slot_logits: torch.Tensor,
    code_matrix: torch.Tensor,
    gold_tuple_idx: torch.Tensor,
    tuple_idx_to_answer: list[str],
    gold_answers: list[str],
    run_id: str,
    source: str,
    topk: int = 10,
    answer_filter_used: str = "none",
    gold_year_rate: float = 0.0,
    gold_qid_rate: float = 0.0,
    answer_filter_row_count: int = 0,
) -> dict:
    logits_np = slot_logits.detach().cpu().numpy()
    code_matrix_np = code_matrix.detach().cpu().numpy()
    pred_codes_np = constrained_decode_by_logprobs(logits_np, code_matrix_np)
    tuple2idx = {tuple(row.tolist()): idx for idx, row in enumerate(code_matrix_np)}
    pred_tuple_idx = [tuple2idx.get(tuple(row.tolist()), -1) for row in pred_codes_np]
    pred_answers = [tuple_idx_to_answer[idx] if idx >= 0 else "__OOV__" for idx in pred_tuple_idx]
    gold_tuple_idx_list = gold_tuple_idx.detach().cpu().tolist()

    n_eval = len(gold_answers)
    code_em = sum(int(p == g) for p, g in zip(pred_tuple_idx, gold_tuple_idx_list)) / n_eval if n_eval else 0.0
    answer_em = sum(int(exact_match(p, g)) for p, g in zip(pred_answers, gold_answers)) / n_eval if n_eval else 0.0

    # argmax path
    argmax_codes = slot_logits.argmax(dim=-1).detach().cpu().numpy()
    argmax_pred_idx = [tuple2idx.get(tuple(row.tolist()), -1) for row in argmax_codes]
    argmax_pred_answers = [tuple_idx_to_answer[idx] if idx >= 0 else "__OOV__" for idx in argmax_pred_idx]
    code_em_argmax = sum(int(p == g) for p, g in zip(argmax_pred_idx, gold_tuple_idx_list)) / n_eval if n_eval else 0.0
    answer_em_argmax = (
        sum(int(exact_match(p, g)) for p, g in zip(argmax_pred_answers, gold_answers)) / n_eval if n_eval else 0.0
    )

    negative_margin = negative_margin_stats(logits_np, code_matrix_np[gold_tuple_idx_list])

    # hist
    pred_hist = {}
    for ans in pred_answers:
        pred_hist[ans] = pred_hist.get(ans, 0) + 1
    gold_hist = {}
    for ans in gold_answers:
        gold_hist[ans] = gold_hist.get(ans, 0) + 1
    pred_hist_topk = [{"answer": k, "count": v} for k, v in sorted(pred_hist.items(), key=lambda x: (-x[1], x[0]))[:topk]]
    gold_hist_topk = [{"value": k, "count": v} for k, v in sorted(gold_hist.items(), key=lambda x: (-x[1], x[0]))[:topk]]

    return {
        "code_em": code_em,
        "answer_em": answer_em,
        "orbit_consistency": {"count": 0, "rate": None},
        "negative_margin": negative_margin,
        "answer_filter_used": answer_filter_used,
        "gold_year_rate": gold_year_rate,
        "gold_qid_rate": gold_qid_rate,
        "answer_filter_row_count": answer_filter_row_count,
        "em_breakdown": {
            "answer_em_constrained": answer_em,
            "answer_em_argmax": answer_em_argmax,
            "code_em_constrained": code_em,
            "code_em_argmax": code_em_argmax,
        },
        "decode_mode_used": "constrained",
        "pred_hist_topk": pred_hist_topk,
        "gold_hist_topk": gold_hist_topk,
        "n_eval": n_eval,
        "candidate_size": int(code_matrix.shape[0]),
        "run_id": run_id,
        "source": source,
    }


def _write_overfit_report(*, report: dict, report_dir: Path, out_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.json"
    payload = json.dumps(report, indent=2)
    report_path.write_text(payload)
    latest_path = out_dir / "report.json"
    latest_path.write_text(payload)
    print(f"wrote overfit report: {report_path.resolve()}")
    return report_path


def _compute_negative_margin(
    model: CFMModel,
    records: List[Dict[str, object]],
    answer_codes: Dict[str, List[int]],
) -> float:
    model.eval()
    device = next(model.parameters()).device
    margins = []
    with torch.no_grad():
        for rec in records:
            pos_answer = str(rec["object_label"])
            pos_codes = answer_codes.get(pos_answer)
            if pos_codes is None:
                continue
            negs = rec["hard_negatives"]
            if negs is None or len(negs) == 0:
                continue
            neg_codes = None
            for neg in list(negs):
                neg_answer = str(neg["object_label"])
                if neg_answer in answer_codes:
                    neg_codes = answer_codes[neg_answer]
                    break
            if neg_codes is None:
                continue
            pos_codes_t = torch.tensor([pos_codes], device=device)
            neg_codes_t = torch.tensor([neg_codes], device=device)
            v_pos = model.decode_value(pos_codes_t)
            v_neg = model.decode_value(neg_codes_t)
            cos = F.cosine_similarity(v_pos, v_neg).item()
            margins.append(1.0 - cos)
    if not margins:
        return 0.0
    return float(np.mean(margins))


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to config YAML"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Disable resume from latest checkpoint"),
    overfit_trex: bool = typer.Option(False, "--overfit-trex", help="Overfit a tiny T-REx slice to sanity-check learning"),
    overfit_lama_years: bool = typer.Option(
        False,
        "--overfit-lama-years",
        help="Overfit a tiny Google_RE year slice to sanity-check literal learning",
    ),
    overfit_lama_qids: bool = typer.Option(
        False,
        "--overfit-lama-qids",
        help="Overfit a tiny LAMA slice restricted to QID answers",
    ),
    overfit_lama_subset: Optional[str] = typer.Option(
        None,
        "--overfit-lama-subset",
        help="Overfit a tiny LAMA subset (google_re, trex, conceptnet)",
    ),
    overfit_holdout_frac: float = typer.Option(
        0.25,
        "--overfit-holdout-frac",
        help="Holdout fraction for overfit subset gating",
    ),
    overfit_n: int = typer.Option(128, "--overfit-n", help="Overfit sample count"),
    overfit_seed: int = typer.Option(0, "--overfit-seed", help="Overfit sampling seed"),
    overfit_balanced: bool = typer.Option(True, "--overfit-balanced/--no-overfit-balanced", help="Balance overfit sampling"),
) -> None:
    cfg = yaml.safe_load(config.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)
    global _ORBIT_ZERO_WARNING_SHOWN, _ORBIT_ZERO_ZERO_STREAK, _FIRST_NONFINITE_COMPONENT_LOGGED

    runtime_cfg = cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", DEFAULT_CPU_THREADS))
    prefer_gpu = bool(runtime_cfg.get("prefer_gpu", True))
    use_faiss = bool(runtime_cfg.get("use_faiss", True))

    torch_info = configure_torch(cpu_threads=cpu_threads, prefer_gpu=prefer_gpu)
    faiss_info = resolve_faiss(prefer_gpu=prefer_gpu, cpu_threads=cpu_threads) if use_faiss else {
        "faiss_available": False,
        "faiss_gpu_available": False,
        "faiss_gpu_count": 0,
        "faiss": None,
    }
    log_runtime(logger, torch_info, faiss_info, cpu_threads)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    inf_cfg = cfg["inference"]
    tuple_loss_weight = float(train_cfg.get("tuple_loss_weight", 0.0))
    slot_loss_weight = float(train_cfg.get("slot_loss_weight", 1.0))

    out_dir = Path("out")
    _ensure_file_logger(out_dir)
    ckpt_dir = out_dir / "ckpt"

    factbank_dir = Path(data_cfg["factbank_dir"])
    codes_dir = Path(data_cfg["codes_dir"])

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))
    tuple_code_matrix = None
    tuple_answers: List[str] = []
    global_code_matrix_np = None
    global_code_matrix_device = None
    global_tuple2idx: Optional[Dict[Tuple[int, ...], int]] = None
    global_tuple_idx_to_answer: List[str] = []
    global_index_offsets: Optional[np.ndarray] = None
    global_index_flat: Optional[np.ndarray] = None
    overfit_run_id: str | None = None
    gold_year_rate = 0.0
    gold_qid_rate = 0.0

    overfit_lama_subset_name = overfit_lama_subset.lower().strip() if overfit_lama_subset else None
    overfit_year_only = False
    overfit_qid_only = False
    if overfit_lama_subset_name and overfit_lama_subset_name not in {"google_re", "trex", "conceptnet"}:
        raise ValueError("overfit_lama_subset must be one of google_re, trex, conceptnet")
    if overfit_lama_years:
        if overfit_lama_qids:
            raise ValueError("overfit-lama-years and overfit-lama-qids are mutually exclusive")
        if overfit_lama_subset_name and overfit_lama_subset_name != "google_re":
            raise ValueError("overfit-lama-years requires subset google_re")
        overfit_lama_subset_name = "google_re"
        overfit_year_only = True
    if overfit_lama_qids:
        if overfit_lama_subset_name is None:
            overfit_lama_subset_name = "trex"
        overfit_qid_only = True

    if overfit_trex and overfit_lama_subset_name:
        raise ValueError("overfit_trex and overfit_lama_subset are mutually exclusive")

    if overfit_trex:
        trex_limit = int(train_cfg.get("overfit_trex_samples", 512))
        logger.info(
            "load_trex_overfit start limit=%s local_files_only=%s cache_dir=%s",
            trex_limit,
            local_files_only,
            cache_dir,
        )
        t0 = time.perf_counter()
        records = _load_trex_overfit_records(trex_limit, cache_dir=cache_dir, local_files_only=local_files_only)
        logger.info("load_trex_overfit done records=%s time=%.2fs", len(records), time.perf_counter() - t0)
        answer_pool = [str(rec.get("object_label", "")).strip() for rec in records]
        answer_codes = _build_hashed_answer_codes(
            answer_pool,
            m=int(model_cfg["m"]),
            k=int(model_cfg["K"]),
        )
        logger.info("overfit_answer_codes built answers=%s", len(answer_codes))
    elif overfit_lama_subset_name:
        overfit_run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        years_limit = int(train_cfg.get("overfit_lama_years_samples", 512))
        logger.info(
            "load_lama_overfit start subset=%s limit=%s local_files_only=%s cache_dir=%s",
            overfit_lama_subset_name,
            years_limit,
            local_files_only,
            cache_dir,
        )
        t0 = time.perf_counter()
        records = _load_lama_overfit_records(
            overfit_lama_subset_name,
            years_limit,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            year_only=False,
            qid_only=False,
        )
        gold_year_total = sum(1 for rec in records if _is_year_literal(str(rec.get("object_label", "")).strip()))
        gold_qid_total = sum(1 for rec in records if _is_qid(str(rec.get("object_label", "")).strip()))
        gold_total = len(records)
        gold_year_rate = gold_year_total / gold_total if gold_total else 0.0
        gold_qid_rate = gold_qid_total / gold_total if gold_total else 0.0
        if overfit_lama_subset_name == "trex" and gold_qid_rate > 0.0:
            overfit_qid_only = True
        if overfit_qid_only or gold_qid_rate >= 0.90:
            records = [rec for rec in records if _is_qid(str(rec.get("object_label", "")).strip())]
            overfit_qid_only = True
        elif overfit_year_only or gold_year_rate >= 0.90:
            records = [rec for rec in records if _is_year_literal(str(rec.get("object_label", "")).strip())]
            overfit_year_only = True
        logger.info(
            "load_lama_overfit done records=%s gold_year_rate=%.4f gold_qid_rate=%.4f time=%.2fs",
            len(records),
            gold_year_rate,
            gold_qid_rate,
            time.perf_counter() - t0,
        )
        import pandas as pd

        logger.info("load_answer_codes start path=%s", codes_dir / "answer_codes.parquet")
        t0 = time.perf_counter()
        answer_df = pd.read_parquet(codes_dir / "answer_codes.parquet")
        answer_to_codes: Dict[str, Tuple[int, ...]] = {}
        filtered_records: List[Dict[str, object]] = []
        for rec in records:
            answer = str(rec.get("object_label", "")).strip()
            if not answer:
                continue
            codes = codes_from_answer(answer_df, answer)
            if codes is None:
                continue
            answer_to_codes[answer] = codes
            filtered_records.append(rec)
        records = filtered_records
        if overfit_balanced and overfit_year_only:
            records = _balanced_sample_records_by_year(records, overfit_n, overfit_seed)
        elif overfit_n > 0:
            rng = random.Random(overfit_seed)
            rng.shuffle(records)
            records = records[:overfit_n]
        answer_codes = {answer: list(codes) for answer, codes in answer_to_codes.items()}
        tuple_answers = sorted({str(rec.get("object_label", "")).strip() for rec in records if str(rec.get("object_label", "")).strip()})
        tuple_code_matrix = torch.tensor([list(answer_to_codes[answer]) for answer in tuple_answers], dtype=torch.long)
        tuple2idx = {tuple(answer_to_codes[answer]): idx for idx, answer in enumerate(tuple_answers)}
        for rec in records:
            answer = str(rec.get("object_label", "")).strip()
            codes = answer_to_codes.get(answer)
            if codes is None:
                continue
            rec["tuple_idx"] = tuple2idx[tuple(codes)]
        logger.info(
            "load_answer_codes done answers=%s tuples=%s time=%.2fs",
            len(answer_codes),
            len(tuple_answers),
            time.perf_counter() - t0,
        )

    else:
        logger.info("load_factbank start")
        t0 = time.perf_counter()
        records = _load_factbank(factbank_dir)
        logger.info("load_factbank done records=%s time=%.2fs", len(records), time.perf_counter() - t0)
        logger.info("load_answer_codes start path=%s", codes_dir / "answer_codes.parquet")
        t0 = time.perf_counter()
        answer_codes = load_answer_codes(codes_dir / "answer_codes.parquet")
        logger.info("load_answer_codes done answers=%s time=%.2fs", len(answer_codes), time.perf_counter() - t0)
        if tuple_loss_weight > 0.0:
            import pandas as pd

            answer_df = pd.read_parquet(codes_dir / "answer_codes.parquet")
            rows = []
            answers = []
            code_len = None
            for _, row in answer_df.iterrows():
                answer = str(row.get("answer", "")).strip()
                codes = row.get("codes", [])
                if not answer:
                    continue
                try:
                    code_list = [int(c) for c in codes]
                except Exception:
                    continue
                if code_len is None:
                    code_len = len(code_list)
                if len(code_list) != code_len:
                    continue
                rows.append(code_list)
                answers.append(answer)
            if rows:
                global_code_matrix_np = np.asarray(rows, dtype=np.int64)
                global_tuple2idx = {tuple(row): idx for idx, row in enumerate(global_code_matrix_np.tolist())}
                global_tuple_idx_to_answer = answers
                code_vocab_size = int(global_code_matrix_np.max()) + 1
                index_path = Path(train_cfg.get("index_path", "data/codes/answer_codes.index.npz"))
                global_index_offsets, global_index_flat = ensure_inverted_index(
                    index_path,
                    global_code_matrix_np,
                    code_vocab_size,
                )
                logger.info(
                    "tuple_candidates_loaded total=%s index_path=%s",
                    len(global_tuple_idx_to_answer),
                    index_path,
                )

    fallback_orbits = int(data_cfg.get("orbits_per_fact") or 0)
    filled_labels, generated_orbits = _normalize_factbank(records, fallback_orbits, seed)
    if filled_labels:
        logger.info("factbank_labels_filled count=%s", filled_labels)
    if generated_orbits:
        logger.info("factbank_orbits_generated count=%s per_fact=%s", generated_orbits, fallback_orbits)

    train_records = records
    holdout_records: List[Dict[str, object]] = []
    if overfit_lama_subset_name:
        train_records, holdout_records = _split_records_holdout(records, overfit_holdout_frac, overfit_seed)
    logger.info("build_examples start")
    t0 = time.perf_counter()
    examples = _build_examples(train_records, answer_codes)
    holdout_examples = _build_examples(holdout_records, answer_codes) if holdout_records else []
    if tuple_code_matrix is not None:
        tuple2idx = {tuple(row): idx for idx, row in enumerate(tuple_code_matrix.tolist())}
        for ex in examples:
            codes_tuple = tuple(ex["codes"])
            ex["tuple_idx"] = tuple2idx.get(codes_tuple, -1)
        for ex in holdout_examples:
            codes_tuple = tuple(ex["codes"])
            ex["tuple_idx"] = tuple2idx.get(codes_tuple, -1)
    logger.info(
        "build_examples done examples=%s holdout=%s orbits_per_fact=%s time=%.2fs",
        len(examples),
        len(holdout_examples),
        len(train_records[0]["question_orbits"]) if train_records else 0,
        time.perf_counter() - t0,
    )
    if (overfit_trex or overfit_lama_subset_name) and not examples:
        raise RuntimeError("overfit dataset produced zero training examples; check codebook coverage and filters")
    gen_loss_weight = float(train_cfg.get("gen_loss_weight", 1.0))
    gen_targets, gen_total, gen_coverage = _gen_target_coverage(examples)
    sample_unique_answers, sample_unique_qids = _sample_unique_answer_stats(examples, sample_size=200)
    logger.info(
        "gen_target_coverage=%.4f with_targets=%s total=%s sample_unique_answers=%s sample_unique_qids=%s",
        gen_coverage,
        gen_targets,
        gen_total,
        sample_unique_answers,
        sample_unique_qids,
    )
    if gen_loss_weight > 0.0 and gen_coverage == 0.0:
        raise RuntimeError("gen_loss enabled but no non-empty generation targets found in examples")
    overfit_enabled = overfit_trex or overfit_lama_subset_name is not None
    overfit_examples = examples if overfit_enabled else []
    texts = [ex["prompt"] for ex in examples] + [ex["answer"] for ex in examples]
    if holdout_examples:
        texts += [ex["prompt"] for ex in holdout_examples] + [ex["answer"] for ex in holdout_examples]
    logger.info("build_tokenizer start texts=%s", len(texts))
    t0 = time.perf_counter()
    tokenizer = SimpleTokenizer.build(texts, vocab_max=model_cfg.get("vocab_max"))
    logger.info("build_tokenizer done vocab_size=%s time=%.2fs", len(tokenizer.vocab), time.perf_counter() - t0)

    out_dir = Path("out")
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    codebook_path = codes_dir / "codebooks.safetensors"
    logger.info("init_model start")
    t0 = time.perf_counter()
    model = CFMModel.from_codebooks(
        codebook_path,
        vocab_size=len(tokenizer.vocab),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        m=int(model_cfg["m"]),
        K=int(model_cfg["K"]),
        backbone=model_cfg.get("backbone", "tiny"),
        hf_model_name=model_cfg.get("hf_model_name"),
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("init_model done params=%s time=%.2fs", param_count, time.perf_counter() - t0)

    device = torch.device("cuda" if torch_info.get("torch_cuda") else "cpu")
    model.to(device)
    logger.info("model_to_device done device=%s", device.type)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    logger.info("optimizer_init done")

    rng = random.Random(seed)
    batch_size = int(train_cfg["batch_size"])
    steps = int(train_cfg["steps"])
    overfit_eval_every = 0
    overfit_eval_samples = 0
    if overfit_enabled:
        steps = int(train_cfg.get("overfit_steps", 1000))
        overfit_eval_every = int(train_cfg.get("overfit_eval_every", 50))
        overfit_eval_samples = int(train_cfg.get("overfit_eval_samples", 256))
        if overfit_trex:
            mode = "trex"
        elif overfit_lama_subset_name:
            mode = overfit_lama_subset_name
        else:
            mode = "lama"
        logger.info(
            "overfit_mode enabled mode=%s steps=%s eval_every=%s eval_samples=%s",
            mode,
            steps,
            overfit_eval_every,
            overfit_eval_samples,
        )
    max_seq_len = int(model_cfg["max_seq_len"])
    orbits_per_fact_in_batch = max(2, int(train_cfg.get("orbits_per_fact_in_batch", 2)))
    facts_per_batch = max(1, batch_size // orbits_per_fact_in_batch)
    effective_batch = facts_per_batch * orbits_per_fact_in_batch
    if effective_batch != batch_size:
        logger.info("batch_size_adjusted requested=%s effective=%s", batch_size, effective_batch)
    batch_size = effective_batch
    cloze_ratio = float(train_cfg.get("cloze_ratio", 0.8))
    if cloze_ratio < 0.0 or cloze_ratio > 1.0:
        logger.warning("cloze_ratio_out_of_range value=%.3f; clamping to [0,1]", cloze_ratio)
        cloze_ratio = min(max(cloze_ratio, 0.0), 1.0)
    gen_loss_weight = float(train_cfg.get("gen_loss_weight", gen_loss_weight))
    require_gen_loss = bool(train_cfg.get("require_gen_loss", False))
    require_gen_loss_steps = int(train_cfg.get("require_gen_loss_steps", 200))
    amp_requested = bool(train_cfg.get("amp", False))
    amp_enabled = amp_requested and bool(torch_info.get("torch_cuda"))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    max_nan_skips = int(train_cfg.get("max_nan_skips", 3))
    cloze_addr_weight = float(train_cfg.get("cloze_addr_weight", 2.0))
    cloze_orbit_boost = float(train_cfg.get("cloze_orbit_boost", 2.0))
    orbit_consistency_weight = float(train_cfg.get("orbit_consistency_weight", 0.2))
    entropy_weight = float(train_cfg.get("entropy_weight", 0.01))
    obj_prior_weight = float(train_cfg.get("obj_prior_weight", 0.0))
    collapse_window = int(train_cfg.get("collapse_window", 50))
    nonfinite_lr_reduce_every = int(train_cfg.get("nonfinite_lr_reduce_every", 0))
    nonfinite_lr_reduce_factor = float(train_cfg.get("nonfinite_lr_reduce_factor", 0.5))
    allow_vocab_resize_on_resume = bool(train_cfg.get("allow_vocab_resize_on_resume", False))

    facts = _build_fact_index(records, answer_codes)
    relation_ids, relation_to_indices = _build_relation_index(facts)
    if not facts:
        orbits_ready = sum(1 for rec in records if rec.get("question_orbits"))
        answers_ready = sum(1 for rec in records if _resolve_answer(rec, answer_codes)[0] is not None)
        raise ValueError(
            "No training facts with orbits and answer codes found. "
            f"records={len(records)} with_orbits={orbits_ready} with_answer_codes={answers_ready}"
        )
    if not relation_ids:
        raise ValueError("No relation ids found for relation-balanced sampling.")
    code_to_label = _build_code_to_label(answer_codes)
    steps_per_epoch = max(1, len(facts) // max(facts_per_batch, 1))
    epoch_idx = 0
    epoch_label_counts: Counter = Counter()
    epoch_code_counts: List[Counter] = [Counter() for _ in range(int(model_cfg["m"]))]
    collapse_window = max(0, collapse_window)
    pred_window: deque[list[str]] = deque(maxlen=collapse_window) if collapse_window > 0 else deque()
    pred_counter: Counter = Counter()
    pred_total = 0
    collapse_detected = False
    collapse_active = False

    resume = bool(train_cfg.get("resume", True))
    if no_resume:
        resume = False
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))
    checkpoint_on_start = bool(train_cfg.get("checkpoint_on_start", True))
    start_step = 0
    if resume:
        ckpt_path = _find_latest_checkpoint(ckpt_dir)
        if ckpt_path:
            logger.info("resume_checkpoint start path=%s", ckpt_path)
            checkpoint = _load_checkpoint(ckpt_path)
            vocab_resized = False
            if "model" in checkpoint:
                state = checkpoint["model"]
                vocab_resized = _resize_vocab_state(model, state, allow_vocab_resize_on_resume)
                model.load_state_dict(state)
            else:
                logger.warning("resume_checkpoint missing=model_state path=%s", ckpt_path)
            if "optimizer" in checkpoint:
                if vocab_resized:
                    logger.info("resume_checkpoint optimizer_reset reason=vocab_resize")
                else:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    _move_optimizer_state(optimizer, device)
            else:
                logger.warning("resume_checkpoint missing=optimizer_state path=%s", ckpt_path)
            ckpt_step = get_ckpt_step(checkpoint)
            start_step = int(ckpt_step) if ckpt_step is not None else 0
            if "epoch_idx" in checkpoint:
                epoch_idx = int(checkpoint["epoch_idx"])
            else:
                epoch_idx = start_step // steps_per_epoch
            if "rng_state" in checkpoint:
                rng.setstate(checkpoint["rng_state"])
            if "np_state" in checkpoint:
                np.random.set_state(checkpoint["np_state"])
            if "torch_state" in checkpoint:
                torch_state = checkpoint["torch_state"]
                if torch.is_tensor(torch_state):
                    torch_state = torch_state.cpu()
                torch.random.set_rng_state(torch_state)
            if torch_info.get("torch_cuda") and checkpoint.get("torch_cuda_state"):
                torch_cuda_state = checkpoint["torch_cuda_state"]
                torch.cuda.set_rng_state_all([t.cpu() for t in torch_cuda_state])
            logger.info("resume_checkpoint done step=%s epoch=%s", start_step, epoch_idx)
        else:
            logger.info("resume_checkpoint skipped reason=no_checkpoint")
    else:
        logger.info("resume_checkpoint skipped reason=disabled")

    if resume and start_step >= steps:
        logger.info("already_finished start_step=%s steps=%s (exiting)", start_step, steps)
        return

    model.train()
    log_every = int(train_cfg["log_every"])
    time_log_interval = float(train_cfg.get("time_log_interval", 30.0))
    logger.info(
        "training start steps=%s batch_size=%s device=%s log_every=%s time_log_interval=%.1fs",
        steps,
        batch_size,
        device.type,
        log_every,
        time_log_interval,
    )
    logger.info(
        "sampling_config facts_per_batch=%s orbits_per_fact=%s cloze_ratio=%.2f relations=%s",
        facts_per_batch,
        orbits_per_fact_in_batch,
        cloze_ratio,
        len(relation_ids),
    )
    if overfit_year_only:
        tuple_loss_weight = 1.0
        slot_loss_weight = 0.2
    logger.info(
        "loss_weights gen_weight=%.2f cloze_addr=%.2f cloze_orbit_boost=%.2f orbit_weight=%.3f entropy_weight=%.4f obj_prior_weight=%.4f tuple_weight=%.2f slot_weight=%.2f",
        gen_loss_weight,
        cloze_addr_weight,
        cloze_orbit_boost,
        orbit_consistency_weight,
        entropy_weight,
        obj_prior_weight,
        tuple_loss_weight,
        slot_loss_weight,
    )
    gen_loss_disabled_reason = ""
    if gen_loss_weight <= 0.0:
        gen_loss_disabled_reason = "weight_zero"
    elif gen_coverage == 0.0:
        gen_loss_disabled_reason = "no_targets"
    if gen_loss_disabled_reason:
        logger.info("gen_loss_disabled reason=%s", gen_loss_disabled_reason)
        if require_gen_loss:
            raise RuntimeError(f"gen_loss disabled (reason={gen_loss_disabled_reason}) but require_gen_loss is true")
    else:
        logger.info("gen_loss_enabled weight=%.3f coverage=%.4f", gen_loss_weight, gen_coverage)
    logger.info(
        "amp_enabled=%s grad_clip_norm=%.4f max_nan_skips=%s",
        amp_enabled,
        grad_clip_norm,
        max_nan_skips,
    )
    train_start = time.perf_counter()
    last_log_time = train_start
    last_log_step = start_step
    next_time_log = train_start + time_log_interval
    if checkpoint_on_start and start_step == 0:
        _save_checkpoint(ckpt_dir, step=0, model=model, optimizer=optimizer, rng=rng, epoch_idx=epoch_idx)
    scaler = GradScaler(enabled=amp_enabled)
    tuple_code_matrix_device = tuple_code_matrix.to(device) if tuple_code_matrix is not None else None
    if global_code_matrix_np is not None:
        global_code_matrix_device = torch.tensor(global_code_matrix_np, dtype=torch.long, device=device)
    nan_skip_count = 0
    nonfinite_events = 0
    gen_loss_zero_steps = 0
    overfit_success_streak = 0
    overfit_acc_threshold = float(train_cfg.get("overfit_acc_threshold", 0.99))
    overfit_acc_patience = int(train_cfg.get("overfit_acc_patience", 3))

    for step in range(start_step, steps):
        batch = _sample_fact_batch(
            facts,
            relation_ids,
            relation_to_indices,
            rng,
            facts_per_batch=facts_per_batch,
            orbits_per_fact=orbits_per_fact_in_batch,
            cloze_ratio=cloze_ratio,
        )
        (
            input_ids,
            attention_masks,
            labels,
            prompt_ids,
            prompt_masks,
            code_targets,
            tuple_targets,
        ) = _prepare_batch(batch, tokenizer, max_seq_len, tuple_idx_map=global_tuple2idx)
        fact_ids = [ex["fact_id"] for ex in batch]
        is_cloze = [bool(ex["is_cloze"]) for ex in batch]
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        code_targets = code_targets.to(device)
        tuple_targets = tuple_targets.to(device)

        with autocast(enabled=amp_enabled):
            addr_logits_raw, _, _ = model.encode_prompt(prompt_ids, prompt_masks)
            _, gen_logits_raw = model.forward_generation(input_ids, attention_masks, code_targets)
        addr_logits = [logits.float() for logits in addr_logits_raw]
        gen_logits = gen_logits_raw.float()

        addr_loss = torch.tensor(0.0, device=device)
        for i, logits in enumerate(addr_logits):
            weights = torch.tensor(
                [cloze_addr_weight if flag else 1.0 for flag in is_cloze],
                device=logits.device,
                dtype=logits.dtype,
            )
            per_item = F.cross_entropy(logits, code_targets[:, i], reduction="none")
            addr_loss = addr_loss + (per_item * weights).mean()

        prefix_pad = torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype)
        labels = torch.cat([prefix_pad, labels], dim=1)
        gen_target_count = int((labels != -100).sum().item())
        if gen_loss_weight > 0.0 and gen_target_count > 0:
            gen_loss = F.cross_entropy(
                gen_logits.reshape(-1, gen_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        else:
            gen_loss = torch.tensor(0.0, device=device)
            if gen_loss_weight > 0.0 and gen_target_count == 0 and not gen_loss_disabled_reason:
                logger.warning("gen_loss_batch_no_targets step=%s", step + 1)

        tuple_loss = torch.tensor(0.0, device=device)
        tuple_pred_idx = None
        if tuple_loss_weight > 0.0:
            valid_mask = tuple_targets >= 0
            if valid_mask.any():
                slot_logits = torch.stack(addr_logits, dim=1)
                slot_logits = slot_logits[valid_mask]
                if tuple_code_matrix_device is not None:
                    tuple_loss, tuple_pred_idx = tuple_ce_loss(
                        slot_logits,
                        tuple_code_matrix_device,
                        tuple_targets[valid_mask],
                    )
                elif (
                    global_code_matrix_device is not None
                    and global_index_offsets is not None
                    and global_index_flat is not None
                ):
                    gold_idx = tuple_targets[valid_mask]
                    cand_rows = candidate_rows_from_logits(
                        slot_logits,
                        global_index_offsets,
                        global_index_flat,
                        topk_per_slot=int(train_cfg.get("candidate_topk_per_slot", 16)),
                        max_candidates=int(train_cfg.get("candidate_max", 4096)),
                        always_include=gold_idx,
                    )
                    if cand_rows:
                        max_len = max(len(rows) for rows in cand_rows)
                    else:
                        max_len = 0
                    if max_len > 0:
                        cand_idx = torch.empty((len(cand_rows), max_len), dtype=torch.long)
                        code_matrix_size = int(global_code_matrix_device.size(0))
                        for row_i, rows in enumerate(cand_rows):
                            rows_list = [int(r) for r in rows.tolist()] if rows.size else []
                            gold = int(gold_idx[row_i].item())
                            if not rows_list:
                                rows_list = [gold]
                            if rows_list[0] != gold:
                                if gold in rows_list:
                                    rows_list.remove(gold)
                                rows_list.insert(0, gold)
                            used = set(rows_list)
                            if len(rows_list) < max_len:
                                rng = np.random.default_rng(row_i)
                                while len(rows_list) < max_len:
                                    cand = int(rng.integers(0, code_matrix_size))
                                    if cand in used:
                                        continue
                                    rows_list.append(cand)
                                    used.add(cand)
                            cand_idx[row_i] = torch.tensor(rows_list[:max_len], dtype=torch.long)
                        cand_idx = cand_idx.to(device)
                        cand_codes = global_code_matrix_device[cand_idx]
                        tuple_logprobs = candidate_tuple_logprobs(slot_logits, cand_codes)
                        tuple_loss = tuple_ce_loss_candidates(tuple_logprobs)
                        pred_in_cand = tuple_logprobs.argmax(dim=1)
                        tuple_pred_idx = cand_idx.gather(1, pred_in_cand.unsqueeze(1)).squeeze(1)

        orbit_loss, orbit_pairs_in_batch = _orbit_consistency_loss(addr_logits, fact_ids, is_cloze, cloze_orbit_boost)
        orbit_loss = torch.nan_to_num(
            orbit_loss * orbit_consistency_weight,
            nan=0.0,
            posinf=1e4,
            neginf=1e4,
        )
        if orbit_pairs_in_batch == 0:
            _ORBIT_ZERO_ZERO_STREAK += 1
            if _ORBIT_ZERO_ZERO_STREAK >= _ORBIT_ZERO_WARNING_THRESHOLD and not _ORBIT_ZERO_WARNING_SHOWN:
                logger.warning(
                    "orbit_pairs_zero streak=%s steps without pairs (step=%s)",
                    _ORBIT_ZERO_ZERO_STREAK,
                    step + 1,
                )
                _ORBIT_ZERO_WARNING_SHOWN = True
        else:
            _ORBIT_ZERO_ZERO_STREAK = 0
        entropy_reg = _batch_code_entropy(addr_logits)

        contrast_loss = torch.tensor(0.0, device=device)
        neg_codes = []
        neg_from_hard = 0
        for ex in batch:
            negs = ex["negatives"]
            neg_code = None
            for neg in negs:
                neg_answer = str(neg["object_label"])
                if neg_answer in answer_codes:
                    neg_code = answer_codes[neg_answer]
                    neg_from_hard += 1
                    break
            if neg_code is None:
                if len(batch) > 1:
                    neg_code = batch[rng.randrange(len(batch))]["codes"]
                else:
                    neg_code = ex["codes"]
            neg_codes.append(neg_code)
        neg_codes = torch.tensor(neg_codes, device=device)
        v_pos = model.decode_value(code_targets)
        v_neg = model.decode_value(neg_codes)
        contrast_loss = contrastive_margin_loss(v_pos, v_neg, margin=float(train_cfg["contrast_margin"]))

        obj_prior_loss = torch.tensor(0.0, device=device)
        if obj_prior_weight > 0.0 and collapse_active:
            k = int(model_cfg["K"])
            for logits in addr_logits:
                probs = logits.softmax(dim=-1).mean(dim=0)
                obj_prior_loss = obj_prior_loss + (probs * (probs + 1e-9).log()).sum() + math.log(k)
            obj_prior_loss = obj_prior_loss / max(len(addr_logits), 1)

        loss = (
            slot_loss_weight * addr_loss
            + gen_loss_weight * gen_loss
            + tuple_loss_weight * tuple_loss
            + contrast_loss
            + orbit_loss
            - entropy_weight * entropy_reg
            + obj_prior_weight * obj_prior_loss
        )

        component_map = {
            "addr": addr_loss,
            "gen": gen_loss,
            "contrast": contrast_loss,
            "orbit": orbit_loss,
            "tuple": tuple_loss,
            "entropy": entropy_reg,
            "obj_prior": obj_prior_loss,
        }
        nonfinite_components = [name for name, tensor in component_map.items() if not torch.isfinite(tensor)]
        if not torch.isfinite(loss):
            logger.warning(
                "nonfinite_loss step=%s addr=%s gen=%s contrast=%s orbit=%s entropy=%s",
                step + 1,
                addr_loss.item(),
                gen_loss.item(),
                contrast_loss.item(),
                orbit_loss.item(),
                entropy_reg.item(),
            )
            logger.warning(
                "nonfinite_stats step=%s orbit_pairs=%s addr_stats=%s gen_stats=%s",
                step + 1,
                orbit_pairs_in_batch,
                [_tensor_stats(logits) for logits in addr_logits],
                _tensor_stats(gen_logits),
            )
            if nonfinite_components and not _FIRST_NONFINITE_COMPONENT_LOGGED:
                logger.warning(
                    "first_nonfinite_components step=%s components=%s",
                    step + 1,
                    nonfinite_components,
                )
                _FIRST_NONFINITE_COMPONENT_LOGGED = True
            _save_nonfinite_debug(
                step + 1,
                batch,
                code_targets,
                orbit_pairs_in_batch,
                addr_logits,
                gen_logits,
                nonfinite_components,
                {name: float(tensor.detach().cpu().item()) for name, tensor in component_map.items()},
            )
            nan_skip_count += 1
            nonfinite_events += 1
            if nonfinite_lr_reduce_every > 0 and nonfinite_events % nonfinite_lr_reduce_every == 0:
                for group in optimizer.param_groups:
                    group["lr"] = float(group.get("lr", 0.0)) * nonfinite_lr_reduce_factor
                logger.warning(
                    "nonfinite_lr_reduce step=%s events=%s factor=%.3f",
                    step + 1,
                    nonfinite_events,
                    nonfinite_lr_reduce_factor,
                )
            if nan_skip_count >= max_nan_skips:
                _save_checkpoint(
                    ckpt_dir,
                    step=step + 1,
                    model=model,
                    optimizer=optimizer,
                    rng=rng,
                    epoch_idx=epoch_idx,
                )
                logger.error("max_nan_skips reached step=%s exit=1", step + 1)
                sys.exit(1)
            optimizer.zero_grad(set_to_none=True)
            continue

        if require_gen_loss:
            if gen_loss.item() <= 1e-8:
                gen_loss_zero_steps += 1
                if gen_loss_zero_steps >= require_gen_loss_steps:
                    raise RuntimeError(
                        f"gen_loss stayed at 0 for {gen_loss_zero_steps} steps; "
                        "check targets, tokenizer, and gen_loss_weight"
                    )
            else:
                gen_loss_zero_steps = 0

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if grad_clip_norm > 0.0:
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)
        else:
            grad_norm = _compute_grad_norm(model)
        if amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        nan_skip_count = 0

        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            _save_checkpoint(
                ckpt_dir,
                step=step + 1,
                model=model,
                optimizer=optimizer,
                rng=rng,
                epoch_idx=epoch_idx,
            )

        with torch.no_grad():
            pred_codes = torch.stack([logits.argmax(dim=-1) for logits in addr_logits], dim=1).cpu().tolist()
        batch_labels: List[str] = []
        for code_tuple in pred_codes:
            label = code_to_label.get(tuple(code_tuple), "unknown")
            batch_labels.append(label)
            epoch_label_counts[label] += 1
            for idx, code in enumerate(code_tuple):
                epoch_code_counts[idx][int(code)] += 1
        if collapse_window > 0:
            if len(pred_window) == pred_window.maxlen:
                old = pred_window.popleft()
                pred_total -= len(old)
                pred_counter.subtract(old)
                for key in list(pred_counter.keys()):
                    if pred_counter[key] <= 0:
                        del pred_counter[key]
            pred_window.append(batch_labels)
            pred_counter.update(batch_labels)
            pred_total += len(batch_labels)

        now = time.perf_counter()
        log_due = (step + 1) % log_every == 0 or now >= next_time_log
        if log_due:
            steps_since = (step + 1) - last_log_step
            step_time = (now - last_log_time) / max(steps_since, 1)
            avg_time = (now - train_start) / (step + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            amp_flag = "on" if amp_enabled else "off"
            logger.info(
                "train step=%s/%s loss=%.4f addr=%.4f gen=%.4f tuple=%.4f contrast=%.4f orbit=%.4f entropy=%.4f obj_prior=%.4f orbit_pairs=%s lr=%.6g grad_norm=%.4f amp=%s neg_hard=%s step_s=%.3f avg_s=%.3f",
                step + 1,
                steps,
                loss.item(),
                addr_loss.item(),
                gen_loss.item(),
                tuple_loss.item(),
                contrast_loss.item(),
                orbit_loss.item(),
                entropy_reg.item(),
                obj_prior_loss.item(),
                orbit_pairs_in_batch,
                current_lr,
                grad_norm,
                amp_flag,
                neg_from_hard,
                step_time,
                avg_time,
            )
            _get_console().print(
                f"step {step+1}/{steps} loss={loss.item():.4f} addr={addr_loss.item():.4f} gen={gen_loss.item():.4f} tuple={tuple_loss.item():.4f} contrast={contrast_loss.item():.4f} orbit={orbit_loss.item():.4f} entropy={entropy_reg.item():.4f} obj_prior={obj_prior_loss.item():.4f} orbit_pairs={orbit_pairs_in_batch}"
            )
            if global_code_matrix_np is not None and global_tuple2idx is not None and global_tuple_idx_to_answer:
                with torch.no_grad():
                    slot_logits_np = torch.stack(addr_logits, dim=1).detach().cpu().numpy()
                    gold_idx = tuple_targets.detach().cpu().tolist()
                    valid_rows = [i for i, idx in enumerate(gold_idx) if idx >= 0]
                    if valid_rows:
                        argmax_codes = np.argmax(slot_logits_np, axis=2)
                        argmax_idx = [
                            global_tuple2idx.get(tuple(argmax_codes[i].tolist()), -1) for i in valid_rows
                        ]
                        constrained_codes = constrained_decode_by_logprobs(slot_logits_np, global_code_matrix_np)
                        constrained_idx = [
                            global_tuple2idx.get(tuple(constrained_codes[i].tolist()), -1) for i in valid_rows
                        ]
                        gold_idx_valid = [gold_idx[i] for i in valid_rows]
                        gold_answers = [str(batch[i]["answer"]) for i in valid_rows]
                        argmax_answers = [
                            global_tuple_idx_to_answer[idx] if idx >= 0 else "__OOV__" for idx in argmax_idx
                        ]
                        constrained_answers = [
                            global_tuple_idx_to_answer[idx] if idx >= 0 else "__OOV__" for idx in constrained_idx
                        ]
                        code_em_argmax = (
                            sum(int(p == g) for p, g in zip(argmax_idx, gold_idx_valid)) / len(valid_rows)
                        )
                        code_em_constrained = (
                            sum(int(p == g) for p, g in zip(constrained_idx, gold_idx_valid)) / len(valid_rows)
                        )
                        answer_em_argmax = (
                            sum(int(exact_match(p, g)) for p, g in zip(argmax_answers, gold_answers)) / len(valid_rows)
                        )
                        answer_em_constrained = (
                            sum(int(exact_match(p, g)) for p, g in zip(constrained_answers, gold_answers))
                            / len(valid_rows)
                        )
                        logger.info(
                            "train_em step=%s code_em_argmax=%.4f answer_em_argmax=%.4f code_em_constrained=%.4f answer_em_constrained=%.4f n=%s",
                            step + 1,
                            code_em_argmax,
                            answer_em_argmax,
                            code_em_constrained,
                            answer_em_constrained,
                            len(valid_rows),
                        )
            if collapse_window > 0 and pred_total > 0:
                top_label, top_count = max(pred_counter.items(), key=lambda x: x[1])
                top1_freq = top_count / pred_total
                top1_qid = ""
                top1_qid_freq = 0.0
                qid_counts = {label: count for label, count in pred_counter.items() if _is_qid(label)}
                if qid_counts:
                    top1_qid, top1_qid_count = max(qid_counts.items(), key=lambda x: x[1])
                    top1_qid_freq = top1_qid_count / pred_total
                unique_pct = len(pred_counter) / pred_total
                if len(pred_counter) > 1:
                    probs = np.array(list(pred_counter.values()), dtype=np.float32) / pred_total
                    entropy = float(-(probs * np.log(probs + 1e-9)).sum() / math.log(len(pred_counter)))
                else:
                    entropy = 0.0
                collapse_detected = (
                    top1_freq >= _COLLAPSE_TOP1_THRESHOLD
                    or unique_pct <= _COLLAPSE_UNIQUE_THRESHOLD
                    or entropy <= _COLLAPSE_ENTROPY_THRESHOLD
                )
                collapse_active = obj_prior_weight > 0.0 and collapse_detected
                logger.info(
                    "collapse_diag window=%s top1_freq=%.4f top_label=%s top1_qid_freq=%.4f top1_qid=%s entropy=%.4f unique_pct=%.4f collapse_detected=%s collapse_active=%s",
                    collapse_window,
                    top1_freq,
                    top_label,
                    top1_qid_freq,
                    top1_qid,
                    entropy,
                    unique_pct,
                    collapse_detected,
                    collapse_active,
                )
            last_log_time = now
            last_log_step = step + 1
            if now >= next_time_log:
                next_time_log = now + time_log_interval

        if overfit_enabled and overfit_eval_every > 0 and (step + 1) % overfit_eval_every == 0:
            acc = _compute_code_em(
                model,
                overfit_examples,
                tokenizer,
                max_seq_len=max_seq_len,
                max_samples=overfit_eval_samples,
            )
            tuple_acc, slot_acc = _compute_code_accuracy(
                model,
                overfit_examples,
                tokenizer,
                max_seq_len=max_seq_len,
                max_samples=overfit_eval_samples,
            )
            pred_tuple_found_rate = _compute_pred_tuple_found_rate(
                model,
                overfit_examples,
                tokenizer,
                max_seq_len=max_seq_len,
                code_to_label=code_to_label,
                max_samples=overfit_eval_samples,
            )
            overfit_tuple_acc = tuple_acc
            top1_year = ""
            top1_freq = 0.0
            if tuple_pred_idx is not None and tuple_answers:
                preds = tuple_pred_idx.detach().cpu().tolist()
                valid_targets = [int(t) for t in tuple_targets.detach().cpu().tolist() if t >= 0]
                if preds:
                    counts = {}
                    for idx in preds:
                        answer = tuple_answers[int(idx)]
                        counts[answer] = counts.get(answer, 0) + 1
                    top1_year = max(counts.items(), key=lambda x: x[1])[0]
                    top1_freq = max(counts.values()) / max(len(preds), 1)
                    if valid_targets:
                        matches = [int(idx == tgt) for idx, tgt in zip(preds, valid_targets)]
                        overfit_tuple_acc = sum(matches) / max(len(matches), 1)
            logger.info(
                "overfit_acc step=%s acc=%.4f tuple_acc=%.4f slot_acc=%.4f pred_tuple_found_rate=%.4f top1_year=%s top1_freq=%.4f samples=%s",
                step + 1,
                acc,
                overfit_tuple_acc,
                slot_acc,
                pred_tuple_found_rate,
                top1_year,
                top1_freq,
                min(len(overfit_examples), overfit_eval_samples),
            )
            model.train()
            overfit_success_streak, should_exit = _should_overfit_early_exit(
                overfit_success_streak,
                acc,
                overfit_acc_threshold,
                overfit_acc_patience,
            )
            if should_exit:
                logger.info(
                    "overfit_early_exit step=%s acc=%.4f tuple_acc=%.4f pred_tuple_found_rate=%.4f streak=%s",
                    step + 1,
                    acc,
                    overfit_tuple_acc,
                    pred_tuple_found_rate,
                    overfit_success_streak,
                )
                break

        if (step + 1) % steps_per_epoch == 0:
            epoch_idx += 1
            total = sum(epoch_label_counts.values())
            top_labels = sorted(epoch_label_counts.items(), key=lambda x: (-x[1], x[0]))[:5]
            logger.info("epoch_end epoch=%s top_pred_labels=%s total=%s", epoch_idx, top_labels, total)
            for idx, counts in enumerate(epoch_code_counts):
                total_counts = sum(counts.values())
                if total_counts == 0:
                    entropy = 0.0
                else:
                    probs = np.array([v / total_counts for v in counts.values()], dtype=np.float32)
                    entropy = float(-(probs * np.log(probs + 1e-9)).sum() / math.log(int(model_cfg["K"])))
                logger.info("epoch_end epoch=%s code_entropy codebook=%s entropy=%.4f", epoch_idx, idx, entropy)
            epoch_label_counts = Counter()
            epoch_code_counts = [Counter() for _ in range(int(model_cfg["m"]))]

    logger.info("training done time=%.2fs", time.perf_counter() - train_start)
    if overfit_lama_subset_name and tuple_code_matrix is not None:
        model.eval()
        run_id = overfit_run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        source = "overfit-lama-years" if overfit_year_only and overfit_lama_subset_name == "google_re" else f"overfit-lama-{overfit_lama_subset_name}"
        answer_filter_used = "qid" if overfit_qid_only else ("year" if overfit_year_only else "none")
        answer_filter_row_count = len(tuple_answers) if answer_filter_used != "none" else 0
        report_train = _run_overfit_report_for_examples(
            model,
            overfit_examples,
            tokenizer,
            max_seq_len,
            tuple_code_matrix,
            tuple_answers,
            run_id,
            source,
            answer_filter_used,
            gold_year_rate if overfit_lama_subset_name else 0.0,
            gold_qid_rate if overfit_lama_subset_name else 0.0,
            answer_filter_row_count,
        )
        report_holdout = _run_overfit_report_for_examples(
            model,
            holdout_examples,
            tokenizer,
            max_seq_len,
            tuple_code_matrix,
            tuple_answers,
            run_id,
            source,
            answer_filter_used,
            gold_year_rate if overfit_lama_subset_name else 0.0,
            gold_qid_rate if overfit_lama_subset_name else 0.0,
            answer_filter_row_count,
        )
        combined = _compose_overfit_split_report(report_train, report_holdout)
        report_tag = "overfit-lama-years" if overfit_year_only and overfit_lama_subset_name == "google_re" else f"overfit-lama-{overfit_lama_subset_name}"
        report_dir = out_dir / "reports" / report_tag / run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_meta = {}
        tokenizer_path = out_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer_meta = _save_overfit_tokenizer(tokenizer_path, report_dir)
        overfit_ckpt_path = report_dir / "model_overfit.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "step": steps,
                "run_id": run_id,
                "source": "overfit",
            },
            overfit_ckpt_path,
        )
        print(f"overfit_checkpoint={overfit_ckpt_path.resolve()}")
        (report_dir / "report_train.json").write_text(json.dumps(report_train, indent=2))
        (report_dir / "report_holdout.json").write_text(json.dumps(report_holdout, indent=2))
        combined.update(tokenizer_meta)
        report_path = _write_overfit_report(report=combined, report_dir=report_dir, out_dir=out_dir)
        logger.info("overfit_report_saved path=%s", report_path)
        model.train()
    _save_checkpoint(ckpt_dir, step=steps, model=model, optimizer=optimizer, rng=rng, epoch_idx=epoch_idx)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    logger.info("checkpoint_saved path=%s", ckpt_dir / "model.pt")
    if not overfit_enabled:
        logger.info("eval start")
        report = {
            "code_em": _compute_code_em(model, examples, tokenizer, max_seq_len=max_seq_len),
            "answer_em": _compute_answer_em(
                model,
                examples,
                tokenizer,
                max_seq_len=max_seq_len,
                max_new_tokens=int(inf_cfg["max_gen_tokens"]),
            ),
            "orbit_consistency": _compute_orbit_consistency(model, records, tokenizer, max_seq_len=max_seq_len),
            "negative_margin": _compute_negative_margin(model, records, answer_codes),
        }
        logger.info("eval done")

        out_dir = Path("out")
        ckpt_dir = out_dir / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        logger.info("checkpoint_saved path=%s", ckpt_dir / "model.pt")

        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("report_saved path=%s", report_path)
        _get_console().print(f"Saved report to {report_path}")


if __name__ == "__main__":
    app()
