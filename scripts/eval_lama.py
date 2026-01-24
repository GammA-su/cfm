from __future__ import annotations

import hashlib
import json
import logging
import sys
import tarfile
import threading
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from collections import Counter

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
_set_startup_stage("import safetensors")
from safetensors.torch import load_file
_set_startup_stage("import typer")
import typer
_set_startup_stage("import yaml")
import yaml
_set_startup_stage("import datasets")
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager
_set_startup_stage("import pandas")
import pandas as pd
_set_startup_stage("import rich.console")
from rich.console import Console

from forge_omega_500.data.rvq import load_code_to_label, lookup_code_label
from scripts.ckpt_io import get_ckpt_step, load_checkpoint
from codebook import (
    build_code_matrix,
    build_reverse_codebook,
    candidate_rows_from_logits,
    constrained_decode_by_logprobs,
    constrained_decode_candidates_by_logprobs,
    build_slot_index,
    code_vocab_size_from_df,
    codes_from_answer,
    decode_codes,
    ensure_inverted_index,
    is_year_literal,
    load_answer_codebook,
    normalize_lama_answer,
    normalize_wikidata_qid,
    year_row_indices,
)
from metrics_kbqa import (
    code_tuple_em,
    exact_match,
    negative_margin_stats,
    orbit_consistency,
)

from forge_omega_500.eval.metrics import calibration_curve
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

console = Console()
logger = setup_logger("eval_lama")
app = typer.Typer(add_completion=False)
_FILE_LOG_READY = False

_set_startup_stage("ready")

_DEFAULT_LAMA_SUBSETS = ("google_re", "trex", "conceptnet", "squad")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_tokenizer_path(checkpoint_path: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    candidate = checkpoint_path.parent / "tokenizer.json"
    if candidate.exists():
        return candidate
    return Path("out") / "tokenizer.json"


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "model_overfit.pt"
        if direct.is_file():
            return direct
        candidates = list(path.rglob("model_overfit.pt"))
        if candidates:
            return max(candidates, key=lambda item: item.stat().st_mtime)
        hint = f"find {path} -name 'model_overfit.pt' -type f -printf '%T@ %p\\n' | sort -n | tail -1"
        raise ValueError(
            "checkpoint must be a file; got directory: "
            f"{path}\nLooked for: {direct} and {path}/**/model_overfit.pt\nHint: {hint}"
        )
    raise ValueError(f"checkpoint path does not exist: {path}")


def _ckpt_vocab_size(state: dict) -> int | None:
    for key in ("backbone.token_emb.weight", "backbone.emb.weight"):
        weight = state.get(key)
        if weight is not None:
            try:
                return int(weight.shape[0])
            except Exception:
                continue
    return None


def _ensure_file_logger(out_dir: Path) -> None:
    global _FILE_LOG_READY
    if _FILE_LOG_READY:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_lama.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("file_logging_enabled path=%s", log_path)
    _FILE_LOG_READY = True

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"
_MASK_TOKENS = ("[MASK]", "<mask>", "MASK")


def _download_lama_files(cache_dir: Path, local_files_only: bool) -> tuple[Path, Path]:
    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    logger.info("download_lama start url=%s", LAMA_DATA_URL)
    archive_path = Path(dl_manager.download(LAMA_DATA_URL))
    logger.info("download_lama done path=%s", archive_path)
    logger.info("download_relations start url=%s", LAMA_RELATIONS_URL)
    relations_path = Path(dl_manager.download(LAMA_RELATIONS_URL))
    logger.info("download_relations done path=%s", relations_path)
    return archive_path, relations_path


def _load_relations(relations_path: Path) -> Dict[str, Dict[str, str]]:
    relations: Dict[str, Dict[str, str]] = {}
    with relations_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            rel = str(data.get("relation", ""))
            if rel:
                relations[rel] = data
    return relations


def _iter_lama_records(
    cfg_name: str,
    archive_path: Path,
    relations_path: Path,
    max_samples: int,
) -> Iterable[Dict[str, object]]:
    if cfg_name == "trex":
        relations = _load_relations(relations_path)
        patterns = ["TREx/*"]
    elif cfg_name == "google_re":
        relations = {}
        patterns = [
            "Google_RE/date_of_birth_test.jsonl",
            "Google_RE/place_of_birth_test.jsonl",
            "Google_RE/place_of_death_test.jsonl",
        ]
    elif cfg_name == "conceptnet":
        relations = {}
        patterns = ["ConceptNet/test.jsonl"]
    elif cfg_name == "squad":
        relations = {}
        patterns = ["Squad/test.jsonl"]
    else:
        return []

    yielded = 0
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if not any(fnmatch(name, pat) for pat in patterns):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            for raw in fileobj:
                data = json.loads(raw.decode("utf-8"))
                if cfg_name == "trex":
                    pred = relations.get(str(data.get("predicate_id", "")), {})
                    predicate_id = _first_value(data, ("predicate_id", "relation_id", "relation"))
                    sub_uri = _first_value(data, ("sub_uri", "subj_uri", "subject_uri"))
                    obj_uri = _first_value(data, ("obj_uri", "object_uri"))
                    gold_label = _first_value(data, ("obj_label", "object_label", "obj", "answer"))
                    for evidence in data.get("evidences", []):
                        yield {
                            "masked_sentence": _first_value(evidence, ("masked_sentence", "masked_sentences")),
                            "template": str(pred.get("template", "")),
                            "predicate_id": predicate_id,
                            "sub_label": str(data.get("sub_label", "")),
                            "obj_label": str(data.get("obj_label", "")),
                            "sub_uri": sub_uri,
                            "obj_uri": obj_uri,
                            "gold_label": gold_label,
                        }
                        yielded += 1
                        if max_samples and yielded >= max_samples:
                            return
                else:
                    record = dict(data)
                    record["masked_sentence"] = _first_value(data, ("masked_sentence", "masked_sentences"))
                    predicate_id = _first_value(data, ("predicate_id", "relation_id", "relation"))
                    record["predicate_id"] = predicate_id
                    if cfg_name == "google_re":
                        relation_value = _first_value(data, ("relation",))
                        if not relation_value and predicate_id:
                            record["relation"] = predicate_id
                        if not predicate_id:
                            record["relation"] = ""
                            record["literal_only"] = True
                    record["sub_uri"] = _first_value(data, ("sub_uri", "subj_uri", "subject_uri"))
                    record["obj_uri"] = _first_value(data, ("obj_uri", "object_uri"))
                    record["gold_label"] = _first_value(data, ("obj_label", "object_label", "obj", "answer"))
                    yield record
                    yielded += 1
                    if max_samples and yielded >= max_samples:
                        return


def _load_lama_fallback(cfg_name: str, cache_dir: Path, local_files_only: bool, max_samples: int) -> List[Dict[str, object]]:
    logger.info("fallback_loader start cfg=%s max_samples=%s", cfg_name, max_samples)
    archive_path, relations_path = _download_lama_files(cache_dir, local_files_only=local_files_only)
    records = list(_iter_lama_records(cfg_name, archive_path, relations_path, max_samples=max_samples))
    logger.info("fallback_loader done cfg=%s records=%s", cfg_name, len(records))
    return records


def _replace_mask_tokens(text: str) -> str:
    for token in _MASK_TOKENS:
        text = text.replace(token, "____")
    return text


def _iter_field_values(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        items: List[str] = []
        for item in value:
            items.extend(_iter_field_values(item))
        return items
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _iter_field_values(value.item())
        return _iter_field_values(value.tolist())
    text = str(value).strip()
    return [text] if text else []


def _first_value(row: Dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        values = _iter_field_values(row.get(key))
        if values:
            return values[0]
    return ""


def _extract_masked_sentence(example: Dict[str, object]) -> Tuple[str, str]:
    values = _iter_field_values(example.get("masked_sentence"))
    if values:
        return values[0], "masked_sentence"
    values = _iter_field_values(example.get("masked_sentences"))
    if values:
        return values[0], "masked_sentences"
    values = _iter_field_values(example.get("masked_prompt"))
    if values:
        return values[0], "masked_prompt"
    values = _iter_field_values(example.get("sentence"))
    if values:
        return values[0], "sentence"
    return "", ""


def _extract_subject(example: Dict[str, object]) -> str:
    for key in ("sub_label", "subject_label", "sub", "subject"):
        values = _iter_field_values(example.get(key))
        if values:
            return values[0]
    return ""


def _build_prompt_with_field(example: Dict[str, object]) -> Tuple[str | None, str]:
    masked_sentence, field = _extract_masked_sentence(example)
    if masked_sentence:
        prompt = "Fill in the blank: " + _replace_mask_tokens(masked_sentence)
        return prompt, field or "masked_sentence"
    template_values = _iter_field_values(example.get("template"))
    if template_values:
        template = template_values[0]
        subject = _extract_subject(example)
        if not subject:
            return None, "template"
        prompt = str(template).replace("[X]", subject).replace("[Y]", "____")
        prompt = _replace_mask_tokens(prompt)
        return "Fill in the blank: " + prompt, "template"
    subject = _extract_subject(example)
    if not subject:
        return None, "subject"
    return f"What is the answer for {subject}?", "subject"


def _build_prompt(example: Dict[str, object]) -> str:
    prompt, _ = _build_prompt_with_field(example)
    return prompt or ""


def _gold_answer(example: Dict[str, object]) -> str:
    for key in ["obj_label", "obj", "object_label", "answer"]:
        values = _iter_field_values(example.get(key))
        if values:
            return values[0]
    return ""


def _normalize_answer(text: str) -> str:
    normalized = text.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1].strip()
    normalized = normalized.rstrip(".,;:!?").strip()
    return normalized


def _normalize_for_em(pred: str, gold: str) -> Tuple[str, str]:
    pred_norm = _normalize_answer(pred).lower()
    gold_norm = _normalize_answer(gold).lower()
    if gold_norm:
        gold_tokens = gold_norm.split()
        if len(gold_tokens) == 1:
            pred_tokens = pred_norm.split()
            pred_norm = pred_tokens[0] if pred_tokens else ""
    return pred_norm, gold_norm


def _is_numeric_literal(text: str) -> bool:
    normalized = _normalize_answer(text)
    return normalized.isdigit()


def _load_qid_to_label(path: Path) -> Dict[str, str]:
    if not path.exists():
        logger.warning("qid_to_label_missing path=%s", path)
        return {}
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning("qid_to_label_load_failed path=%s err=%s", path, exc)
        return {}
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        qid = str(row.get("qid", "")).strip()
        label = str(row.get("label", "")).strip()
        if qid and label and qid not in mapping:
            mapping[qid] = label
    logger.info("qid_to_label_loaded entries=%s", len(mapping))
    return mapping


def _map_pred_to_label(pred: str, qid_to_label: Dict[str, str]) -> Tuple[str, str]:
    raw = pred.strip()
    if not raw:
        return pred, ""
    qid = normalize_wikidata_qid(raw) or ""
    if qid:
        label = qid_to_label.get(qid, "")
        if label:
            return label, qid
    return pred, qid


def _answer_is_qid(answer: str) -> bool:
    return bool(answer and normalize_wikidata_qid(answer) == answer)


def _answer_type_counts(answers: Iterable[str]) -> Dict[str, int]:
    counts = {"qid": 0, "literal": 0}
    for answer in answers:
        if not answer:
            continue
        if _answer_is_qid(answer):
            counts["qid"] += 1
        else:
            counts["literal"] += 1
    return counts


def _code_accuracy(pred_codes: List[int], gold_codes: List[int]) -> Tuple[int, int, int]:
    slot_total = min(len(pred_codes), len(gold_codes))
    if slot_total == 0:
        return 0, 0, 0
    slot_correct = sum(1 for idx in range(slot_total) if pred_codes[idx] == gold_codes[idx])
    tuple_correct = 1 if slot_correct == slot_total and len(pred_codes) == len(gold_codes) else 0
    return tuple_correct, slot_correct, slot_total


def _validate_code_space(pred_codes: List[int], code_vocab_size: int) -> Tuple[int, int]:
    if not pred_codes:
        return 0, 0
    codes_min = min(pred_codes)
    codes_max = max(pred_codes)
    if codes_min < 0 or codes_max >= code_vocab_size:
        raise ValueError(
            "pred_codes out of codebook range; likely token IDs or wrong head. "
            f"min={codes_min} max={codes_max} code_vocab_size={code_vocab_size}"
        )
    return codes_min, codes_max



def _constrained_decode(
    logp: torch.Tensor,
    code_mat: np.ndarray,
    answers: List[str],
    chunk_size: int = 4096,
) -> Tuple[List[str], np.ndarray]:
    if logp.numel() == 0 or code_mat.size == 0:
        return [], np.zeros((0, 0), dtype=np.int64)
    if logp.dim() != 3:
        raise ValueError(f"logp must have shape [B,S,C], got {tuple(logp.shape)}")
    batch_size, slot_count, _ = logp.shape
    best_scores = torch.full((batch_size,), -float("inf"), device=logp.device)
    best_idx = torch.zeros((batch_size,), dtype=torch.long, device=logp.device)
    total = code_mat.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = torch.tensor(code_mat[start:end], device=logp.device, dtype=torch.long)
        scores = None
        for slot in range(slot_count):
            slot_logp = logp[:, slot, :]
            idx = chunk[:, slot].unsqueeze(0).expand(batch_size, -1)
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
    pred_answers = [answers[int(idx)] for idx in best_idx_cpu]
    pred_codes = code_mat[best_idx_cpu]
    return pred_answers, pred_codes

def _select_nearest_candidate(
    candidates: Iterable[str],
    answer_codes: Dict[str, List[int]],
    pred_codes: List[int],
) -> str | None:
    best_answer = None
    best_dist = None
    for answer in candidates:
        codes = answer_codes.get(answer)
        if codes is None:
            continue
        max_len = max(len(codes), len(pred_codes))
        dist = 0
        for idx in range(max_len):
            a = codes[idx] if idx < len(codes) else None
            b = pred_codes[idx] if idx < len(pred_codes) else None
            if a != b:
                dist += 1
        if best_dist is None or dist < best_dist or (dist == best_dist and answer < best_answer):
            best_dist = dist
            best_answer = answer
    return best_answer


def _project_oov(
    pred_codes: List[int],
    slot_index: List[Dict[int, List[str]]],
    answer_codes: Dict[str, List[int]],
    slot_conf: List[float] | None = None,
) -> str | None:
    if not pred_codes or not slot_index or len(pred_codes) != len(slot_index):
        return None
    slot_order = list(range(len(pred_codes)))
    if slot_conf and len(slot_conf) == len(pred_codes):
        slot_order = [idx for idx, _ in sorted(enumerate(slot_conf), key=lambda item: item[1])]
    for drop_count in range(0, len(pred_codes) + 1):
        drop_slots = set(slot_order[:drop_count])
        candidates = None
        for idx in range(len(pred_codes)):
            if idx in drop_slots:
                continue
            answers = slot_index[idx].get(pred_codes[idx], [])
            if candidates is None:
                candidates = set(answers)
            else:
                candidates &= set(answers)
            if not candidates:
                break
        if candidates:
            projected = _select_nearest_candidate(candidates, answer_codes, pred_codes)
            if projected:
                return projected
    return None


def _score_answer(pred_answer: str, gold_answer: str) -> Tuple[float, float | None, float | None]:
    acc = 1.0 if pred_answer and gold_answer and pred_answer == gold_answer else 0.0
    if _answer_is_qid(gold_answer):
        return acc, acc, None
    if gold_answer:
        return acc, None, acc
    return 0.0, None, None


def _decode_pred(tokenizer: SimpleTokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids)


def _build_code_to_answer_map(path: Path) -> Dict[Tuple[int, ...], str]:
    if not path.exists():
        logger.warning("code_to_label_missing path=%s", path)
        return {}
    try:
        mapping = load_code_to_label(path)
    except Exception as exc:
        logger.warning("code_to_label_load_failed path=%s err=%s", path, exc)
        return {}
    logger.info("code_to_label_loaded entries=%s", len(mapping))
    return mapping


def _fallback_token_from_logits(
    model: CFMModel,
    prompt_ids: torch.Tensor,
    prompt_masks: torch.Tensor,
    codes: torch.Tensor,
    tokenizer: SimpleTokenizer,
) -> str:
    batch_size = prompt_ids.size(0)
    generated = torch.full((batch_size, 1), tokenizer.bos_id, device=prompt_ids.device, dtype=torch.long)
    attention = torch.ones_like(generated)
    input_ids = torch.cat([prompt_ids, torch.full_like(prompt_ids[:, :1], tokenizer.sep_id), generated], dim=1)
    attn = torch.cat([prompt_masks, torch.ones_like(prompt_ids[:, :1]), attention], dim=1)
    if model.max_seq_len and input_ids.size(1) > model.max_seq_len:
        overflow = input_ids.size(1) - model.max_seq_len
        input_ids = input_ids[:, overflow:]
        attn = attn[:, overflow:]
    _, logits = model.forward_generation(input_ids, attn, codes)
    next_logits = logits[:, -1]
    topk = torch.topk(next_logits, k=min(10, next_logits.size(-1)), dim=-1)
    skip_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id}
    for token_id in topk.indices[0].tolist():
        if token_id not in skip_ids:
            return tokenizer.inv_vocab.get(token_id, tokenizer.unk_token)
    top_id = int(topk.indices[0][0].item())
    return tokenizer.inv_vocab.get(top_id, tokenizer.unk_token)


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to config YAML"),
    ckpt: Path = typer.Option(
        Path("out/ckpt/model.pt"),
        help="Checkpoint path",
    ),
    checkpoint: Path | None = typer.Option(
        None,
        "--checkpoint",
        help="Override checkpoint path",
    ),
    tokenizer_path: Path | None = typer.Option(
        None,
        "--tokenizer",
        help="Override tokenizer path",
    ),
    subset: str = typer.Option(
        "all",
        help="LAMA subset: trex, google_re, conceptnet, squad, or all",
    ),
    limit: int | None = typer.Option(
        None,
        help="Max samples to evaluate (default: eval.max_samples from config)",
    ),
    max_samples: int | None = typer.Option(
        None,
        help="Override eval.max_samples from config",
    ),
    store_samples: int = typer.Option(
        5,
        help="How many sample predictions to store per subset",
    ),
    pred_hist_topk: int = typer.Option(
        50,
        help="Top-k size for prediction histogram",
    ),
    decode: str = typer.Option(
        "constrained",
        help="Decode mode: argmax or constrained",
    ),
    use_candidates: bool | None = typer.Option(
        None,
        "--use-candidates/--no-candidates",
        help="Override eval.use_candidates",
    ),
    force_answer_filter: str = typer.Option(
        "auto",
        help="Force answer filter: auto, none, year, qid",
    ),
    orbit_key: str = typer.Option(
        "entity",
        help="Key to group orbits for consistency (fallback prompt or disabled)",
    ),
    margin_metrics: bool = typer.Option(
        True,
        "--margin-metrics/--no-margin-metrics",
        help="Compute negative margin diagnostics",
    ),
    project_oov: bool = typer.Option(
        False,
        help="Project OOV code tuples to nearest codebook answer",
    ),
    out_report: Path = typer.Option(
        Path("out/lama_report.json"),
        "--out-report",
        help="Output report path",
    ),
) -> None:
    cfg = yaml.safe_load(config.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)

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

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    if max_samples is not None:
        eval_cfg["max_samples"] = int(max_samples)
    if use_candidates is not None:
        eval_cfg["use_candidates"] = bool(use_candidates)
    force_answer_filter = force_answer_filter.strip().lower()
    if force_answer_filter not in {"auto", "none", "year", "qid"}:
        raise ValueError("force-answer-filter must be one of auto, none, year, qid")
    if checkpoint is not None:
        ckpt = checkpoint
    inf_cfg = cfg.get("inference", {})

    out_dir = Path("out")
    _ensure_file_logger(out_dir)
    out_path = out_report
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = _resolve_checkpoint_path(Path(ckpt))
    resolved_tokenizer_path = _resolve_tokenizer_path(ckpt_path, str(tokenizer_path) if tokenizer_path else None)
    if not resolved_tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {resolved_tokenizer_path}")
    logger.info("load_tokenizer start path=%s", resolved_tokenizer_path)
    t0 = time.perf_counter()
    tokenizer = SimpleTokenizer.load(resolved_tokenizer_path)
    tokenizer_vocab_size = len(tokenizer.vocab)
    tokenizer_sha256 = _sha256_file(resolved_tokenizer_path)
    logger.info(
        "load_tokenizer done vocab_size=%s sha256=%s time=%.2fs",
        tokenizer_vocab_size,
        tokenizer_sha256,
        time.perf_counter() - t0,
    )

    codes_dir = Path(data_cfg["codes_dir"])
    codebook_path = codes_dir / "codebooks.safetensors"
    logger.info("load_codebooks_meta start path=%s", codebook_path)
    codebook_tensors = load_file(str(codebook_path))
    codebooks = codebook_tensors["codebooks"]
    codebook_m = int(codebooks.shape[0])
    codebook_K = int(codebooks.shape[1])
    cfg_m = int(model_cfg["m"])
    cfg_K = int(model_cfg["K"])
    if codebook_m != cfg_m or codebook_K != cfg_K:
        logger.warning(
            "override_model_codebooks_shape config_m=%s config_K=%s codebook_m=%s codebook_K=%s",
            cfg_m,
            cfg_K,
            codebook_m,
            codebook_K,
        )
    logger.info("init_model start")
    t0 = time.perf_counter()
    model = CFMModel.from_codebooks(
        codebook_path,
        vocab_size=len(tokenizer.vocab),
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        m=codebook_m,
        K=codebook_K,
        backbone=model_cfg.get("backbone", "tiny"),
        hf_model_name=model_cfg.get("hf_model_name"),
    )
    logger.info("init_model done time=%.2fs", time.perf_counter() - t0)
    logger.info("load_checkpoint start path=%s", ckpt_path)
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    ckpt_step = get_ckpt_step(checkpoint)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    ckpt_vocab_size = _ckpt_vocab_size(state) if isinstance(state, dict) else None
    if ckpt_vocab_size is not None and ckpt_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "checkpoint/tokenizer vocab mismatch: "
            f"checkpoint_vocab_size={ckpt_vocab_size} tokenizer_vocab_size={tokenizer_vocab_size} "
            f"tokenizer_path={resolved_tokenizer_path}"
        )
    model.load_state_dict(state)
    logger.info("load_checkpoint done")
    logger.info("eval_ckpt path=%s step=%s", ckpt_path, ckpt_step)
    model.eval()

    device = torch.device("cuda" if torch_info.get("torch_cuda") else "cpu")
    model.to(device)
    logger.info("model_to_device done device=%s", device.type)

    code_to_answer = _build_code_to_answer_map(codes_dir / "code_to_label.parquet")
    answer_codebook_df = pd.read_parquet(codes_dir / "answer_codes.parquet")
    answer_codebook = load_answer_codebook(codes_dir / "answer_codes.parquet")
    reverse_codebook = build_reverse_codebook(answer_codebook_df)
    slot_index = build_slot_index(answer_codebook_df)
    code_mat = build_code_matrix(answer_codebook_df)
    code_vocab_size = code_vocab_size_from_df(answer_codebook_df)
    year_rows = year_row_indices(answer_codebook_df)
    year_code_mat = code_mat[year_rows] if year_rows.size else np.zeros((0, 0), dtype=np.int64)
    qid_rows = np.asarray(
        [i for i, val in enumerate(answer_codebook_df.get("answer", []).astype(str).tolist()) if normalize_wikidata_qid(val)],
        dtype=np.int64,
    )
    qid_code_mat = code_mat[qid_rows] if qid_rows.size else np.zeros((0, 0), dtype=np.int64)
    use_candidates = bool(eval_cfg.get("use_candidates", False))
    candidate_topk_per_slot = int(eval_cfg.get("candidate_topk_per_slot", 16))
    candidate_max = int(eval_cfg.get("candidate_max", 4096))
    index_path = Path(eval_cfg.get("index_path", "data/codes/answer_codes.index.npz"))
    index_offsets = None
    index_flat = None
    if use_candidates:
        index_offsets, index_flat = ensure_inverted_index(index_path, code_mat, code_vocab_size)
    year_index_offsets = None
    year_index_flat = None
    if use_candidates and year_rows.size:
        year_index_offsets, year_index_flat = ensure_inverted_index(
            Path(str(index_path) + ".years"),
            year_code_mat,
            code_vocab_size,
        )
    qid_index_offsets = None
    qid_index_flat = None
    if use_candidates and qid_rows.size:
        qid_index_offsets, qid_index_flat = ensure_inverted_index(
            Path(str(index_path) + ".qids"),
            qid_code_mat,
            code_vocab_size,
        )
    factbank_dir = Path(data_cfg["factbank_dir"])
    qid_to_label = _load_qid_to_label(factbank_dir / "qid_to_label.parquet")

    subset = subset.strip().lower()
    valid_subsets = set(_DEFAULT_LAMA_SUBSETS)
    if subset not in valid_subsets and subset != "all":
        raise typer.BadParameter("subset must be one of trex, google_re, conceptnet, squad, all")

    configured_subsets = eval_cfg.get("subsets", list(_DEFAULT_LAMA_SUBSETS))
    if isinstance(configured_subsets, str):
        configured_subsets = [configured_subsets]
    base_subsets = [str(name).strip().lower() for name in configured_subsets if str(name).strip()]
    invalid = sorted({name for name in base_subsets if name not in valid_subsets})
    if invalid:
        raise ValueError(f"Invalid eval.subsets entries: {invalid}")
    if subset != "all":
        base_subsets = [name for name in base_subsets if name == subset]
        if not base_subsets:
            raise typer.BadParameter(f"subset {subset} not enabled in eval.subsets")

    limit_samples = int(eval_cfg["max_samples"]) if limit is None else int(limit)
    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))
    use_fallback = False
    logger.info("discover_lama_configs start")
    try:
        available = get_dataset_config_names("facebook/lama")
        target_configs = [c for c in base_subsets if c in available]
    except Exception as exc:
        logger.warning("get_dataset_config_names failed: %s; falling back to tar loader", exc)
        target_configs = list(base_subsets)
        use_fallback = True
    logger.info("discover_lama_configs done use_fallback=%s", use_fallback)
    report = {}
    logger.info("lama_configs=%s", ",".join(target_configs))

    max_gen_tokens = int(inf_cfg.get("max_gen_tokens", 8))
    if max_gen_tokens < 4:
        logger.warning("max_gen_tokens_too_small value=%s clamped=4", max_gen_tokens)
        max_gen_tokens = 4
    abstain_on_empty = bool(inf_cfg.get("abstain_on_empty", False))

    store_samples = max(0, int(store_samples))
    pred_hist_topk = max(0, int(pred_hist_topk))

    for cfg_name in target_configs:
        max_samples = limit_samples
        logger.info("load_dataset start cfg=%s", cfg_name)
        t0 = time.perf_counter()
        if use_fallback:
            ds = _load_lama_fallback(cfg_name, cache_dir, local_files_only, max_samples)
        else:
            ds = load_dataset("facebook/lama", cfg_name, split="train")
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
        logger.info("load_dataset done cfg=%s samples=%s time=%.2fs", cfg_name, len(ds), time.perf_counter() - t0)
        gold_year_total = 0
        gold_qid_total = 0
        gold_total = 0
        for example in ds:
            gold = _gold_answer(example)
            gold_uri = _first_value(example, ("obj_uri", "object_uri"))
            gold_answer = normalize_lama_answer(gold, gold_uri)
            if gold_answer:
                gold_total += 1
                if is_year_literal(gold_answer):
                    gold_year_total += 1
                if normalize_wikidata_qid(gold_answer):
                    gold_qid_total += 1
        gold_year_rate = gold_year_total / gold_total if gold_total else 0.0
        gold_qid_rate = gold_qid_total / gold_total if gold_total else 0.0
        if force_answer_filter == "year":
            answer_filter_used = "year" if year_rows.size else "none"
        elif force_answer_filter == "qid":
            answer_filter_used = "qid" if qid_rows.size else "none"
        elif force_answer_filter == "none":
            answer_filter_used = "none"
        else:
            answer_filter_used = "year" if gold_year_rate >= 0.90 and year_rows.size else "none"

        if answer_filter_used == "year":
            decode_matrix = year_code_mat
            decode_offsets = year_index_offsets
            decode_flat = year_index_flat
        elif answer_filter_used == "qid":
            decode_matrix = qid_code_mat
            decode_offsets = qid_index_offsets
            decode_flat = qid_index_flat
        else:
            decode_matrix = code_mat
            decode_offsets = index_offsets
            decode_flat = index_flat

        logger.info(
            "subset_runtime cfg=%s device=%s gpu_enabled=%s gpu_count=%s faiss_available=%s faiss_gpu_enabled=%s",
            cfg_name,
            device.type,
            torch_info.get("torch_cuda"),
            torch_info.get("gpu_count"),
            faiss_info.get("faiss_available"),
            faiss_info.get("faiss_gpu_available"),
        )
        logger.info("eval_config=%s samples=%s", cfg_name, len(ds))
        confidences = []
        accuracies = []
        accuracies_uri = []
        accuracies_text = []
        predictions = []
        gold_hist = Counter()
        pred_collapse_hist = Counter()
        gold_answers: List[str] = []
        gold_in_codebook = 0
        pred_in_codebook = 0
        pred_total = 0
        skipped_no_code = 0
        pred_oov_count = 0
        pred_tuple_found = 0
        tuple_correct = 0
        tuple_total = 0
        slot_correct = 0
        slot_total = 0
        projected_oov_count = 0
        argmax_pred_total = 0
        argmax_pred_oov_count = 0
        argmax_pred_tuple_found = 0
        argmax_tuple_correct = 0
        argmax_tuple_total = 0
        argmax_slot_correct = 0
        argmax_slot_total = 0
        constrained_pred_total = 0
        constrained_pred_oov_count = 0
        constrained_pred_tuple_found = 0
        constrained_tuple_correct = 0
        constrained_tuple_total = 0
        constrained_slot_correct = 0
        constrained_slot_total = 0
        answer_em_total = 0
        answer_em_argmax_correct = 0
        answer_em_constrained_correct = 0
        code_em_total = 0
        code_em_argmax_correct = 0
        code_em_constrained_correct = 0
        gold_answer_nonempty = 0
        gold_codes_found = 0
        pred_codes_in_range = 0
        gold_codes_min = None
        gold_codes_max = None
        gold_codes_shape = None
        orbit_preds: Dict[str, List[str]] = {}
        margin_logits: List[np.ndarray] = []
        margin_golds: List[np.ndarray] = []
        pred_codes_min = None
        pred_codes_max = None
        pred_codes_shape = None
        pred_codes_head: List[List[int]] = []
        gold_codes_head: List[List[int] | None] = []
        pred_codes_all: List[List[int]] = []
        gold_codes_all: List[List[int]] = []
        candidate_sizes: List[int] = []
        candidate_hit_total = 0
        candidate_hit_found = 0
        tuple_to_row = {tuple(row.tolist()): idx for idx, row in enumerate(code_mat)}
        decode_row_to_local = None
        if answer_filter_used == "year" and year_rows.size:
            decode_row_to_local = {int(row): idx for idx, row in enumerate(year_rows.tolist())}
        elif answer_filter_used == "qid" and qid_rows.size:
            decode_row_to_local = {int(row): idx for idx, row in enumerate(qid_rows.tolist())}
        log_every = int(eval_cfg.get("log_every", 50))
        time_log_interval = float(eval_cfg.get("time_log_interval", 30.0))
        eval_start = time.perf_counter()
        last_log_time = eval_start
        last_log_idx = 0
        next_time_log = eval_start + time_log_interval
        processed = 0
        decode_mode = decode.lower().strip()

        for idx, example in enumerate(ds, start=1):
            prompt, prompt_field = _build_prompt_with_field(example)
            if not prompt:
                logger.warning(
                    "skip_sample cfg=%s index=%s prompt_field=%s keys=%s",
                    cfg_name,
                    idx,
                    prompt_field,
                    sorted(example.keys()),
                )
                continue
            logger.info("prompt_field_used cfg=%s index=%s field=%s", cfg_name, idx, prompt_field)
            prompt_ids = [[tokenizer.bos_id] + tokenizer.encode(prompt)]
            prompt_ids, prompt_masks = pad_sequences(prompt_ids, tokenizer.pad_id, max_len=int(model_cfg["max_seq_len"]))
            prompt_ids = prompt_ids.to(device)
            prompt_masks = prompt_masks.to(device)

            addr_logits, conf, _ = model.encode_prompt(prompt_ids, prompt_masks)
            generated, codes = model.generate_answer(
                prompt_ids,
                prompt_masks,
                max_new_tokens=max_gen_tokens,
                eos_id=tokenizer.eos_id,
                bos_id=tokenizer.bos_id,
                sep_id=tokenizer.sep_id,
            )
            generated_token_count = max(0, int(generated.size(1)) - 1)
            pred_text = _decode_pred(tokenizer, generated[0].tolist())
            pred_codes_np = np.asarray(codes.cpu().numpy(), dtype=np.int64)
            if pred_codes_shape is None:
                pred_codes_shape = list(pred_codes_np.shape)
            fallback_used = False
            mapping_hit = False
            if not pred_text.strip() and not abstain_on_empty:
                fallback, mapping_hit = lookup_code_label(code_to_answer, codes[0].tolist())
                if fallback:
                    pred_text = fallback
                    fallback_used = True
                else:
                    fallback_token = _fallback_token_from_logits(model, prompt_ids, prompt_masks, codes, tokenizer)
                    pred_text = fallback_token or tokenizer.unk_token
                    fallback_used = True
            decoded_len = len(pred_text)
            logger.info(
                "pred_stats cfg=%s index=%s generated_token_count=%s decoded_len=%s fallback_used=%s mapping_hit=%s",
                cfg_name,
                idx,
                generated_token_count,
                decoded_len,
                fallback_used,
                mapping_hit,
            )
            gold = _gold_answer(example)
            gold_uri = _first_value(example, ("obj_uri", "object_uri"))
            mapped_label, pred_qid = _map_pred_to_label(pred_text, qid_to_label)
            gold_answer = normalize_lama_answer(gold, gold_uri)
            gold_codes = codes_from_answer(answer_codebook_df, gold_answer)
            gold_codes_np = None if gold_codes is None else np.asarray(gold_codes, dtype=np.int64)
            if gold_answer:
                gold_answer_nonempty += 1
            if gold_codes_np is not None:
                gold_codes_found += 1
                codes_min_g = int(gold_codes_np.min())
                codes_max_g = int(gold_codes_np.max())
                gold_codes_min = codes_min_g if gold_codes_min is None else min(gold_codes_min, codes_min_g)
                gold_codes_max = codes_max_g if gold_codes_max is None else max(gold_codes_max, codes_max_g)
            pred_codes_argmax = pred_codes_np[0]
            codes_min, codes_max = _validate_code_space(pred_codes_argmax.tolist(), code_vocab_size)
            pred_codes_in_range += 1
            pred_codes_min = codes_min if pred_codes_min is None else min(pred_codes_min, codes_min)
            pred_codes_max = codes_max if pred_codes_max is None else max(pred_codes_max, codes_max)
            if len(pred_codes_head) < 5:
                pred_codes_head.append(pred_codes_argmax.tolist())
            if len(gold_codes_head) < 5:
                gold_codes_head.append(gold_codes_np.tolist() if gold_codes_np is not None else None)
            pred_codes_all.append(pred_codes_argmax.tolist())
            if gold_codes_np is not None:
                gold_codes_all.append(gold_codes_np.tolist())

            slot_logits = torch.stack(addr_logits, dim=1)
            if margin_metrics and gold_codes_np is not None:
                margin_logits.append(slot_logits[0].detach().cpu().numpy())
                margin_golds.append(gold_codes_np)
            if use_candidates:
                if decode_offsets is None or decode_flat is None:
                    raise ValueError("candidate index not available")
                cand_rows = candidate_rows_from_logits(
                    slot_logits,
                    decode_offsets,
                    decode_flat,
                    topk_per_slot=candidate_topk_per_slot,
                    max_candidates=candidate_max,
                )
                if cand_rows:
                    candidate_sizes.append(int(len(cand_rows[0])))
                pred_rows, pred_codes = constrained_decode_candidates_by_logprobs(
                    slot_logits,
                    decode_matrix,
                    cand_rows,
                )
                constrained_codes_row = pred_codes[0].detach().cpu().numpy()
                if gold_codes_np is not None:
                    gold_row = tuple_to_row.get(tuple(gold_codes_np.tolist()))
                    if gold_row is not None:
                        candidate_hit_total += 1
                        cand_row_set = set(int(x) for x in cand_rows[0].tolist()) if cand_rows else set()
                        if decode_row_to_local is not None:
                            local_idx = decode_row_to_local.get(int(gold_row), None)
                            if local_idx is not None and local_idx in cand_row_set:
                                candidate_hit_found += 1
                        else:
                            if int(gold_row) in cand_row_set:
                                candidate_hit_found += 1
            else:
                constrained_codes_matrix = constrained_decode_by_logprobs(
                    slot_logits[0].detach().cpu().numpy()[None, ...],
                    decode_matrix,
                    chunk=int(eval_cfg.get("constrained_chunk", 4096)),
                )
                constrained_codes_row = np.asarray(constrained_codes_matrix[0], dtype=np.int64)
            constrained_answer = decode_codes(reverse_codebook, constrained_codes_row.tolist()) or "__OOV__"

            orbit_id = None
            if orbit_key in example:
                orbit_id = str(example.get(orbit_key))
            elif orbit_key == "prompt":
                orbit_id = prompt
            if orbit_id:
                orbit_preds.setdefault(orbit_id, []).append(constrained_answer)

            decoded_argmax = decode_codes(reverse_codebook, pred_codes_argmax.tolist())
            argmax_answer = decoded_argmax if decoded_argmax is not None else "__OOV__"
            argmax_pred_total += 1
            if decoded_argmax is not None:
                argmax_pred_tuple_found += 1
            if argmax_answer == "__OOV__":
                argmax_pred_oov_count += 1

            constrained_pred_total += 1
            if constrained_answer == "__OOV__":
                constrained_pred_oov_count += 1
            else:
                constrained_pred_tuple_found += 1

            if decode_mode not in {"argmax", "constrained"}:
                raise ValueError(f"invalid decode mode: {decode}")
            if decode_mode == "argmax":
                pred_answer = argmax_answer
                pred_codes_used = pred_codes_argmax
                if pred_answer == "__OOV__" and project_oov:
                    slot_conf = [float(torch.softmax(logits, dim=-1)[0].max().item()) for logits in addr_logits]
                    projected = _project_oov(pred_codes_used.tolist(), slot_index, answer_codebook, slot_conf=slot_conf)
                    if projected:
                        pred_answer = projected
                        projected_oov_count += 1
            else:
                pred_answer = constrained_answer
                pred_codes_used = constrained_codes_row

            if gold_answer:
                answer_em_total += 1
                if exact_match(argmax_answer, gold_answer):
                    answer_em_argmax_correct += 1
                if exact_match(constrained_answer, gold_answer):
                    answer_em_constrained_correct += 1
            if gold_codes_np is not None:
                code_em_total += 1
                if code_tuple_em(tuple(pred_codes_argmax.tolist()), tuple(gold_codes_np.tolist())):
                    code_em_argmax_correct += 1
                if code_tuple_em(tuple(constrained_codes_row.tolist()), tuple(gold_codes_np.tolist())):
                    code_em_constrained_correct += 1

            if pred_answer != "__OOV__":
                pred_tuple_found += 1

            acc, acc_uri, acc_text = _score_answer(pred_answer, gold_answer)


            confidences.append(float(conf.item()))
            accuracies.append(acc)
            if acc_uri is not None:
                accuracies_uri.append(acc_uri)
            if acc_text is not None:
                accuracies_text.append(acc_text)
            if gold_answer:
                gold_answers.append(gold_answer)
                if gold_codes_np is not None:
                    gold_in_codebook += 1
                    tuple_total += 1
                    tuple_hit, slots_hit, slots_total = _code_accuracy(pred_codes_used.tolist(), gold_codes_np.tolist())
                    tuple_correct += tuple_hit
                    slot_correct += slots_hit
                    slot_total += slots_total

                    argmax_tuple_total += 1
                    tuple_hit_a, slots_hit_a, slots_total_a = _code_accuracy(
                        pred_codes_argmax.tolist(), gold_codes_np.tolist()
                    )
                    argmax_tuple_correct += tuple_hit_a
                    argmax_slot_correct += slots_hit_a
                    argmax_slot_total += slots_total_a

                    constrained_tuple_total += 1
                    tuple_hit_c, slots_hit_c, slots_total_c = _code_accuracy(
                        constrained_codes_row.tolist(), gold_codes_np.tolist()
                    )
                    constrained_tuple_correct += tuple_hit_c
                    constrained_slot_correct += slots_hit_c
                    constrained_slot_total += slots_total_c
                else:
                    skipped_no_code += 1
            pred_total += 1
            if pred_answer == "__OOV__":
                pred_oov_count += 1
            else:
                pred_in_codebook += 1
            pred_key = pred_answer or _normalize_answer(pred_text).lower()
            if pred_key:
                pred_collapse_hist[pred_key] += 1
            gold_key = gold_answer or _normalize_answer(gold).lower()
            if gold_key:
                gold_hist[gold_key] += 1
            if len(predictions) < store_samples:
                predictions.append({
                    "prompt": prompt,
                    "pred": pred_text,
                    "pred_mapped": mapped_label if mapped_label != pred_text else "",
                    "pred_qid": pred_qid,
                    "pred_answer": pred_answer,
                    "gold": gold,
                    "gold_uri": gold_uri,
                    "gold_answer": gold_answer,
                    "conf": float(conf.item()),
                })
            processed += 1
            now = time.perf_counter()
            log_due = processed % log_every == 0 or now >= next_time_log
            if log_due:
                items_since = processed - last_log_idx
                per_item = (now - last_log_time) / max(items_since, 1)
                avg_item = (now - eval_start) / max(processed, 1)
                logger.info(
                    "eval_progress cfg=%s item=%s/%s acc=%.3f step_s=%.3f avg_s=%.3f",
                    cfg_name,
                    processed,
                    len(ds),
                    float(np.mean(accuracies)) if accuracies else 0.0,
                    per_item,
                    avg_item,
                )
                last_log_time = now
                last_log_idx = processed
                if now >= next_time_log:
                    next_time_log = now + time_log_interval

        pred_codes_np_full = np.asarray(pred_codes_all, dtype=np.int64) if pred_codes_all else np.zeros((0, 0), dtype=np.int64)
        pred_codes_shape = list(pred_codes_np_full.shape)
        total_evaluated_confirm = int(pred_codes_np_full.shape[0])
        if total_evaluated_confirm != processed:
            raise ValueError(
                f"pred_codes shape mismatch: pred_rows={total_evaluated_confirm} total_evaluated={processed}"
            )
        if pred_codes_np_full.size > 0:
            pred_codes_min = int(pred_codes_np_full.min())
            pred_codes_max = int(pred_codes_np_full.max())
        gold_codes_np_full = np.asarray(gold_codes_all, dtype=np.int64) if gold_codes_all else np.zeros((0, 0), dtype=np.int64)
        gold_codes_shape = list(gold_codes_np_full.shape)
        if gold_codes_np_full.size > 0:
            gold_codes_min = int(gold_codes_np_full.min())
            gold_codes_max = int(gold_codes_np_full.max())
        acc_overall = float(np.mean(accuracies)) if accuracies else 0.0
        acc_uri = float(np.mean(accuracies_uri)) if accuracies_uri else 0.0
        acc_text = float(np.mean(accuracies_text)) if accuracies_text else 0.0
        unique_pred_count = len(pred_collapse_hist)
        top1_freq = 0.0
        if processed and pred_collapse_hist:
            top1_freq = max(pred_collapse_hist.values()) / processed
        gold_type_counts = _answer_type_counts(gold_answers)
        tuple_acc = tuple_correct / tuple_total if tuple_total else 0.0
        slot_acc = slot_correct / slot_total if slot_total else 0.0
        pred_tuple_found_rate = pred_tuple_found / pred_total if pred_total else 0.0
        argmax_tuple_acc = argmax_tuple_correct / argmax_tuple_total if argmax_tuple_total else 0.0
        argmax_slot_acc = argmax_slot_correct / argmax_slot_total if argmax_slot_total else 0.0
        argmax_pred_tuple_found_rate = argmax_pred_tuple_found / argmax_pred_total if argmax_pred_total else 0.0
        constrained_tuple_acc = constrained_tuple_correct / constrained_tuple_total if constrained_tuple_total else 0.0
        constrained_slot_acc = constrained_slot_correct / constrained_slot_total if constrained_slot_total else 0.0
        constrained_pred_tuple_found_rate = constrained_pred_tuple_found / constrained_pred_total if constrained_pred_total else 0.0
        answer_em_constrained = answer_em_constrained_correct / answer_em_total if answer_em_total else 0.0
        answer_em_argmax = answer_em_argmax_correct / answer_em_total if answer_em_total else 0.0
        code_em_constrained = code_em_constrained_correct / code_em_total if code_em_total else 0.0
        code_em_argmax = code_em_argmax_correct / code_em_total if code_em_total else 0.0
        if decode_mode == "constrained":
            code_em = code_em_constrained
            answer_em = answer_em_constrained
        else:
            code_em = code_em_argmax
            answer_em = answer_em_argmax
        orbit_stats = orbit_consistency(orbit_preds) if orbit_preds else {"count": 0, "rate": None}
        if margin_metrics and margin_logits:
            margin_stats = negative_margin_stats(
                np.asarray(margin_logits, dtype=np.float32),
                np.asarray(margin_golds, dtype=np.int64),
            )
        else:
            margin_stats = {"negative_margin_rate": 0.0, "margin_min_mean": 0.0, "margin_min_p50": 0.0, "margin_min_p05": 0.0}
        gold_answer_nonempty_rate = gold_answer_nonempty / processed if processed else 0.0
        gold_codes_found_rate = gold_codes_found / processed if processed else 0.0
        pred_codes_in_range_rate = pred_codes_in_range / pred_total if pred_total else 0.0
        candidate_size_mean = None
        candidate_size_p95 = None
        if use_candidates and candidate_sizes:
            candidate_size_mean = float(np.mean(candidate_sizes))
            candidate_size_p95 = float(np.percentile(candidate_sizes, 95))
        candidate_hit_rate = None
        if use_candidates:
            candidate_hit_rate = candidate_hit_found / candidate_hit_total if candidate_hit_total else 0.0
        report[cfg_name] = {
            "accuracy": acc_overall,
            "accuracy_uri": acc_uri,
            "accuracy_text": acc_text,
            "calibration": calibration_curve(confidences, accuracies, bins=int(eval_cfg["calib_bins"])),
            "checkpoint_path": str(ckpt_path.resolve()),
            "checkpoint_path_used": str(ckpt_path.resolve()),
            "checkpoint_step": ckpt_step,
            "tokenizer_path_used": str(resolved_tokenizer_path.resolve()),
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "tokenizer_sha256": tokenizer_sha256,
            "use_candidates": use_candidates,
            "total_evaluated": processed,
            "stored_samples": len(predictions),
            "stored_samples_limit": store_samples,
            "unique_pred_count": unique_pred_count,
            "top1_freq": top1_freq,
            "gold_answer_type_counts": gold_type_counts,
            "code_em": code_em,
            "answer_em": answer_em,
            "orbit_consistency": orbit_stats,
            "negative_margin": margin_stats,
            "em_breakdown": {
                "answer_em_constrained": answer_em_constrained,
                "answer_em_argmax": answer_em_argmax,
                "code_em_constrained": code_em_constrained,
                "code_em_argmax": code_em_argmax,
            },
            "alignment_sanity": {
                "gold_answer_nonempty_rate": gold_answer_nonempty_rate,
                "gold_codes_found_rate": gold_codes_found_rate,
                "pred_codes_in_range_rate": pred_codes_in_range_rate,
            },
            "skipped_no_code": skipped_no_code,
            "pred_codes_shape": pred_codes_shape,
            "pred_codes_min": pred_codes_min,
            "pred_codes_max": pred_codes_max,
            "code_vocab_size": code_vocab_size,
            "total_evaluated_confirm": total_evaluated_confirm,
            "gold_codes_shape": gold_codes_shape,
            "gold_codes_min": gold_codes_min,
            "gold_codes_max": gold_codes_max,
            "pred_codes_head": pred_codes_head,
            "gold_codes_head": gold_codes_head,
            "pred_oov_count": pred_oov_count,
            "pred_tuple_found": {"count": pred_tuple_found, "rate": pred_tuple_found_rate},
            "argmax_pred_tuple_found_rate": argmax_pred_tuple_found_rate,
            "argmax_pred_oov_count": argmax_pred_oov_count,
            "constrained_pred_tuple_found_rate": constrained_pred_tuple_found_rate,
            "constrained_pred_oov_count": constrained_pred_oov_count,
            "decode_mode_used": decode_mode,
            "projected_oov_count": projected_oov_count,
            "code_acc_tuple": {"correct": tuple_correct, "total": tuple_total, "acc": tuple_acc},
            "code_acc_slot": {"correct": slot_correct, "total": slot_total, "acc": slot_acc},
            "code_acc_tuple_argmax": {
                "correct": argmax_tuple_correct,
                "total": argmax_tuple_total,
                "acc": argmax_tuple_acc,
            },
            "code_acc_slot_argmax": {
                "correct": argmax_slot_correct,
                "total": argmax_slot_total,
                "acc": argmax_slot_acc,
            },
            "code_acc_tuple_constrained": {
                "correct": constrained_tuple_correct,
                "total": constrained_tuple_total,
                "acc": constrained_tuple_acc,
            },
            "code_acc_slot_constrained": {
                "correct": constrained_slot_correct,
                "total": constrained_slot_total,
                "acc": constrained_slot_acc,
            },
            "codebook_coverage": {
                "gold_in_codebook": gold_in_codebook,
                "gold_total": len(gold_answers),
                "pred_in_codebook": pred_in_codebook,
                "pred_total": pred_total,
                "skipped_no_code": skipped_no_code,
            },
            "pred_hist_topk": [
                {"answer": answer, "count": count}
                for answer, count in pred_collapse_hist.most_common(pred_hist_topk)
            ],
            "gold_hist_topk": [
                {"value": value, "count": count}
                for value, count in gold_hist.most_common(pred_hist_topk)
            ],
            "samples": predictions,
            "answer_filter_used": answer_filter_used,
            "answer_filter_year_rows": int(year_rows.size),
            "gold_year_rate": gold_year_rate,
            "gold_qid_rate": gold_qid_rate if gold_total else 0.0,
        }
        if use_candidates:
            report[cfg_name]["candidate_size_mean"] = candidate_size_mean
            report[cfg_name]["candidate_size_p95"] = candidate_size_p95
            report[cfg_name]["candidate_hit_rate"] = candidate_hit_rate
        logger.info(
            "eval_config done cfg=%s accuracy=%.4f acc_uri=%.4f acc_text=%.4f",
            cfg_name,
            acc_overall,
            acc_uri,
            acc_text,
        )
        hit_info = f"{candidate_hit_rate:.4f}" if candidate_hit_rate is not None else "na"
        print(
            f"subset={cfg_name} ckpt={ckpt_path.resolve()} step={ckpt_step} "
            f"candidates={use_candidates} filter={answer_filter_used} "
            f"hit={hit_info} gold_year_rate={gold_year_rate:.4f} em={answer_em:.4f}"
        )

    out_path.write_text(json.dumps(report, indent=2))
    logger.info("report_saved path=%s", out_path)
    console.print(f"Saved LAMA report to {out_path}")


if __name__ == "__main__":
    app()
