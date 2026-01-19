from __future__ import annotations

import json
import logging
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
_set_startup_stage("import rich.console")
from rich.console import Console

from forge_omega_500.data.rvq import load_answer_codes
from forge_omega_500.model.cfm import CFMModel, contrastive_margin_loss
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

console = Console()
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
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _build_examples(records: List[Dict[str, object]], answer_codes: Dict[str, List[int]]) -> List[Dict[str, object]]:
    examples = []
    for rec in records:
        answer = str(rec["object_label"])
        if answer not in answer_codes:
            continue
        codes = answer_codes[answer]
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
        answer = str(rec["object_label"])
        codes = answer_codes.get(answer)
        if codes is None:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = []
    prompt_masks = []
    input_ids = []
    attention_masks = []
    labels = []
    code_targets = []
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

    input_ids, attention_masks = pad_sequences(input_ids, tokenizer.pad_id, max_len=max_seq_len)
    labels, _ = pad_sequences(labels, -100, max_len=max_seq_len)
    prompt_ids, prompt_masks = pad_sequences(prompt_ids, tokenizer.pad_id, max_len=max_seq_len)
    code_targets = torch.tensor(code_targets, dtype=torch.long)

    return input_ids, attention_masks, labels, prompt_ids, prompt_masks, code_targets


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
def main(config: Path = typer.Option(..., help="Path to config YAML")) -> None:
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

    out_dir = Path("out")
    _ensure_file_logger(out_dir)
    ckpt_dir = out_dir / "ckpt"

    factbank_dir = Path(data_cfg["factbank_dir"])
    codes_dir = Path(data_cfg["codes_dir"])

    logger.info("load_factbank start")
    t0 = time.perf_counter()
    records = _load_factbank(factbank_dir)
    logger.info("load_factbank done records=%s time=%.2fs", len(records), time.perf_counter() - t0)
    logger.info("load_answer_codes start path=%s", codes_dir / "answer_codes.parquet")
    t0 = time.perf_counter()
    answer_codes = load_answer_codes(codes_dir / "answer_codes.parquet")
    logger.info("load_answer_codes done answers=%s time=%.2fs", len(answer_codes), time.perf_counter() - t0)

    logger.info("build_examples start")
    t0 = time.perf_counter()
    examples = _build_examples(records, answer_codes)
    logger.info(
        "build_examples done examples=%s orbits_per_fact=%s time=%.2fs",
        len(examples),
        len(records[0]["question_orbits"]) if records else 0,
        time.perf_counter() - t0,
    )
    texts = [ex["prompt"] for ex in examples] + [ex["answer"] for ex in examples]
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
        raise ValueError("No training facts with orbits and answer codes found.")
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
            start_step = int(checkpoint.get("step", 0))
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
    logger.info(
        "loss_weights cloze_addr=%.2f cloze_orbit_boost=%.2f orbit_weight=%.3f entropy_weight=%.4f obj_prior_weight=%.4f",
        cloze_addr_weight,
        cloze_orbit_boost,
        orbit_consistency_weight,
        entropy_weight,
        obj_prior_weight,
    )
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
    nan_skip_count = 0
    nonfinite_events = 0

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
        ) = _prepare_batch(batch, tokenizer, max_seq_len)
        fact_ids = [ex["fact_id"] for ex in batch]
        is_cloze = [bool(ex["is_cloze"]) for ex in batch]
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        code_targets = code_targets.to(device)

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
        gen_loss = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), labels.reshape(-1), ignore_index=-100)

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
            addr_loss
            + gen_loss
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
                "train step=%s/%s loss=%.4f addr=%.4f gen=%.4f contrast=%.4f orbit=%.4f entropy=%.4f obj_prior=%.4f orbit_pairs=%s lr=%.6g grad_norm=%.4f amp=%s neg_hard=%s step_s=%.3f avg_s=%.3f",
                step + 1,
                steps,
                loss.item(),
                addr_loss.item(),
                gen_loss.item(),
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
            console.print(
                f"step {step+1}/{steps} loss={loss.item():.4f} addr={addr_loss.item():.4f} gen={gen_loss.item():.4f} contrast={contrast_loss.item():.4f} orbit={orbit_loss.item():.4f} entropy={entropy_reg.item():.4f} obj_prior={obj_prior_loss.item():.4f} orbit_pairs={orbit_pairs_in_batch}"
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
    _save_checkpoint(ckpt_dir, step=steps, model=model, optimizer=optimizer, rng=rng, epoch_idx=epoch_idx)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    logger.info("checkpoint_saved path=%s", ckpt_dir / "model.pt")
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
    console.print(f"Saved report to {report_path}")


if __name__ == "__main__":
    app()
