from __future__ import annotations

import logging
import random
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
_set_startup_stage("import typer")
import typer
_set_startup_stage("import yaml")
import yaml

from forge_omega_500.data.rvq import load_code_to_label, lookup_code_label
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

logger = setup_logger("eval_factbank_overlap")
app = typer.Typer(add_completion=False)
_FILE_LOG_READY = False

_MASK_TOKENS = ("[MASK]", "<mask>", "MASK")

_set_startup_stage("ready")


def _ensure_file_logger(out_dir: Path) -> None:
    global _FILE_LOG_READY
    if _FILE_LOG_READY:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "eval_factbank_overlap.log"
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("file_logging_enabled path=%s", log_path)
    _FILE_LOG_READY = True


def _load_factbank(factbank_dir: Path) -> List[Dict[str, object]]:
    import pandas as pd

    df = pd.read_parquet(factbank_dir / "facts.parquet")
    return df.to_dict(orient="records")


def _normalize_text(text: str) -> str:
    return str(text).strip().lower()


def _replace_mask_tokens(text: str) -> str:
    for token in _MASK_TOKENS:
        text = text.replace(token, "____")
    return text


def _render_cloze(orbit: str) -> str:
    text = str(orbit).strip()
    if not text:
        return ""
    text = _replace_mask_tokens(text)
    if "____" not in text:
        return ""
    if text.lower().startswith("fill in the blank:"):
        return text
    return "Fill in the blank: " + text


def _extract_cloze_prompts(orbits: Iterable[object]) -> List[str]:
    prompts = []
    seen = set()
    for orbit in orbits:
        prompt = _render_cloze(str(orbit))
        if not prompt:
            continue
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
    return prompts


def _normalize_orbits(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if v]
    text = str(value).strip()
    return [text] if text else []


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
def main(config: Path = typer.Option(..., help="Path to config YAML")) -> None:
    cfg = yaml.safe_load(config.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)
    rng = random.Random(seed)

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
    eval_cfg = cfg.get("eval", {})
    inf_cfg = cfg.get("inference", {})

    out_dir = Path("out")
    _ensure_file_logger(out_dir)

    factbank_dir = Path(data_cfg["factbank_dir"])
    logger.info("load_factbank start")
    t0 = time.perf_counter()
    records = _load_factbank(factbank_dir)
    logger.info("load_factbank done records=%s time=%.2fs", len(records), time.perf_counter() - t0)

    tokenizer = SimpleTokenizer.load(out_dir / "tokenizer.json")
    codes_dir = Path(data_cfg["codes_dir"])
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
    logger.info("init_model done time=%.2fs", time.perf_counter() - t0)
    ckpt = out_dir / "ckpt/model.pt"
    logger.info("load_checkpoint start path=%s", ckpt)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    logger.info("load_checkpoint done")
    model.eval()

    device = torch.device("cuda" if torch_info.get("torch_cuda") else "cpu")
    model.to(device)
    logger.info("model_to_device done device=%s", device.type)

    code_to_answer = _build_code_to_answer_map(codes_dir / "code_to_label.parquet")
    max_gen_tokens = int(inf_cfg.get("max_gen_tokens", 8))
    if max_gen_tokens < 4:
        logger.warning("max_gen_tokens_too_small value=%s clamped=4", max_gen_tokens)
        max_gen_tokens = 4
    abstain_on_empty = bool(inf_cfg.get("abstain_on_empty", False))

    max_facts = int(eval_cfg.get("factbank_samples", eval_cfg.get("max_samples", 500)))
    max_orbits_per_fact = int(eval_cfg.get("max_orbits_per_fact", 5))

    fact_items = []
    for rec in records:
        obj_label = str(rec.get("object_label", "")).strip()
        if not obj_label:
            continue
        orbits = _normalize_orbits(rec.get("question_orbits"))
        prompts = _extract_cloze_prompts(orbits)
        if not prompts:
            continue
        fact_items.append({
            "fact_id": rec.get("fact_id", ""),
            "object_label": obj_label,
            "prompts": prompts,
        })
    logger.info("factbank_cloze_ready facts=%s", len(fact_items))
    if not fact_items:
        raise RuntimeError("No FactBank entries with cloze prompts found.")

    if max_facts and max_facts < len(fact_items):
        sample_indices = rng.sample(range(len(fact_items)), k=max_facts)
        sampled = [fact_items[i] for i in sample_indices]
    else:
        sampled = list(fact_items)

    logger.info(
        "eval_factbank start facts=%s max_orbits_per_fact=%s",
        len(sampled),
        max_orbits_per_fact,
    )
    correct = 0
    total = 0
    consistent = 0
    pred_counts: Counter = Counter()

    with torch.no_grad():
        for rec in sampled:
            prompts = rec["prompts"]
            if max_orbits_per_fact and len(prompts) > max_orbits_per_fact:
                prompt_list = rng.sample(prompts, k=max_orbits_per_fact)
            else:
                prompt_list = prompts
            prompt_ids = [[tokenizer.bos_id] + tokenizer.encode(p) for p in prompt_list]
            prompt_ids, prompt_masks = pad_sequences(
                prompt_ids,
                tokenizer.pad_id,
                max_len=int(model_cfg["max_seq_len"]),
            )
            prompt_ids = prompt_ids.to(device)
            prompt_masks = prompt_masks.to(device)

            addr_logits, _, _ = model.encode_prompt(prompt_ids, prompt_masks)
            generated, codes = model.generate_answer(
                prompt_ids,
                prompt_masks,
                max_new_tokens=max_gen_tokens,
                eos_id=tokenizer.eos_id,
                bos_id=tokenizer.bos_id,
                sep_id=tokenizer.sep_id,
            )
            if codes is None:
                codes = torch.stack([logits.argmax(dim=-1) for logits in addr_logits], dim=1)

            code_tuples = []
            for idx in range(generated.size(0)):
                pred_text = tokenizer.decode(generated[idx].tolist())
                fallback_used = False
                mapping_hit = False
                if not pred_text.strip() and not abstain_on_empty:
                    fallback, mapping_hit = lookup_code_label(code_to_answer, codes[idx].tolist())
                    if fallback:
                        pred_text = fallback
                        fallback_used = True
                    else:
                        fallback_token = _fallback_token_from_logits(
                            model,
                            prompt_ids[idx:idx + 1],
                            prompt_masks[idx:idx + 1],
                            codes[idx:idx + 1],
                            tokenizer,
                        )
                        pred_text = fallback_token or tokenizer.unk_token
                        fallback_used = True
                if fallback_used:
                    logger.debug(
                        "fallback_used fact_id=%s mapping_hit=%s",
                        rec.get("fact_id", ""),
                        mapping_hit,
                    )
                pred_norm = _normalize_text(pred_text)
                gold_norm = _normalize_text(rec["object_label"])
                correct += 1 if pred_norm == gold_norm else 0
                total += 1
                pred_counts[pred_norm] += 1
                code_tuples.append(tuple(int(c) for c in codes[idx].tolist()))

            if len(code_tuples) <= 1 or all(ct == code_tuples[0] for ct in code_tuples[1:]):
                consistent += 1

    accuracy = correct / max(total, 1)
    orbit_consistency = consistent / max(len(sampled), 1)
    logger.info("eval_factbank done accuracy=%.4f orbit_consistency=%.4f", accuracy, orbit_consistency)
    print(f"accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"orbit_consistency: {orbit_consistency:.4f} ({consistent}/{len(sampled)})")
    print("top_answers:")
    for answer, count in pred_counts.most_common(10):
        print(f"{answer}\t{count}")


if __name__ == "__main__":
    app()
