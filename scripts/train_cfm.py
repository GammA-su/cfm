from __future__ import annotations

import json
import logging
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

_set_startup_stage("ready")


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
    train_start = time.perf_counter()
    last_log_time = train_start
    last_log_step = 0
    next_time_log = train_start + time_log_interval
    for step in range(steps):
        batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
        (
            input_ids,
            attention_masks,
            labels,
            prompt_ids,
            prompt_masks,
            code_targets,
        ) = _prepare_batch(batch, tokenizer, max_seq_len)

        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = prompt_masks.to(device)
        code_targets = code_targets.to(device)

        addr_logits, _, _ = model.encode_prompt(prompt_ids, prompt_masks)
        addr_loss = 0.0
        for i, logits in enumerate(addr_logits):
            addr_loss = addr_loss + F.cross_entropy(logits, code_targets[:, i])

        _, logits = model.forward_generation(input_ids, attention_masks, code_targets)
        prefix_pad = torch.full((labels.size(0), 1), -100, device=labels.device, dtype=labels.dtype)
        labels = torch.cat([prefix_pad, labels], dim=1)
        gen_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

        contrast_loss = torch.tensor(0.0, device=device)
        neg_codes = []
        for ex in batch:
            negs = ex["negatives"]
            neg_code = None
            for neg in negs:
                neg_answer = str(neg["object_label"])
                if neg_answer in answer_codes:
                    neg_code = answer_codes[neg_answer]
                    break
            if neg_code is None:
                neg_code = ex["codes"]
            neg_codes.append(neg_code)
        neg_codes = torch.tensor(neg_codes, device=device)
        v_pos = model.decode_value(code_targets)
        v_neg = model.decode_value(neg_codes)
        contrast_loss = contrastive_margin_loss(v_pos, v_neg, margin=float(train_cfg["contrast_margin"]))

        loss = addr_loss + gen_loss + contrast_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        now = time.perf_counter()
        log_due = (step + 1) % log_every == 0 or now >= next_time_log
        if log_due:
            steps_since = (step + 1) - last_log_step
            step_time = (now - last_log_time) / max(steps_since, 1)
            avg_time = (now - train_start) / (step + 1)
            logger.info(
                "train step=%s/%s loss=%.4f addr=%.4f gen=%.4f contrast=%.4f step_s=%.3f avg_s=%.3f",
                step + 1,
                steps,
                loss.item(),
                addr_loss.item(),
                gen_loss.item(),
                contrast_loss.item(),
                step_time,
                avg_time,
            )
            console.print(
                f"step {step+1}/{steps} loss={loss.item():.4f} addr={addr_loss.item():.4f} gen={gen_loss.item():.4f} contrast={contrast_loss.item():.4f}"
            )
            last_log_time = now
            last_log_step = step + 1
            if now >= next_time_log:
                next_time_log = now + time_log_interval

    logger.info("training done time=%.2fs", time.perf_counter() - train_start)
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
