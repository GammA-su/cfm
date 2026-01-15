from __future__ import annotations

import json
import logging
import sys
import tarfile
import threading
import time
from fnmatch import fnmatch
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
_set_startup_stage("import datasets")
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager
_set_startup_stage("import rich.console")
from rich.console import Console

from forge_omega_500.eval.metrics import calibration_curve
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

console = Console()
logger = setup_logger("eval_lama")
app = typer.Typer(add_completion=False)
_FILE_LOG_READY = False

_set_startup_stage("ready")


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
                    for evidence in data.get("evidences", []):
                        yield {
                            "masked_sentence": str(evidence.get("masked_sentence", "")),
                            "template": str(pred.get("template", "")),
                            "sub_label": str(data.get("sub_label", "")),
                            "obj_label": str(data.get("obj_label", "")),
                        }
                        yielded += 1
                        if max_samples and yielded >= max_samples:
                            return
                else:
                    yield data
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


def _extract_masked_sentence(example: Dict[str, object]) -> Tuple[str, str]:
    if example.get("masked_sentence"):
        return str(example["masked_sentence"]), "masked_sentence"
    if example.get("masked_sentences"):
        values = example["masked_sentences"]
        if isinstance(values, list):
            for item in values:
                if item:
                    return str(item), "masked_sentences"
        elif values:
            return str(values), "masked_sentences"
    if example.get("masked_prompt"):
        return str(example["masked_prompt"]), "masked_prompt"
    if example.get("sentence"):
        return str(example["sentence"]), "sentence"
    return "", ""


def _extract_subject(example: Dict[str, object]) -> str:
    for key in ("sub_label", "subject_label", "sub", "subject"):
        value = example.get(key)
        if isinstance(value, list):
            for item in value:
                if item:
                    return str(item).strip()
        elif value:
            return str(value).strip()
    return ""


def _build_prompt_with_field(example: Dict[str, object]) -> Tuple[str | None, str]:
    masked_sentence, field = _extract_masked_sentence(example)
    if masked_sentence:
        prompt = "Fill in the blank: " + _replace_mask_tokens(masked_sentence)
        return prompt, field or "masked_sentence"
    template = example.get("template")
    if template:
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
        if key in example and example[key]:
            return str(example[key])
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


def _decode_with_fallback(tokenizer: SimpleTokenizer, token_ids: List[int]) -> str:
    decoded = tokenizer.decode(token_ids)
    if decoded:
        return decoded
    raw_tokens = [tokenizer.inv_vocab.get(i, tokenizer.unk_token) for i in token_ids]
    return " ".join(raw_tokens).strip()


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

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]
    inf_cfg = cfg["inference"]

    out_dir = Path("out")
    _ensure_file_logger(out_dir)

    logger.info("load_tokenizer start")
    t0 = time.perf_counter()
    tokenizer = SimpleTokenizer.load(out_dir / "tokenizer.json")
    logger.info("load_tokenizer done vocab_size=%s time=%.2fs", len(tokenizer.vocab), time.perf_counter() - t0)

    codebook_path = Path(data_cfg["codes_dir"]) / "codebooks.safetensors"
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

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))
    use_fallback = False
    logger.info("discover_lama_configs start")
    try:
        available = get_dataset_config_names("facebook/lama")
        target_configs = [c for c in ["google_re", "trex", "conceptnet", "squad"] if c in available]
    except Exception as exc:
        logger.warning("get_dataset_config_names failed: %s; falling back to tar loader", exc)
        target_configs = ["google_re", "trex", "conceptnet", "squad"]
        use_fallback = True
    logger.info("discover_lama_configs done use_fallback=%s", use_fallback)
    report = {}
    logger.info("lama_configs=%s", ",".join(target_configs))

    for cfg_name in target_configs:
        max_samples = int(eval_cfg["max_samples"])
        logger.info("load_dataset start cfg=%s", cfg_name)
        t0 = time.perf_counter()
        if use_fallback:
            ds = _load_lama_fallback(cfg_name, cache_dir, local_files_only, max_samples)
        else:
            ds = load_dataset("facebook/lama", cfg_name, split="train")
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))
        logger.info("load_dataset done cfg=%s samples=%s time=%.2fs", cfg_name, len(ds), time.perf_counter() - t0)

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
        predictions = []
        log_every = int(eval_cfg.get("log_every", 50))
        time_log_interval = float(eval_cfg.get("time_log_interval", 30.0))
        eval_start = time.perf_counter()
        last_log_time = eval_start
        last_log_idx = 0
        next_time_log = eval_start + time_log_interval
        processed = 0

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
            generated, _ = model.generate_answer(
                prompt_ids,
                prompt_masks,
                max_new_tokens=int(inf_cfg["max_gen_tokens"]),
                eos_id=tokenizer.eos_id,
                bos_id=tokenizer.bos_id,
                sep_id=tokenizer.sep_id,
            )
            pred_text = _decode_with_fallback(tokenizer, generated[0].tolist())
            gold = _gold_answer(example)
            pred_norm, gold_norm = _normalize_for_em(pred_text, gold)
            acc = 1.0 if pred_norm == gold_norm else 0.0

            confidences.append(float(conf.item()))
            accuracies.append(acc)
            predictions.append({"prompt": prompt, "pred": pred_text, "gold": gold, "conf": float(conf.item())})
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

        report[cfg_name] = {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "calibration": calibration_curve(confidences, accuracies, bins=int(eval_cfg["calib_bins"])),
            "samples": predictions[:5],
        }
        logger.info("eval_config done cfg=%s accuracy=%.4f", cfg_name, report[cfg_name]["accuracy"])

    out_path = out_dir / "lama_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("report_saved path=%s", out_path)
    console.print(f"Saved LAMA report to {out_path}")


if __name__ == "__main__":
    app()
