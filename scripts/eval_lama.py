from __future__ import annotations

import json
import tarfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List

from forge_omega_500.runtime import (
    DEFAULT_CPU_THREADS,
    configure_env,
    configure_torch,
    log_runtime,
    resolve_faiss,
    setup_logger,
)

configure_env(DEFAULT_CPU_THREADS)

import numpy as np
import torch
import typer
import yaml
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager
from rich.console import Console

from forge_omega_500.eval.metrics import calibration_curve, exact_match
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

console = Console()
logger = setup_logger("eval_lama")
app = typer.Typer(add_completion=False)

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"


def _download_lama_files(cache_dir: Path, local_files_only: bool) -> tuple[Path, Path]:
    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    archive_path = Path(dl_manager.download(LAMA_DATA_URL))
    relations_path = Path(dl_manager.download(LAMA_RELATIONS_URL))
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
    archive_path, relations_path = _download_lama_files(cache_dir, local_files_only=local_files_only)
    return list(_iter_lama_records(cfg_name, archive_path, relations_path, max_samples=max_samples))


def _build_prompt(example: Dict[str, object]) -> str:
    if "masked_sentence" in example and example["masked_sentence"]:
        return "Fill in the blank: " + str(example["masked_sentence"]).replace("[MASK]", "____")
    template = example.get("template")
    subj = example.get("sub_label") or example.get("subject_label") or ""
    if template:
        prompt = template.replace("[X]", str(subj)).replace("[Y]", "____")
        return "Fill in the blank: " + prompt
    return f"What is the answer for {subj}?"


def _gold_answer(example: Dict[str, object]) -> str:
    for key in ["obj_label", "obj", "object_label", "answer"]:
        if key in example and example[key]:
            return str(example[key])
    return ""


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

    tokenizer = SimpleTokenizer.load(Path("out/tokenizer.json"))

    codebook_path = Path(data_cfg["codes_dir"]) / "codebooks.safetensors"
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
    ckpt = Path("out/ckpt/model.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch_info.get("torch_cuda") else "cpu")
    model.to(device)

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))
    use_fallback = False
    try:
        available = get_dataset_config_names("facebook/lama")
        target_configs = [c for c in ["google_re", "trex", "conceptnet", "squad"] if c in available]
    except Exception as exc:
        logger.warning("get_dataset_config_names failed: %s; falling back to tar loader", exc)
        target_configs = ["google_re", "trex", "conceptnet", "squad"]
        use_fallback = True
    report = {}
    logger.info("lama_configs=%s", ",".join(target_configs))

    for cfg_name in target_configs:
        max_samples = int(eval_cfg["max_samples"])
        if use_fallback:
            ds = _load_lama_fallback(cfg_name, cache_dir, local_files_only, max_samples)
        else:
            ds = load_dataset("facebook/lama", cfg_name, split="train")
            if max_samples:
                ds = ds.select(range(min(max_samples, len(ds))))

        logger.info("eval_config=%s samples=%s", cfg_name, len(ds))
        confidences = []
        accuracies = []
        predictions = []

        for example in ds:
            prompt = _build_prompt(example)
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
            pred_text = tokenizer.decode(generated[0].tolist())
            gold = _gold_answer(example)
            acc = exact_match(pred_text, gold)

            confidences.append(float(conf.item()))
            accuracies.append(acc)
            predictions.append({"prompt": prompt, "pred": pred_text, "gold": gold, "conf": float(conf.item())})

        report[cfg_name] = {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "calibration": calibration_curve(confidences, accuracies, bins=int(eval_cfg["calib_bins"])),
            "samples": predictions[:5],
        }

    out_path = Path("out/lama_report.json")
    out_path.write_text(json.dumps(report, indent=2))
    console.print(f"Saved LAMA report to {out_path}")


if __name__ == "__main__":
    app()
