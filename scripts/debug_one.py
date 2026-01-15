from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from forge_omega_500.runtime import (
    DEFAULT_CPU_THREADS,
    configure_env,
    configure_torch,
    log_runtime,
    resolve_faiss,
    setup_logger,
)

configure_env(DEFAULT_CPU_THREADS)

import torch
import typer
import yaml
from datasets import get_dataset_config_names, load_dataset

from eval_lama import _build_prompt, _load_lama_fallback
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed

logger = setup_logger("debug_one")
app = typer.Typer(add_completion=False)


def _load_subset(
    cfg_name: str,
    cache_dir: Path,
    local_files_only: bool,
    max_samples: int,
) -> Tuple[Iterable[dict], bool]:
    try:
        available = get_dataset_config_names("facebook/lama")
        if cfg_name not in available:
            raise ValueError(f"subset_not_found: {cfg_name}")
        ds = load_dataset("facebook/lama", cfg_name, split="train")
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return ds, False
    except Exception as exc:
        logger.warning("load_dataset failed: %s; falling back to tar loader", exc)
        ds = _load_lama_fallback(cfg_name, cache_dir, local_files_only, max_samples)
        return ds, True


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to config YAML"),
    subset: str = typer.Option(..., help="LAMA subset: google_re/trex/conceptnet/squad"),
    index: int = typer.Option(..., min=0, help="Sample index in the subset"),
    max_new_tokens: int = typer.Option(8, min=1, help="Max tokens to generate"),
) -> None:
    subset = subset.strip().lower()
    valid = {"google_re", "trex", "conceptnet", "squad"}
    if subset not in valid:
        raise typer.BadParameter(f"subset must be one of {sorted(valid)}")

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
    eval_cfg = cfg.get("eval", {})

    out_dir = Path("out")
    tokenizer = SimpleTokenizer.load(out_dir / "tokenizer.json")

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
    ckpt = out_dir / "ckpt/model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch_info.get("torch_cuda") else "cpu")
    model.to(device)
    logger.info("model_to_device done device=%s", device.type)

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))
    max_samples = int(eval_cfg.get("max_samples", 0))
    ds, used_fallback = _load_subset(subset, cache_dir, local_files_only, max_samples)
    ds_len = len(ds)
    logger.info("load_dataset done cfg=%s samples=%s use_fallback=%s", subset, ds_len, used_fallback)
    if ds_len == 0:
        raise typer.BadParameter(f"subset {subset} has no samples")
    if index < 0 or index >= ds_len:
        raise typer.BadParameter(f"index out of range: {index} (len={ds_len})")

    example = ds[index]
    prompt = _build_prompt(example)
    prompt_ids = [[tokenizer.bos_id] + tokenizer.encode(prompt)]
    prompt_ids, prompt_masks = pad_sequences(
        prompt_ids,
        tokenizer.pad_id,
        max_len=int(model_cfg["max_seq_len"]),
    )
    prompt_ids = prompt_ids.to(device)
    prompt_masks = prompt_masks.to(device)

    addr_logits, conf, _ = model.encode_prompt(prompt_ids, prompt_masks)
    code_tuple = torch.stack([logits.argmax(dim=-1) for logits in addr_logits], dim=1)

    if hasattr(model, "generate_answer"):
        generated, gen_codes = model.generate_answer(
            prompt_ids,
            prompt_masks,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            bos_id=tokenizer.bos_id,
            sep_id=tokenizer.sep_id,
        )
        if gen_codes is not None:
            code_tuple = gen_codes
        raw_token_ids = generated[0].tolist()
        pred_text = tokenizer.decode(raw_token_ids)
        print("prompt:", prompt)
        print("codes:", [int(x) for x in code_tuple[0].tolist()])
        print("conf:", float(conf.item()))
        print("token_ids:", raw_token_ids)
        print("pred:", repr(pred_text))
        print("pred_len:", len(pred_text))
        return

    generated = torch.full((1, 1), tokenizer.bos_id, device=device, dtype=torch.long)
    attention = torch.ones_like(generated)
    input_ids = torch.cat([prompt_ids, torch.full_like(prompt_ids[:, :1], tokenizer.sep_id), generated], dim=1)
    attn = torch.cat([prompt_masks, torch.ones_like(prompt_ids[:, :1]), attention], dim=1)
    if model.max_seq_len and input_ids.size(1) > model.max_seq_len:
        overflow = input_ids.size(1) - model.max_seq_len
        input_ids = input_ids[:, overflow:]
        attn = attn[:, overflow:]
    _, logits = model.forward_generation(input_ids, attn, code_tuple)
    topk = torch.topk(logits[:, -1], k=min(5, logits.size(-1)), dim=-1)
    topk_pairs = list(zip(topk.indices[0].tolist(), topk.values[0].tolist()))
    print("prompt:", prompt)
    print("codes:", [int(x) for x in code_tuple[0].tolist()])
    print("conf:", float(conf.item()))
    print("logits_top5:", topk_pairs)
    print("pred:", repr(""))
    print("pred_len:", 0)


if __name__ == "__main__":
    app()
