from __future__ import annotations

import json
from pathlib import Path

from forge_omega_500.runtime import (
    DEFAULT_CPU_THREADS,
    configure_env,
    configure_torch,
    log_runtime,
    resolve_faiss,
    setup_logger,
)

configure_env(DEFAULT_CPU_THREADS)

import typer
import yaml
from rich.console import Console

from forge_omega_500.data.build_factbank import build_factbank_records, save_factbank
from forge_omega_500.data.embeddings import build_answer_embeddings
from forge_omega_500.data.hf_ingest import ingest_wikidata5m, ingest_wikidata5m_tar, load_triples
from forge_omega_500.data.rvq import (
    assign_codes,
    save_answer_codes,
    save_code_to_label,
    save_codebooks,
    train_rvq,
)
from forge_omega_500.model.utils import set_seed

console = Console()
logger = setup_logger("build_all")
app = typer.Typer(add_completion=False)


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
    raw_dir = Path(data_cfg["raw_dir"])
    factbank_dir = Path(data_cfg["factbank_dir"])
    emb_dir = Path(data_cfg["emb_dir"])
    codes_dir = Path(data_cfg["codes_dir"])

    use_direct_tar = bool(data_cfg.get("use_direct_tar", True))
    logger.info(
        "ingest_wikidata5m start use_direct_tar=%s use_streaming=%s shuffle=%s local_files_only=%s max_triples=%s",
        use_direct_tar,
        bool(data_cfg.get("use_streaming", True)),
        bool(data_cfg.get("shuffle", False)),
        bool(data_cfg.get("local_files_only", False)),
        int(data_cfg["max_triples"]),
    )
    raw_path = raw_dir / "wikidata5m.parquet"
    if use_direct_tar:
        ingest_wikidata5m_tar(
            raw_path,
            max_triples=int(data_cfg["max_triples"]),
            sources=list(data_cfg.get("wikidata5m_sources", ["transductive"])),
            cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
            local_files_only=bool(data_cfg.get("local_files_only", False)),
        )
    else:
        ingest_wikidata5m(
            raw_path,
            max_triples=int(data_cfg["max_triples"]),
            seed=seed,
            use_streaming=bool(data_cfg.get("use_streaming", True)),
            shuffle=bool(data_cfg.get("shuffle", False)),
            shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
            local_files_only=bool(data_cfg.get("local_files_only", False)),
        )

    logger.info("build_factbank start")
    triples = load_triples(raw_path)
    logger.info("triples_loaded=%s", len(triples))
    records = build_factbank_records(
        triples,
        max_facts=int(data_cfg["max_facts"]),
        orbits_per_fact=int(data_cfg["orbits_per_fact"]),
        negatives_per_fact=int(data_cfg["negatives_per_fact"]),
        seed=seed,
    )
    save_factbank(records, factbank_dir)
    logger.info("facts_built=%s", len(records))

    logger.info("build_answer_embeddings start")
    model_cfg = cfg["model"]
    answers, embeddings = build_answer_embeddings(records, dim=int(model_cfg["d_code"]), emb_dir=emb_dir)
    logger.info("answers=%s emb_dim=%s", len(answers), int(model_cfg["d_code"]))

    rvq_cfg = cfg["rvq"]
    require_faiss_gpu = bool(rvq_cfg.get("require_faiss_gpu", False))
    if require_faiss_gpu:
        if not use_faiss:
            raise RuntimeError("rvq.require_faiss_gpu=true requires runtime.use_faiss=true")
        if not prefer_gpu:
            raise RuntimeError("rvq.require_faiss_gpu=true requires runtime.prefer_gpu=true")
        if not faiss_info["faiss_gpu_available"]:
            raise RuntimeError("rvq.require_faiss_gpu=true but faiss GPU is unavailable; install faiss-gpu")

    logger.info("train_rvq start (faiss=%s faiss_gpu=%s)", bool(faiss_info["faiss"]), faiss_info["faiss_gpu_available"])
    gpu_search_batch = rvq_cfg.get("gpu_search_batch", None)
    if gpu_search_batch is not None:
        gpu_search_batch = int(gpu_search_batch)
    codebooks = train_rvq(
        embeddings,
        m=int(model_cfg["m"]),
        K=int(model_cfg["K"]),
        iters=int(rvq_cfg["iters"]),
        seed=int(rvq_cfg["seed"]),
        faiss_mod=faiss_info["faiss"],
        use_gpu=bool(faiss_info["faiss_gpu_available"]),
        require_gpu=require_faiss_gpu,
        gpu_search_batch=gpu_search_batch,
    )
    codes = assign_codes(
        embeddings,
        codebooks,
        faiss_mod=faiss_info["faiss"],
        use_gpu=bool(faiss_info["faiss_gpu_available"]),
        require_gpu=require_faiss_gpu,
        gpu_search_batch=gpu_search_batch,
    )

    save_codebooks(codebooks, codes_dir / "codebooks.safetensors")
    save_answer_codes(answers, codes, codes_dir / "answer_codes.parquet")
    save_code_to_label(records, answers, codes, codes_dir / "code_to_label.parquet")

    metadata_path = factbank_dir / "build_summary.json"
    metadata_path.write_text(json.dumps({"facts": len(records), "answers": len(answers)}, indent=2))

    console.print("\n[bold]Next steps[/bold]")
    console.print("uv run python scripts/train_cfm.py --config configs/default.yaml")
    console.print("uv run python scripts/eval_lama.py --config configs/default.yaml")
    console.print("uv run pytest -q")


if __name__ == "__main__":
    app()
