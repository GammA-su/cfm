from __future__ import annotations

import logging
import os
from typing import Any, Dict

DEFAULT_CPU_THREADS = 16


def configure_env(cpu_threads: int = DEFAULT_CPU_THREADS) -> None:
    keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    for key in keys:
        os.environ[key] = str(cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def configure_torch(cpu_threads: int, prefer_gpu: bool) -> Dict[str, Any]:
    info = {"torch_cuda": False, "gpu_count": 0, "device": "cpu"}
    try:
        import torch
    except Exception:
        return info

    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if prefer_gpu and torch.cuda.is_available():
        info["torch_cuda"] = True
        info["gpu_count"] = int(torch.cuda.device_count())
        info["device"] = "cuda"

    return info


def resolve_faiss(prefer_gpu: bool, cpu_threads: int) -> Dict[str, Any]:
    info = {
        "faiss_available": False,
        "faiss_gpu_available": False,
        "faiss_gpu_count": 0,
        "faiss": None,
    }
    try:
        import faiss
    except Exception:
        return info

    info["faiss_available"] = True
    info["faiss"] = faiss
    try:
        faiss.omp_set_num_threads(cpu_threads)
    except Exception:
        pass

    try:
        gpu_count = int(faiss.get_num_gpus())
    except Exception:
        gpu_count = 0
    info["faiss_gpu_count"] = gpu_count
    info["faiss_gpu_available"] = prefer_gpu and gpu_count > 0
    return info


def log_runtime(logger: logging.Logger, torch_info: Dict[str, Any], faiss_info: Dict[str, Any], cpu_threads: int) -> None:
    if torch_info.get("torch_cuda"):
        logger.info(
            "gpu_enabled=true gpu_count=%s torch_cuda=true device=%s",
            torch_info.get("gpu_count"),
            torch_info.get("device"),
        )
    else:
        logger.info(
            "gpu_enabled=false torch_cuda=false device=cpu cpu_threads=%s",
            cpu_threads,
        )

    if faiss_info.get("faiss_available"):
        logger.info(
            "faiss_available=true faiss_gpu_enabled=%s faiss_gpu_count=%s",
            str(faiss_info.get("faiss_gpu_available")).lower(),
            faiss_info.get("faiss_gpu_count"),
        )
    else:
        logger.info("faiss_available=false faiss_gpu_enabled=false faiss_gpu_count=0")

    if torch_info.get("torch_cuda") and not faiss_info.get("faiss_gpu_available"):
        logger.info("faiss_gpu_enabled=false gpu_available=true hint=install_faiss_gpu_with_uv")


__all__ = ["configure_env", "setup_logger", "configure_torch", "resolve_faiss", "log_runtime", "DEFAULT_CPU_THREADS"]
