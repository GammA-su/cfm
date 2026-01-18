from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

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
from forge_omega_500.data.templates import load_relation_templates
from forge_omega_500.model.utils import set_seed

console = Console()
logger = setup_logger("build_all")
app = typer.Typer(add_completion=False)

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"
_LAMA_GOOGLE_RE_FILES = (
    "Google_RE/date_of_birth_test.jsonl",
    "Google_RE/place_of_birth_test.jsonl",
    "Google_RE/place_of_death_test.jsonl",
)
WIKIDATA5M_TAR_URLS = (
    "https://data.dgl.ai/dataset/wikidata5m/wikidata5m_transductive.tar.gz",
    "https://data.dgl.ai/dataset/wikidata5m/wikidata5m_transductive.tar.gz?download=1",
)
WIKIDATA5M_URL_ENV = "WIKIDATA5M_URLS"
WIKIDATA5M_EXPECTED = (
    "wikidata5m_transductive_train.txt",
    "wikidata5m_transductive_valid.txt",
    "wikidata5m_transductive_test.txt",
)


def _save_qid_to_label(records: list[dict[str, object]], factbank_dir: Path) -> None:
    mapping: dict[str, str] = {}
    for rec in records:
        for qid_key, label_key in (("subject_qid", "subject_label"), ("object_qid", "object_label")):
            qid = str(rec.get(qid_key, "")).strip()
            label = str(rec.get(label_key, "")).strip()
            if not qid or not qid.startswith("Q") or not qid[1:].isdigit():
                continue
            if not label:
                continue
            current = mapping.get(qid)
            if current is None or label < current:
                mapping[qid] = label
    rows = [{"qid": qid, "label": mapping[qid]} for qid in sorted(mapping.keys())]
    df = pd.DataFrame(rows)
    path = factbank_dir / "qid_to_label.parquet"
    df.to_parquet(path, index=False)
    logger.info("qid_to_label_saved path=%s rows=%s", path, len(rows))


def _triple_from_ids(subject_id: str, relation_id: str, object_id: str) -> dict[str, str]:
    subject_qid = subject_id if subject_id.startswith("Q") and subject_id[1:].isdigit() else ""
    relation_pid = relation_id if relation_id.startswith("P") and relation_id[1:].isdigit() else ""
    object_qid = object_id if object_id.startswith("Q") and object_id[1:].isdigit() else ""
    object_literal = "" if object_qid else object_id
    return {
        "subject_internal_id": subject_id,
        "relation_internal_id": relation_id,
        "object_internal_id": object_id,
        "subject_qid": subject_qid,
        "relation_pid": relation_pid,
        "object_qid": object_qid,
        "object_literal": object_literal,
        "subject_id": subject_id,
        "subject_label": subject_id,
        "relation_id": relation_id,
        "relation_label": relation_id,
        "object_id_or_value": object_id,
        "object_label": object_id,
    }


def _resolve_wikidata5m_urls() -> tuple[str, ...]:
    env_value = os.getenv(WIKIDATA5M_URL_ENV, "").strip()
    if not env_value:
        return WIKIDATA5M_TAR_URLS
    urls = tuple(part.strip() for part in env_value.split(",") if part.strip())
    return urls if urls else WIKIDATA5M_TAR_URLS


def _download_wikidata5m_archive(archive_path: Path) -> None:
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (compatible; CFM/1.0; +https://github.com)"},
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://data.dgl.ai/dataset/wikidata5m/",
        },
    ]
    last_error: Exception | None = None
    urls = _resolve_wikidata5m_urls()
    for url in urls:
        candidate = Path(url)
        if candidate.exists():
            shutil.copyfile(candidate, archive_path)
            if archive_path.stat().st_size > 0:
                return
            continue
        for headers in headers_list:
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request) as response, archive_path.open("wb") as f:
                    shutil.copyfileobj(response, f)
                if archive_path.stat().st_size > 0:
                    return
            except Exception as exc:
                last_error = exc
                continue
    raise RuntimeError(f"failed to download wikidata5m urls={urls} last_error={last_error}")


def _ensure_wikidata5m_transductive(transductive_dir: Path, allow_download: bool) -> None:
    missing = [name for name in WIKIDATA5M_EXPECTED if not (transductive_dir / name).exists()]
    if not missing:
        return
    if not allow_download:
        raise FileNotFoundError(f"missing wikidata5m splits at {transductive_dir}: {missing}")
    base_dir = transductive_dir.parent
    archive_path = base_dir / "wikidata5m_transductive.tar.gz"
    base_dir.mkdir(parents=True, exist_ok=True)
    urls = _resolve_wikidata5m_urls()
    logger.info(
        "wikidata5m_download start urls=%s dest=%s env=%s",
        urls,
        archive_path,
        WIKIDATA5M_URL_ENV,
    )
    _download_wikidata5m_archive(archive_path)
    logger.info("wikidata5m_download done bytes=%s", archive_path.stat().st_size)
    transductive_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=transductive_dir)
    _recover_wikidata5m_files(transductive_dir)
    missing = [name for name in WIKIDATA5M_EXPECTED if not (transductive_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing wikidata5m splits after extract: {missing}")


def _recover_wikidata5m_files(transductive_dir: Path) -> None:
    for name in WIKIDATA5M_EXPECTED:
        target = transductive_dir / name
        if target.exists():
            continue
        found = next(transductive_dir.rglob(name), None)
        if found is None:
            continue
        if found.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(found), str(target))
        logger.info("wikidata5m_recovered file=%s source=%s", target, found)


def _load_wikidata5m_triples(
    transductive_dir: Path,
    include_valid: bool,
    include_test: bool,
    max_facts: int,
    seed: int,
    allow_download: bool,
    boost_lama_relations: bool,
    min_per_pid: int,
    max_per_pid: int,
    cache_dir: Path,
    local_files_only: bool,
) -> list[dict[str, str]]:
    train_path = transductive_dir / "wikidata5m_transductive_train.txt"
    valid_path = transductive_dir / "wikidata5m_transductive_valid.txt"
    test_path = transductive_dir / "wikidata5m_transductive_test.txt"

    _ensure_wikidata5m_transductive(transductive_dir, allow_download)

    paths = [train_path]
    if include_valid:
        paths.append(valid_path)
    if include_test:
        paths.append(test_path)

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing wikidata5m split: {path}")

    if max_facts <= 0:
        triples: list[dict[str, str]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    subject_id, relation_id, object_id = (p.strip() for p in parts)
                    if not subject_id or not relation_id or not object_id:
                        continue
                    triples.append(_triple_from_ids(subject_id, relation_id, object_id))
        logger.info(
            "wikidata5m_loaded splits=%s total_seen=%s sampled=%s",
            len(paths),
            len(triples),
            len(triples),
        )
        return triples

    rng = random.Random(seed)
    boosted: list[dict[str, str]] = []
    boosted_keys: set[tuple[str, str, str]] = set()
    lama_pid_set: set[str] = set()
    pid_heaps: dict[str, list[tuple[int, dict[str, str]]]] = {}
    pid_seen: dict[str, int] = {}

    if boost_lama_relations:
        lama_pid_set = _load_lama_relation_ids(cache_dir, local_files_only)
        logger.info("lama_boost_relations loaded=%s", len(lama_pid_set))
        pid_heaps = {pid: [] for pid in lama_pid_set}

        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3:
                        continue
                    subject_id, relation_id, object_id = (p.strip() for p in parts)
                    if not subject_id or not relation_id or not object_id:
                        continue
                    if relation_id not in lama_pid_set:
                        continue
                    pid_seen[relation_id] = pid_seen.get(relation_id, 0) + 1
                    heap = pid_heaps[relation_id]
                    triple_hash = _stable_hash(seed, subject_id, relation_id, object_id)
                    triple = _triple_from_ids(subject_id, relation_id, object_id)
                    if len(heap) < max_per_pid:
                        heap.append((triple_hash, triple))
                        if len(heap) == max_per_pid:
                            heap.sort(key=lambda x: x[0], reverse=True)
                        continue
                    if heap and triple_hash < heap[0][0]:
                        heap[0] = (triple_hash, triple)
                        heap.sort(key=lambda x: x[0], reverse=True)

        for pid, heap in pid_heaps.items():
            kept = [item[1] for item in heap]
            for triple in kept:
                key = (
                    triple["subject_id"],
                    triple["relation_id"],
                    triple["object_id_or_value"],
                )
                if key in boosted_keys:
                    continue
                boosted_keys.add(key)
                boosted.append(triple)
            seen_count = pid_seen.get(pid, 0)
            kept_count = len(kept)
            if kept_count < min_per_pid:
                logger.warning(
                    "lama_boost_missing pid=%s seen=%s kept=%s min=%s",
                    pid,
                    seen_count,
                    kept_count,
                    min_per_pid,
                )

        logger.info(
            "lama_boost_done relations=%s boosted=%s",
            len(lama_pid_set),
            len(boosted),
        )

    budget = max_facts - len(boosted)
    if budget <= 0:
        logger.info(
            "wikidata5m_loaded splits=%s lama_seen=%s sampled=%s boosted_only=true",
            len(paths),
            sum(pid_seen.values()),
            len(boosted),
        )
        return boosted[:max_facts]

    reservoir: list[dict[str, str]] = []
    seen = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                subject_id, relation_id, object_id = (p.strip() for p in parts)
                if not subject_id or not relation_id or not object_id:
                    continue
                key = (subject_id, relation_id, object_id)
                if key in boosted_keys:
                    continue
                triple = _triple_from_ids(subject_id, relation_id, object_id)
                seen += 1
                if len(reservoir) < budget:
                    reservoir.append(triple)
                    continue
                j = rng.randrange(seen)
                if j < budget:
                    reservoir[j] = triple

    logger.info(
        "wikidata5m_loaded splits=%s total_seen=%s sampled=%s boosted=%s",
        len(paths),
        seen,
        len(reservoir) + len(boosted),
        len(boosted),
    )
    return boosted + reservoir


def _download_lama_archive(cache_dir: Path, local_files_only: bool) -> Path:
    from datasets import DownloadConfig
    from datasets.download.download_manager import DownloadManager

    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    logger.info("download_lama start url=%s", LAMA_DATA_URL)
    archive_path = Path(dl_manager.download(LAMA_DATA_URL))
    logger.info("download_lama done path=%s", archive_path)
    return archive_path


def _download_lama_relations(cache_dir: Path, local_files_only: bool) -> Path:
    from datasets import DownloadConfig
    from datasets.download.download_manager import DownloadManager

    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    logger.info("download_lama_relations start url=%s", LAMA_RELATIONS_URL)
    relations_path = Path(dl_manager.download(LAMA_RELATIONS_URL))
    logger.info("download_lama_relations done path=%s", relations_path)
    return relations_path


def _load_lama_relation_ids(cache_dir: Path, local_files_only: bool) -> set[str]:
    relations_path = _download_lama_relations(cache_dir, local_files_only)
    relation_ids: set[str] = set()
    with relations_path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            rel = str(data.get("relation", "")).strip()
            if rel.startswith("P") and rel[1:].isdigit():
                relation_ids.add(rel)
    return relation_ids


def _stable_hash(seed: int, subject_id: str, relation_id: str, object_id: str) -> int:
    payload = f"{seed}|{subject_id}|{relation_id}|{object_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _extract_lama_literals(archive_path: Path, max_samples: int | None = None) -> list[dict[str, str]]:
    rows = []
    seen = set()
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name
            if name not in _LAMA_GOOGLE_RE_FILES:
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            for raw in fileobj:
                data = json.loads(raw.decode("utf-8"))
                sub_uri = str(data.get("sub_uri", "")).strip()
                obj_uri = str(data.get("obj_uri", "")).strip()
                predicate_id = str(data.get("predicate_id", "")).strip()
                masked_sentence = str(data.get("masked_sentence", "")).strip()
                gold_label = str(data.get("obj_label") or data.get("object_label") or data.get("answer") or "").strip()
                if not gold_label:
                    continue
                if gold_label.startswith("Q") and gold_label[1:].isdigit():
                    continue
                key = (sub_uri, predicate_id, gold_label, masked_sentence)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "sub_uri": sub_uri,
                    "obj_uri": obj_uri,
                    "predicate_id": predicate_id,
                    "masked_sentence": masked_sentence,
                    "obj_literal": gold_label,
                    "gold_label": gold_label,
                })
                if max_samples and len(rows) >= max_samples:
                    return rows
    return rows


def _save_lama_literals(
    factbank_dir: Path,
    cache_dir: Path,
    local_files_only: bool,
    max_samples: int | None = None,
) -> None:
    archive_path = _download_lama_archive(cache_dir, local_files_only)
    rows = _extract_lama_literals(archive_path, max_samples=max_samples)
    if not rows:
        logger.info("lama_literals_empty")
        return
    rows.sort(key=lambda r: (r["predicate_id"], r["sub_uri"], r["obj_literal"]))
    df = pd.DataFrame(rows)
    path = factbank_dir / "literals.parquet"
    df.to_parquet(path, index=False)
    logger.info("lama_literals_saved path=%s rows=%s", path, len(rows))


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to config YAML"),
    source: str = typer.Option("hf", help="Data source: hf or wikidata5m"),
    max_facts: int | None = typer.Option(None, help="Override max_facts from config"),
    include_valid: bool = typer.Option(False, help="Include wikidata5m valid split"),
    include_test: bool = typer.Option(False, help="Include wikidata5m test split"),
    boost_lama_relations: bool = typer.Option(True, help="Boost LAMA relation coverage"),
    min_per_pid: int = typer.Option(2000, help="Min facts per LAMA predicate ID"),
    max_per_pid: int = typer.Option(20000, help="Max facts per LAMA predicate ID"),
) -> None:
    cfg = yaml.safe_load(config.read_text())
    seed = int(cfg["seed"])
    set_seed(seed)
    source = source.strip().lower()
    if source not in {"hf", "wikidata5m"}:
        raise typer.BadParameter("source must be one of hf or wikidata5m")

    runtime_cfg = cfg.get("runtime", {})
    cpu_threads = int(runtime_cfg.get("cpu_threads", DEFAULT_CPU_THREADS))
    prefer_gpu = bool(runtime_cfg.get("prefer_gpu", True))
    use_faiss = bool(runtime_cfg.get("use_faiss", True))

    torch_info = configure_torch(cpu_threads=cpu_threads, prefer_gpu=prefer_gpu)

    rvq_cfg = cfg["rvq"]
    rvq_prefer_gpu = bool(rvq_cfg.get("prefer_gpu", False))
    faiss_info = resolve_faiss(prefer_gpu=rvq_prefer_gpu, cpu_threads=cpu_threads) if use_faiss else {
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

    effective_max_facts = int(max_facts) if max_facts is not None else int(data_cfg["max_facts"])
    triples = []
    if source == "wikidata5m":
        transductive_dir = Path("data") / "wikidata5m" / "transductive"
        logger.info(
            "load_wikidata5m start dir=%s include_valid=%s include_test=%s max_facts=%s",
            transductive_dir,
            include_valid,
            include_test,
            effective_max_facts,
        )
        try:
            triples = _load_wikidata5m_triples(
                transductive_dir=transductive_dir,
                include_valid=include_valid,
                include_test=include_test,
                max_facts=effective_max_facts,
                seed=seed,
                allow_download=not bool(data_cfg.get("local_files_only", False)),
                boost_lama_relations=boost_lama_relations,
                min_per_pid=min_per_pid,
                max_per_pid=max_per_pid,
                cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
                local_files_only=bool(data_cfg.get("local_files_only", False)),
            )
        except Exception as exc:
            logger.warning(
                "wikidata5m_transductive_unavailable error=%s fallback=hf",
                exc,
            )
            use_direct_tar = bool(data_cfg.get("use_direct_tar", True))
            raw_path = raw_dir / "wikidata5m.parquet"
            if use_direct_tar:
                ingest_wikidata5m_tar(
                    raw_path,
                    max_triples=effective_max_facts,
                    sources=list(data_cfg.get("wikidata5m_sources", ["transductive"])),
                    cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
                    local_files_only=bool(data_cfg.get("local_files_only", False)),
                )
            else:
                ingest_wikidata5m(
                    raw_path,
                    max_triples=effective_max_facts,
                    seed=seed,
                    use_streaming=bool(data_cfg.get("use_streaming", True)),
                    shuffle=bool(data_cfg.get("shuffle", False)),
                    shuffle_buffer=int(data_cfg.get("shuffle_buffer", 1000)),
                    local_files_only=bool(data_cfg.get("local_files_only", False)),
                    cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
                )
            triples = load_triples(raw_path)
    else:
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
                cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
            )
        triples = load_triples(raw_path)

    logger.info("load_relation_templates start")
    relation_templates = load_relation_templates(
        cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
        local_files_only=bool(data_cfg.get("local_files_only", False)),
    )
    logger.info("load_relation_templates done relations=%s", len(relation_templates))

    logger.info("build_factbank start")
    logger.info("triples_loaded=%s", len(triples))
    records = build_factbank_records(
        triples,
        max_facts=0 if source == "wikidata5m" else effective_max_facts,
        orbits_per_fact=int(data_cfg["orbits_per_fact"]),
        negatives_per_fact=int(data_cfg["negatives_per_fact"]),
        seed=seed,
        templates_by_relation=relation_templates,
        min_cloze=10,
    )
    save_factbank(records, factbank_dir)
    _save_qid_to_label(records, factbank_dir)
    logger.info("facts_built=%s", len(records))
    literal_count = sum(1 for rec in records if str(rec.get("object_literal", "")).strip())
    logger.info("literal_facts=%s", literal_count)

    logger.info("build_answer_embeddings start")
    model_cfg = cfg["model"]
    answers, embeddings = build_answer_embeddings(records, dim=int(model_cfg["d_code"]), emb_dir=emb_dir)
    logger.info("answers=%s emb_dim=%s", len(answers), int(model_cfg["d_code"]))

    require_faiss_gpu = bool(rvq_cfg.get("require_faiss_gpu", False))
    if require_faiss_gpu:
        if not use_faiss:
            raise RuntimeError("rvq.require_faiss_gpu=true requires runtime.use_faiss=true")
        if not faiss_info["faiss_gpu_available"]:
            raise RuntimeError(
                "rvq.require_faiss_gpu=true but faiss GPU is unavailable; set rvq.prefer_gpu=true or disable"
            )

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

    build_lama_literals = bool(data_cfg.get("build_lama_literals", False))
    if build_lama_literals:
        if literal_count > 0:
            logger.info("lama_literals_skipped reason=source_has_literals")
        else:
            max_samples = data_cfg.get("lama_literals_max_samples", None)
            if max_samples is not None:
                max_samples = int(max_samples)
            _save_lama_literals(
                factbank_dir=factbank_dir,
                cache_dir=Path(data_cfg.get("hf_cache_dir", ".cache/huggingface")),
                local_files_only=bool(data_cfg.get("local_files_only", False)),
                max_samples=max_samples,
            )

    metadata_path = factbank_dir / "build_summary.json"
    metadata_path.write_text(json.dumps({"facts": len(records), "answers": len(answers)}, indent=2))

    console.print("\n[bold]Next steps[/bold]")
    console.print("uv run python scripts/train_cfm.py --config configs/default.yaml")
    console.print("uv run python scripts/eval_lama.py --config configs/default.yaml")
    console.print("uv run pytest -q")


if __name__ == "__main__":
    app()
