from __future__ import annotations

import json
import string
import tarfile
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

configure_env(DEFAULT_CPU_THREADS)

import numpy as np
import pandas as pd
import typer
import yaml
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager

logger = setup_logger("check_lama_coverage")
app = typer.Typer(add_completion=False)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"


def _normalize(text: str) -> str:
    return text.lower().strip().translate(_PUNCT_TABLE)


def _normalize_id(value: str) -> str:
    text = str(value).strip().strip("<>")
    if not text:
        return ""
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if text.startswith("wd:"):
        text = text[3:]
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


def _extract_uri(row: Dict[str, object], keys: Iterable[str]) -> str:
    return _first_value(row, keys)


def _extract_subject(row: Dict[str, object]) -> str:
    for key in ("sub_label", "subject_label", "sub", "subject"):
        values = _iter_field_values(row.get(key))
        if values:
            return values[0]
    return ""


def _extract_object(row: Dict[str, object]) -> str:
    for key in ("obj_label", "object_label", "obj", "answer"):
        values = _iter_field_values(row.get(key))
        if values:
            return values[0]
    return ""


def _extract_relation(row: Dict[str, object]) -> Tuple[str, str]:
    rel_id = ""
    rel_label = ""
    for key in ("predicate_id", "relation_id", "relation"):
        values = _iter_field_values(row.get(key))
        if values:
            rel_id = values[0]
            break
    for key in ("relation_label", "predicate_label"):
        values = _iter_field_values(row.get(key))
        if values:
            rel_label = values[0]
            break
    template_values = _iter_field_values(row.get("template"))
    if not rel_label and template_values:
        rel_label = template_values[0]
    return rel_id, rel_label


def _iter_facts(df: pd.DataFrame) -> Iterable[Dict[str, str]]:
    for _, row in df.iterrows():
        yield {
            "subject_id": str(row.get("subject_id", "")),
            "subject_label": str(row.get("subject_label", "")),
            "relation_id": str(row.get("relation_id", "")),
            "relation_label": str(row.get("relation_label", "")),
            "object_id_or_value": str(row.get("object_id_or_value", "")),
            "object_label": str(row.get("object_label", "")),
        }


def _download_lama_files(cache_dir: Path, local_files_only: bool) -> Tuple[Path, Path]:
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
                            "predicate_label": str(pred.get("label", "")),
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


def _load_lama_fallback(
    cfg_name: str,
    cache_dir: Path,
    local_files_only: bool,
    max_samples: int,
) -> List[Dict[str, object]]:
    logger.info("fallback_loader start cfg=%s max_samples=%s", cfg_name, max_samples)
    archive_path, relations_path = _download_lama_files(cache_dir, local_files_only=local_files_only)
    records = list(_iter_lama_records(cfg_name, archive_path, relations_path, max_samples=max_samples))
    logger.info("fallback_loader done cfg=%s records=%s", cfg_name, len(records))
    return records


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), help="Path to config YAML"),
    subset: str = typer.Option(..., help="LAMA subset: google_re/trex/conceptnet/squad"),
    limit: int = typer.Option(2000, min=1, help="Max samples to check"),
) -> None:
    subset = subset.strip().lower()
    if subset not in {"google_re", "trex", "conceptnet", "squad"}:
        raise typer.BadParameter("subset must be one of google_re, trex, conceptnet, squad")

    cfg = yaml.safe_load(config.read_text())
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
    factbank_dir = Path(data_cfg["factbank_dir"])
    facts_path = factbank_dir / "facts.parquet"
    if not facts_path.exists():
        raise typer.BadParameter(f"missing facts.parquet at {facts_path}")

    logger.info("load_factbank start path=%s", facts_path)
    t0 = time.perf_counter()
    df = pd.read_parquet(facts_path)
    rows = len(df)
    any_set = set()
    strict_label_set = set()
    strict_rel_id_set = set()
    id_keys = set()
    for fact in _iter_facts(df):
        subj_id = _normalize_id(fact.get("subject_id", ""))
        rel_id_raw = _normalize_id(fact.get("relation_id", ""))
        obj_id = _normalize_id(fact.get("object_id_or_value", ""))
        if subj_id and rel_id_raw and obj_id:
            id_keys.add((subj_id, rel_id_raw, obj_id))
        subj = _normalize(fact["subject_label"])
        obj = _normalize(fact["object_label"])
        if not subj or not obj:
            continue
        any_set.add((subj, obj))
        rel_id = _normalize(fact["relation_id"])
        rel_label = _normalize(fact["relation_label"])
        if rel_id:
            strict_rel_id_set.add((subj, rel_id, obj))
        if rel_label:
            strict_label_set.add((subj, rel_label, obj))
    strict_id_set = id_keys
    logger.info(
        "load_factbank done rows=%s id_keys=%s any_keys=%s strict_label=%s strict_id=%s time=%.2fs",
        rows,
        len(id_keys),
        len(any_set),
        len(strict_label_set),
        len(strict_id_set),
        time.perf_counter() - t0,
    )

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))

    logger.info("load_lama start subset=%s limit=%s", subset, limit)
    use_fallback = False
    try:
        available = get_dataset_config_names("facebook/lama")
        if subset not in available:
            raise ValueError(f"subset_not_found: {subset}")
        ds = load_dataset("facebook/lama", subset, split="train")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
    except Exception as exc:
        logger.warning("load_dataset failed: %s; falling back to tar loader", exc)
        use_fallback = True
        ds = _load_lama_fallback(subset, cache_dir, local_files_only, limit)
    logger.info("load_lama done samples=%s use_fallback=%s", len(ds), use_fallback)

    total = 0
    any_hits = 0
    strict_hits = 0
    strict_eligible = 0
    literal_hits = 0
    literal_total = 0
    matched = []
    unmatched = []

    for row in ds:
        sub_uri = _extract_uri(row, ("sub_uri", "subj_uri", "subject_uri"))
        obj_uri = _extract_uri(row, ("obj_uri", "object_uri"))
        rel_id_raw, rel_label_raw = _extract_relation(row)

        if subset == "trex":
            sub_id = _normalize_id(sub_uri)
            obj_id = _normalize_id(obj_uri)
            rel_id = _normalize_id(rel_id_raw)
            if not sub_id or not obj_id or not rel_id:
                continue
            total += 1
            id_match = (sub_id, rel_id, obj_id) in strict_id_set
            if id_match:
                any_hits += 1
                strict_hits += 1
            strict_eligible += 1
            if len(matched) < 5 and id_match:
                matched.append({"subject": sub_uri, "relation": rel_id, "object": obj_uri})
            if len(unmatched) < 5 and not id_match:
                unmatched.append({"subject": sub_uri, "relation": rel_id, "object": obj_uri})
            continue

        subj_raw = _extract_subject(row)
        obj_raw = _extract_object(row)
        if not subj_raw or not obj_raw:
            continue
        subj = _normalize(subj_raw)
        obj = _normalize(obj_raw)
        if not subj or not obj:
            continue
        total += 1

        any_match = (subj, obj) in any_set
        if any_match:
            any_hits += 1

        if subset == "google_re":
            literal_total += 1
            if any_match:
                literal_hits += 1
            if len(matched) < 5 and any_match:
                entry = {"subject": subj_raw, "relation": rel_id_raw or rel_label_raw, "object": obj_raw}
                if sub_uri:
                    entry["sub_uri"] = sub_uri
                if obj_uri:
                    entry["obj_uri"] = obj_uri
                matched.append(entry)
            if len(unmatched) < 5 and not any_match:
                entry = {"subject": subj_raw, "relation": rel_id_raw or rel_label_raw, "object": obj_raw}
                if sub_uri:
                    entry["sub_uri"] = sub_uri
                if obj_uri:
                    entry["obj_uri"] = obj_uri
                unmatched.append(entry)
            continue

        rel_id = _normalize(rel_id_raw)
        rel_label = _normalize(rel_label_raw)
        strict_match = False
        if rel_id or rel_label:
            strict_eligible += 1
            if rel_id and (subj, rel_id, obj) in strict_rel_id_set:
                strict_match = True
            elif rel_label and (subj, rel_label, obj) in strict_label_set:
                strict_match = True
            if strict_match:
                strict_hits += 1

        if len(matched) < 5 and (any_match or strict_match):
            entry = {"subject": subj_raw, "relation": rel_id_raw or rel_label_raw, "object": obj_raw}
            if sub_uri:
                entry["sub_uri"] = sub_uri
            if obj_uri:
                entry["obj_uri"] = obj_uri
            matched.append(entry)
        if len(unmatched) < 5 and not any_match:
            entry = {"subject": subj_raw, "relation": rel_id_raw or rel_label_raw, "object": obj_raw}
            if sub_uri:
                entry["sub_uri"] = sub_uri
            if obj_uri:
                entry["obj_uri"] = obj_uri
            unmatched.append(entry)

    if subset == "google_re":
        coverage_literal = literal_hits / max(literal_total, 1)
        print(f"coverage_literal: {coverage_literal:.4f} ({literal_hits}/{literal_total})")
    else:
        coverage_any = any_hits / max(total, 1)
        coverage_strict = strict_hits / max(strict_eligible, 1)
        print(f"coverage_any: {coverage_any:.4f} ({any_hits}/{total})")
        print(f"coverage_strict: {coverage_strict:.4f} ({strict_hits}/{strict_eligible})")
    print("matched_examples:")
    for ex in matched:
        print(f"  - {ex}")
    print("unmatched_examples:")
    for ex in unmatched:
        print(f"  - {ex}")


if __name__ == "__main__":
    app()
