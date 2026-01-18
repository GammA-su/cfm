from __future__ import annotations

import json
import tarfile
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import typer
import yaml
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager

from forge_omega_500.runtime import DEFAULT_CPU_THREADS, configure_env, setup_logger

app = typer.Typer(add_completion=False)
logger = setup_logger("eval_lama_oracle")

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"


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
    text = str(value).strip()
    return [text] if text else []


def _first_value(row: Dict[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        values = _iter_field_values(row.get(key))
        if values:
            return values[0]
    return ""


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
                    if cfg_name == "google_re" and not predicate_id:
                        record["literal_only"] = True
                    record["sub_uri"] = _first_value(data, ("sub_uri", "subj_uri", "subject_uri"))
                    record["obj_uri"] = _first_value(data, ("obj_uri", "object_uri"))
                    record["gold_label"] = _first_value(data, ("obj_label", "object_label", "obj", "answer"))
                    yield record
                    yielded += 1
                    if max_samples and yielded >= max_samples:
                        return


def _load_lama_fallback(cfg_name: str, cache_dir: Path, local_files_only: bool, max_samples: int) -> List[Dict[str, object]]:
    logger.info("fallback_loader start cfg=%s max_samples=%s", cfg_name, max_samples)
    archive_path, relations_path = _download_lama_files(cache_dir, local_files_only=local_files_only)
    records = list(_iter_lama_records(cfg_name, archive_path, relations_path, max_samples=max_samples))
    logger.info("fallback_loader done cfg=%s records=%s", cfg_name, len(records))
    return records


def _clean_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _build_fact_index(df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    index: Dict[Tuple[str, str], List[str]] = {}
    for _, row in df.iterrows():
        subject_id = _clean_value(row.get("subject_id"))
        relation_id = _clean_value(row.get("relation_id"))
        object_id = _clean_value(row.get("object_id_or_value"))
        if not subject_id or not relation_id or not object_id:
            continue
        key = (subject_id, relation_id)
        values = index.setdefault(key, [])
        values.append(object_id)
    return index


def _extract_fields(example: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    masked_sentence = _first_value(example, ("masked_sentence", "masked_sentences", "masked_prompt", "sentence"))
    predicate_id = _first_value(example, ("predicate_id", "relation_id", "relation"))
    sub_uri = _first_value(example, ("sub_uri", "subj_uri", "subject_uri"))
    obj_uri = _first_value(example, ("obj_uri", "object_uri"))
    gold_label = _first_value(example, ("obj_label", "object_label", "obj", "answer"))
    return sub_uri, predicate_id, obj_uri, masked_sentence, gold_label


@app.command()
def main(
    config: Path = typer.Option(..., help="Path to config YAML"),
    subset: str = typer.Option(..., help="LAMA subset: trex, google_re, conceptnet, squad"),
    limit: int = typer.Option(2000, help="Max samples to evaluate"),
) -> None:
    configure_env(DEFAULT_CPU_THREADS)
    t0 = time.perf_counter()

    cfg = yaml.safe_load(config.read_text())
    data_cfg = cfg.get("data", {})
    factbank_dir = Path(data_cfg.get("factbank_dir", "data/factbank"))
    fact_path = factbank_dir / "facts.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"missing factbank: {fact_path}")

    subset = subset.strip().lower()
    if subset not in {"trex", "google_re", "conceptnet", "squad"}:
        raise typer.BadParameter("subset must be one of trex, google_re, conceptnet, squad")

    logger.info("load_factbank start path=%s", fact_path)
    df = pd.read_parquet(fact_path, columns=["subject_id", "relation_id", "object_id_or_value"])
    fact_index = _build_fact_index(df)
    logger.info("load_factbank done rows=%s index_keys=%s", len(df), len(fact_index))

    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))

    use_fallback = False
    logger.info("discover_lama_configs start")
    try:
        available = get_dataset_config_names("facebook/lama")
        if subset not in available:
            use_fallback = True
    except Exception as exc:
        logger.warning("get_dataset_config_names failed: %s; falling back to tar loader", exc)
        use_fallback = True
    logger.info("discover_lama_configs done use_fallback=%s", use_fallback)

    logger.info("load_dataset start cfg=%s", subset)
    load_t0 = time.perf_counter()
    if use_fallback:
        ds: Iterable[Dict[str, object]] = _load_lama_fallback(subset, cache_dir, local_files_only, limit)
    else:
        dataset = load_dataset("facebook/lama", subset, split="train")
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        ds = dataset
    logger.info("load_dataset done cfg=%s time=%.2fs", subset, time.perf_counter() - load_t0)

    supported = 0
    covered = 0
    uncovered = 0
    ambiguous = 0
    correct = 0
    total = 0
    unsupported = 0

    uncovered_examples: List[Dict[str, str]] = []
    ambiguous_examples: List[Dict[str, str]] = []
    max_examples = 10

    eval_t0 = time.perf_counter()
    for example in ds:
        total += 1
        sub_uri, predicate_id, obj_uri, masked_sentence, gold_label = _extract_fields(example)
        if not sub_uri or not predicate_id or not obj_uri:
            unsupported += 1
            continue
        supported += 1
        key = (sub_uri, predicate_id)
        objects = fact_index.get(key)
        if not objects:
            uncovered += 1
            if len(uncovered_examples) < max_examples:
                uncovered_examples.append({
                    "sub_uri": sub_uri,
                    "predicate_id": predicate_id,
                    "obj_uri": obj_uri,
                    "masked_sentence": masked_sentence,
                    "gold_label": gold_label,
                })
            continue
        covered += 1
        if len(set(objects)) > 1:
            ambiguous += 1
            if len(ambiguous_examples) < max_examples:
                ambiguous_examples.append({
                    "sub_uri": sub_uri,
                    "predicate_id": predicate_id,
                    "obj_uri": obj_uri,
                    "masked_sentence": masked_sentence,
                    "gold_label": gold_label,
                    "object_count": str(len(set(objects))),
                })
        if obj_uri in objects:
            correct += 1

    accuracy_oracle = (correct / supported) if supported else 0.0
    covered_rate = (covered / supported) if supported else 0.0
    ambiguity_rate = (ambiguous / covered) if covered else 0.0

    report = {
        "subset": subset,
        "limit": limit,
        "total_samples": total,
        "supported_samples": supported,
        "unsupported_samples": unsupported,
        "covered_samples": covered,
        "uncovered_samples": uncovered,
        "ambiguous_samples": ambiguous,
        "accuracy_oracle": accuracy_oracle,
        "covered_rate": covered_rate,
        "ambiguity_rate": ambiguity_rate,
        "examples": {
            "uncovered": uncovered_examples,
            "ambiguous": ambiguous_examples,
        },
    }

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lama_oracle_{subset}.json"
    out_path.write_text(json.dumps(report, indent=2))

    logger.info(
        "oracle_done subset=%s total=%s supported=%s covered=%s correct=%s ambiguous=%s time=%.2fs",
        subset,
        total,
        supported,
        covered,
        correct,
        ambiguous,
        time.perf_counter() - eval_t0,
    )
    logger.info("report_saved path=%s total_time=%.2fs", out_path, time.perf_counter() - t0)


if __name__ == "__main__":
    app()
