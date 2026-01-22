from __future__ import annotations

import json
import sys
import tarfile
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List

import typer
import yaml
from datasets import DownloadConfig, get_dataset_config_names, load_dataset
from datasets.download.download_manager import DownloadManager

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from codebook import ensure_answers_in_codebook, normalize_lama_answer, load_answer_codebook

app = typer.Typer(add_completion=False)

LAMA_DATA_URL = "https://dl.fbaipublicfiles.com/LAMA/negated_data.tar.gz"
LAMA_RELATIONS_URL = "https://s3.amazonaws.com/datasets.huggingface.co/lama/relations.jsonl"
_DEFAULT_LAMA_SUBSETS = ("google_re", "trex", "conceptnet", "squad")


def _download_lama_files(cache_dir: Path, local_files_only: bool) -> tuple[Path, Path]:
    dl_config = DownloadConfig(cache_dir=str(cache_dir), local_files_only=local_files_only)
    dl_manager = DownloadManager(download_config=dl_config)
    archive_path = Path(dl_manager.download(LAMA_DATA_URL))
    relations_path = Path(dl_manager.download(LAMA_RELATIONS_URL))
    return archive_path, relations_path


def _first_value(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def _iter_lama_records(
    cfg_name: str,
    archive_path: Path,
    relations_path: Path,
    max_samples: int,
) -> Iterable[Dict[str, object]]:
    if cfg_name == "trex":
        patterns = ["TREx/*"]
    elif cfg_name == "google_re":
        patterns = [
            "Google_RE/date_of_birth_test.jsonl",
            "Google_RE/place_of_birth_test.jsonl",
            "Google_RE/place_of_death_test.jsonl",
        ]
    elif cfg_name == "conceptnet":
        patterns = ["ConceptNet/test.jsonl"]
    elif cfg_name == "squad":
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
                    gold_label = _first_value(data.get("obj_label") or data.get("object_label") or data.get("obj"))
                    obj_uri = _first_value(data.get("obj_uri") or data.get("object_uri"))
                    for evidence in data.get("evidences", []):
                        yield {
                            "gold": gold_label,
                            "gold_uri": obj_uri,
                            "masked_sentence": _first_value(evidence.get("masked_sentence")),
                        }
                        yielded += 1
                        if max_samples and yielded >= max_samples:
                            return
                else:
                    record = dict(data)
                    record["gold"] = _first_value(
                        data.get("obj_label") or data.get("object_label") or data.get("obj") or data.get("answer")
                    )
                    record["gold_uri"] = _first_value(data.get("obj_uri") or data.get("object_uri"))
                    yield record
                    yielded += 1
                    if max_samples and yielded >= max_samples:
                        return


def _load_lama_records(
    cfg_name: str,
    cache_dir: Path,
    local_files_only: bool,
    max_samples: int,
    use_fallback: bool,
) -> Iterable[Dict[str, object]]:
    if use_fallback:
        archive_path, relations_path = _download_lama_files(cache_dir, local_files_only)
        return list(_iter_lama_records(cfg_name, archive_path, relations_path, max_samples=max_samples))
    dataset = load_dataset("facebook/lama", cfg_name, split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


@app.command()
def main(config: Path = typer.Option(..., help="Path to config YAML")) -> None:
    cfg = yaml.safe_load(config.read_text())
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    model_cfg = cfg["model"]

    codes_dir = Path(data_cfg["codes_dir"])
    codebook_path = codes_dir / "answer_codes.parquet"
    code_len = int(model_cfg["m"])
    code_vocab = int(model_cfg["K"])

    subsets = eval_cfg.get("subsets", list(_DEFAULT_LAMA_SUBSETS))
    if isinstance(subsets, str):
        subsets = [subsets]
    subsets = [str(name).strip().lower() for name in subsets if str(name).strip()]
    if not subsets:
        subsets = list(_DEFAULT_LAMA_SUBSETS)

    max_samples = int(eval_cfg.get("max_samples", 0))
    local_files_only = bool(data_cfg.get("local_files_only", False))
    cache_dir = Path(data_cfg.get("hf_cache_dir", ".cache/huggingface"))

    use_fallback = False
    try:
        available = get_dataset_config_names("facebook/lama")
        subsets = [s for s in subsets if s in available]
    except Exception:
        use_fallback = True

    answers: List[str] = []
    for subset in subsets:
        records = _load_lama_records(
            subset,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_samples=max_samples,
            use_fallback=use_fallback,
        )
        for rec in records:
            gold = _first_value(rec.get("gold"))
            gold_uri = _first_value(rec.get("gold_uri"))
            answers.append(normalize_lama_answer(gold, gold_uri))

    stats = ensure_answers_in_codebook(
        codebook_path,
        answers,
        code_len=code_len,
        code_vocab=code_vocab,
        method="sha1_mod",
    )
    updated = load_answer_codebook(codebook_path)
    still_missing = len({a for a in answers if a} - set(updated.keys()))
    print(
        "answers_total={total} answers_unique={unique} added={added} still_missing={missing}".format(
            total=stats["total_answers"],
            unique=stats["unique_answers"],
            added=stats["added"],
            missing=still_missing,
        )
    )


if __name__ == "__main__":
    app()
