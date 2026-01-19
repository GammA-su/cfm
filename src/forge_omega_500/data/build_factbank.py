from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _clean_label(value: object, fallback: str) -> str:
    text = str(value).strip()
    return text if text else fallback


def _is_qid(value: str) -> bool:
    return value.startswith("Q") and value[1:].isdigit()


def _is_pid(value: str) -> bool:
    return value.startswith("P") and value[1:].isdigit()


def _render_template(template: str, subject: str, obj_placeholder: str) -> str:
    text = str(template)
    text = text.replace("[X]", subject).replace("[Y]", obj_placeholder)
    return text


def _generate_orbits(
    subject_label: str,
    relation_label: str,
    object_label: str,
    count: int,
    seed: int,
    template: str | None = None,
) -> List[str]:
    rng = random.Random(seed)
    subject = subject_label or "Unknown"
    relation = relation_label or "relation"
    base_templates = [
        "Fill in the blank: {subject} {relation} ____.",
        "{subject} {relation} ____.",
        "What is the {relation} of {subject}?",
        "The {relation} of {subject} is ____.",
        "{subject} has {relation} ____.",
    ]
    if template:
        base_templates.append("Fill in the blank: " + _render_template(template, subject, "____"))
    rng.shuffle(base_templates)
    orbits: List[str] = []
    if count <= 0:
        return orbits
    idx = 0
    while len(orbits) < count:
        tmpl = base_templates[idx % len(base_templates)]
        if "{object}" in tmpl:
            orbit = tmpl.format(subject=subject, relation=relation, object=object_label)
        else:
            orbit = tmpl.format(subject=subject, relation=relation)
        orbits.append(orbit)
        idx += 1
    return orbits


def _prepare_triples(
    triples: Iterable[Dict[str, object]],
    max_facts: int,
    seed: int,
) -> List[Dict[str, object]]:
    records = list(triples)
    records.sort(
        key=lambda r: (
            str(r.get("subject_id", "")),
            str(r.get("relation_id", "")),
            str(r.get("object_id_or_value", "")),
        )
    )
    if max_facts and max_facts > 0 and len(records) > max_facts:
        rng = random.Random(seed)
        indices = list(range(len(records)))
        rng.shuffle(indices)
        indices = indices[:max_facts]
        indices.sort()
        records = [records[i] for i in indices]
    return records


def build_factbank_records(
    triples: Iterable[Dict[str, object]],
    max_facts: int,
    orbits_per_fact: int,
    negatives_per_fact: int,
    seed: int,
    templates_by_relation: Optional[Dict[str, str]] = None,
    min_cloze: int = 10,
) -> List[Dict[str, object]]:
    selected = _prepare_triples(triples, max_facts=max_facts, seed=seed)
    records: List[Dict[str, object]] = []
    for idx, triple in enumerate(selected):
        subject_id = _clean_label(triple.get("subject_id"), "")
        relation_id = _clean_label(triple.get("relation_id"), "")
        object_id = _clean_label(triple.get("object_id_or_value"), "")
        subject_label = _clean_label(triple.get("subject_label"), subject_id)
        relation_label = _clean_label(triple.get("relation_label"), relation_id)
        object_label = _clean_label(triple.get("object_label"), object_id)
        template = None
        if templates_by_relation:
            template = templates_by_relation.get(relation_id)
        orbits = _generate_orbits(
            subject_label,
            relation_label,
            object_label,
            orbits_per_fact,
            seed=seed + idx,
            template=template,
        )
        subject_qid = subject_id if _is_qid(subject_id) else ""
        relation_pid = relation_id if _is_pid(relation_id) else ""
        object_qid = object_id if _is_qid(object_id) else ""
        object_literal = "" if object_qid else object_id
        record: Dict[str, object] = {
            "fact_id": idx,
            "subject_id": subject_id,
            "subject_label": subject_label,
            "relation_id": relation_id,
            "relation_label": relation_label,
            "object_id_or_value": object_id,
            "object_label": object_label,
            "question_orbits": orbits,
            "hard_negatives": [],
            "subject_qid": subject_qid,
            "relation_pid": relation_pid,
            "object_qid": object_qid,
            "object_literal": object_literal,
        }
        records.append(record)

    if records and negatives_per_fact > 0:
        total = len(records)
        for idx, record in enumerate(records):
            negatives: List[Dict[str, object]] = []
            offset = 1
            while len(negatives) < negatives_per_fact and offset < total + negatives_per_fact:
                other = records[(idx + offset) % total]
                if other["object_label"] != record["object_label"]:
                    negatives.append({
                        "subject_id": other["subject_id"],
                        "relation_id": other["relation_id"],
                        "object_id_or_value": other["object_id_or_value"],
                        "object_label": other["object_label"],
                    })
                offset += 1
            record["hard_negatives"] = negatives
    return records


def save_factbank(records: List[Dict[str, object]], factbank_dir: Path) -> None:
    factbank_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(factbank_dir / "facts.parquet", index=False)
