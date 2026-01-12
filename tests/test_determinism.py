import hashlib
import json

from forge_omega_500.data.build_factbank import build_factbank_records


def _hash_records(records):
    payload = json.dumps(records, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_triples(n):
    triples = []
    for i in range(n):
        triples.append({
            "subject_id": f"S{i}",
            "subject_label": f"Subject {i}",
            "relation_id": f"R{i%3}",
            "relation_label": f"rel_{i%3}",
            "object_id_or_value": f"O{i}",
            "object_label": f"Object {i}",
        })
    return triples


def test_deterministic_factbank_build():
    triples = _make_triples(25)
    records_a = build_factbank_records(triples, max_facts=20, orbits_per_fact=10, negatives_per_fact=5, seed=123)
    records_b = build_factbank_records(triples, max_facts=20, orbits_per_fact=10, negatives_per_fact=5, seed=123)
    assert _hash_records(records_a) == _hash_records(records_b)
