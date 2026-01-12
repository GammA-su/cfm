from pathlib import Path

from forge_omega_500.data.build_factbank import build_factbank_records
from forge_omega_500.data.embeddings import build_answer_embeddings
from forge_omega_500.data.rvq import assign_codes, train_rvq


def _make_triples(n):
    triples = []
    for i in range(n):
        triples.append({
            "subject_id": f"S{i}",
            "subject_label": f"Subject {i}",
            "relation_id": f"R{i%5}",
            "relation_label": f"rel_{i%5}",
            "object_id_or_value": f"O{i}",
            "object_label": f"Object {i}",
        })
    return triples


def test_negative_codes_differ():
    triples = _make_triples(30)
    records = build_factbank_records(triples, max_facts=30, orbits_per_fact=10, negatives_per_fact=5, seed=11)

    answers, embeddings = build_answer_embeddings(records, dim=32, emb_dir=Path("/tmp/emb_neg"))
    codebooks = train_rvq(embeddings, m=2, K=32, iters=4, seed=11)
    codes = assign_codes(embeddings, codebooks)
    answer_codes = {ans: codes[i].tolist() for i, ans in enumerate(answers)}

    same = 0
    total = 0
    for rec in records:
        pos_codes = answer_codes[rec["object_label"]]
        for neg in rec["hard_negatives"]:
            neg_label = neg["object_label"]
            if neg_label not in answer_codes:
                continue
            neg_codes = answer_codes[neg_label]
            total += 1
            if neg_codes == pos_codes:
                same += 1

    rate = same / max(total, 1)
    assert rate < 0.2
