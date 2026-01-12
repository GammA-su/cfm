import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from forge_omega_500.data.build_factbank import build_factbank_records
from forge_omega_500.data.embeddings import build_answer_embeddings
from forge_omega_500.data.rvq import assign_codes, train_rvq
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed


def _make_triples(n):
    triples = []
    for i in range(n):
        triples.append({
            "subject_id": f"S{i}",
            "subject_label": f"Subject {i}",
            "relation_id": f"R{i%4}",
            "relation_label": f"rel_{i%4}",
            "object_id_or_value": f"O{i}",
            "object_label": f"Object {i}",
        })
    return triples


def _orbit_consistency(model, records, tokenizer, max_seq_len):
    model.eval()
    consistent = 0
    with torch.no_grad():
        for rec in records:
            prompts = [[tokenizer.bos_id] + tokenizer.encode(o) for o in rec["question_orbits"]]
            prompt_ids, prompt_masks = pad_sequences(prompts, tokenizer.pad_id, max_len=max_seq_len)
            codes = model.predict_codes(prompt_ids, prompt_masks).cpu().numpy()
            if np.all(codes == codes[0]):
                consistent += 1
    return consistent / len(records)


def test_orbit_invariance_improves():
    set_seed(7)
    triples = _make_triples(20)
    records = build_factbank_records(triples, max_facts=20, orbits_per_fact=10, negatives_per_fact=5, seed=7)
    answers, embeddings = build_answer_embeddings(records, dim=32, emb_dir=Path("/tmp/emb_orbit"))

    codebooks = train_rvq(embeddings, m=2, K=32, iters=4, seed=7)
    codes = assign_codes(embeddings, codebooks)
    answer_codes = {ans: codes[i].tolist() for i, ans in enumerate(answers)}

    examples = []
    for rec in records:
        for orbit in rec["question_orbits"]:
            examples.append({"prompt": orbit, "codes": answer_codes[rec["object_label"]]})

    texts = [ex["prompt"] for ex in examples]
    tokenizer = SimpleTokenizer.build(texts)

    model = CFMModel(
        vocab_size=len(tokenizer.vocab),
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=48,
        m=2,
        K=32,
        codebooks=torch.tensor(codebooks),
    )

    baseline = _orbit_consistency(model, records, tokenizer, max_seq_len=48)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    rng = random.Random(7)

    for _ in range(60):
        batch = [examples[rng.randrange(len(examples))] for _ in range(16)]
        prompt_ids = [[tokenizer.bos_id] + tokenizer.encode(ex["prompt"]) for ex in batch]
        prompt_ids, prompt_masks = pad_sequences(prompt_ids, tokenizer.pad_id, max_len=48)
        code_targets = torch.tensor([ex["codes"] for ex in batch], dtype=torch.long)

        addr_logits, _, _ = model.encode_prompt(prompt_ids, prompt_masks)
        addr_loss = 0.0
        for i, logits in enumerate(addr_logits):
            addr_loss = addr_loss + F.cross_entropy(logits, code_targets[:, i])

        optimizer.zero_grad()
        addr_loss.backward()
        optimizer.step()

    improved = _orbit_consistency(model, records, tokenizer, max_seq_len=48)
    assert improved >= baseline + 0.1
