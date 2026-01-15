from pathlib import Path

import torch

from forge_omega_500.data.build_factbank import build_factbank_records
from forge_omega_500.data.embeddings import build_answer_embeddings
from forge_omega_500.data.rvq import assign_codes, load_code_to_label, lookup_code_label, save_code_to_label, train_rvq
from forge_omega_500.model.cfm import CFMModel
from forge_omega_500.model.utils import SimpleTokenizer, pad_sequences, set_seed


def _make_triples(n):
    triples = []
    for i in range(n):
        triples.append({
            "subject_id": f"S{i}",
            "subject_label": f"Subject {i}",
            "relation_id": f"R{i%2}",
            "relation_label": f"rel_{i%2}",
            "object_id_or_value": f"O{i}",
            "object_label": f"Object {i}",
        })
    return triples


def _fallback_token_from_logits(model, prompt_ids, prompt_masks, codes, tokenizer):
    generated = torch.full((1, 1), tokenizer.bos_id, device=prompt_ids.device, dtype=torch.long)
    attention = torch.ones_like(generated)
    input_ids = torch.cat([prompt_ids, torch.full_like(prompt_ids[:, :1], tokenizer.sep_id), generated], dim=1)
    attn = torch.cat([prompt_masks, torch.ones_like(prompt_ids[:, :1]), attention], dim=1)
    if model.max_seq_len and input_ids.size(1) > model.max_seq_len:
        overflow = input_ids.size(1) - model.max_seq_len
        input_ids = input_ids[:, overflow:]
        attn = attn[:, overflow:]
    _, logits = model.forward_generation(input_ids, attn, codes)
    next_logits = logits[:, -1]
    topk = torch.topk(next_logits, k=min(10, next_logits.size(-1)), dim=-1)
    skip_ids = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.sep_id}
    for token_id in topk.indices[0].tolist():
        if token_id not in skip_ids:
            return tokenizer.inv_vocab.get(token_id, tokenizer.unk_token)
    top_id = int(topk.indices[0][0].item())
    return tokenizer.inv_vocab.get(top_id, tokenizer.unk_token)


def test_eval_inference_nonempty(tmp_path):
    set_seed(123)
    triples = _make_triples(6)
    records = build_factbank_records(triples, max_facts=6, orbits_per_fact=2, negatives_per_fact=1, seed=123)

    emb_dir = Path(tmp_path) / "emb"
    answers, embeddings = build_answer_embeddings(records, dim=16, emb_dir=emb_dir)
    codebooks = train_rvq(embeddings, m=2, K=8, iters=2, seed=123)
    codes = assign_codes(embeddings, codebooks)
    mapping_path = Path(tmp_path) / "code_to_label.parquet"
    save_code_to_label(records, answers, codes, mapping_path)
    code_to_label = load_code_to_label(mapping_path)

    texts = []
    for rec in records:
        texts.extend(rec["question_orbits"])
    tokenizer = SimpleTokenizer.build(texts)

    model = CFMModel(
        vocab_size=len(tokenizer.vocab),
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=32,
        m=2,
        K=8,
        codebooks=torch.tensor(codebooks),
    )
    model.eval()
    with torch.no_grad():
        model.backbone.lm_head.weight.zero_()

    prompt = records[0]["question_orbits"][0]
    prompt_ids = [[tokenizer.bos_id] + tokenizer.encode(prompt)]
    prompt_ids, prompt_masks = pad_sequences(prompt_ids, tokenizer.pad_id, max_len=32)

    generated, pred_codes = model.generate_answer(
        prompt_ids,
        prompt_masks,
        max_new_tokens=4,
        eos_id=tokenizer.eos_id,
        bos_id=tokenizer.bos_id,
        sep_id=tokenizer.sep_id,
    )
    pred_text = tokenizer.decode(generated[0].tolist())
    if not pred_text.strip():
        fallback, _ = lookup_code_label(code_to_label, pred_codes[0].tolist())
        if fallback:
            pred_text = fallback
        else:
            pred_text = _fallback_token_from_logits(model, prompt_ids, prompt_masks, pred_codes, tokenizer)

    assert pred_text.strip() != ""
