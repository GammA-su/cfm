# forge_omega_500

Toy-scale, deterministic CFM pipeline (FactBank build -> RVQ codes -> CFM model training -> LAMA eval).

## Quickstart (uv)

```bash
uv sync
uv run python scripts/build_all.py --config configs/default.yaml
uv run python scripts/train_cfm.py --config configs/default.yaml
uv run python scripts/eval_lama.py --config configs/default.yaml
uv run pytest -q
```

## Setup (Wikidata5M extract)

If you download `wikidata5m_transductive.tar.gz` manually, extract it with:

```bash
tar --no-same-owner --no-same-permissions -xzf wikidata5m_transductive.tar.gz -C data/wikidata5m/transductive
```

The archive records UID/GID values that can be missing on some systems. The resulting warning is harmless, but it causes a non-zero exit unless you use the flags above.

## Notes

- Hugging Face datasets are the *input seeds*. The FactBank and code labels are generated locally in `data/`.
- The pipeline runs on a toy subset by default and is deterministic (fixed seeds + stable ordering).
- Offline runs work after the HF datasets are cached.
- Optional FAISS acceleration:
  - CPU: `uv sync --extra faiss-cpu`
  - GPU: `uv sync --extra faiss-gpu`

## Repo Layout

- `src/forge_omega_500/data/` data pipeline (HF ingest, FactBank build, orbits, negatives, embeddings, RVQ)
- `src/forge_omega_500/model/` CFM model + tiny decoder
- `src/forge_omega_500/eval/` metrics
- `scripts/` build/train/eval entry points
- `tests/` deterministic tests for orbit invariance, negatives, and determinism
- `configs/default.yaml` default configuration
