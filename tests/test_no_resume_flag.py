from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_cfm as train_cfm  # noqa: E402


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def test_no_resume_flag(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    ckpt_dir = tmp_path / "out" / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 5, "model": {}, "optimizer": {}}, ckpt_dir / "latest.pt")

    cfg = {
        "seed": 0,
        "runtime": {"cpu_threads": 1, "prefer_gpu": False, "use_faiss": False},
        "data": {"factbank_dir": "data/factbank", "codes_dir": "data/codes", "orbits_per_fact": 1},
        "model": {
            "backbone": "tiny",
            "d_model": 8,
            "n_layers": 1,
            "n_heads": 1,
            "max_seq_len": 8,
            "m": 1,
            "K": 2,
            "d_code": 8,
        },
        "rvq": {"iters": 1, "seed": 0},
        "train": {
            "batch_size": 1,
            "steps": 0,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "log_every": 1,
            "contrast_margin": 0.2,
            "resume": True,
        },
        "inference": {"max_gen_tokens": 1},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    record = {
        "fact_id": 0,
        "question_orbits": ["Fill in the blank: X relation ____."],
        "object_label": "Answer",
        "relation_id": "P1",
        "hard_negatives": [],
    }
    monkeypatch.setattr(train_cfm, "_load_factbank", lambda _: [record])
    monkeypatch.setattr(train_cfm, "load_answer_codes", lambda *_: {"Answer": [0]})
    monkeypatch.setattr(train_cfm.CFMModel, "from_codebooks", classmethod(lambda cls, *a, **k: _DummyModel()))
    monkeypatch.setattr(train_cfm, "_compute_code_em", lambda *a, **k: 0.0)
    monkeypatch.setattr(train_cfm, "_compute_answer_em", lambda *a, **k: 0.0)
    monkeypatch.setattr(train_cfm, "_compute_orbit_consistency", lambda *a, **k: 0.0)
    monkeypatch.setattr(train_cfm, "_compute_negative_margin", lambda *a, **k: 0.0)

    def _raise_resume(*_args, **_kwargs):
        raise AssertionError("resume attempted")

    monkeypatch.setattr(train_cfm, "load_checkpoint", _raise_resume)

    result = runner.invoke(train_cfm.app, ["--config", str(cfg_path), "--no-resume"])
    assert result.exit_code == 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "resume_checkpoint start" not in combined
