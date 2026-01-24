from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forge_omega_500.model.utils import SimpleTokenizer  # noqa: E402
import scripts.train_cfm as train_cfm  # noqa: E402


def test_train_overfit_saves_tokenizer(tmp_path: Path) -> None:
    tokenizer = SimpleTokenizer.build(["hello world"])
    source = tmp_path / "tokenizer.json"
    tokenizer.save(source)

    run_dir = tmp_path / "run"
    meta = train_cfm._save_overfit_tokenizer(source, run_dir)

    saved_path = Path(meta["tokenizer_path_saved"])
    assert saved_path.exists()
    assert meta["tokenizer_vocab_size"] == len(tokenizer.vocab)
    assert meta["tokenizer_sha256"] == train_cfm._sha256_file(saved_path)
