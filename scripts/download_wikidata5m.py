from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path


URLS = [
    "https://data.dgl.ai/dataset/wikidata5m/wikidata5m_transductive.tar.gz",
    "https://data.dgl.ai/dataset/wikidata5m/wikidata5m_transductive.tar.gz?download=1",
]
URLS_ENV = "WIKIDATA5M_URLS"

EXPECTED_FILES = [
    "wikidata5m_transductive_train.txt",
    "wikidata5m_transductive_valid.txt",
    "wikidata5m_transductive_test.txt",
]


def _resolve_urls() -> list[str]:
    env_value = os.getenv(URLS_ENV, "").strip()
    if not env_value:
        return URLS
    urls = [part.strip() for part in env_value.split(",") if part.strip()]
    return urls or URLS


def _download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    headers_list = [
        {"User-Agent": "Mozilla/5.0 (compatible; CFM/1.0; +https://github.com)"},
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Referer": "https://data.dgl.ai/dataset/wikidata5m/",
        },
    ]
    last_error: Exception | None = None
    for url in urls:
        candidate = Path(url)
        if candidate.exists():
            shutil.copyfile(candidate, dest)
            if dest.stat().st_size > 0:
                return
            continue
        for headers in headers_list:
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request) as response, dest.open("wb") as f:
                    shutil.copyfileobj(response, f)
                if dest.stat().st_size > 0:
                    return
            except Exception as exc:
                last_error = exc
                continue
    raise RuntimeError(f"failed to download wikidata5m from urls={urls} last_error={last_error}")


def _extract(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(path=out_dir)


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Wikidata5M transductive split.")
    parser.add_argument(
        "--output-dir",
        default="data/wikidata5m",
        help="Base output directory (default: data/wikidata5m)",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    archive_path = base_dir / "wikidata5m_transductive.tar.gz"
    extract_dir = base_dir / "transductive"

    urls = _resolve_urls()
    print(f"downloading wikidata5m from urls={urls} env={URLS_ENV}")
    _download(urls, archive_path)
    _extract(archive_path, extract_dir)

    for name in EXPECTED_FILES:
        path = extract_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing expected file: {path}")
        count = _count_lines(path)
        print(f"{path}: {count} lines")


if __name__ == "__main__":
    main()
