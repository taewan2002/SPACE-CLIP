"""Readers shared by public result checks; measurements can be plain or gzip JSONL."""

import gzip
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    if not path.exists() and path.suffix == ".jsonl":
        path = path.with_suffix(".jsonl.gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
