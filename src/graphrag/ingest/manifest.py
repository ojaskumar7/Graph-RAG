from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    path: str
    start: int
    end: int
    text_hash: str


MANIFEST_NAME = "manifest.jsonl"


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_manifest(processed_dir: Path, records: list[ChunkRecord]) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def load_manifest(processed_dir: Path) -> dict[str, ChunkRecord]:
    path = processed_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rec = ChunkRecord(**data)
            result[rec.chunk_id] = rec
    return result


def get_unchanged_chunks(
    processed_dir: Path, new_records: list[ChunkRecord]
) -> tuple[list[ChunkRecord], list[ChunkRecord]]:
    """Split new_records into (unchanged, changed) based on text_hash."""
    existing = load_manifest(processed_dir)
    unchanged = []
    changed = []
    for rec in new_records:
        if (
            rec.chunk_id in existing
            and existing[rec.chunk_id].text_hash == rec.text_hash
        ):
            unchanged.append(rec)
        else:
            changed.append(rec)
    return unchanged, changed
