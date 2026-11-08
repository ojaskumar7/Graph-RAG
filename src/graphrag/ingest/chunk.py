from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> Iterator[Chunk]:
    """Simple sliding-window chunking by characters (approximate tokens)."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be > overlap")
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end]
        chunk_id = hashlib.sha1(chunk_text.encode()).hexdigest()[:16]
        yield Chunk(chunk_id=chunk_id, doc_id="", text=chunk_text, start=start, end=end)
        if end == n:
            break
        start = end - overlap


def chunk_document(
    doc_id: str, text: str, chunk_size: int = 800, overlap: int = 100
) -> list[Chunk]:
    chunks = list(chunk_text(text, chunk_size, overlap))
    for c in chunks:
        c.doc_id = doc_id
    return chunks
