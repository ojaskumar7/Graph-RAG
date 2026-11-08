from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Document:
    doc_id: str
    path: Path
    text: str


def load_documents(raw_dir: Path) -> list[Document]:
    """Load .md and .txt files from raw_dir."""
    docs = []
    for ext in ("*.md", "*.txt"):
        for path in raw_dir.rglob(ext):
            text = path.read_text(encoding="utf-8")
            doc_id = (
                path.relative_to(raw_dir).as_posix().replace(".", "_").replace("/", "_")
            )
            docs.append(Document(doc_id=doc_id, path=path, text=text))
    return docs
