from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from fastembed import TextEmbedding

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    cache_dir: Path | None = None
    batch_size: int = 32
    max_length: int = 512

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path(
                os.environ.get("GRAPHRAG_EMBEDDINGS_CACHE", "./data/embeddings")
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class LocalEmbedder:
    """Local ONNX-based text embedder using fastembed."""

    def __init__(self, config: EmbeddingConfig | None = None):
        if not FASTEMBED_AVAILABLE:
            raise RuntimeError(
                "fastembed not installed. Install with: pip install fastembed"
            )
        self.config = config or EmbeddingConfig()
        self._model: TextEmbedding | None = None
        self._dim: int | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self.config.model_name,
                cache_dir=str(self.config.cache_dir),
            )
            self._dim = len(list(self._model.embed(["test"]))[0])
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            _ = self.model
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n_texts, dim) array."""
        embeddings = list(self.model.embed(texts, batch_size=self.config.batch_size))
        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dim,) array."""
        return self.embed([text])[0]

    def save_index(
        self,
        path: Path,
        embeddings: np.ndarray,
        ids: list[str],
        texts: dict[str, str] | None = None,
    ) -> None:
        """Save embeddings, IDs, and texts to disk."""
        data = {
            "embeddings": embeddings,
            "ids": ids,
            "texts": texts or {},
            "model_name": self.config.model_name,
            "dimension": self.dimension,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_index(self, path: Path) -> tuple[np.ndarray, list[str], dict[str, str]]:
        """Load embeddings, IDs, and texts from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("model_name") != self.config.model_name:
            raise ValueError(
                f"Model mismatch: {data['model_name']} != {self.config.model_name}"
            )
        return data["embeddings"], data["ids"], data.get("texts", {})


class VectorIndex:
    """Simple in-memory vector index with cosine similarity."""

    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        self.embeddings: np.ndarray | None = None
        self.ids: list[str] = []
        self.texts: dict[str, str] = {}

    def add(self, ids: list[str], texts: list[str]) -> None:
        """Add texts to index."""
        new_embeddings = self.embedder.embed(texts)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        self.ids.extend(ids)
        for id_, text in zip(ids, texts):
            self.texts[id_] = text

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """Search for top-k similar texts. Returns (id, score, text)."""
        if self.embeddings is None or len(self.ids) == 0:
            return []

        query_emb = self.embedder.embed_single(query)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        doc_embs_normed = self.embeddings / doc_norms

        scores = doc_embs_normed @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self.ids[i], float(scores[i]), self.texts[self.ids[i]])
            for i in top_indices
        ]

    def save(self, path: Path) -> None:
        self.embedder.save_index(path, self.embeddings, self.ids, self.texts)

    def load(self, path: Path) -> None:
        self.embeddings, self.ids, self.texts = self.embedder.load_index(path)
