from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.embeddings import embed_texts


class FaissVectorStore:
    """
    Local vector store backed by:
    - FAISS index for similarity search
    - JSON file for chunk metadata/text

    This is a local stand-in for Supabase pgvector.
    """

    def __init__(self, store_dir: str = "storage/data"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "chunks.json"

        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

        self._load()

    def _load(self) -> None:
        """
        Load existing index + metadata from disk.

        Problem solved:
        - App restarts should not force re-embedding everything.
        """
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.index = None
            self.chunks = []

    def _save(self) -> None:
        """Persist index + metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> None:
        """
        Clear the store.

        Useful when you want to re-ingest PDFs with new chunking parameters.
        """
        self.index = None
        self.chunks = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()

    def add_chunks(self, chunk_dicts: list[dict[str, Any]]) -> None:
        """
        Add chunks to the index.

        Each chunk_dict should include at least:
        - text
        - page
        - chunk_index
        - source_path

        We embed the chunk text and insert vectors into FAISS.
        """
        if not chunk_dicts:
            return

        texts = [c["text"] for c in chunk_dicts]
        vectors = embed_texts(texts)  # shape (N, D), float32, normalized

        # Build index on first insert
        if self.index is None:
            dim = vectors.shape[1]
            # Inner product index; with normalized vectors this behaves like cosine similarity.
            self.index = faiss.IndexFlatIP(dim)

        # Add vectors
        self.index.add(vectors)

        # Append metadata in same order; FAISS ids correspond to list positions
        self.chunks.extend(chunk_dicts)

        # Persist so we can reuse after restart
        self._save()

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Search for top-k chunks matching the query.

        Returns chunk dicts plus a similarity score.
        """
        if self.index is None or not self.chunks:
            return []

        q_vec = embed_texts([query])  # shape (1, D)
        scores, idxs = self.index.search(q_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[idx])
            item["score"] = float(score)
            results.append(item)

        return results
