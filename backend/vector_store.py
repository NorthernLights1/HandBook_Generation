from __future__ import annotations

import os

from backend.vector_store_faiss import FaissVectorStore
from backend.vector_store_supabase import SupabaseVectorStore


def get_vector_store(store_dir: str | None = None):
    """
    Factory function that returns the configured vector backend.

    Controlled by env variable:
        VECTOR_BACKEND = "faiss" | "supabase"

    Default: supabase (target architecture)
    """

    backend = os.getenv("VECTOR_BACKEND", "supabase").lower()

    if backend == "faiss":
        if store_dir is None:
            raise ValueError("FAISS backend requires store_dir")
        return FaissVectorStore(store_dir=store_dir)

    if backend == "supabase":
        return SupabaseVectorStore()

    raise ValueError(f"Unsupported VECTOR_BACKEND: {backend}")
