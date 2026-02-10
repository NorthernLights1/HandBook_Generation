from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Model cache (loaded once)
# -----------------------------
# Loading embedding models is slow. This avoids reloading on every rerun.
_MODEL: SentenceTransformer | None = None


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Return a cached embedding model.

    Problem solved:
    - Streamlit reruns would otherwise reload the model repeatedly (slow).
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts into vectors suitable for FAISS.

    Why normalize_embeddings=True:
    - Normalized vectors allow cosine similarity via inner product (fast & standard).

    Returns:
    - float32 numpy array shaped (N, D)
    """
    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vectors.astype("float32")
