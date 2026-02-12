from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from supabase import create_client, Client

from backend.embeddings import embed_texts


class SupabaseVectorStore:
    """
    Supabase pgvector-backed store.

    Contract matches FaissVectorStore:
      - add_chunks(list[dict]) -> None
      - search(query: str, k: int) -> list[dict]  (with score)

    Requires env vars:
      SUPABASE_URL
      SUPABASE_SERVICE_ROLE_KEY   (recommended for demo server-side inserts)
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)")

        self.client: Client = create_client(url, key)

    def _get_or_create_document(self, source_path: str) -> str:
        """
        Ensure there's a row in documents for a given PDF.
        Returns document_id (uuid as string).
        """
        # Normalize: store only filename to avoid local absolute paths in DB
        normalized = Path(source_path).name

        # Try fetch
        res = self.client.table("documents").select("id").eq("source_path", normalized).execute()
        if res.data:
            return res.data[0]["id"]

        # Insert
        ins = self.client.table("documents").insert({
            "source_path": normalized,
            "title": normalized
        }).execute()

        return ins.data[0]["id"]

    def add_chunks(self, chunk_dicts: List[Dict[str, Any]]) -> None:
        """
        Insert chunks + embeddings into Supabase.

        chunk_dict expected keys:
          - text
          - page
          - chunk_index
          - source_path
        """
        if not chunk_dicts:
            return

        # 1) Embed all chunk texts
        texts = [c["text"] for c in chunk_dicts]
        vectors = embed_texts(texts)  # numpy float32, shape (N, D)

        # 2) Batch insert chunks. Group by document for cleaner metadata.
        rows = []
        for i, c in enumerate(chunk_dicts):
            doc_id = self._get_or_create_document(c["source_path"])

            rows.append({
                "document_id": doc_id,
                "content": c["text"],
                "metadata": {
                    "page": int(c["page"]),
                    "chunk_index": int(c["chunk_index"]),
                    "source_path": Path(c["source_path"]).name
                },
                # supabase client will serialize list fine
                "embedding": vectors[i].tolist()
            })

        # Insert in chunks to avoid payload limits
        BATCH = 200
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            self.client.table("chunks").insert(batch).execute()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Similarity search via RPC match_chunks.
        Returns list in the same shape as FAISS store.search() results.
        """
        q_vec = embed_texts([query])[0].tolist()

        rpc = self.client.rpc("match_chunks", {
            "query_embedding": q_vec,
            "match_count": int(k),
            "filter": {}
        }).execute()

        results = []
        for row in (rpc.data or []):
            md = row.get("metadata") or {}
            results.append({
                "text": row.get("content", ""),
                "page": md.get("page", None),
                "chunk_index": md.get("chunk_index", None),
                "source_path": md.get("source_path", "unknown"),
                "score": float(row.get("score", 0.0))
            })

        return results

    def reset(self) -> None:
        """
        Optional: Clears all rows (for dev only).
        WARNING: destructive.
        """
        # Supabase doesn't support delete all easily without conditions.
        # For dev, you can run SQL:
        #   truncate table public.chunks, public.documents restart identity;
        raise NotImplementedError("Reset in Supabase should be done via SQL TRUNCATE in dev.")
