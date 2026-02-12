# backend/vector_store_supabase.py
import os
from typing import Any, Dict, List, Optional

from supabase import create_client, Client


class SupabaseVectorStore:
    """
    Supabase pgvector store (minimal, app-compatible).

    Assumes:
      - tables: documents, chunks
      - RPC: match_chunks(query_embedding vector, match_count int)
    Env:
      - SUPABASE_URL
      - SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError(
                "Missing SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY)."
            )
        self.client: Client = create_client(url, key)

    # ---- Documents ---------------------------------------------------------

    def insert_document(self, source: str, title: Optional[str] = None) -> str:
        payload = {"source": source, "title": title}
        res = self.client.table("documents").insert(payload).execute()
        if not res.data:
            raise RuntimeError("Failed to insert document into Supabase.")
        return res.data[0]["id"]

    # ---- Chunks ------------------------------------------------------------

    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
        Insert chunks for a single document_id.

        chunks: list of dicts like:
          {
            "chunk_index": int,
            "page_start": int,
            "page_end": int,
            "content": str,
            "metadata": dict,
            "embedding": list[float]
          }
        """
        rows = []
        for c in chunks:
            rows.append(
                {
                    "document_id": document_id,
                    "chunk_index": c.get("chunk_index"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "content": c["content"],
                    "metadata": c.get("metadata", {}),
                    "embedding": c.get("embedding"),
                }
            )

        if rows:
            self.client.table("chunks").insert(rows).execute()

    def add_chunks(self, chunk_dicts: List[Dict[str, Any]]) -> None:
        """
        Compatibility method expected by app.py: store.add_chunks(chunk_dicts)

        Expects each chunk dict to contain:
          - document_id (uuid string)
          - content (str)
          - embedding (list[float])

        Optional:
          - chunk_index, page_start, page_end, metadata
        """
        if not chunk_dicts:
            return

        rows = []
        for c in chunk_dicts:
            doc_id = c.get("document_id")
            if not doc_id:
                raise RuntimeError("add_chunks: each chunk must include 'document_id'")

            if "content" not in c:
                raise RuntimeError("add_chunks: each chunk must include 'content'")

            rows.append(
                {
                    "document_id": doc_id,
                    "chunk_index": c.get("chunk_index"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "content": c["content"],
                    "metadata": c.get("metadata", {}),
                    "embedding": c.get("embedding"),
                }
            )

        # Bulk insert
        self.client.table("chunks").insert(rows).execute()

    # ---- Retrieval ---------------------------------------------------------

    def similarity_search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        res = self.client.rpc(
            "match_chunks",
            {"query_embedding": query_embedding, "match_count": k},
        ).execute()
        return res.data or []

    # Some codebases use retrieve() naming; keep a safe alias if needed.
    def retrieve(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        return self.similarity_search(query_embedding=query_embedding, k=k)
