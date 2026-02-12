# backend/vector_store_supabase.py
import os
from typing import Any, Dict, List, Optional

from supabase import create_client, Client


class SupabaseVectorStore:
    """
    Minimal Supabase pgvector store.
    Assumes:
      - tables: documents, chunks
      - RPC: match_chunks(query_embedding vector, match_count int)
    """

    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL and/or SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY).")

        self.client: Client = create_client(url, key)

    def insert_document(self, source: str, title: Optional[str] = None) -> str:
        payload = {"source": source, "title": title}
        res = self.client.table("documents").insert(payload).execute()
        return res.data[0]["id"]

    def insert_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> None:
        """
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

    def similarity_search(self, query_embedding: List[float], k: int = 10) -> List[Dict[str, Any]]:
        res = self.client.rpc("match_chunks", {"query_embedding": query_embedding, "match_count": k}).execute()
        return res.data or []

