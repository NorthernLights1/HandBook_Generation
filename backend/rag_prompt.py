from __future__ import annotations

from typing import List, Dict
from pathlib import Path

def format_retrieved_chunks(results: List[Dict], max_chars_per_chunk: int = 1500) -> str:
    """
    Turn retrieved chunks into a compact evidence block.

    Problem solved:
    - LLM needs evidence + provenance (source/page) to cite correctly.

    If skipped:
    - You can't enforce citations; model will guess.
    """
    lines = []
    for r in results:
        source = Path(r["source_path"]).name
        page = r["page"]
        chunk_index = r["chunk_index"]

        text = (r.get("text") or "").strip().replace("\n", " ")
        text = text[:max_chars_per_chunk]  # keep context compact

        lines.append(f"[{source} | p.{page} | c{chunk_index}] {text}")
    return "\n".join(lines)

def build_rag_messages(user_question: str, retrieved_context: str) -> list[dict]:
    """
    Build messages that enforce grounding.

    Problem solved:
    - Prompt structure makes it harder for the model to ignore evidence.

    If skipped:
    - The model answers from general knowledge.
    """
    system = {
        "role": "system",
        "content": (
            "You are a strict RAG assistant. Use ONLY the provided PDF context. "
            "If the answer is not explicitly supported by the context, say: "
            "\"I don't have enough information in the uploaded PDFs.\" "
            "Every factual claim must include an inline citation using the exact "
            "bracket label from the context, e.g. [file.pdf | p.12 | c3]. "
            "Do not invent sources, pages, or citations."
        ),
    }

    user = {
        "role": "user",
        "content": (
            "PDF CONTEXT:\n"
            f"{retrieved_context}\n\n"
            "QUESTION:\n"
            f"{user_question}\n\n"
            "Answer clearly and include citations for claims."
        ),
    }

    return [system, user]
