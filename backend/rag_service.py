from __future__ import annotations

from typing import Tuple, List, Dict, Any
import re

from backend.rag_prompt import format_retrieved_chunks, build_rag_messages
from backend.llm_xai import chat_completion


def _make_subqueries(question: str) -> List[str]:
    """
    Create 2–4 sub-queries for compound questions.
    Goal: retrieve evidence for each part separately, then merge.
    """
    q = question.strip()
    subs = [q]

    # Split on "and" (most common)
    if " and " in q.lower():
        parts = re.split(r"\s+and\s+", q, flags=re.IGNORECASE)
        subs.extend([p.strip(" .,:;") for p in parts if p.strip()])

    # Split on multiple question punctuation
    if ";" in q or "?" in q:
        parts = re.split(r"[;?]+", q)
        subs.extend([p.strip(" .,:;") for p in parts if p.strip()])

    # Optional: long questions with commas
    if len(q) > 120 and "," in q:
        parts = q.split(",")
        subs.extend([p.strip(" .,:;") for p in parts if p.strip()])

    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in subs:
        s2 = re.sub(r"\s+", " ", s).strip()
        if s2 and s2.lower() not in seen:
            out.append(s2)
            seen.add(s2.lower())

    return out[:4]  # cap to avoid too many searches


def rag_answer(
    question: str,
    store,
    k: int = 6,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    RAG orchestrator: retrieve → decide → prompt → LLM.

    Upgraded retrieval:
    - For multi-part questions, run multiple searches and merge results.
    """
    # 1) Retrieve evidence (multi-query)
    subqueries = _make_subqueries(question)

    merged: List[Dict[str, Any]] = []
    k_per = max(4, k)  # each subquery gets enough candidates

    for sq in subqueries:
        merged.extend(store.search(sq, k=k_per))

    # Dedupe by stable chunk identity (source/page/chunk)
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for r in merged:
        key = (r.get("source_path"), r.get("page"), r.get("chunk_index"))
        if key not in deduped:
            deduped[key] = r
        else:
            # keep the higher score version
            if float(r.get("score", 0.0)) > float(deduped[key].get("score", 0.0)):
                deduped[key] = r

    # Keep top-k overall
    retrieved = sorted(
        deduped.values(),
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True
    )[:k]

    # 2) If no evidence, refuse (prevents hallucination)
    if not retrieved:
        return (
            "I don't have enough information in the uploaded PDFs.",
            [],
            ""
        )

    # 3) Build evidence block + prompt messages
    context_text = format_retrieved_chunks(retrieved)
    messages = build_rag_messages(question, context_text)

    # 4) Call Grok (LLM transport only)
    answer = chat_completion(messages=messages)

    return answer, retrieved, context_text
