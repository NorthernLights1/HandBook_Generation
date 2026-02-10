from __future__ import annotations

from typing import Tuple, List, Dict, Any

from backend.rag_prompt import format_retrieved_chunks, build_rag_messages
from backend.llm_xai import chat_completion


def rag_answer(
    question: str,
    store,
    k: int = 6,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """
    RAG orchestrator: retrieve → decide → prompt → LLM.

    Problem solved:
    - Keeps RAG logic out of Streamlit UI code.
    - Makes the same function reusable for chat AND handbook sections.

    Returns:
      answer: final assistant answer (string)
      retrieved: list of retrieved chunk dicts (for debugging/citations)
      context_text: the formatted evidence text sent to the LLM (for debug)
    """
    # 1) Retrieve evidence
    retrieved = store.search(question, k=k)

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
