from __future__ import annotations

import os
import requests

XAI_BASE_URL = "https://api.x.ai"
CHAT_COMPLETIONS_URL = f"{XAI_BASE_URL}/v1/chat/completions"

def chat_completion(messages: list[dict], model: str | None = None) -> str:
    """
    Send messages to Grok and return text (LLM client only).

    Single responsibility:
    - No retrieval logic
    - No refusal logic
    - Just transport
    """
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing XAI_API_KEY env var")

    model = model or os.getenv("XAI_MODEL")
    if not model:
        raise RuntimeError("Missing XAI_MODEL env var (set to an allowed Grok model id)")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    resp = requests.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code} {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]
