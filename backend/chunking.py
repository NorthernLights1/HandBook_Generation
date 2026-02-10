from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Chunk:
    """
    A single retrievable unit of text.

    Why we store metadata:
    - page: enables citations and debugging ("page X")
    - chunk_index: stable ordering within a page
    - source_path: lets us trace back to the exact PDF
    """
    text: str
    page: int
    chunk_index: int
    source_path: str


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines as separators.

    Why paragraphs:
    - better semantic coherence than fixed character splits
    - reduces chance of cutting a concept mid-sentence

    We also strip whitespace and remove empty segments.
    """
    raw_parts = text.split("\n\n")
    paragraphs = [p.strip() for p in raw_parts if p.strip()]
    return paragraphs


def _approx_token_count(s: str) -> int:
    """
    Approximate token count cheaply.

    Why approximation:
    - True tokenization depends on the model tokenizer.
    - For chunking, a rough estimate is enough.

    Rule of thumb: ~4 characters per token in English (varies by language).
    We'll use word count as a stable approximation.
    """
    return max(1, len(s.split()))


def chunk_pages(
    pages: List[Dict],
    source_path: str,
    max_tokens: int = 800,
    overlap_tokens: int = 120,
) -> List[Chunk]:
    """
    Convert extracted pages into chunks.

    Inputs:
      pages: [{"page": 1, "text": "..."}, ...]
      source_path: which PDF this came from (for traceability)

    Key design choices:
    - page-aware: we chunk within each page so citations remain stable
    - paragraph-first: preserve meaning better than raw slicing
    - max_tokens: keeps retrieval precise and avoids huge chunks
    - overlap_tokens: prevents losing context at chunk boundaries

    Returns:
      List[Chunk]
    """
    chunks: List[Chunk] = []

    # Validate parameters defensibly
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    for page_obj in pages:
        page_num = int(page_obj["page"])
        text = page_obj.get("text", "") or ""

        # If page has no extracted text (e.g., scanned), skip for now.
        # Later, OCR would fill this.
        if not text.strip():
            continue

        paragraphs = _split_into_paragraphs(text)

        # We build chunks by accumulating paragraphs until we hit max_tokens
        current_parts: List[str] = []
        current_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = _approx_token_count(para)

            # If adding this paragraph would exceed max size, we finalize current chunk
            if current_parts and (current_tokens + para_tokens > max_tokens):
                chunk_text = "\n\n".join(current_parts).strip()

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        page=page_num,
                        chunk_index=chunk_index,
                        source_path=source_path,
                    )
                )
                chunk_index += 1

                # Overlap handling:
                # We keep the last overlap_tokens worth of text as the start of the next chunk.
                # This reduces the chance that important context is split across boundary.
                if overlap_tokens > 0:
                    # Simple overlap strategy: keep last N words of the chunk
                    words = chunk_text.split()
                    overlap_words = words[-overlap_tokens:] if len(words) > overlap_tokens else words
                    current_parts = [" ".join(overlap_words)]
                    current_tokens = _approx_token_count(current_parts[0])
                else:
                    current_parts = []
                    current_tokens = 0

            # Add paragraph to current chunk buffer
            current_parts.append(para)
            current_tokens += para_tokens

        # Flush any remaining content as final chunk for the page
        if current_parts:
            chunk_text = "\n\n".join(current_parts).strip()
            chunks.append(
                Chunk(
                    text=chunk_text,
                    page=page_num,
                    chunk_index=chunk_index,
                    source_path=source_path,
                )
            )

    return chunks
