from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from backend.llm_xai import chat_completion
from backend.rag_prompt import format_retrieved_chunks
from backend.handbook_prompts import outline_messages, section_messages


def parse_toc(toc_text: str) -> List[str]:
    """
    Very simple TOC parser:
    - Extracts lines that look like numbered headings.
    - For MVP, we treat each top-level line as a section title.

    You can refine later, but this gets you working fast.
    """
    lines = [ln.strip() for ln in toc_text.splitlines() if ln.strip()]
    section_titles = []

    for ln in lines:
        # Keep lines that start with "1." "2." etc.
        if len(ln) >= 2 and ln[0].isdigit() and "." in ln[:4]:
            section_titles.append(ln)

    # Fallback: if parsing fails, treat whole toc as one section
    return section_titles or ["1. Introduction"]


def generate_handbook(topic: str, store, out_dir: str = "storage/handbooks") -> Tuple[str, str]:
    """
    Generate a long handbook by:
    1) generating a TOC/outline
    2) generating each section with its own retrieval

    Returns:
      handbook_text, saved_markdown_path
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    md_file = out_path / f"handbook_{topic.replace(' ', '_')[:40]}.md"

    # -----------------------------
    # 1) Get outline / TOC
    # -----------------------------
    toc = chat_completion(outline_messages(topic))

    # Save header + TOC early (so you have progress even if it crashes)
    md_file.write_text(f"# Handbook: {topic}\n\n## Table of Contents\n\n{toc}\n\n", encoding="utf-8")

    # -----------------------------
    # 2) Parse sections
    # -----------------------------
    sections = parse_toc(toc)

    # -----------------------------
    # 3) Generate sections (loop)
    # -----------------------------
    for sec in sections:
        section_title = sec

        # Retrieve evidence for this section (section-driven retrieval)
        retrieved = store.search(f"{topic} — {section_title}", k=8)
        evidence = format_retrieved_chunks(retrieved, max_chars_per_chunk=1200)

        # Section goals are simple for MVP (later: derive from outline bullets)
        section_goals = f"Cover this topic thoroughly: {section_title}"

        section_text = chat_completion(section_messages(section_title, section_goals, evidence))

        # Append section to markdown
        with md_file.open("a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## {section_title}\n\n")
            f.write(section_text)
            f.write("\n")

    handbook_text = md_file.read_text(encoding="utf-8")
    return handbook_text, str(md_file)
