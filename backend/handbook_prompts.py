from __future__ import annotations

def outline_messages(topic: str) -> list[dict]:
    """
    Creates a structured outline request.

    Output must be:
    - Table of contents
    - Section titles
    - Short bullets per section (what to cover)
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an expert technical author. Create a detailed handbook outline. "
                "Return ONLY a numbered table of contents with sections and subsections. "
                "Keep it comprehensive enough to support a 20,000+ word handbook."
            ),
        },
        {
            "role": "user",
            "content": f"Create a handbook outline on: {topic}",
        },
    ]


def section_messages(section_title: str, section_goals: str, evidence_block: str) -> list[dict]:
    """
    Writes ONE section, grounded in evidence.
    Requires citations using the evidence labels.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are writing one section of a long technical handbook. "
                "Use ONLY the provided evidence. If evidence is insufficient, state that clearly. "
                "Cite claims inline using the exact bracket labels from the evidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"SECTION TITLE:\n{section_title}\n\n"
                f"SECTION GOALS:\n{section_goals}\n\n"
                f"EVIDENCE:\n{evidence_block}\n\n"
                "Write this section in a clear handbook style with headings, bullet lists where useful, "
                "and citations for claims."
            ),
        },
    ]
