from pathlib import Path
import pdfplumber

def extract_text_by_page(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, page by page.

    Why page-by-page?
    - Enables citations ("page X")
    - Makes chunking controllable
    - Makes debugging extraction quality easy

    Returns:
        A list of dicts:
        [
          {"page": 1, "text": "..."},
          {"page": 2, "text": "..."},
        ]
    """
    pdf_file = Path(pdf_path)

    # Defensive check: fail early with a clear error if file is missing
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    # pdfplumber reads layout-aware text from each page
    with pdfplumber.open(pdf_file) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            # Extract text; may return None on empty/scanned pages
            text = page.extract_text() or ""

            # Normalize whitespace a bit so chunking doesn't get weird
            text = text.replace("\r", "\n").strip()

            pages.append({
                "page": idx,
                "text": text
            })

    return pages
