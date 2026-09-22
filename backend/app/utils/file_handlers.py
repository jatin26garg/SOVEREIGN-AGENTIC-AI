"""
FILE HANDLERS
Extract text from PDF, DOCX, TXT files.

PDF extraction uses pdfplumber (preserves tables) with pypdf as fallback.
"""

import io
import os
from typing import Optional


# ========== Imports ==========
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


# ===================================================
# MAIN ENTRY POINT
# ===================================================

def extract_text_from_file(file_content: bytes, filename: str) -> Optional[str]:
    """
    Extract text from a file based on its extension.

    Args:
        file_content: Raw bytes of the file
        filename: Original filename (used to detect extension)

    Returns:
        Extracted text as a string

    Raises:
        ValueError: If the file type is unsupported or extraction fails
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_pdf_text(file_content)

    elif ext == ".docx":
        return extract_docx_text(file_content)

    elif ext == ".txt":
        return extract_txt_text(file_content)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ===================================================
# PDF EXTRACTION (with table support)
# ===================================================

def extract_pdf_text(file_content: bytes) -> str:
    """
    Extract text AND tables from a PDF.

    Tries pdfplumber first (better table handling).
    Falls back to pypdf if pdfplumber isn't available.
    """
    # Try pdfplumber first (preserves tables)
    if pdfplumber is not None:
        try:
            return _extract_with_pdfplumber(file_content)
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}. Falling back to pypdf...")

    # Fallback to pypdf (plain text, no table structure)
    if PdfReader is not None:
        return _extract_with_pypdf(file_content)

    raise ImportError(
        "Neither pdfplumber nor pypdf is installed. "
        "Run: pip install pdfplumber pypdf"
    )


def _extract_with_pdfplumber(file_content: bytes) -> str:
    """
    Extract text + tables using pdfplumber.

    NOTE: pdfplumber.open() accepts a BytesIO wrapper OR a file path.
    We use BytesIO here because we have raw bytes in memory.
    """
    # pdfplumber needs a file-like object (BytesIO) OR a path
    pdf_stream = io.BytesIO(file_content)

    text_parts = []

    with pdfplumber.open(pdf_stream) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # ---------- Regular text ----------
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(text)

            # ---------- Tables (as markdown) ----------
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue

                md_lines = []
                for row_idx, row in enumerate(table):
                    # Clean each cell
                    cleaned = [
                        str(cell or "").replace("\n", " ").strip()
                        for cell in row
                    ]
                    md_lines.append("| " + " | ".join(cleaned) + " |")

                    # Add markdown separator after header row
                    if row_idx == 0:
                        md_lines.append(
                            "|" + "|".join(["---"] * len(cleaned)) + "|"
                        )

                text_parts.append("\n\n" + "\n".join(md_lines) + "\n\n")

    return "\n\n".join(text_parts)


def _extract_with_pypdf(file_content: bytes) -> str:
    """
    Extract plain text using pypdf (no table structure preserved).

    NOTE: pypdf accepts raw bytes directly OR a BytesIO.
    We pass raw bytes here for simplicity.
    """
    # pypdf.PdfReader accepts raw bytes directly
    reader = PdfReader(io.BytesIO(file_content))

    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            text_parts.append(text)

    return "\n\n".join(text_parts)


# ===================================================
# DOCX EXTRACTION
# ===================================================

def extract_docx_text(file_content: bytes) -> str:
    """
    Extract text from a DOCX file.

    NOTE: python-docx's Document() accepts a file-like object (BytesIO).
    We must wrap the bytes in BytesIO first.
    """
    if Document is None:
        raise ImportError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    # Wrap bytes in BytesIO — Document() requires a file-like object
    doc_stream = io.BytesIO(file_content)
    doc = Document(doc_stream)

    text_parts = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    return "\n\n".join(text_parts)


# ===================================================
# TXT EXTRACTION
# ===================================================

def extract_txt_text(file_content: bytes) -> str:
    """
    Extract text from a TXT file.

    NOTE: file_content is raw bytes. We decode it directly
    (no need for BytesIO here).
    """
    # Try common encodings
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_content.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "Could not decode TXT file. Please ensure it's UTF-8 encoded."
    )