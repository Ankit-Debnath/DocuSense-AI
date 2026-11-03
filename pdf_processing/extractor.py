"""
PDF Text Extractor Module
Handles extracting text, page numbers, and character metrics from PDF files using pypdf.
Enforces configurable maximum page and character limits to protect against resource exhaustion.
"""

from typing import Any, Dict, List, Optional
import pypdf
from pypdf.errors import PdfReadError

DEFAULT_MAX_PDF_PAGES = 100
DEFAULT_MAX_EXTRACTED_CHARACTERS = 250000


def get_max_pdf_pages() -> int:
    """
    Retrieves the maximum allowed page count from application configuration.
    """
    try:
        from flask import current_app
        if current_app:
            val = current_app.config.get("MAX_PDF_PAGES", DEFAULT_MAX_PDF_PAGES)
            if isinstance(val, int) and val > 0:
                return val
    except (RuntimeError, ImportError):
        pass
    return DEFAULT_MAX_PDF_PAGES


def get_max_extracted_characters() -> int:
    """
    Retrieves the maximum allowed extracted character ceiling from configuration.
    """
    try:
        from flask import current_app
        if current_app:
            val = current_app.config.get("MAX_EXTRACTED_CHARACTERS", DEFAULT_MAX_EXTRACTED_CHARACTERS)
            if isinstance(val, int) and val > 0:
                return val
    except (RuntimeError, ImportError):
        pass
    return DEFAULT_MAX_EXTRACTED_CHARACTERS


def extract_text_from_pdf(
    file_path: str,
    max_pages: Optional[int] = None,
    max_characters: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a PDF file with bounded page count and text limits.

    Parameters:
        file_path (str): The local filesystem path to the PDF document.
        max_pages (int, optional): Maximum permitted page count override.
        max_characters (int, optional): Maximum permitted character count override.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether extraction succeeded.
            - total_pages (int): Total number of pages in the document.
            - total_characters (int): Sum of characters across all pages.
            - has_extractable_text (bool): True if at least 1 non-whitespace character exists.
            - pages (list): List of dicts per page: [{'page_number': int, 'text': str, 'char_count': int}]
            - error (str | None): User-friendly error message if an issue occurred.
    """
    effective_max_pages = max_pages if (isinstance(max_pages, int) and max_pages > 0) else get_max_pdf_pages()
    effective_max_chars = max_characters if (isinstance(max_characters, int) and max_characters > 0) else get_max_extracted_characters()

    result: Dict[str, Any] = {
        "success": False,
        "total_pages": 0,
        "total_characters": 0,
        "has_extractable_text": False,
        "pages": [],
        "error": None,
    }

    try:
        with open(file_path, "rb") as stream:
            reader = pypdf.PdfReader(stream)

            # Handle encrypted / password-protected PDFs
            if reader.is_encrypted:
                try:
                    # Attempt empty password in case of standard encryption without user password
                    decrypted = reader.decrypt("")
                    if decrypted == 0:
                        result["error"] = "This PDF is password-protected. Please provide an unencrypted PDF."
                        return result
                except Exception:
                    result["error"] = "This PDF is password-protected and cannot be read."
                    return result

            num_pages = len(reader.pages)
            result["total_pages"] = num_pages

            if num_pages == 0:
                result["success"] = True
                result["error"] = "This PDF document contains no pages."
                return result

            # Pre-extraction guard: reject documents exceeding the page count ceiling
            if num_pages > effective_max_pages:
                result["success"] = False
                result["error"] = (
                    f"Document exceeds the maximum allowed limit of {effective_max_pages} pages "
                    f"(found {num_pages} pages). Please upload a smaller document."
                )
                return result

            total_chars = 0
            extracted_pages: List[Dict[str, Any]] = []

            for index, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                page_char_count = len(page_text.strip())
                total_chars += page_char_count

                # Resource guard: reject decompression bombs / excessive raw text
                if total_chars > effective_max_chars:
                    result["success"] = False
                    result["pages"] = []
                    result["total_characters"] = 0
                    result["has_extractable_text"] = False
                    result["error"] = (
                        f"Document exceeds the maximum allowed extractable text limit "
                        f"({effective_max_chars:,} characters). Processing aborted."
                    )
                    return result

                extracted_pages.append({
                    "page_number": index,
                    "text": page_text,
                    "char_count": page_char_count,
                })

            result["success"] = True
            result["pages"] = extracted_pages
            result["total_characters"] = total_chars
            result["has_extractable_text"] = total_chars > 0

        return result

    except PdfReadError as pe:
        result["error"] = f"Unable to read the PDF file. It may be corrupted or damaged. ({str(pe)})"
        return result
    except Exception as e:
        result["error"] = f"An unexpected error occurred while processing the PDF: {str(e)}"
        return result
