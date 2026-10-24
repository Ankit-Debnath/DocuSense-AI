"""
PDF Text Extractor Module
Handles extracting text, page numbers, and character metrics from PDF files using pypdf.
"""

from typing import Any, Dict, List
import pypdf
from pypdf.errors import PdfReadError


def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a PDF file.

    Parameters:
        file_path (str): The local filesystem path to the PDF document.

    Returns:
        dict: A dictionary containing:
            - success (bool): Whether extraction succeeded.
            - total_pages (int): Total number of pages in the document.
            - total_characters (int): Sum of characters across all pages.
            - has_extractable_text (bool): True if at least 1 non-whitespace character exists.
            - pages (list): List of dicts per page: [{'page_number': int, 'text': str, 'char_count': int}]
            - error (str | None): User-friendly error message if an issue occurred.
    """
    result: Dict[str, Any] = {
        "success": False,
        "total_pages": 0,
        "total_characters": 0,
        "has_extractable_text": False,
        "pages": [],
        "error": None,
    }

    try:
        reader = pypdf.PdfReader(file_path)

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

        total_chars = 0
        extracted_pages: List[Dict[str, Any]] = []

        for index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_char_count = len(page_text.strip())
            total_chars += page_char_count

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
