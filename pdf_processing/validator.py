import os
import uuid
from werkzeug.utils import secure_filename

# Magic bytes signature that every valid PDF file starts with: '%PDF-'
PDF_MAGIC_BYTES = b"%PDF-"

def is_allowed_extension(filename: str, allowed_extensions: set) -> bool:
    """
    Checks if the filename has an allowed extension.
    Example: 'report.pdf' -> True, 'script.py' -> False
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def validate_pdf_file(file_storage, allowed_extensions: set) -> tuple[bool, str]:
    """
    Validates an uploaded file object from request.files:
    1. Checks if a file was provided with a valid filename.
    2. Validates the file extension.
    3. Inspects the file's binary header for the '%PDF-' magic bytes signature.

    Returns:
        tuple[bool, str]: (is_valid, error_message_or_empty)
    """
    if file_storage is None or file_storage.filename == "":
        return False, "No file selected. Please choose a PDF file to upload."

    filename = file_storage.filename

    # Step 1: Validate file extension
    if not is_allowed_extension(filename, allowed_extensions):
        return False, "Invalid file format. Only PDF files (.pdf) are allowed."

    # Step 2: Validate PDF magic bytes signature
    header = file_storage.stream.read(5)
    # Reset stream position to the beginning so saving later works properly
    file_storage.stream.seek(0)

    if not header.startswith(PDF_MAGIC_BYTES):
        return False, "Invalid file content. The uploaded file is not a valid PDF document."

    return True, ""


def generate_safe_filename(original_filename: str) -> str:
    """
    Sanitizes the original filename to prevent directory traversal attacks
    and prepends a unique short UUID to prevent filename collisions.
    """
    clean_name = secure_filename(original_filename)
    if not clean_name:
        clean_name = "document.pdf"

    unique_prefix = uuid.uuid4().hex[:8]
    return f"{unique_prefix}_{clean_name}"
