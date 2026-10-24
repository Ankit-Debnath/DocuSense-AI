"""
PDF Routes Blueprint
Handles PDF upload, history viewing, API key management, cached summarization, and grounded Q&A.
Features SQLite summary caching and structured performance timing logs.
"""

import logging
import os
import sys
import time
from urllib.parse import urlsplit
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, session

from pdf_processing.validator import validate_pdf_file, generate_safe_filename
from pdf_processing.extractor import extract_text_from_pdf
from database.models import (
    insert_document,
    get_all_documents,
    save_document_summary,
    get_document_summary,
    delete_document_summary
)
from ai.analyzer import summarize_document, answer_pdf_question
from ai.llm_client import save_api_key

# Structured route performance logger
logger = logging.getLogger("pdf_analyzer.routes")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

pdf_bp = Blueprint("pdf", __name__)


def is_safe_redirect_url(target: str) -> bool:
    """
    Validates that a redirect target is strictly a safe, same-origin relative path.
    Rejects:
      - None, empty, or whitespace strings
      - Strings not starting with a leading '/'
      - Protocol-relative URLs ('//evil.com')
      - Backslash-based bypasses ('/\\evil.com', '\\evil.com')
      - Absolute URLs with scheme or netloc ('https://evil.com', 'http://...')
      - Paths containing backslashes
      - Dangerous URL schemes (javascript:, data:)
    """
    if not target or not isinstance(target, str):
        return False

    trimmed = target.strip()
    if not trimmed:
        return False

    # Must start with exactly one forward slash; reject protocol-relative or backslash-based paths
    if not trimmed.startswith("/") or trimmed.startswith("//") or trimmed.startswith("/\\") or trimmed.startswith("\\"):
        return False

    try:
        parts = urlsplit(trimmed)
    except ValueError:
        return False

    # For safe relative paths, both scheme and netloc (hostname) must be empty
    if parts.scheme or parts.netloc:
        return False

    # Disallow backslashes anywhere in the path
    if "\\" in parts.path:
        return False

    return True


def _validate_stored_filename(stored_filename: str, upload_folder: str) -> tuple[bool, str]:
    """
    Validates that stored_filename is a safe, single-element basename
    and that the resolved path is strictly contained inside upload_folder.
    Protects against directory traversal (../, absolute paths, special characters).
    """
    if not stored_filename:
        return False, ""

    # Strictly reject path separators or relative directory navigation
    if "/" in stored_filename or "\\" in stored_filename or ".." in stored_filename:
        return False, ""

    clean_name = os.path.basename(stored_filename)
    if not clean_name or clean_name != stored_filename:
        return False, ""

    resolved_folder = os.path.abspath(upload_folder)
    resolved_path = os.path.abspath(os.path.join(resolved_folder, clean_name))

    # Must reside strictly within the uploads directory
    try:
        if os.path.commonpath([resolved_folder, resolved_path]) != resolved_folder:
            return False, ""
    except ValueError:
        return False, ""

    if not os.path.isfile(resolved_path):
        return False, ""

    return True, resolved_path


@pdf_bp.route("/", methods=["GET"])
def index():
    """
    Renders the homepage containing the PDF upload interface.
    """
    return render_template("index.html")


@pdf_bp.route("/upload", methods=["POST"])
def upload_pdf():
    """
    Handles PDF file uploads:
    - Verifies 'pdf_file' exists in request.files
    - Validates file extension and PDF signature
    - Saves the file safely with a sanitized, collision-free filename
    - Extracts text page-by-page using pypdf (with latency measurement)
    - Saves document metadata to SQLite
    - Checks for existing cached summary
    - Renders the results page or flashes errors
    """
    upload_t0 = time.perf_counter()

    # Check if the file part is present in the request
    if "pdf_file" not in request.files:
        flash("No file part found in request.", "error")
        return redirect(url_for("pdf.index"))

    file = request.files["pdf_file"]

    # Validate the file (presence, extension, and PDF header signature)
    is_valid, error_message = validate_pdf_file(
        file, 
        current_app.config["ALLOWED_EXTENSIONS"]
    )

    if not is_valid:
        flash(error_message, "error")
        return redirect(url_for("pdf.index"))

    # Generate a collision-free and safe filename
    safe_filename = generate_safe_filename(file.filename)
    upload_folder = current_app.config["UPLOAD_FOLDER"]

    # Ensure uploads directory exists
    os.makedirs(upload_folder, exist_ok=True)

    # Save the file to the dedicated uploads folder
    destination_path = os.path.join(upload_folder, safe_filename)
    file.save(destination_path)

    # Extract text and metadata from the saved PDF with timing
    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(destination_path)
    extract_ms = (time.perf_counter() - extract_t0) * 1000

    # Handle extraction errors (e.g., password-protected or corrupted PDFs)
    if not extraction_result["success"]:
        flash(extraction_result["error"] or "Failed to extract text from the PDF.", "error")
        return redirect(url_for("pdf.index"))

    # Persist document metadata in the SQLite database
    insert_document(
        filename=safe_filename,
        original_filename=file.filename,
        page_count=extraction_result["total_pages"],
        character_count=extraction_result["total_characters"],
        status="processed"
    )

    total_upload_ms = (time.perf_counter() - upload_t0) * 1000
    logger.info(
        "[PERF] Upload & Extraction Complete | Filename: %s | Pages: %d | Chars: %d | Extraction: %.2f ms | Total Upload: %.2f ms",
        safe_filename,
        extraction_result["total_pages"],
        extraction_result["total_characters"],
        extract_ms,
        total_upload_ms
    )

    # Check if summary is already cached
    cached_row = get_document_summary(safe_filename)
    cached_summary_result = None
    if cached_row:
        cached_summary_result = {
            "success": True,
            "summary": cached_row["summary"],
            "error": None,
            "model_used": cached_row["model_used"],
            "cached": True
        }

    # Clear previous Q&A session history for newly uploaded file
    session.pop(f"qa_history_{safe_filename}", None)

    # Render results page with page-by-page extracted content and AI controls
    return render_template(
        "result.html",
        filename=file.filename,
        stored_filename=safe_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=extraction_result["pages"],
        summary_result=cached_summary_result,
        qa_history=[]
    )


@pdf_bp.route("/history", methods=["GET"])
def history():
    """
    Renders the document history page displaying previously processed documents, newest first.
    """
    documents = get_all_documents()
    return render_template("history.html", documents=documents)


@pdf_bp.route("/save-key", methods=["POST"])
def save_key():
    """
    Saves a Gemini API key into the local .env configuration securely.
    """
    api_key = request.form.get("gemini_api_key", "").strip()
    return_url = request.form.get("return_url", "").strip()

    if not api_key:
        flash("API key cannot be empty.", "error")
    else:
        save_api_key(api_key)
        flash("Google Gemini API key saved successfully into .env!", "success")

    if return_url and is_safe_redirect_url(return_url):
        return redirect(return_url)
    return redirect(url_for("pdf.index"))


@pdf_bp.route("/summarize", methods=["POST"])
def summarize():
    """
    Generates an executive summary for an uploaded document using Gemini.
    Checks SQLite cache first to eliminate duplicate API requests.
    Supports force-regeneration via 'regenerate=true'.
    """
    summarize_t0 = time.perf_counter()

    stored_filename = request.form.get("stored_filename", "").strip()
    original_filename = request.form.get("original_filename", stored_filename)
    regenerate = (request.form.get("regenerate") or "").strip().lower() in ("true", "1")

    is_valid, file_path = _validate_stored_filename(stored_filename, current_app.config["UPLOAD_FOLDER"])
    if not is_valid:
        flash("Invalid document reference or file not found.", "error")
        return redirect(url_for("pdf.index"))

    # 1. Check SQLite summary cache unless force-regenerating
    cached_row = get_document_summary(stored_filename) if not regenerate else None
    if cached_row:
        total_ms = (time.perf_counter() - summarize_t0) * 1000
        logger.info(
            "[PERF] Summarize CACHE HIT | File: %s | Model: %s | Latency: %.2f ms (Saved Gemini Call)",
            stored_filename,
            cached_row["model_used"],
            total_ms
        )
        summary_result = {
            "success": True,
            "summary": cached_row["summary"],
            "error": None,
            "model_used": cached_row["model_used"],
            "cached": True,
            "timing_ms": {"prep_ms": 0.0, "api_ms": 0.0, "total_ms": total_ms}
        }

        # Need pages for rendering result.html
        extract_t0 = time.perf_counter()
        extraction_result = extract_text_from_pdf(file_path)
        extract_ms = (time.perf_counter() - extract_t0) * 1000

        if request.is_json:
            return jsonify(summary_result)

        qa_history = session.get(f"qa_history_{stored_filename}", [])
        return render_template(
            "result.html",
            filename=original_filename,
            stored_filename=stored_filename,
            total_pages=extraction_result["total_pages"],
            total_characters=extraction_result["total_characters"],
            has_extractable_text=extraction_result["has_extractable_text"],
            pages=extraction_result["pages"],
            summary_result=summary_result,
            qa_history=qa_history
        )

    # 2. Cache miss or explicit regeneration: extract and call Gemini
    if regenerate:
        delete_document_summary(stored_filename)

    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(file_path)
    extract_ms = (time.perf_counter() - extract_t0) * 1000

    summary_result = summarize_document(extraction_result["pages"])
    summary_result["cached"] = False

    # 3. Cache successful summary in SQLite
    if summary_result["success"]:
        save_document_summary(
            filename=stored_filename,
            summary=summary_result["summary"],
            model_used=summary_result["model_used"]
        )

    total_summarize_ms = (time.perf_counter() - summarize_t0) * 1000
    logger.info(
        "[PERF] Summarize Finished | File: %s | Extraction: %.2f ms | Success: %s | Total Route: %.2f ms",
        stored_filename,
        extract_ms,
        summary_result["success"],
        total_summarize_ms
    )

    if request.is_json:
        return jsonify(summary_result)

    qa_history = session.get(f"qa_history_{stored_filename}", [])
    return render_template(
        "result.html",
        filename=original_filename,
        stored_filename=stored_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=extraction_result["pages"],
        summary_result=summary_result,
        qa_history=qa_history
    )


@pdf_bp.route("/ask", methods=["POST"])
def ask():
    """
    Answers a user question grounded in the uploaded document using Gemini.
    Logs structured timings for PDF extraction, chunk retrieval, and Gemini inference.
    """
    ask_t0 = time.perf_counter()

    stored_filename = request.form.get("stored_filename", "").strip()
    original_filename = request.form.get("original_filename", stored_filename)
    question = request.form.get("question", "").strip()

    is_valid, file_path = _validate_stored_filename(stored_filename, current_app.config["UPLOAD_FOLDER"])
    if not is_valid:
        flash("Invalid document reference or file not found.", "error")
        return redirect(url_for("pdf.index"))

    history_key = f"qa_history_{stored_filename}"

    # Handle explicit clear conversation request
    if request.form.get("clear_chat") == "true":
        session.pop(history_key, None)
        qa_history = []
        qa_result = None
        question = ""
    else:
        if not question:
            flash("Please enter a question about the document.", "error")
            return redirect(url_for("pdf.index"))
        extract_t0 = time.perf_counter()
        extraction_result = extract_text_from_pdf(file_path)
        extract_ms = (time.perf_counter() - extract_t0) * 1000

        qa_result = answer_pdf_question(extraction_result["pages"], question)

        total_ask_ms = (time.perf_counter() - ask_t0) * 1000
        logger.info(
            "[PERF] QA Finished | File: %s | Extraction: %.2f ms | Success: %s | Total Route: %.2f ms",
            stored_filename,
            extract_ms,
            qa_result["success"],
            total_ask_ms
        )

        qa_history = session.get(history_key, [])
        if qa_result.get("success"):
            qa_history.append({
                "question": question,
                "answer": qa_result.get("answer", ""),
                "page_citations": qa_result.get("page_citations", []),
                "evidence_passages": qa_result.get("evidence_passages", []),
                "model_used": qa_result.get("model_used", ""),
                "success": True
            })
            session[history_key] = qa_history

    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(file_path)

    # Also pass existing cached summary if available
    cached_row = get_document_summary(stored_filename)
    summary_result = None
    if cached_row:
        summary_result = {
            "success": True,
            "summary": cached_row["summary"],
            "model_used": cached_row["model_used"],
            "cached": True
        }

    if request.is_json:
        return jsonify(qa_result if qa_result else {"success": True, "cleared": True})

    return render_template(
        "result.html",
        filename=original_filename,
        stored_filename=stored_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=extraction_result["pages"],
        summary_result=summary_result,
        qa_result=qa_result,
        qa_history=qa_history,
        last_question=question
    )
