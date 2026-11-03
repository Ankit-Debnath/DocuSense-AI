"""
PDF Routes Blueprint
Handles PDF upload, history viewing, API key management, cached summarization, and grounded Q&A.
Features SQLite summary caching and structured performance timing logs.
"""

import hashlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify, session
from flask_wtf.csrf import generate_csrf

from pdf_processing.validator import validate_pdf_file, generate_safe_filename
from pdf_processing.extractor import extract_text_from_pdf
from config import is_production_mode
from database.models import (
    insert_document,
    get_all_documents,
    get_document_by_id,
    get_document_by_content_hash,
    get_document_by_filename,
    save_document_summary,
    get_document_summary,
    delete_document_summary,
    save_page_explanation,
    get_page_explanation,
    get_document_page_explanations,
    delete_page_explanation
)
from ai.analyzer import summarize_document, answer_pdf_question, explain_page_content
from ai.llm_client import save_api_key

# Structured route performance logger
logger = logging.getLogger("pdf_analyzer.routes")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

pdf_bp = Blueprint("pdf", __name__)

# Defaults for Q&A conversational turns retained in client-side session cookie
DEFAULT_MAX_QA_HISTORY_TURNS = 5
DEFAULT_MAX_QA_QUESTION_CHARS = 120
DEFAULT_MAX_QA_ANSWER_CHARS = 140


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

    resolved_folder = os.path.realpath(upload_folder)
    resolved_path = os.path.realpath(os.path.join(resolved_folder, clean_name))

    # Must not resolve to the directory itself
    if resolved_path == resolved_folder:
        return False, ""

    # Must reside strictly within the uploads directory
    try:
        if os.path.commonpath([resolved_folder, resolved_path]) != resolved_folder:
            return False, ""
    except ValueError:
        return False, ""

    if not os.path.isfile(resolved_path):
        return False, ""

    return True, resolved_path


def _enrich_pages_with_explanations(pages: List[Dict[str, Any]], stored_filename: str) -> List[Dict[str, Any]]:
    """
    Enriches page dictionaries with cached AI walkthrough explanations from SQLite.
    """
    cached_map = get_document_page_explanations(stored_filename)
    for p in pages:
        cached = cached_map.get(p["page_number"])
        p["explanation"] = cached["explanation"] if cached else None
        p["explanation_model"] = cached["model_used"] if cached else None
    return pages


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
    - Computes SHA-256 content hash and checks for duplicate uploads
    - Reuses existing stored document, summary, and page explanations if hash matches and file is valid
    - Otherwise saves the file safely with a sanitized, collision-free filename
    - Extracts text page-by-page using pypdf (with latency measurement)
    - Saves document metadata and content hash to SQLite
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

    # Compute SHA-256 hash of the uploaded PDF content
    hasher = hashlib.sha256()
    while chunk := file.stream.read(65536):
        hasher.update(chunk)
    content_hash = hasher.hexdigest()
    # Rewind stream immediately for any subsequent operations
    file.stream.seek(0)

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_folder, exist_ok=True)

    # Check for existing stored document with matching content hash
    existing_doc = get_document_by_content_hash(content_hash)
    if existing_doc:
        is_safe, existing_path = _validate_stored_filename(existing_doc["filename"], upload_folder)
        if is_safe and os.path.isfile(existing_path):
            extract_t0 = time.perf_counter()
            extraction_result = extract_text_from_pdf(existing_path)
            extract_ms = (time.perf_counter() - extract_t0) * 1000

            if extraction_result["success"]:
                safe_filename = existing_doc["filename"]
                total_upload_ms = (time.perf_counter() - upload_t0) * 1000
                logger.info(
                    "[PERF] Duplicate Upload CACHE HIT | File: %s | Reused: %s | Hash: %s | Extraction: %.2f ms | Total: %.2f ms",
                    file.filename,
                    safe_filename,
                    content_hash,
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

                # Render results page with page-by-page extracted content and cached AI controls
                enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], safe_filename)
                return render_template(
                    "result.html",
                    filename=existing_doc["original_filename"],
                    stored_filename=safe_filename,
                    total_pages=extraction_result["total_pages"],
                    total_characters=extraction_result["total_characters"],
                    has_extractable_text=extraction_result["has_extractable_text"],
                    pages=enriched_pages,
                    summary_result=cached_summary_result,
                    qa_history=[]
                )

    # Cache miss or missing stored file: generate a collision-free and safe filename
    safe_filename = generate_safe_filename(file.filename)

    # Save the file to the dedicated uploads folder
    destination_path = os.path.join(upload_folder, safe_filename)
    try:
        file.save(destination_path)
    except OSError as save_exc:
        logger.error(
            "[UPLOAD ERROR] Failed to save uploaded file '%s' to '%s': %s",
            file.filename,
            destination_path,
            save_exc
        )
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "[UPLOAD ERROR] Failed to clean up partial file '%s': %s",
                    destination_path,
                    cleanup_exc
                )
        flash("An error occurred while saving the uploaded file. Please try again.", "error")
        return redirect(url_for("pdf.index"))

    # Extract text and metadata from the saved PDF with timing
    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(destination_path)
    extract_ms = (time.perf_counter() - extract_t0) * 1000

    # Handle extraction errors (e.g., password-protected or corrupted PDFs)
    if not extraction_result["success"]:
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "[UPLOAD ERROR] Failed to clean up invalid PDF file '%s': %s",
                    destination_path,
                    cleanup_exc
                )
        flash(extraction_result["error"] or "Failed to extract text from the PDF.", "error")
        return redirect(url_for("pdf.index"))

    # Persist document metadata and content hash in the SQLite database
    try:
        insert_document(
            filename=safe_filename,
            original_filename=file.filename,
            page_count=extraction_result["total_pages"],
            character_count=extraction_result["total_characters"],
            status="processed",
            content_hash=content_hash
        )
    except Exception as db_exc:
        logger.error(
            "[UPLOAD ERROR] Database insertion failed for '%s' (%s). Cleaning up file '%s': %s",
            file.filename,
            safe_filename,
            destination_path,
            db_exc
        )
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "[UPLOAD ERROR] Failed to clean up orphaned file '%s': %s",
                    destination_path,
                    cleanup_exc
                )
        flash("A database error occurred while registering the document. Please try again.", "error")
        return redirect(url_for("pdf.index"))

    total_upload_ms = (time.perf_counter() - upload_t0) * 1000
    logger.info(
        "[PERF] Upload & Extraction Complete | Filename: %s | Pages: %d | Chars: %d | Hash: %s | Extraction: %.2f ms | Total Upload: %.2f ms",
        safe_filename,
        extraction_result["total_pages"],
        extraction_result["total_characters"],
        content_hash,
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
    enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], safe_filename)
    return render_template(
        "result.html",
        filename=file.filename,
        stored_filename=safe_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=enriched_pages,
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


@pdf_bp.route("/history/<int:doc_id>", methods=["GET"])
@pdf_bp.route("/analysis/<path:stored_filename>", methods=["GET"])
@pdf_bp.route("/analysis/<int:doc_id>", methods=["GET"])
def view_analysis(doc_id: Optional[int] = None, stored_filename: Optional[str] = None):
    """
    Renders the complete extraction and analysis results for an existing document from history.
    Validates document record in database and file on disk to prevent path traversal or missing data errors.
    """
    if doc_id is not None:
        doc = get_document_by_id(doc_id)
    elif stored_filename:
        if stored_filename.isdigit():
            doc = get_document_by_id(int(stored_filename))
        else:
            doc = None
        if not doc:
            doc = get_document_by_filename(stored_filename)
    else:
        doc = None

    if not doc:
        flash("Document record not found in history.", "error")
        return redirect(url_for("pdf.history"))

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    is_valid, file_path = _validate_stored_filename(doc["filename"], upload_folder)
    if not is_valid:
        flash("Document file not found on disk or invalid filename.", "error")
        return redirect(url_for("pdf.history"))

    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(file_path)
    extract_ms = (time.perf_counter() - extract_t0) * 1000

    if not extraction_result["success"]:
        flash(extraction_result["error"] or "Failed to read document.", "error")
        return redirect(url_for("pdf.history"))

    cached_row = get_document_summary(doc["filename"])
    cached_summary_result = None
    if cached_row:
        cached_summary_result = {
            "success": True,
            "summary": cached_row["summary"],
            "error": None,
            "model_used": cached_row["model_used"],
            "cached": True
        }

    qa_history = session.get(f"qa_history_{doc['filename']}", [])
    enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], doc["filename"])

    return render_template(
        "result.html",
        filename=doc["original_filename"],
        stored_filename=doc["filename"],
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=enriched_pages,
        summary_result=cached_summary_result,
        qa_history=qa_history
    )


@pdf_bp.route("/save-key", methods=["POST"])
def save_key():
    """
    Handles API key management.
    In production environments, modifying server credentials via public web requests
    is strictly forbidden to protect against credential tampering.
    In development environments, allows local updates only if explicitly permitted.
    """
    is_prod = is_production_mode(current_app.config)
    web_config_enabled = current_app.config.get("ENABLE_WEB_KEY_CONFIG", not is_prod)

    if is_prod or not web_config_enabled:
        err_msg = "API key modification via web request is disabled in production. Please set GEMINI_API_KEY in your environment."
        if request.is_json:
            return jsonify({"success": False, "error": err_msg}), 403
        flash(err_msg, "error")
        return render_template("index.html"), 403

    api_key = request.form.get("gemini_api_key", "").strip()
    return_url = request.form.get("return_url", "").strip()

    if not api_key:
        flash("API key cannot be empty.", "error")
    else:
        success = save_api_key(api_key)
        if success:
            flash("Google Gemini API key saved successfully into .env!", "success")
        else:
            flash("Failed to save API key. Please check filesystem permissions or set GEMINI_API_KEY in your environment.", "error")

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
        enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], stored_filename)
        return render_template(
            "result.html",
            filename=original_filename,
            stored_filename=stored_filename,
            total_pages=extraction_result["total_pages"],
            total_characters=extraction_result["total_characters"],
            has_extractable_text=extraction_result["has_extractable_text"],
            pages=enriched_pages,
            summary_result=summary_result,
            qa_history=qa_history
        )

    # 2. Cache miss or explicit regeneration: extract and call Gemini
    if regenerate:
        delete_document_summary(stored_filename)

    extract_t0 = time.perf_counter()
    extraction_result = extract_text_from_pdf(file_path)
    extract_ms = (time.perf_counter() - extract_t0) * 1000

    summary_result = summarize_document(
        extraction_result["pages"],
        document_title=original_filename,
        total_pages=extraction_result["total_pages"]
    )
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
    enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], stored_filename)
    return render_template(
        "result.html",
        filename=original_filename,
        stored_filename=stored_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=enriched_pages,
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

    # Support both Form submission and JSON payload
    data = request.get_json(silent=True) if request.is_json else request.form
    if data is None:
        data = request.form

    stored_filename = str(data.get("stored_filename", "")).strip()
    original_filename = str(data.get("original_filename", stored_filename)).strip()
    question = str(data.get("question", "")).strip()

    is_valid, file_path = _validate_stored_filename(stored_filename, current_app.config["UPLOAD_FOLDER"])
    if not is_valid:
        flash("Invalid document reference or file not found.", "error")
        return redirect(url_for("pdf.index"))

    history_key = f"qa_history_{stored_filename}"

    # Handle explicit clear conversation request
    is_clear = str(data.get("clear_chat", "")).strip().lower() in ("true", "1")
    if is_clear:
        session.pop(history_key, None)
        qa_history = []
        qa_result = None
        question = ""
        if request.is_json or request.headers.get("Accept") == "application/json":
            return jsonify({"success": True, "cleared": True})
        extract_t0 = time.perf_counter()
        extraction_result = extract_text_from_pdf(file_path)
    else:
        if not question:
            flash("Please enter a question about the document.", "error")
            return redirect(url_for("pdf.index"))
        extract_t0 = time.perf_counter()
        extraction_result = extract_text_from_pdf(file_path)
        extract_ms = (time.perf_counter() - extract_t0) * 1000

        # Resolve document record ID for semantic & hybrid retrieval
        doc_record = get_document_by_filename(stored_filename)
        if not doc_record:
            doc_id = insert_document(
                filename=stored_filename,
                original_filename=original_filename,
                page_count=extraction_result["total_pages"],
                character_count=extraction_result["total_characters"],
                status="processed"
            )
        else:
            doc_id = doc_record["id"]

        qa_result = answer_pdf_question(
            extraction_result["pages"],
            question,
            document_id=doc_id,
            file_path=file_path,
            total_pages=extraction_result["total_pages"]
        )

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
            max_turns = current_app.config.get("MAX_QA_HISTORY_TURNS", DEFAULT_MAX_QA_HISTORY_TURNS)
            max_q_chars = current_app.config.get("MAX_QA_QUESTION_CHARS", DEFAULT_MAX_QA_QUESTION_CHARS)
            max_a_chars = current_app.config.get("MAX_QA_ANSWER_CHARS", DEFAULT_MAX_QA_ANSWER_CHARS)

            raw_answer = qa_result.get("answer", "").strip()
            compact_answer = raw_answer[:max_a_chars] if len(raw_answer) > max_a_chars else raw_answer

            citations_for_history = qa_result.get("verified_citations") or qa_result.get("page_citations") or []

            qa_history.append({
                "question": question[:max_q_chars],
                "answer": compact_answer,
                "page_citations": citations_for_history[:10],
                "verified_citations": (qa_result.get("verified_citations") or [])[:10],
                "citation_status": qa_result.get("citation_status", "not_applicable"),
                "insufficient_evidence": qa_result.get("insufficient_evidence", False),
                "model_used": qa_result.get("model_used", ""),
                "success": True
            })
            # Bound history entries to maintain safe client-side cookie limits
            if len(qa_history) > max_turns:
                qa_history = qa_history[-max_turns:]

            # Prune any old qa_history keys from other documents to prevent cookie accumulation
            for k in list(session.keys()):
                if k.startswith("qa_history_") and k != history_key:
                    session.pop(k, None)

            session[history_key] = qa_history

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

    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify(qa_result if qa_result else {"success": True, "cleared": True})

    # Prepare display history: current turn displays full response while cookie retains bounded excerpt
    rendered_history = [dict(item) for item in qa_history]
    if qa_result and qa_result.get("success") and rendered_history:
        rendered_history[-1]["answer"] = qa_result.get("answer", "")
        rendered_history[-1]["evidence_passages"] = qa_result.get("evidence_passages", [])
        rendered_history[-1]["citation_status"] = qa_result.get("citation_status", "not_applicable")
        rendered_history[-1]["verified_citations"] = qa_result.get("verified_citations", [])
    elif qa_result and qa_result.get("citation_status") == "invalid":
        # Unsuccessful invalid turns are not persisted to session, but must be rendered for the active response
        rendered_history.append({
            "question": question,
            "answer": qa_result.get("answer", ""),
            "page_citations": qa_result.get("page_citations", []),
            "verified_citations": qa_result.get("verified_citations", []),
            "citation_status": "invalid",
            "insufficient_evidence": False,
            "evidence_passages": qa_result.get("evidence_passages", []),
            "model_used": qa_result.get("model_used", ""),
            "success": False
        })

    enriched_pages = _enrich_pages_with_explanations(extraction_result["pages"], stored_filename)
    return render_template(
        "result.html",
        filename=original_filename,
        stored_filename=stored_filename,
        total_pages=extraction_result["total_pages"],
        total_characters=extraction_result["total_characters"],
        has_extractable_text=extraction_result["has_extractable_text"],
        pages=enriched_pages,
        summary_result=summary_result,
        qa_result=qa_result,
        qa_history=rendered_history,
        last_question=question
    )


@pdf_bp.route("/explain-page", methods=["POST"])
def explain_page():
    """
    Generates and caches a publication-grade walkthrough of an individual PDF page.
    Returns JSON response with the structured explanation and cache status.
    """
    explain_t0 = time.perf_counter()

    # Support both Form submission and JSON payload
    data = request.get_json(silent=True) or request.form

    stored_filename = (data.get("stored_filename") or "").strip()
    original_filename = (data.get("original_filename") or stored_filename).strip()
    page_num_str = str(data.get("page_number", "")).strip()
    regenerate = str(data.get("regenerate", "")).strip().lower() in ("true", "1")

    if not page_num_str.isdigit():
        return jsonify({"success": False, "error": "Invalid or missing page number."}), 400

    page_number = int(page_num_str)

    is_valid, file_path = _validate_stored_filename(stored_filename, current_app.config["UPLOAD_FOLDER"])
    if not is_valid:
        return jsonify({"success": False, "error": "Invalid document reference or file not found."}), 404

    # 1. Check SQLite cache unless force-regenerating
    cached = get_page_explanation(stored_filename, page_number) if not regenerate else None
    if cached:
        total_ms = (time.perf_counter() - explain_t0) * 1000
        logger.info(
            "[PERF] Page Explain CACHE HIT | File: %s | Page: %d | Latency: %.2f ms",
            stored_filename,
            page_number,
            total_ms
        )
        return jsonify({
            "success": True,
            "page_number": page_number,
            "explanation": cached["explanation"],
            "model_used": cached["model_used"],
            "cached": True,
            "timing_ms": {"api_ms": 0.0, "total_ms": total_ms}
        })

    # 2. Extract text from PDF file
    extraction_result = extract_text_from_pdf(file_path)
    if not extraction_result["success"]:
        return jsonify({"success": False, "error": extraction_result["error"] or "Failed to read PDF."}), 500

    target_page = next((p for p in extraction_result["pages"] if p["page_number"] == page_number), None)
    if not target_page:
        return jsonify({"success": False, "error": f"Page {page_number} not found in document."}), 404

    if regenerate:
        delete_page_explanation(stored_filename, page_number)

    # 3. Generate structured explanation with Gemini
    result = explain_page_content(
        page_number=page_number,
        page_text=target_page["text"],
        total_pages=extraction_result["total_pages"],
        document_title=original_filename
    )

    # 4. Cache successful explanation in SQLite
    if result["success"]:
        save_page_explanation(
            filename=stored_filename,
            page_number=page_number,
            explanation=result["explanation"],
            model_used=result["model_used"]
        )

    total_ms = (time.perf_counter() - explain_t0) * 1000
    result["cached"] = False
    if "timing_ms" in result:
        result["timing_ms"]["total_ms"] = total_ms

    logger.info(
        "[PERF] Page Explain Complete | File: %s | Page: %d | Success: %s | Total: %.2f ms",
        stored_filename,
        page_number,
        result["success"],
        total_ms
    )

    return jsonify(result)


@pdf_bp.route("/csrf-token", methods=["GET"])
def get_csrf_token():
    """
    Returns a fresh CSRF token for asynchronous client requests.
    Enables automatic token renewal without requiring a full page refresh.
    """
    token = generate_csrf()
    return jsonify({"success": True, "csrf_token": token})
