"""
Operational Observability & Request Correlation Module
Provides:
- Cryptographic / UUID4 request correlation ID generation and propagation (X-Request-ID)
- Structured access and latency logging (standard Python logging infrastructure)
- Sensitive data exclusion (query strings, bodies, cookies, and keys are never logged)
- Correlated unhandled server exception logging and safe error response generation
"""

import logging
import re
import sys
import time
import uuid
from typing import Optional
from flask import Flask, g, request, jsonify, render_template, flash, has_request_context, current_app
from werkzeug.exceptions import HTTPException

# Request ID validation: alphanumeric, underscores, hyphens, maximum 64 characters.
# Prevents newline injection (CRLF / log forging) and unbounded header sizes.
REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# Dedicated logger for request lifecycle and observability
_logger_initialized = False


def get_request_logger() -> logging.Logger:
    """
    Returns the structured request logger. Configures a stdout handler only once
    to prevent duplicate log records when create_app() is called repeatedly in tests.
    """
    global _logger_initialized
    logger = logging.getLogger("pdf_analyzer.request")
    if not _logger_initialized:
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        _logger_initialized = True
    return logger


def generate_request_id(incoming_header: Optional[str] = None) -> str:
    """
    Generates a secure, unique request correlation ID.
    By default, server-generated UUID4 (hex) is authoritative to guarantee uniqueness,
    prevent client forgery, and eliminate log-injection risks.
    If TRUST_INCOMING_REQUEST_ID config is explicitly enabled and a valid X-Request-ID
    header is present matching ^[a-zA-Z0-9_-]{1,64}$, that sanitized ID is used.
    Otherwise, a fresh UUID4 is always generated.
    """
    if incoming_header is None and has_request_context():
        incoming_header = request.headers.get("X-Request-ID")

    trust_client = False
    try:
        if current_app:
            trust_client = bool(current_app.config.get("TRUST_INCOMING_REQUEST_ID", False))
    except (RuntimeError, AttributeError):
        trust_client = False

    if trust_client and incoming_header:
        clean = incoming_header.strip()
        if REQUEST_ID_REGEX.match(clean):
            return clean

    return uuid.uuid4().hex


def get_request_id() -> str:
    """
    Retrieves the current request ID from flask.g if inside an active request context.
    Returns empty string if outside request context or not yet assigned.
    """
    if has_request_context():
        return getattr(g, "request_id", "")
    return ""


def init_observability(app: Flask) -> None:
    """
    Registers request lifecycle hooks on the Flask application instance:
    - before_request: generates and binds unique request ID and records start time.
    - after_request: attaches X-Request-ID header and records structured access log.
    - errorhandler(Exception): logs unhandled exceptions with request ID and returns safe 500 response.
    """
    request_logger = get_request_logger()

    @app.before_request
    def assign_request_id():
        g.request_id = generate_request_id()
        g.request_start_time = time.perf_counter()

    @app.after_request
    def record_request_observability(response):
        # Ensure request ID is present on response header and flask.g
        req_id = getattr(g, "request_id", None)
        if not req_id:
            req_id = generate_request_id()
            g.request_id = req_id

        response.headers["X-Request-ID"] = req_id

        # Compute request duration in milliseconds (strictly non-negative)
        start_time = getattr(g, "request_start_time", None)
        if start_time is not None:
            duration_ms = max(0.0, (time.perf_counter() - start_time) * 1000.0)
        else:
            duration_ms = 0.0

        # Structured request logging
        status_code = response.status_code
        method = request.method
        # Strictly use request.path to omit query strings and prevent sensitive param logging
        path = request.path

        log_msg = (
            f"request_id={req_id} "
            f"method={method} "
            f"path={path} "
            f"status={status_code} "
            f"duration_ms={duration_ms:.2f}ms"
        )
        extra = {
            "request_id": req_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
        }

        if status_code < 400:
            request_logger.info(log_msg, extra=extra)
        elif status_code < 500:
            request_logger.warning(log_msg, extra=extra)
        else:
            request_logger.error(log_msg, extra=extra)

        return response

    @app.errorhandler(Exception)
    def handle_unhandled_exception(error):
        # Pass standard HTTP exceptions (404, 405, 400, etc.) through to Flask default handling
        if isinstance(error, HTTPException):
            return error

        req_id = getattr(g, "request_id", None)
        if not req_id:
            req_id = generate_request_id()
            g.request_id = req_id

        # Correlated error logging with stack trace on server side
        request_logger.error(
            f"request_id={req_id} unhandled_exception={type(error).__name__}",
            exc_info=True,
            extra={"request_id": req_id, "error_type": type(error).__name__}
        )

        is_ajax_or_api = (
            request.is_json
            or request.path in ("/explain-page", "/healthz", "/ready")
            or request.path.startswith("/api/")
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in request.headers.get("Accept", "")
        )
        if is_ajax_or_api:
            resp = jsonify({
                "success": False,
                "error": "An internal server error occurred.",
                "request_id": req_id
            })
            resp.status_code = 500
            return resp

        try:
            flash("An unexpected server error occurred. Please try again.", "error")
            return render_template("index.html"), 500
        except Exception:
            return (
                "An unexpected server error occurred. Please try again.",
                500,
                {"Content-Type": "text/plain; charset=utf-8"}
            )
