"""
Health & Operational Observability Routes
Provides liveness (/healthz) and readiness (/ready) endpoints for orchestration
and monitoring platforms (e.g., Kubernetes, Render, Railway, AWS ECS).
"""

import logging
from flask import Blueprint, jsonify
from database.db import get_db

logger = logging.getLogger("pdf_analyzer.observability")

health_bp = Blueprint("health", __name__)


@health_bp.route("/healthz", methods=["GET", "HEAD"])
def healthz():
    """
    Liveness probe: confirms the Flask application process is running.
    Executes no database queries, no external network calls, no Gemini API requests,
    no PDF processing, and no filesystem writes.
    Returns HTTP 200 with small safe JSON payload: {"status": "ok"}.
    """
    return jsonify({"status": "ok"}), 200


@health_bp.route("/ready", methods=["GET", "HEAD"])
def ready():
    """
    Readiness probe: verifies whether the application can access its essential local
    dependency: SQLite database at the configured DATABASE_PATH.
    Executes a lightweight read-only query (SELECT 1;).
    Does not create or modify application data, and does not invoke Gemini or external APIs.
    Returns HTTP 200 if ready, or HTTP 503 if the database check fails.
    Never exposes internal filesystem paths, exception details, or secrets in the response.
    """
    try:
        db = get_db()
        cursor = db.execute("SELECT 1;")
        row = cursor.fetchone()
        if row and row[0] == 1:
            return jsonify({
                "status": "ready",
                "checks": {
                    "database": "ok"
                }
            }), 200
        else:
            return jsonify({
                "status": "not_ready",
                "checks": {
                    "database": "error"
                }
            }), 503
    except Exception as exc:
        logger.warning("Readiness probe database check failed: %s", type(exc).__name__)
        return jsonify({
            "status": "not_ready",
            "checks": {
                "database": "error"
            }
        }), 503
