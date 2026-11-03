"""
Focused unit, integration, and security tests for:
- Health check endpoints (/healthz, /ready)
- Request ID generation, propagation (X-Request-ID), and injection prevention
- Structured request logging, non-negative duration, and sensitive-data exclusion
- Error correlation across redirects, 404/400/405 client errors, and 500 server errors
"""

import io
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from app import create_app
from config import Config
from observability import get_request_logger, generate_request_id, REQUEST_ID_REGEX


class ObservabilityTestConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    GEMINI_API_KEY = "test-mock-gemini-key-not-real"
    SECRET_KEY = "test-secret-key-for-observability-32chars"


class ObservabilityTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_upload_dir = os.path.join(self.test_dir, "uploads")
        self.test_instance_dir = os.path.join(self.test_dir, "instance")
        self.test_db_path = os.path.join(self.test_instance_dir, "test.db")

        class CustomConfig(ObservabilityTestConfig):
            UPLOAD_FOLDER = self.test_upload_dir
            INSTANCE_FOLDER = self.test_instance_dir
            DATABASE_PATH = self.test_db_path

        self.CustomConfig = CustomConfig
        self.app = create_app(CustomConfig)
        self.client = self.app.test_client()

        # Capture log records emitted to pdf_analyzer.request
        self.log_records = []
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_records.append(record)
        self.logger = get_request_logger()
        self.logger.addHandler(self.handler)

    def tearDown(self):
        if self.handler in self.logger.handlers:
            self.logger.removeHandler(self.handler)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # A. Health Check Probes (/healthz and /ready)
    # =========================================================================

    def test_healthz_returns_200(self):
        """1. /healthz returns HTTP 200 with status 'ok' on both GET and HEAD."""
        res_get = self.client.get("/healthz")
        self.assertEqual(res_get.status_code, 200)
        self.assertEqual(res_get.get_json(), {"status": "ok"})
        self.assertEqual(res_get.content_type, "application/json")

        res_head = self.client.head("/healthz")
        self.assertEqual(res_head.status_code, 200)
        self.assertEqual(res_head.data, b"")

    def test_healthz_works_when_database_unavailable(self):
        """2. /healthz responds 200 even when database is unavailable or corrupted."""
        with patch("routes.health_routes.get_db") as mock_get_db:
            mock_get_db.side_effect = sqlite3.OperationalError("Database disk is full or locked")
            res = self.client.get("/healthz")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.get_json(), {"status": "ok"})
            mock_get_db.assert_not_called()

    def test_healthz_does_not_invoke_gemini(self):
        """3. /healthz does not initialize or invoke Gemini / Google GenAI SDK."""
        with patch("google.genai.Client") as mock_genai_client, \
             patch("ai.llm_client.create_gemini_client") as mock_create_client:
            res = self.client.get("/healthz")
            self.assertEqual(res.status_code, 200)
            mock_genai_client.assert_not_called()
            mock_create_client.assert_not_called()

    def test_ready_returns_200_when_sqlite_accessible(self):
        """4. /ready returns HTTP 200 with checks.database = 'ok' when SQLite is accessible."""
        res = self.client.get("/ready")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {
            "status": "ready",
            "checks": {
                "database": "ok"
            }
        })

    def test_ready_returns_503_when_database_fails(self):
        """5. /ready returns HTTP 503 with checks.database = 'error' when database check fails."""
        with patch("routes.health_routes.get_db") as mock_get_db:
            mock_get_db.side_effect = sqlite3.OperationalError("Unable to open database file")
            res = self.client.get("/ready")
            self.assertEqual(res.status_code, 503)
            self.assertEqual(res.get_json(), {
                "status": "not_ready",
                "checks": {
                    "database": "error"
                }
            })

    def test_ready_uses_configured_external_database_path(self):
        """6. /ready executes against the configured external DATABASE_PATH."""
        custom_external_db = os.path.join(self.test_dir, "custom_ext", "prod.sqlite")

        class ExternalConfig(self.CustomConfig):
            DATABASE_PATH = custom_external_db

        app = create_app(ExternalConfig)
        client = app.test_client()

        res = client.get("/ready")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ready")
        # Verify the physical database file exists at the custom external location
        self.assertTrue(os.path.exists(custom_external_db))

    def test_ready_does_not_modify_database_records(self):
        """7. /ready executes read-only SELECT 1; and does not modify application records."""
        # Insert a canary document into documents table
        with self.app.app_context():
            from database.db import get_db
            db = get_db()
            db.execute(
                "INSERT INTO documents (filename, original_filename, page_count, character_count, status) "
                "VALUES ('canary.pdf', 'canary.pdf', 1, 100, 'ready');"
            )
            db.commit()
            count_before = db.execute("SELECT count(*) FROM documents;").fetchone()[0]

        # Call /ready multiple times
        for _ in range(5):
            res = self.client.get("/ready")
            self.assertEqual(res.status_code, 200)

        # Confirm count and state remain identical
        with self.app.app_context():
            db = get_db()
            count_after = db.execute("SELECT count(*) FROM documents;").fetchone()[0]
            self.assertEqual(count_before, count_after)

    def test_health_endpoints_return_safe_json_without_secrets_or_paths(self):
        """8. Both /healthz and /ready (success and error) return safe JSON without secrets or paths."""
        sensitive_strings = [
            self.test_db_path,
            self.test_upload_dir,
            self.test_instance_dir,
            "test-mock-gemini-key-not-real",
            "test-secret-key-for-observability-32chars",
            "OperationalError",
            "SELECT 1",
        ]

        # Check /healthz
        res_hz = self.client.get("/healthz")
        hz_text = res_hz.data.decode("utf-8")
        for s in sensitive_strings:
            self.assertNotIn(s, hz_text)

        # Check /ready (success)
        res_rd = self.client.get("/ready")
        rd_text = res_rd.data.decode("utf-8")
        for s in sensitive_strings:
            self.assertNotIn(s, rd_text)

        # Check /ready (failure)
        with patch("routes.health_routes.get_db") as mock_get_db:
            mock_get_db.side_effect = sqlite3.OperationalError(
                f"Disk error accessing {self.test_db_path}"
            )
            res_fail = self.client.get("/ready")
            self.assertEqual(res_fail.status_code, 503)
            fail_text = res_fail.data.decode("utf-8")
            for s in sensitive_strings:
                self.assertNotIn(s, fail_text)

    # =========================================================================
    # B. Request ID Generation and Propagation
    # =========================================================================

    def test_every_request_receives_unique_request_id(self):
        """9. Every incoming request receives a unique, non-empty request ID."""
        request_ids = set()
        for _ in range(10):
            res = self.client.get("/healthz")
            req_id = res.headers.get("X-Request-ID")
            self.assertIsNotNone(req_id)
            self.assertRegex(req_id, r"^[a-zA-Z0-9_\-]{16,64}$")
            request_ids.add(req_id)

        # All 10 IDs must be strictly unique
        self.assertEqual(len(request_ids), 10)

    def test_response_header_matches_logged_request_id(self):
        """10. The X-Request-ID header matches the exact request_id in the structured log."""
        self.log_records.clear()
        res = self.client.get("/healthz")
        header_id = res.headers.get("X-Request-ID")

        self.assertTrue(len(self.log_records) >= 1)
        record = self.log_records[-1]
        self.assertEqual(record.request_id, header_id)
        self.assertIn(f"request_id={header_id}", record.getMessage())

    def test_client_cannot_inject_newlines_into_log_ids(self):
        """11. Malicious client cannot inject CRLF / newlines or forged log entries via headers."""
        # 1. Direct unit test of generate_request_id with newline injection payloads
        crlf_payloads = [
            "evil_id\r\nINJECTED_ENTRY [CRITICAL] server compromised",
            "evil_id\nINJECTED_ENTRY",
            "evil_id\rINJECTED_ENTRY",
            "malicious\r\nSet-Cookie: admin=1",
            "id with spaces and $pecial chars",
            "a" * 128,  # Exceeds max length of 64
        ]

        for payload in crlf_payloads:
            with self.subTest(payload=payload):
                # When TRUST_INCOMING_REQUEST_ID is False (default)
                with self.app.app_context():
                    safe_id = generate_request_id(incoming_header=payload)
                    self.assertNotIn("\r", safe_id)
                    self.assertNotIn("\n", safe_id)
                    self.assertNotIn("INJECTED_ENTRY", safe_id)
                    self.assertTrue(REQUEST_ID_REGEX.match(safe_id))

        # 2. Test with TRUST_INCOMING_REQUEST_ID=True to verify strict sanitization regex
        class TrustingConfig(self.CustomConfig):
            TRUST_INCOMING_REQUEST_ID = True

        trusting_app = create_app(TrustingConfig)

        with trusting_app.app_context():
            # CRLF payload must be rejected by regex and fall back to clean UUID4
            safe_fallback = generate_request_id(incoming_header="evil\r\nattack")
            self.assertNotIn("\r", safe_fallback)
            self.assertNotIn("\n", safe_fallback)
            self.assertNotEqual(safe_fallback, "evil\r\nattack")
            self.assertTrue(REQUEST_ID_REGEX.match(safe_fallback))

            # Legitimate client ID matching regex must be accepted
            clean_client_id = "upstream-edge-gateway-12345"
            accepted_id = generate_request_id(incoming_header=clean_client_id)
            self.assertEqual(accepted_id, clean_client_id)

        # 3. HTTP Client test with safe header variation
        self.log_records.clear()
        res = self.client.get("/healthz", headers={"X-Request-ID": "valid-client-id-ignored-by-default"})
        assigned_id = res.headers.get("X-Request-ID")
        self.assertNotIn("\r", assigned_id)
        self.assertNotIn("\n", assigned_id)
        self.assertTrue(REQUEST_ID_REGEX.match(assigned_id))
        self.assertNotEqual(assigned_id, "valid-client-id-ignored-by-default")

    def test_redirect_responses_include_request_id(self):
        """12. HTTP 302 redirect responses include the X-Request-ID response header."""
        # /analysis/<nonexistent> redirects to index with 302
        res = self.client.get("/analysis/nonexistent_file_to_redirect.pdf")
        self.assertEqual(res.status_code, 302)
        self.assertIn("X-Request-ID", res.headers)
        self.assertTrue(len(res.headers["X-Request-ID"]) > 0)

    def test_error_responses_include_request_id(self):
        """13. Client and server error responses (400, 404, 405, 500) include X-Request-ID."""
        # 404 Not Found
        res_404 = self.client.get("/non-existent-probe-path")
        self.assertEqual(res_404.status_code, 404)
        self.assertIn("X-Request-ID", res_404.headers)

        # 405 Method Not Allowed
        res_405 = self.client.post("/healthz")
        self.assertEqual(res_405.status_code, 405)
        self.assertIn("X-Request-ID", res_405.headers)

        # 400 Bad Request (CSRF rejection on POST /save-key)
        res_400 = self.client.post("/save-key", data={"gemini_api_key": "some_key"})
        self.assertEqual(res_400.status_code, 400)
        self.assertIn("X-Request-ID", res_400.headers)

        # 500 Internal Server Error (unhandled exception in route)
        with patch.dict(self.app.view_functions, {"pdf.index": MagicMock(side_effect=RuntimeError("Simulated failure"))}):
            res_500 = self.client.get("/")
            self.assertEqual(res_500.status_code, 500)
            self.assertIn("X-Request-ID", res_500.headers)
            self.assertTrue(len(res_500.headers["X-Request-ID"]) > 0)

    def test_multiple_app_instances_do_not_share_request_context(self):
        """14. Multiple Flask app instances isolate their request context and request IDs."""
        app1 = create_app(self.CustomConfig)
        app2 = create_app(self.CustomConfig)

        client1 = app1.test_client()
        client2 = app2.test_client()

        res1 = client1.get("/healthz")
        res2 = client2.get("/healthz")

        id1 = res1.headers.get("X-Request-ID")
        id2 = res2.headers.get("X-Request-ID")

        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)

    # =========================================================================
    # C. Structured Request Logging
    # =========================================================================

    def test_method_path_status_request_id_and_duration_are_logged(self):
        """15. Access log captures method, path, status, request_id, and duration_ms."""
        self.log_records.clear()
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)

        record = self.log_records[-1]
        self.assertEqual(record.method, "GET")
        self.assertEqual(record.path, "/healthz")
        self.assertEqual(record.status_code, 200)
        self.assertEqual(record.request_id, res.headers["X-Request-ID"])
        self.assertTrue(hasattr(record, "duration_ms"))
        self.assertGreaterEqual(record.duration_ms, 0.0)

        # Formatted log string verification
        msg = record.getMessage()
        self.assertIn(f"request_id={res.headers['X-Request-ID']}", msg)
        self.assertIn("method=GET", msg)
        self.assertIn("path=/healthz", msg)
        self.assertIn("status=200", msg)
        self.assertIn("duration_ms=", msg)

    def test_duration_is_non_negative(self):
        """16. Logged duration_ms is strictly greater than or equal to 0.0."""
        self.log_records.clear()
        for path in ["/", "/healthz", "/ready"]:
            self.client.get(path)

        self.assertTrue(len(self.log_records) >= 3)
        for record in self.log_records:
            self.assertGreaterEqual(record.duration_ms, 0.0)

    def test_query_strings_are_not_logged(self):
        """17. Sensitive query string parameters are strictly excluded from request logs."""
        self.log_records.clear()
        secret_token = "secret-token-super-confidential-987"
        self.client.get(f"/healthz?token={secret_token}&api_key=should_not_appear")

        record = self.log_records[-1]
        self.assertEqual(record.path, "/healthz")
        self.assertNotIn(secret_token, record.getMessage())
        self.assertNotIn("should_not_appear", record.getMessage())
        self.assertNotIn("token=", record.getMessage())

    def test_request_bodies_cookies_and_keys_are_not_logged(self):
        """18. Request bodies, uploaded PDF payloads, session cookies, and API keys are not logged."""
        self.log_records.clear()

        secret_body_text = "CONFIDENTIAL_PATIENT_RECORDS_TEXT"
        fake_auth = "Bearer secret-bearer-token-12345"

        # POST with sensitive payload and headers
        self.client.post(
            "/save-key",
            data={"gemini_api_key": "sensitive-api-key-abcde"},
            headers={"Authorization": fake_auth},
        )

        record = self.log_records[-1]
        msg = record.getMessage()
        self.assertNotIn("sensitive-api-key-abcde", msg)
        self.assertNotIn("secret-bearer-token-12345", msg)
        self.assertNotIn(fake_auth, msg)
        self.assertNotIn(secret_body_text, msg)

    def test_repeated_application_creation_does_not_duplicate_log_entries(self):
        """19. Repeated create_app() calls do not duplicate logging handlers or log records."""
        # Initial handler count on request logger
        initial_handlers_count = len(self.logger.handlers)

        # Repeatedly create new Flask app instances
        app_a = create_app(self.CustomConfig)
        app_b = create_app(self.CustomConfig)
        app_c = create_app(self.CustomConfig)

        # Handler count on logger must not expand
        self.assertEqual(len(self.logger.handlers), initial_handlers_count)

        # Send 1 request to app_c
        self.log_records.clear()
        client_c = app_c.test_client()
        client_c.get("/healthz")

        # Must record exactly 1 log record for that single request
        request_entries = [r for r in self.log_records if r.path == "/healthz"]
        self.assertEqual(len(request_entries), 1)


if __name__ == "__main__":
    unittest.main()
